"""Application service for processing and persisting PDF documents."""

import asyncio
import hashlib
import logging
import os
import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from garden_center.config import Settings
from garden_center.document_processing import (
    PDF_CONTENT_TYPE,
    chunk_pages,
    extract_pdf_pages,
    validate_upload,
)
from garden_center.models import Document, DocumentChunk, ProcessingStatus

logger = logging.getLogger(__name__)


class DuplicateDocumentError(ValueError):
    """Raised when the same PDF content has already been uploaded."""


def _process_document(content: bytes, settings: Settings):
    """Extract and chunk PDF content."""
    pages = extract_pdf_pages(
        content,
        max_pages=settings.max_pdf_pages,
        max_characters=settings.max_extracted_characters,
    )
    return chunk_pages(
        pages,
        chunk_size=settings.chunk_size_tokens,
        overlap=settings.chunk_overlap_tokens,
    )


def _document_path(storage_path: Path, document_id: uuid.UUID) -> Path:
    return storage_path / f"{document_id}.pdf"


def _store_file(storage_path: Path, document_id: uuid.UUID, content: bytes) -> Path:
    storage_path.mkdir(parents=True, exist_ok=True)
    destination = _document_path(storage_path, document_id)
    temporary = destination.with_suffix(".pdf.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return destination


async def create_document(
    session: AsyncSession,
    upload: UploadFile,
    settings: Settings,
) -> Document:
    """Validate, extract, chunk, store, and persist one PDF atomically."""
    content = await upload.read(settings.max_upload_size_bytes + 1)
    validate_upload(
        upload.filename,
        upload.content_type,
        content,
        settings.max_upload_size_bytes,
    )
    chunks = await asyncio.to_thread(_process_document, content, settings)
    content_hash = hashlib.sha256(content).hexdigest()

    existing_id = await session.scalar(
        select(Document.id).where(Document.content_hash == content_hash)
    )
    if existing_id is not None:
        raise DuplicateDocumentError("This PDF has already been uploaded.")

    document = Document(
        id=uuid.uuid4(),
        filename=upload.filename or "document.pdf",
        content_type=PDF_CONTENT_TYPE,
        file_size=len(content),
        content_hash=content_hash,
        processing_status=ProcessingStatus.PROCESSING,
        chunks=[
            DocumentChunk(
                chunk_index=chunk.chunk_index,
                text_content=chunk.text,
                source_page=chunk.source_page,
                token_count=chunk.token_count,
            )
            for chunk in chunks
        ],
    )
    destination = await asyncio.to_thread(
        _store_file, settings.document_storage_path, document.id, content
    )
    session.add(document)

    try:
        await session.commit()
        await session.refresh(document)
    except IntegrityError as exc:
        await session.rollback()
        await asyncio.to_thread(destination.unlink, missing_ok=True)
        if getattr(exc.orig, "sqlstate", None) == "23505":
            raise DuplicateDocumentError("This PDF has already been uploaded.") from exc
        raise
    except Exception:
        await session.rollback()
        await asyncio.to_thread(destination.unlink, missing_ok=True)
        raise

    logger.info("Processed document %s with %d chunks", document.id, len(chunks))
    return await get_document(session, document.id) or document


async def list_documents(
    session: AsyncSession,
    limit: int,
    offset: int,
) -> tuple[list[tuple[Document, int]], int]:
    """Return one document page with database-computed chunk counts."""
    total = await session.scalar(select(func.count()).select_from(Document))
    result = await session.execute(
        select(Document, func.count(DocumentChunk.id).label("chunk_count"))
        .outerjoin(DocumentChunk, DocumentChunk.document_id == Document.id)
        .group_by(Document.id)
        .order_by(Document.created_at.desc(), Document.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return [(document, chunk_count) for document, chunk_count in result], total or 0


async def get_document(
    session: AsyncSession, document_id: uuid.UUID
) -> Document | None:
    """Return one document with chunks in deterministic order."""
    return await session.scalar(
        select(Document)
        .where(Document.id == document_id)
        .options(selectinload(Document.chunks))
    )


async def delete_document(
    session: AsyncSession,
    document: Document,
    settings: Settings,
) -> None:
    """Delete database data first, then remove the stored PDF."""
    await session.delete(document)
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        raise

    path = _document_path(settings.document_storage_path, document.id)
    try:
        await asyncio.to_thread(path.unlink, missing_ok=True)
    except OSError:
        logger.exception("Could not remove stored PDF for document %s", document.id)
