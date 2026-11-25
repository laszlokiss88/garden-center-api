"""Application service for processing and persisting PDF documents."""

import asyncio
import hashlib
import logging
import uuid

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
from garden_center.file_storage import FileStorage
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


async def _delete_after_failed_persistence(
    storage: FileStorage, storage_key: str
) -> None:
    """Cleanup that preserves the original database error."""
    try:
        await storage.delete(storage_key)
    except OSError:
        logger.exception(
            "Could not remove stored file after database persistence failure: %s",
            storage_key,
        )


async def create_document(
    session: AsyncSession,
    upload: UploadFile,
    settings: Settings,
    storage: FileStorage,
) -> Document:
    """Validate, process, store, and persist one PDF with failure cleanup."""
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

    stored_file = await storage.upload(
        content=content,
        content_type=PDF_CONTENT_TYPE,
        directory="documents",
        extension=".pdf",
    )
    document = Document(
        id=uuid.uuid4(),
        filename=upload.filename or "document.pdf",
        content_type=PDF_CONTENT_TYPE,
        file_size=len(content),
        content_hash=content_hash,
        storage_key=stored_file.storage_key,
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
    session.add(document)

    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        await _delete_after_failed_persistence(storage, document.storage_key)
        if getattr(exc.orig, "sqlstate", None) == "23505":
            raise DuplicateDocumentError("This PDF has already been uploaded.") from exc
        raise
    except Exception:
        await session.rollback()
        raise

    await session.refresh(document)
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
    storage: FileStorage,
) -> None:
    """Delete database data first, then remove the stored PDF."""
    await session.delete(document)
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        raise

    try:
        await storage.delete(document.storage_key)
    except OSError:
        logger.exception(
            "Could not remove stored file for deleted document %s: %s",
            document.id,
            document.storage_key,
        )
