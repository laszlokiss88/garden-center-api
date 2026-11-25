"""HTTP routes for PDF document management."""

import logging
import uuid
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from garden_center.config import Settings, get_settings
from garden_center.database import get_database_session
from garden_center.dependencies import get_embedding_provider, get_file_storage
from garden_center.document_processing import DocumentValidationError
from garden_center.document_service import (
    DuplicateDocumentError,
    create_document,
    delete_document,
    get_document,
    list_documents,
)
from garden_center.embeddings import EmbeddingProvider, EmbeddingProviderError
from garden_center.file_storage import FileStorage
from garden_center.models import Document
from garden_center.schemas import (
    DocumentChunkResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentResponse,
    ErrorResponse,
)
from garden_center.search_service import index_document_chunks

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
_DOCUMENT_STORAGE_ERRORS = (SQLAlchemyError, OSError)
SessionDependency = Annotated[AsyncSession, Depends(get_database_session)]
SettingsDependency = Annotated[Settings, Depends(get_settings)]
ProviderDependency = Annotated[EmbeddingProvider, Depends(get_embedding_provider)]
StorageDependency = Annotated[FileStorage, Depends(get_file_storage)]


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


def _document_response(
    document: Document, chunk_count: int | None = None
) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        content_type=document.content_type,
        file_size=document.file_size,
        content_hash=document.content_hash,
        processing_status=document.processing_status,
        created_at=document.created_at,
        updated_at=document.updated_at,
        chunk_count=len(document.chunks) if chunk_count is None else chunk_count,
    )


def _document_detail(document: Document) -> DocumentDetailResponse:
    summary = _document_response(document)
    return DocumentDetailResponse(
        **summary.model_dump(),
        chunks=[
            DocumentChunkResponse.model_validate(chunk) for chunk in document.chunks
        ],
    )


async def _cleanup_failed_upload(
    session: AsyncSession,
    document_id: uuid.UUID,
    storage: FileStorage,
) -> None:
    """Remove all persisted artifacts for an upload that did not finish."""
    try:
        document = await get_document(session, document_id)
        if document is not None:
            await delete_document(session, document, storage)
    except SQLAlchemyError:
        logger.exception("Could not clean up failed document %s", document_id)
        raise _error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "database_unavailable",
            "The failed document could not be cleaned up.",
        ) from None


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def upload_document(
    session: SessionDependency,
    settings: SettingsDependency,
    provider: ProviderDependency,
    storage: StorageDependency,
    response: Response,
    file: Annotated[UploadFile, File(description="PDF document")],
) -> DocumentResponse:
    """Upload, extract, chunk, and persist a PDF."""
    try:
        try:
            document = await create_document(session, file, settings, storage)
            document_id = document.id
        except DocumentValidationError as exc:
            raise _error(status.HTTP_400_BAD_REQUEST, exc.code, exc.message) from exc
        except DuplicateDocumentError as exc:
            raise _error(
                status.HTTP_409_CONFLICT, "duplicate_document", str(exc)
            ) from exc
        except _DOCUMENT_STORAGE_ERRORS:
            logger.exception("Document upload failed")
            raise _error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "document_storage_unavailable",
                "The document could not be stored.",
            ) from None

        try:
            await index_document_chunks(
                session,
                provider,
                settings.embedding_batch_size,
                document,
            )
            indexed_document = await get_document(session, document_id)
            if indexed_document is None:
                raise SQLAlchemyError("Indexed document could not be reloaded")
            document = indexed_document
        except EmbeddingProviderError as exc:
            logger.warning("Document %s indexing failed: %s", document_id, exc.code)
            await _cleanup_failed_upload(session, document_id, storage)
            raise _error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                exc.code,
                exc.message,
            ) from exc
        except SQLAlchemyError:
            logger.exception("Document %s embedding persistence failed", document_id)
            await _cleanup_failed_upload(session, document_id, storage)
            raise _error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "database_unavailable",
                "Document embeddings could not be persisted.",
            ) from None
    finally:
        await file.close()
    response.headers["Location"] = f"/api/v1/documents/{document.id}"
    return _document_response(document)


@router.get("", response_model=DocumentListResponse)
async def get_documents(
    session: SessionDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentListResponse:
    """List uploaded documents newest first with offset pagination."""
    try:
        documents, total = await list_documents(session, limit, offset)
    except SQLAlchemyError:
        logger.exception("Could not list documents")
        raise _error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "database_unavailable",
            "Documents are temporarily unavailable.",
        ) from None
    return DocumentListResponse(
        items=[
            _document_response(document, chunk_count)
            for document, chunk_count in documents
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_document_detail(
    document_id: uuid.UUID, session: SessionDependency
) -> DocumentDetailResponse:
    """Return document metadata and extracted chunks."""
    try:
        document = await get_document(session, document_id)
    except SQLAlchemyError:
        logger.exception("Could not load document %s", document_id)
        raise _error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "database_unavailable",
            "The document is temporarily unavailable.",
        ) from None
    if document is None:
        raise _error(status.HTTP_404_NOT_FOUND, "document_not_found", "Not found.")
    return _document_detail(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_document(
    document_id: uuid.UUID,
    session: SessionDependency,
    storage: StorageDependency,
) -> None:
    """Delete a document, its chunks and embeddings, and its stored PDF."""
    try:
        document = await get_document(session, document_id)
    except SQLAlchemyError:
        logger.exception("Could not load document %s for deletion", document_id)
        raise _error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "database_unavailable",
            "The document could not be deleted.",
        ) from None
    if document is None:
        raise _error(status.HTTP_404_NOT_FOUND, "document_not_found", "Not found.")
    try:
        await delete_document(session, document, storage)
    except SQLAlchemyError:
        logger.exception("Could not delete document %s", document_id)
        raise _error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "database_unavailable",
            "The document could not be deleted.",
        ) from None
