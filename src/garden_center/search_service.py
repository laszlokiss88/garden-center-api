"""Application services for chunk indexing and semantic retrieval."""

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import cast, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from garden_center.embeddings import EmbeddingProvider, EmbeddingProviderError
from garden_center.models import (
    ChunkEmbedding,
    Document,
    DocumentChunk,
    ProcessingStatus,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class IndexResult:
    """Counts returned after an idempotent indexing operation."""

    indexed_chunks: int
    skipped_chunks: int


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A semantically similar chunk and its source metadata."""

    document_id: uuid.UUID
    filename: str
    page_number: int
    chunk_index: int
    text: str
    similarity: float


def _batches(  # noqa: UP047 - keep compatible with pre-3.12 tooling
    values: Sequence[T], size: int
) -> list[Sequence[T]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


async def index_document_chunks(
    session: AsyncSession,
    provider: EmbeddingProvider,
    batch_size: int,
    document_id: uuid.UUID | None = None,
) -> IndexResult:
    """Generate only missing embeddings for the selected embedding model."""
    document_filter = (
        DocumentChunk.document_id == document_id if document_id is not None else None
    )
    total_query = select(func.count()).select_from(DocumentChunk)
    if document_filter is not None:
        total_query = total_query.where(document_filter)
    total_chunks = int(await session.scalar(total_query) or 0)

    already_indexed = (
        select(ChunkEmbedding.id)
        .where(
            ChunkEmbedding.chunk_id == DocumentChunk.id,
            ChunkEmbedding.embedding_provider == provider.provider_name,
            ChunkEmbedding.embedding_model == provider.model,
        )
        .exists()
    )
    missing_query = (
        select(DocumentChunk)
        .where(~already_indexed)
        .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
    )
    if document_filter is not None:
        missing_query = missing_query.where(document_filter)
    missing_chunks = list(await session.scalars(missing_query))
    affected_document_ids = {chunk.document_id for chunk in missing_chunks}

    if not affected_document_ids:
        if document_id is not None and total_chunks > 0:
            await _set_document_statuses(
                session, {document_id}, ProcessingStatus.COMPLETED
            )
        return IndexResult(indexed_chunks=0, skipped_chunks=total_chunks)

    await _set_document_statuses(
        session, affected_document_ids, ProcessingStatus.PROCESSING
    )

    indexed = 0
    try:
        for batch in _batches(missing_chunks, batch_size):
            vectors = await provider.embed([chunk.text_content for chunk in batch])
            if len(vectors) != len(batch) or any(
                len(vector) != provider.dimensions for vector in vectors
            ):
                raise EmbeddingProviderError(
                    "invalid_embedding_response",
                    "The embedding provider returned incompatible vectors.",
                )
            session.add_all(
                [
                    ChunkEmbedding(
                        chunk_id=chunk.id,
                        embedding_provider=provider.provider_name,
                        embedding_model=provider.model,
                        embedding_dimensions=provider.dimensions,
                        vector_data=vector,
                    )
                    for chunk, vector in zip(batch, vectors, strict=True)
                ]
            )
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            indexed += len(batch)
            logger.info(
                "Indexed %d chunks with %s/%s",
                len(batch),
                provider.provider_name,
                provider.model,
            )
    except EmbeddingProviderError:
        await session.rollback()
        await _set_document_statuses(
            session, affected_document_ids, ProcessingStatus.FAILED
        )
        raise

    await _set_document_statuses(
        session, affected_document_ids, ProcessingStatus.COMPLETED
    )

    return IndexResult(indexed_chunks=indexed, skipped_chunks=total_chunks - indexed)


async def _set_document_statuses(
    session: AsyncSession,
    document_ids: set[uuid.UUID],
    processing_status: ProcessingStatus,
) -> None:
    """Persist a lifecycle status for selected documents."""
    await session.execute(
        update(Document)
        .where(Document.id.in_(document_ids))
        .values(processing_status=processing_status)
    )
    await session.commit()


async def semantic_search(
    session: AsyncSession,
    provider: EmbeddingProvider,
    query: str,
    top_k: int,
    similarity_threshold: float,
) -> list[SearchResult]:
    """Find compatible chunk embeddings ordered by cosine similarity."""
    query_vectors = await provider.embed([query])
    if len(query_vectors) != 1 or len(query_vectors[0]) != provider.dimensions:
        raise EmbeddingProviderError(
            "invalid_embedding_response",
            "The embedding provider returned an incompatible query vector.",
        )
    query_vector = query_vectors[0]

    compatible = (
        select(
            ChunkEmbedding.chunk_id,
            ChunkEmbedding.vector_data,
        )
        .where(
            ChunkEmbedding.embedding_provider == provider.provider_name,
            ChunkEmbedding.embedding_model == provider.model,
            ChunkEmbedding.embedding_dimensions == provider.dimensions,
        )
        .cte("compatible_embeddings")
        .prefix_with("MATERIALIZED")
    )
    vector = cast(compatible.c.vector_data, VECTOR(provider.dimensions))
    distance = vector.cosine_distance(query_vector)
    similarity = (1.0 - distance).label("similarity")
    statement = (
        select(
            Document.id,
            Document.filename,
            DocumentChunk.source_page,
            DocumentChunk.chunk_index,
            DocumentChunk.text_content,
            similarity,
        )
        .join(DocumentChunk, DocumentChunk.id == compatible.c.chunk_id)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(similarity >= similarity_threshold)
        .order_by(distance)
        .limit(top_k)
    )
    rows = (await session.execute(statement)).all()
    return [
        SearchResult(
            document_id=row.id,
            filename=row.filename,
            page_number=row.source_page,
            chunk_index=row.chunk_index,
            text=row.text_content,
            similarity=float(row.similarity),
        )
        for row in rows
    ]
