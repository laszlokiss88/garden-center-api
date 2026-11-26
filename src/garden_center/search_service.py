"""Application services for chunk indexing and semantic retrieval."""

import logging
import uuid
from dataclasses import dataclass

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import cast, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from garden_center.embeddings import EmbeddingProvider, EmbeddingProviderError
from garden_center.models import (
    ChunkEmbedding,
    Document,
    DocumentChunk,
    ProcessingStatus,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A semantically similar chunk and its source metadata."""

    document_id: uuid.UUID
    filename: str
    page_number: int
    chunk_index: int
    text: str
    similarity: float


async def index_document_chunks(
    session: AsyncSession,
    provider: EmbeddingProvider,
    batch_size: int,
    document: Document,
) -> None:
    """Generate and persist embeddings for a newly uploaded document."""
    try:
        indexed_count = 0
        for start in range(0, len(document.chunks), batch_size):
            batch = document.chunks[start : start + batch_size]
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
            await session.flush()
            indexed_count += len(batch)
        document.processing_status = ProcessingStatus.COMPLETED
        await session.commit()
        logger.info(
            "Indexed %d chunks with %s/%s",
            indexed_count,
            provider.provider_name,
            provider.model,
        )
    except (EmbeddingProviderError, SQLAlchemyError):
        await session.rollback()
        raise


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
        .where(
            Document.processing_status == ProcessingStatus.COMPLETED,
            similarity >= similarity_threshold,
        )
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
