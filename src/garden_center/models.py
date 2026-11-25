"""SQLAlchemy models for persisted documents and their text chunks."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative model base."""


class ProcessingStatus(StrEnum):
    """Lifecycle states for synchronous document processing."""

    PROCESSING = "processing"
    COMPLETED = "completed"


class Document(Base):
    """An uploaded PDF and its processing state."""

    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint("file_size > 0", name="ck_documents_file_size_positive"),
        CheckConstraint(
            "processing_status IN ('processing', 'completed')",
            name="ck_documents_processing_status",
        ),
        Index("ix_documents_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    file_size: Mapped[int] = mapped_column(BigInteger)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    processing_status: Mapped[str] = mapped_column(
        String(20), default=ProcessingStatus.PROCESSING
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="DocumentChunk.chunk_index",
    )


class DocumentChunk(Base):
    """A deterministic token-aware text segment extracted from a PDF page."""

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_id", "chunk_index", name="uq_document_chunks_document_index"
        ),
        CheckConstraint("chunk_index >= 0", name="ck_document_chunks_index"),
        CheckConstraint("source_page > 0", name="ck_document_chunks_source_page"),
        CheckConstraint("token_count > 0", name="ck_document_chunks_token_count"),
        Index("ix_document_chunks_document_id", "document_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE")
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    text_content: Mapped[str] = mapped_column(Text)
    source_page: Mapped[int] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer)

    document: Mapped[Document] = relationship(back_populates="chunks")
    embeddings: Mapped[list[ChunkEmbedding]] = relationship(
        back_populates="chunk",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ChunkEmbedding(Base):
    """An embedding tied to one provider/model/dimension embedding space."""

    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "chunk_id",
            "embedding_provider",
            "embedding_model",
            "embedding_dimensions",
            name="uq_chunk_embeddings_chunk_model",
        ),
        CheckConstraint(
            "embedding_dimensions > 0", name="ck_chunk_embeddings_dimensions"
        ),
        CheckConstraint(
            "vector_dims(vector_data) = embedding_dimensions",
            name="ck_chunk_embeddings_vector_dimensions",
        ),
        Index(
            "ix_chunk_embeddings_space",
            "embedding_provider",
            "embedding_model",
            "embedding_dimensions",
        ),
        Index("ix_chunk_embeddings_chunk_id", "chunk_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="CASCADE")
    )
    embedding_provider: Mapped[str] = mapped_column(String(50))
    embedding_model: Mapped[str] = mapped_column(String(255))
    embedding_dimensions: Mapped[int] = mapped_column(Integer)
    vector_data: Mapped[list[float]] = mapped_column(VECTOR())

    chunk: Mapped[DocumentChunk] = relationship(back_populates="embeddings")
