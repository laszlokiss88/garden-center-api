"""Add document, chunk, and embedding tables.

Revision ID: 20261124_02
Revises: 20261124_01
Create Date: 2026-11-24
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import VECTOR

revision: str = "20261124_02"
down_revision: str | None = "20261124_01"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Create the document persistence schema."""
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("content_type", sa.String(length=100), nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("processing_status", sa.String(length=20), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("file_size > 0", name="ck_documents_file_size_positive"),
        sa.CheckConstraint(
            "processing_status IN ('processing', 'completed', 'failed')",
            name="ck_documents_processing_status",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_documents_content_hash", "documents", ["content_hash"], unique=True
    )
    op.create_index("ix_documents_created_at", "documents", ["created_at"])

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text_content", sa.Text(), nullable=False),
        sa.Column("source_page", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.CheckConstraint("chunk_index >= 0", name="ck_document_chunks_index"),
        sa.CheckConstraint("source_page > 0", name="ck_document_chunks_source_page"),
        sa.CheckConstraint("token_count > 0", name="ck_document_chunks_token_count"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id", "chunk_index", name="uq_document_chunks_document_index"
        ),
    )
    op.create_index(
        "ix_document_chunks_document_id", "document_chunks", ["document_id"]
    )

    op.create_table(
        "chunk_embeddings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column("embedding_provider", sa.String(length=50), nullable=False),
        sa.Column("embedding_model", sa.String(length=255), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column("vector_data", VECTOR(), nullable=False),
        sa.CheckConstraint(
            "embedding_dimensions > 0", name="ck_chunk_embeddings_dimensions"
        ),
        sa.CheckConstraint(
            "vector_dims(vector_data) = embedding_dimensions",
            name="ck_chunk_embeddings_vector_dimensions",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"], ["document_chunks.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "chunk_id",
            "embedding_provider",
            "embedding_model",
            name="uq_chunk_embeddings_chunk_model",
        ),
    )
    op.create_index("ix_chunk_embeddings_chunk_id", "chunk_embeddings", ["chunk_id"])
    op.create_index(
        "ix_chunk_embeddings_space",
        "chunk_embeddings",
        ["embedding_provider", "embedding_model", "embedding_dimensions"],
    )


def downgrade() -> None:
    """Remove the document persistence schema."""
    op.drop_table("chunk_embeddings")
    op.drop_table("document_chunks")
    op.drop_table("documents")
