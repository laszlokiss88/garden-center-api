"""Include dimensions in embedding-space uniqueness.

Revision ID: 20261125_03
Revises: 20261124_02
Create Date: 2026-11-25
"""

from alembic import op

revision: str = "20261125_03"
down_revision: str | None = "20261124_02"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Allow dimensions to distinguish embeddings for the same model."""
    op.drop_constraint(
        "uq_chunk_embeddings_chunk_model", "chunk_embeddings", type_="unique"
    )
    op.create_unique_constraint(
        "uq_chunk_embeddings_chunk_model",
        "chunk_embeddings",
        [
            "chunk_id",
            "embedding_provider",
            "embedding_model",
            "embedding_dimensions",
        ],
    )


def downgrade() -> None:
    """Restore the original provider/model uniqueness rule."""
    op.drop_constraint(
        "uq_chunk_embeddings_chunk_model", "chunk_embeddings", type_="unique"
    )
    op.create_unique_constraint(
        "uq_chunk_embeddings_chunk_model",
        "chunk_embeddings",
        ["chunk_id", "embedding_provider", "embedding_model"],
    )
