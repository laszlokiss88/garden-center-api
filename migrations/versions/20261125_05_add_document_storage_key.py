"""Persist provider-generated document storage keys.

Revision ID: 20261125_05
Revises: 20261125_04
Create Date: 2026-11-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261125_05"
down_revision: str | None = "20261125_04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add keys while retaining the legacy UUID-based file locations."""
    op.add_column(
        "documents", sa.Column("storage_key", sa.String(length=500), nullable=True)
    )
    op.execute("UPDATE documents SET storage_key = 'documents/' || id::text || '.pdf'")
    op.alter_column("documents", "storage_key", nullable=False)
    op.create_unique_constraint(
        "uq_documents_storage_key", "documents", ["storage_key"]
    )


def downgrade() -> None:
    """Remove persisted storage keys."""
    op.drop_constraint("uq_documents_storage_key", "documents", type_="unique")
    op.drop_column("documents", "storage_key")
