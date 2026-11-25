"""Remove the obsolete persisted failed status.

Revision ID: 20261125_04
Revises: 20261125_03
Create Date: 2026-11-25
"""

from alembic import op

revision: str = "20261125_04"
down_revision: str | None = "20261125_03"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Restrict persisted documents to active or completed processing."""
    op.drop_constraint("ck_documents_processing_status", "documents", type_="check")
    op.create_check_constraint(
        "ck_documents_processing_status",
        "documents",
        "processing_status IN ('processing', 'completed')",
    )


def downgrade() -> None:
    """Restore the original lifecycle values."""
    op.drop_constraint("ck_documents_processing_status", "documents", type_="check")
    op.create_check_constraint(
        "ck_documents_processing_status",
        "documents",
        "processing_status IN ('processing', 'completed', 'failed')",
    )
