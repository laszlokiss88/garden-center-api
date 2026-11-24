"""Enable the pgvector extension.

Revision ID: 20261124_01
Revises:
Create Date: 2026-11-24
"""

from alembic import op

revision: str = "20261124_01"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Install pgvector in the application database."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    """Remove pgvector from the application database."""
    op.execute("DROP EXTENSION IF EXISTS vector")
