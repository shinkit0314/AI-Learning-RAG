"""Add optimistic revision control to indexed document slots.

Revision ID: 20261112_0002
Revises: 20261107_0001
Create Date: 2026-11-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261112_0002"
down_revision: str | None = "20261107_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add a monotonic compare-and-swap revision without changing active generations."""

    op.add_column(
        "indexed_documents",
        sa.Column(
            "revision",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        schema="rag",
    )
    op.execute(
        """
        UPDATE rag.indexed_documents
        SET revision = 1
        WHERE active_generation_id IS NOT NULL
        """
    )
    op.create_check_constraint(
        "ck_indexed_documents_revision",
        "indexed_documents",
        "revision >= 0",
        schema="rag",
    )


def downgrade() -> None:
    """Remove only optimistic revision metadata; preserve indexed content."""

    op.drop_constraint(
        "ck_indexed_documents_revision",
        "indexed_documents",
        type_="check",
        schema="rag",
    )
    op.drop_column("indexed_documents", "revision", schema="rag")
