"""Create the traceable PDF vector index schema.

Revision ID: 20261107_0001
Revises:
Create Date: 2026-11-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import UserDefinedType

revision: str = "20261107_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


class VectorType(UserDefinedType[object]):
    """Self-contained pgvector DDL type for this immutable migration."""

    cache_ok = True

    def get_col_spec(self, **_kwargs: object) -> str:
        return "vector"


def upgrade() -> None:
    """Create four constrained tables after an administrator installs pgvector."""

    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
                RAISE EXCEPTION
                    'pgvector missing; administrator must run CREATE EXTENSION vector';
            END IF;
        END
        $$
        """
    )

    op.create_table(
        "vector_indexes",
        sa.Column("index_id", sa.String(length=68), primary_key=True),
        sa.Column("logical_name", sa.String(length=200), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("identity", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "audit_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("required_pgvector_version", sa.String(length=50), nullable=False),
        sa.Column(
            "status",
            sa.String(length=20),
            server_default=sa.text("'building'"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "embedding_dimension > 0",
            name="ck_vector_indexes_dimension_positive",
        ),
        sa.CheckConstraint(
            "status IN ('building', 'ready', 'active', 'failed', 'retired')",
            name="ck_vector_indexes_status",
        ),
        sa.UniqueConstraint(
            "index_id",
            "embedding_dimension",
            name="uq_vector_indexes_id_dimension",
        ),
        schema="rag",
    )
    op.create_index(
        "uq_vector_indexes_one_active_logical_name",
        "vector_indexes",
        ["logical_name"],
        unique=True,
        schema="rag",
        postgresql_where=sa.text("status = 'active'"),
    )

    op.create_table(
        "indexed_documents",
        sa.Column("index_id", sa.String(length=68), primary_key=True),
        sa.Column("document_id", sa.String(length=255), primary_key=True),
        sa.Column("active_generation_id", sa.String(length=68), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["index_id"],
            ["rag.vector_indexes.index_id"],
            name="fk_indexed_documents_index",
            ondelete="CASCADE",
        ),
        schema="rag",
    )

    op.create_table(
        "document_generations",
        sa.Column("generation_id", sa.String(length=68), primary_key=True),
        sa.Column("index_id", sa.String(length=68), nullable=False),
        sa.Column("document_id", sa.String(length=255), nullable=False),
        sa.Column("file_sha256", sa.String(length=64), nullable=False),
        sa.Column("routing_profile", sa.Text(), nullable=False),
        sa.Column("parser_profile", sa.Text(), nullable=False),
        sa.Column("ocr_decision_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("chunking_profile", sa.Text(), nullable=False),
        sa.Column("gate_profile", sa.String(length=100), nullable=False),
        sa.Column("provenance_registry_fingerprint", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.String(length=20),
            server_default=sa.text("'active'"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("superseded_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('active', 'superseded')",
            name="ck_document_generations_status",
        ),
        sa.ForeignKeyConstraint(
            ["index_id", "document_id"],
            ["rag.indexed_documents.index_id", "rag.indexed_documents.document_id"],
            name="fk_document_generations_document_slot",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "generation_id",
            "index_id",
            "document_id",
            name="uq_document_generations_identity_scope",
        ),
        schema="rag",
    )
    op.create_foreign_key(
        "fk_indexed_documents_active_generation",
        "indexed_documents",
        "document_generations",
        ["active_generation_id", "index_id", "document_id"],
        ["generation_id", "index_id", "document_id"],
        source_schema="rag",
        referent_schema="rag",
    )

    op.create_table(
        "vector_records",
        sa.Column("record_id", sa.String(length=68), primary_key=True),
        sa.Column("index_id", sa.String(length=68), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("generation_id", sa.String(length=68), nullable=False),
        sa.Column("document_id", sa.String(length=255), nullable=False),
        sa.Column("chunk_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_type", sa.String(length=50), nullable=False),
        sa.Column("section_path", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("embedding_input_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding_input_token_count", sa.Integer(), nullable=False),
        sa.Column("embedding", VectorType(), nullable=False),
        sa.Column(
            "provenance_schema_version",
            sa.Integer(),
            server_default=sa.text("1"),
            nullable=False,
        ),
        sa.Column("provenance", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint("chunk_index >= 0", name="ck_vector_records_chunk_index"),
        sa.CheckConstraint(
            "embedding_input_token_count > 0",
            name="ck_vector_records_token_count",
        ),
        sa.CheckConstraint(
            "page_start >= 1 AND page_end >= page_start",
            name="ck_vector_records_page_range",
        ),
        sa.CheckConstraint(
            "provenance_schema_version = 1",
            name="ck_vector_records_provenance_schema",
        ),
        sa.CheckConstraint(
            "vector_dims(embedding) = embedding_dimension",
            name="ck_vector_records_embedding_dimension",
        ),
        sa.ForeignKeyConstraint(
            ["index_id", "embedding_dimension"],
            ["rag.vector_indexes.index_id", "rag.vector_indexes.embedding_dimension"],
            name="fk_vector_records_index_dimension",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["generation_id", "index_id", "document_id"],
            [
                "rag.document_generations.generation_id",
                "rag.document_generations.index_id",
                "rag.document_generations.document_id",
            ],
            name="fk_vector_records_generation",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "generation_id",
            "chunk_fingerprint",
            name="uq_vector_records_generation_chunk",
        ),
        schema="rag",
    )
    op.create_index(
        "ix_vector_records_active_lookup",
        "vector_records",
        ["index_id", "generation_id"],
        schema="rag",
    )


def downgrade() -> None:
    """Remove only objects owned by this migration; keep extension and schema."""

    op.drop_index(
        "ix_vector_records_active_lookup",
        table_name="vector_records",
        schema="rag",
    )
    op.drop_table("vector_records", schema="rag")
    op.drop_constraint(
        "fk_indexed_documents_active_generation",
        "indexed_documents",
        type_="foreignkey",
        schema="rag",
    )
    op.drop_table("document_generations", schema="rag")
    op.drop_table("indexed_documents", schema="rag")
    op.drop_index(
        "uq_vector_indexes_one_active_logical_name",
        table_name="vector_indexes",
        schema="rag",
    )
    op.drop_table("vector_indexes", schema="rag")
