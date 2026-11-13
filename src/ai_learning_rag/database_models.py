"""SQLAlchemy metadata for the traceable PostgreSQL/pgvector index."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import UserDefinedType

RAG_SCHEMA = "rag"


class VectorType(UserDefinedType[Any]):
    """DDL-only pgvector type; SQL operations use explicit vector casts."""

    cache_ok = True

    def get_col_spec(self, **_kwargs: object) -> str:
        """Compile the extension-owned PostgreSQL type name."""

        return "vector"


class Base(DeclarativeBase):
    """Declarative metadata root used by Alembic."""


class VectorIndexModel(Base):
    """One immutable embedding space and its publish lifecycle."""

    __tablename__ = "vector_indexes"
    __table_args__ = (
        UniqueConstraint(
            "index_id",
            "embedding_dimension",
            name="uq_vector_indexes_id_dimension",
        ),
        CheckConstraint("embedding_dimension > 0", name="ck_vector_indexes_dimension_positive"),
        CheckConstraint(
            "status IN ('building', 'ready', 'active', 'failed', 'retired')",
            name="ck_vector_indexes_status",
        ),
        Index(
            "uq_vector_indexes_one_active_logical_name",
            "logical_name",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
        {"schema": RAG_SCHEMA},
    )

    index_id: Mapped[str] = mapped_column(String(68), primary_key=True)
    logical_name: Mapped[str] = mapped_column(String(200), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    identity: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    audit_metadata: Mapped[dict[str, str]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    required_pgvector_version: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default=text("'building'"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )


class IndexedDocumentModel(Base):
    """Stable row locked while a document generation is replaced."""

    __tablename__ = "indexed_documents"
    __table_args__ = (
        ForeignKeyConstraint(
            ["index_id"],
            [f"{RAG_SCHEMA}.vector_indexes.index_id"],
            name="fk_indexed_documents_index",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["active_generation_id", "index_id", "document_id"],
            [
                f"{RAG_SCHEMA}.document_generations.generation_id",
                f"{RAG_SCHEMA}.document_generations.index_id",
                f"{RAG_SCHEMA}.document_generations.document_id",
            ],
            name="fk_indexed_documents_active_generation",
            use_alter=True,
        ),
        CheckConstraint("revision >= 0", name="ck_indexed_documents_revision"),
        {"schema": RAG_SCHEMA},
    )

    index_id: Mapped[str] = mapped_column(String(68), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    active_generation_id: Mapped[str | None] = mapped_column(String(68), nullable=True)
    revision: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )


class DocumentGenerationModel(Base):
    """Immutable processing identity retained after it is superseded."""

    __tablename__ = "document_generations"
    __table_args__ = (
        UniqueConstraint(
            "generation_id",
            "index_id",
            "document_id",
            name="uq_document_generations_identity_scope",
        ),
        ForeignKeyConstraint(
            ["index_id", "document_id"],
            [
                f"{RAG_SCHEMA}.indexed_documents.index_id",
                f"{RAG_SCHEMA}.indexed_documents.document_id",
            ],
            name="fk_document_generations_document_slot",
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "status IN ('active', 'superseded')",
            name="ck_document_generations_status",
        ),
        {"schema": RAG_SCHEMA},
    )

    generation_id: Mapped[str] = mapped_column(String(68), primary_key=True)
    index_id: Mapped[str] = mapped_column(String(68), nullable=False)
    document_id: Mapped[str] = mapped_column(String(255), nullable=False)
    file_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    routing_profile: Mapped[str] = mapped_column(Text, nullable=False)
    parser_profile: Mapped[str] = mapped_column(Text, nullable=False)
    ocr_decision_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    chunking_profile: Mapped[str] = mapped_column(Text, nullable=False)
    gate_profile: Mapped[str] = mapped_column(String(100), nullable=False)
    provenance_registry_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default=text("'active'"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class VectorRecordModel(Base):
    """A pgvector value plus relational filters and versioned provenance."""

    __tablename__ = "vector_records"
    __table_args__ = (
        ForeignKeyConstraint(
            ["index_id", "embedding_dimension"],
            [
                f"{RAG_SCHEMA}.vector_indexes.index_id",
                f"{RAG_SCHEMA}.vector_indexes.embedding_dimension",
            ],
            name="fk_vector_records_index_dimension",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["generation_id", "index_id", "document_id"],
            [
                f"{RAG_SCHEMA}.document_generations.generation_id",
                f"{RAG_SCHEMA}.document_generations.index_id",
                f"{RAG_SCHEMA}.document_generations.document_id",
            ],
            name="fk_vector_records_generation",
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "generation_id",
            "chunk_fingerprint",
            name="uq_vector_records_generation_chunk",
        ),
        CheckConstraint("chunk_index >= 0", name="ck_vector_records_chunk_index"),
        CheckConstraint(
            "embedding_input_token_count > 0",
            name="ck_vector_records_token_count",
        ),
        CheckConstraint(
            "page_start >= 1 AND page_end >= page_start",
            name="ck_vector_records_page_range",
        ),
        CheckConstraint(
            "provenance_schema_version = 1",
            name="ck_vector_records_provenance_schema",
        ),
        CheckConstraint(
            "vector_dims(embedding) = embedding_dimension",
            name="ck_vector_records_embedding_dimension",
        ),
        Index("ix_vector_records_active_lookup", "index_id", "generation_id"),
        {"schema": RAG_SCHEMA},
    )

    record_id: Mapped[str] = mapped_column(String(68), primary_key=True)
    index_id: Mapped[str] = mapped_column(String(68), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    generation_id: Mapped[str] = mapped_column(String(68), nullable=False)
    document_id: Mapped[str] = mapped_column(String(255), nullable=False)
    chunk_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(String(50), nullable=False)
    section_path: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_input_token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[Any] = mapped_column(VectorType(), nullable=False)
    provenance_schema_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("1"),
    )
    provenance: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )


def _model_names() -> tuple[str, ...]:
    """Keep imported model classes visible to static analysis and Alembic."""

    return (
        VectorIndexModel.__tablename__,
        IndexedDocumentModel.__tablename__,
        DocumentGenerationModel.__tablename__,
        VectorRecordModel.__tablename__,
    )


assert _model_names() == (
    "vector_indexes",
    "indexed_documents",
    "document_generations",
    "vector_records",
)
