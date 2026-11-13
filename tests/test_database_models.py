"""Offline checks for PostgreSQL model and integration-test safety contracts."""

from pathlib import Path

import pytest
from sqlalchemy import make_url
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from ai_learning_rag.database import DatabaseSettings, require_test_database_url
from ai_learning_rag.database_models import Base


def test_metadata_contains_the_four_versioned_rag_tables() -> None:
    """The stable document slot must be represented beside generations and vectors."""

    assert set(Base.metadata.tables) == {
        "rag.vector_indexes",
        "rag.indexed_documents",
        "rag.document_generations",
        "rag.vector_records",
    }


def test_vector_record_ddl_contains_dimension_and_provenance_guards() -> None:
    """Critical pgvector invariants should live in PostgreSQL, not only Python."""

    table = Base.metadata.tables["rag.vector_records"]
    ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))

    assert "embedding vector NOT NULL" in ddl
    assert "vector_dims(embedding) = embedding_dimension" in ddl
    assert "provenance_schema_version = 1" in ddl
    assert "fk_vector_records_index_dimension" in ddl


def test_indexed_document_ddl_contains_optimistic_revision_guard() -> None:
    """Every stable document slot must carry a non-negative CAS revision."""

    table = Base.metadata.tables["rag.indexed_documents"]
    ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))

    assert "revision INTEGER DEFAULT 0 NOT NULL" in ddl
    assert "revision >= 0" in ddl


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+psycopg://user:secret@localhost/ai_learning_rag",
        "postgresql+psycopg://user:secret@localhost/ai_learning_rag_test_backup",
        "sqlite:///ai_learning_rag_test",
    ],
)
def test_integration_database_guard_rejects_unsafe_targets(url: str) -> None:
    """A typo must not make integration tests write into a development database."""

    with pytest.raises(ValueError):
        require_test_database_url(url)


def test_integration_database_guard_accepts_explicit_test_database() -> None:
    """Only a PostgreSQL database whose name ends in _test is eligible."""

    url = require_test_database_url(
        "postgresql+psycopg://user:secret@localhost/ai_learning_rag_test"
    )

    assert url.database == "ai_learning_rag_test"


def test_database_settings_reject_non_postgresql_urls() -> None:
    """The pgvector backend must not silently run against a different database."""

    with pytest.raises(ValueError, match="must use PostgreSQL"):
        DatabaseSettings(url=make_url("sqlite:///db"))


def test_migration_does_not_repeat_admin_schema_bootstrap() -> None:
    """The migrator owns rag but must not need database-level CREATE privilege."""

    migration = (
        Path(__file__).parents[1] / "migrations" / "versions" / "20261107_0001_pdf_vector_index.py"
    ).read_text(encoding="utf-8")

    assert "CREATE SCHEMA" not in migration
    assert "CREATE EXTENSION vector;" not in migration


def test_revision_migration_backfills_existing_active_documents() -> None:
    """Previously indexed documents need a valid non-zero revision after upgrade."""

    migration = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "20261112_0002_add_document_revision.py"
    ).read_text(encoding="utf-8")

    assert 'down_revision: str | None = "20261107_0001"' in migration
    assert '"revision"' in migration
    assert "SET revision = 1" in migration
    assert "WHERE active_generation_id IS NOT NULL" in migration
    assert "revision >= 0" in migration
