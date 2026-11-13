"""Explicit PostgreSQL/pgvector contract test; disabled unless requested."""

import os
from dataclasses import replace
from hashlib import sha256
from uuid import uuid4

import pytest

from ai_learning_rag.database import TEST_DATABASE_URL_ENV, require_test_database_url

if os.getenv("AI_LEARNING_RAG_RUN_POSTGRES_INTEGRATION") != "1":
    pytest.skip("set AI_LEARNING_RAG_RUN_POSTGRES_INTEGRATION=1", allow_module_level=True)

raw_test_url = os.getenv(TEST_DATABASE_URL_ENV, "")
if not raw_test_url:
    pytest.skip(f"{TEST_DATABASE_URL_ENV} is required", allow_module_level=True)

test_url = require_test_database_url(raw_test_url)

from sqlalchemy import create_engine  # noqa: E402

from ai_learning_rag.document_index import (  # noqa: E402
    DocumentGeneration,
    DocumentReplaceStatus,
    IndexStatus,
    VectorIndexManifest,
    VectorRecord,
    VectorSearchQuery,
)
from ai_learning_rag.postgres_vector_index import PostgresVectorIndex  # noqa: E402

pytestmark = pytest.mark.postgres_integration


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def test_postgres_backend_replaces_and_searches_active_generation() -> None:
    """Exercise the real transaction, pgvector dimension check, and cosine query."""

    unique = uuid4().hex
    manifest = VectorIndexManifest.create(
        logical_name=f"integration-{unique}",
        embedding_dimension=2,
        model_id="integration-model",
        model_revision="1",
        tokenizer_id="integration-tokenizer",
        tokenizer_revision="1",
        normalized=True,
        pooling="fixture",
        dtype="float32",
        query_profile="fixture-query-v1",
        document_input_profile="section-path-prefix-v1",
        max_input_tokens=32,
        oversize_policy="reject-v1",
        adapter_version="fixture-adapter-v1",
        required_pgvector_version="0.8.7",
    )
    generation = DocumentGeneration.create(
        index_id=manifest.index_id,
        document_id=f"document-{unique}",
        file_sha256=_digest(f"file-{unique}"),
        routing_profile="fixture-routing-v1",
        parser_profile="fixture-parser-v1",
        ocr_decision_fingerprint=_digest("no-ocr"),
        chunking_profile="fixture-chunking-v1",
        gate_profile="pdf-index-gate-v1",
        provenance_registry_fingerprint=_digest("fixture-provenance"),
    )
    record = VectorRecord(
        record_id=f"rec_{_digest(unique)}",
        index_id=manifest.index_id,
        generation_id=generation.generation_id,
        document_id=generation.document_id,
        chunk_fingerprint=_digest(f"chunk-{unique}"),
        chunk_index=0,
        content="PostgreSQL stores traceable vectors.",
        content_type="body_text",
        section_path=("Integration",),
        page_start=1,
        page_end=1,
        embedding_input_hash=_digest(f"input-{unique}"),
        embedding_input_token_count=5,
        vector=(1.0, 0.0),
        provenance={"schema_version": 1, "source_regions": []},
    )

    engine = create_engine(test_url, pool_pre_ping=True)
    try:
        index = PostgresVectorIndex(engine)
        index.ensure_index(manifest)
        index.ensure_index(replace(manifest, status=IndexStatus.READY))
        index.ensure_index(replace(manifest, status=IndexStatus.ACTIVE))

        first = index.replace_document(generation, (record,), expected_revision=0)
        repeated = index.replace_document(generation, (record,), expected_revision=0)
        observed = index.get_document_state(manifest.index_id, generation.document_id)
        concurrent_generation = DocumentGeneration.create(
            index_id=manifest.index_id,
            document_id=generation.document_id,
            file_sha256=_digest(f"file-{unique}-concurrent"),
            routing_profile=generation.routing_profile,
            parser_profile=generation.parser_profile,
            ocr_decision_fingerprint=generation.ocr_decision_fingerprint,
            chunking_profile=generation.chunking_profile,
            gate_profile=generation.gate_profile,
            provenance_registry_fingerprint=generation.provenance_registry_fingerprint,
        )
        concurrent_record = replace(
            record,
            record_id=f"rec_{_digest(f'{unique}-concurrent')}",
            generation_id=concurrent_generation.generation_id,
            chunk_fingerprint=_digest(f"chunk-{unique}-concurrent"),
            embedding_input_hash=_digest(f"input-{unique}-concurrent"),
            vector=(0.0, 1.0),
        )
        published = index.replace_document(
            concurrent_generation,
            (concurrent_record,),
            expected_revision=observed.revision,
        )
        stale_generation = DocumentGeneration.create(
            index_id=manifest.index_id,
            document_id=generation.document_id,
            file_sha256=_digest(f"file-{unique}-stale"),
            routing_profile=generation.routing_profile,
            parser_profile=generation.parser_profile,
            ocr_decision_fingerprint=generation.ocr_decision_fingerprint,
            chunking_profile=generation.chunking_profile,
            gate_profile=generation.gate_profile,
            provenance_registry_fingerprint=generation.provenance_registry_fingerprint,
        )
        stale_record = replace(
            record,
            record_id=f"rec_{_digest(f'{unique}-stale')}",
            generation_id=stale_generation.generation_id,
            chunk_fingerprint=_digest(f"chunk-{unique}-stale"),
            embedding_input_hash=_digest(f"input-{unique}-stale"),
        )
        conflict = index.replace_document(
            stale_generation,
            (stale_record,),
            expected_revision=observed.revision,
        )
        results = index.search(VectorSearchQuery(manifest.index_id, (0.0, 1.0), top_k=1))

        assert first.status is DocumentReplaceStatus.INSERTED
        assert first.document_revision == 1
        assert repeated.status is DocumentReplaceStatus.ALREADY_INDEXED
        assert published.status is DocumentReplaceStatus.REPLACED
        assert published.document_revision == 2
        assert conflict.status is DocumentReplaceStatus.CONCURRENT_UPDATE
        assert conflict.previous_generation_id == concurrent_generation.generation_id
        assert conflict.document_revision == 2
        assert results[0].record.record_id == concurrent_record.record_id
        assert results[0].score == pytest.approx(1.0)
    finally:
        engine.dispose()
