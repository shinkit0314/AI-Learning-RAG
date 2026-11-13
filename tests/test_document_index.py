"""Contract tests for document-level vector index semantics."""

from dataclasses import replace
from hashlib import sha256

import pytest

from ai_learning_rag.document_index import (
    DocumentGeneration,
    DocumentReplaceStatus,
    IndexStatus,
    InMemoryVectorIndex,
    VectorIndexManifest,
    VectorRecord,
    VectorSearchQuery,
)


def digest(value: str) -> str:
    """Return a compact deterministic fixture fingerprint."""

    return sha256(value.encode()).hexdigest()


def make_manifest(
    *,
    logical_name: str = "pdf-knowledge-base",
    dtype: str = "float16",
) -> VectorIndexManifest:
    """Build the smallest complete vector-space manifest used by tests."""

    return VectorIndexManifest.create(
        logical_name=logical_name,
        embedding_dimension=2,
        model_id="Qwen/Qwen3-Embedding-0.6B",
        model_revision="revision-1",
        tokenizer_id="Qwen/Qwen3-Embedding-0.6B",
        tokenizer_revision="revision-1",
        normalized=True,
        pooling="model-default",
        dtype=dtype,
        query_profile="qwen-query-v1",
        document_input_profile="section-path-prefix-v1",
        max_input_tokens=512,
        oversize_policy="reject-v1",
        adapter_version="qwen3-adapter-v1",
        required_pgvector_version="0.8.7",
    )


def make_generation(
    manifest: VectorIndexManifest,
    *,
    document_id: str = "document-a",
    file_marker: str = "file-a",
) -> DocumentGeneration:
    """Build a replacement-sensitive document generation."""

    return DocumentGeneration.create(
        index_id=manifest.index_id,
        document_id=document_id,
        file_sha256=digest(file_marker),
        routing_profile="pdf-page-routing-v1",
        parser_profile="pymupdf-1",
        ocr_decision_fingerprint=digest("ocr-decisions"),
        chunking_profile="pdf-structure-v1",
        gate_profile="pdf-index-gate-v1",
        provenance_registry_fingerprint=digest("provenance-registry"),
    )


def make_record(
    generation: DocumentGeneration,
    *,
    record_id: str,
    vector: tuple[float, ...],
    chunk_index: int = 0,
) -> VectorRecord:
    """Create a valid traceable record for one generation."""

    return VectorRecord(
        record_id=record_id,
        index_id=generation.index_id,
        generation_id=generation.generation_id,
        document_id=generation.document_id,
        chunk_fingerprint=digest(record_id),
        chunk_index=chunk_index,
        content=f"content-{record_id}",
        content_type="body_text",
        section_path=("Section",),
        page_start=1,
        page_end=1,
        embedding_input_hash=digest(f"input-{record_id}"),
        embedding_input_token_count=5,
        vector=vector,
        provenance={"schema_version": 1, "source_regions": []},
    )


def activate(index: InMemoryVectorIndex, manifest: VectorIndexManifest) -> None:
    """Exercise the required build, validate, and publish transitions."""

    index.ensure_index(manifest)
    index.ensure_index(replace(manifest, status=IndexStatus.READY))
    index.ensure_index(replace(manifest, status=IndexStatus.ACTIVE))


def test_vector_identity_changes_when_dtype_changes() -> None:
    """Precision changes must never reuse a semantically different vector space."""

    assert make_manifest(dtype="float16").index_id != make_manifest(dtype="float32").index_id


def test_new_index_cannot_skip_build_and_ready_states() -> None:
    """An index should be validated before it becomes searchable."""

    index = InMemoryVectorIndex()
    manifest = make_manifest()

    with pytest.raises(ValueError, match="must start in building"):
        index.ensure_index(replace(manifest, status=IndexStatus.ACTIVE))

    index.ensure_index(manifest)
    with pytest.raises(ValueError, match="building -> active"):
        index.ensure_index(replace(manifest, status=IndexStatus.ACTIVE))


def test_promoting_a_new_index_retires_the_previous_active_index() -> None:
    """Publishing a new embedding space should atomically preserve one active index."""

    index = InMemoryVectorIndex()
    old = make_manifest(dtype="float16")
    new = make_manifest(dtype="float32")
    activate(index, old)
    activate(index, new)

    assert index.get_manifest(old.index_id).status is IndexStatus.RETIRED  # type: ignore[union-attr]
    assert index.get_manifest(new.index_id).status is IndexStatus.ACTIVE  # type: ignore[union-attr]


def test_replacing_a_document_is_idempotent_and_hides_the_old_generation() -> None:
    """Only complete active generations may participate in retrieval."""

    index = InMemoryVectorIndex()
    manifest = make_manifest()
    activate(index, manifest)
    first = make_generation(manifest, file_marker="first")
    second = make_generation(manifest, file_marker="second")

    inserted = index.replace_document(
        first,
        (make_record(first, record_id="record-old", vector=(1.0, 0.0)),),
        expected_revision=0,
    )
    repeated = index.replace_document(
        first,
        (make_record(first, record_id="ignored", vector=(0.0, 1.0)),),
        expected_revision=0,
    )
    replaced = index.replace_document(
        second,
        (make_record(second, record_id="record-new", vector=(0.0, 1.0)),),
        expected_revision=1,
    )

    assert inserted.status is DocumentReplaceStatus.INSERTED
    assert repeated.status is DocumentReplaceStatus.ALREADY_INDEXED
    assert repeated.record_count == 1
    assert repeated.document_revision == 1
    assert replaced.status is DocumentReplaceStatus.REPLACED
    assert replaced.previous_generation_id == first.generation_id
    assert replaced.document_revision == 2
    assert index.get_active_generation(manifest.index_id, first.document_id) == second
    results = index.search(VectorSearchQuery(manifest.index_id, (1.0, 0.0), top_k=5))
    assert [result.record.record_id for result in results] == ["record-new"]


def test_invalid_candidate_cannot_partially_replace_the_active_generation() -> None:
    """Validation must finish before any visible in-memory state changes."""

    index = InMemoryVectorIndex()
    manifest = make_manifest()
    activate(index, manifest)
    first = make_generation(manifest, file_marker="first")
    invalid = make_generation(manifest, file_marker="invalid")
    index.replace_document(
        first,
        (make_record(first, record_id="record-old", vector=(1.0, 0.0)),),
        expected_revision=0,
    )

    with pytest.raises(ValueError, match="exactly 2 dimensions"):
        index.replace_document(
            invalid,
            (make_record(invalid, record_id="record-bad", vector=(1.0,)),),
            expected_revision=1,
        )

    assert index.get_active_generation(manifest.index_id, first.document_id) == first


def test_search_uses_record_id_as_a_deterministic_tie_breaker() -> None:
    """Stable ordering keeps previews and regression checks reproducible."""

    index = InMemoryVectorIndex()
    manifest = make_manifest()
    activate(index, manifest)
    generation = make_generation(manifest)
    index.replace_document(
        generation,
        (
            make_record(generation, record_id="record-b", vector=(1.0, 0.0), chunk_index=1),
            make_record(generation, record_id="record-a", vector=(1.0, 0.0)),
        ),
        expected_revision=0,
    )

    results = index.search(VectorSearchQuery(manifest.index_id, (1.0, 0.0), top_k=2))

    assert [result.record.record_id for result in results] == ["record-a", "record-b"]


def test_stale_replacement_cannot_overwrite_a_concurrent_generation() -> None:
    """A candidate built from an old revision must not win merely by finishing last."""

    index = InMemoryVectorIndex()
    manifest = make_manifest()
    activate(index, manifest)
    first = make_generation(manifest, file_marker="first")
    concurrent = make_generation(manifest, file_marker="concurrent")
    stale = make_generation(manifest, file_marker="stale")
    index.replace_document(
        first,
        (make_record(first, record_id="record-first", vector=(1.0, 0.0)),),
        expected_revision=0,
    )
    observed = index.get_document_state(manifest.index_id, first.document_id)
    index.replace_document(
        concurrent,
        (make_record(concurrent, record_id="record-concurrent", vector=(0.0, 1.0)),),
        expected_revision=observed.revision,
    )

    conflict = index.replace_document(
        stale,
        (make_record(stale, record_id="record-stale", vector=(1.0, 0.0)),),
        expected_revision=observed.revision,
    )

    assert conflict.status is DocumentReplaceStatus.CONCURRENT_UPDATE
    assert conflict.previous_generation_id == concurrent.generation_id
    assert conflict.document_revision == 2
    state = index.get_document_state(manifest.index_id, first.document_id)
    assert state.active_generation == concurrent
    assert state.revision == 2
    results = index.search(VectorSearchQuery(manifest.index_id, (1.0, 0.0), top_k=5))
    assert [result.record.record_id for result in results] == ["record-concurrent"]
