"""Tests for converting real retrieval preview artifacts into evaluator inputs."""

from hashlib import sha256

import pytest

from ai_learning_rag.document_index import IndexStatus, VectorIndexManifest
from ai_learning_rag.retrieval_evaluation import (
    ChunkIdentity,
    CorpusCompatibilityError,
    CorpusInventory,
    CorpusSample,
)
from ai_learning_rag.retrieval_evaluation_preview import evaluation_inputs_from_preview


def make_manifest() -> VectorIndexManifest:
    """Create one complete active manifest that can survive preview serialization."""

    return VectorIndexManifest.create(
        logical_name="preview-fixture",
        embedding_dimension=2,
        model_id="fixture-model",
        model_revision="revision-1",
        tokenizer_id="fixture-tokenizer",
        tokenizer_revision="revision-1",
        normalized=True,
        pooling="fixture",
        dtype="float32",
        query_profile="fixture-query-v1",
        document_input_profile="fixture-document-v1",
        max_input_tokens=128,
        oversize_policy="reject-v1",
        adapter_version="fixture-adapter-v1",
        required_pgvector_version="0.8.7",
        status=IndexStatus.ACTIVE,
        audit_metadata={"runtime": "fixture"},
    )


def make_payload() -> tuple[dict[str, object], CorpusInventory, VectorIndexManifest]:
    """Build the expanded preview schema emitted by the real indexing script."""

    identity = ChunkIdentity("document-a", sha256(b"chunk-a").hexdigest())
    inventory = CorpusInventory.create(
        corpus_id="fixture-corpus-v1",
        samples=(
            CorpusSample(
                sample_id="sample-a",
                path="data/PDF/a.pdf",
                document_id="document-a",
                file_sha256=sha256(b"file-a").hexdigest(),
            ),
        ),
        chunking_profile="fixture-chunking-v1",
        gate_profile="fixture-gate-v1",
        chunks=(identity,),
    )
    manifest = make_manifest()
    payload: dict[str, object] = {
        "schema_version": "pdf-indexing-retrieval-preview-v1",
        "corpus_inventory": inventory.to_payload(),
        "index_manifest": {
            "index_id": manifest.index_id,
            "status": manifest.status.value,
            "identity": manifest.identity_payload,
            "required_pgvector_version": manifest.required_pgvector_version,
            "audit_metadata": manifest.audit_metadata,
        },
        "queries": [
            {
                "query_id": "query-a",
                "hits": [
                    {
                        "rank": 1,
                        "score": 0.9,
                        "record_id": f"rec_{'1' * 64}",
                        "index_id": manifest.index_id,
                        "generation_id": f"gen_{'2' * 64}",
                        "document_id": identity.document_id,
                        "chunk_fingerprint": identity.chunk_fingerprint,
                        "chunk_index": 0,
                        "content": "Complete retrieved evidence",
                        "content_type": "body_text",
                        "section_path": ["Section"],
                        "page_start": 1,
                        "page_end": 1,
                        "embedding_input_hash": sha256(b"input").hexdigest(),
                        "embedding_input_token_count": 4,
                        "provenance": {"schema_version": 1, "source_regions": []},
                    }
                ],
            }
        ],
    }
    return payload, inventory, manifest


def test_preview_round_trip_preserves_corpus_manifest_and_rankings() -> None:
    """The explicit real run must feed the same identities into offline scoring."""

    payload, expected_inventory, expected_manifest = make_payload()

    inventory, manifest, rankings = evaluation_inputs_from_preview(payload)

    assert inventory == expected_inventory
    assert manifest == expected_manifest
    assert list(rankings) == ["query-a"]
    assert rankings["query-a"][0].record.chunk_fingerprint == inventory.chunks[0].chunk_fingerprint
    assert rankings["query-a"][0].score == 0.9


def test_preview_rejects_tampered_corpus_snapshot() -> None:
    """A hand-edited snapshot cannot claim compatibility with different chunk membership."""

    payload, _inventory, _manifest = make_payload()
    corpus = payload["corpus_inventory"]
    assert isinstance(corpus, dict)
    snapshot = corpus["snapshot"]
    assert isinstance(snapshot, dict)
    snapshot["chunk_set_fingerprint"] = "f" * 64

    with pytest.raises(CorpusCompatibilityError, match="does not match"):
        evaluation_inputs_from_preview(payload)


def test_preview_rejects_non_contiguous_ranks() -> None:
    """Report order must be explicit rather than inferred from malformed rank values."""

    payload, _inventory, _manifest = make_payload()
    queries = payload["queries"]
    assert isinstance(queries, list)
    assert isinstance(queries[0], dict)
    hits = queries[0]["hits"]
    assert isinstance(hits, list)
    assert isinstance(hits[0], dict)
    hits[0]["rank"] = 2

    with pytest.raises(ValueError, match="ranks must be contiguous"):
        evaluation_inputs_from_preview(payload)


def test_preview_rejects_manifest_with_a_forged_index_id() -> None:
    """An index label cannot be detached from the embedding-space identity it names."""

    payload, _inventory, _manifest = make_payload()
    manifest = payload["index_manifest"]
    assert isinstance(manifest, dict)
    manifest["index_id"] = f"idx_{'f' * 64}"

    with pytest.raises(ValueError, match="does not match its manifest"):
        evaluation_inputs_from_preview(payload)
