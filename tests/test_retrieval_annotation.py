"""Tests for human-only candidate relevance review materials."""

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

import pytest

from ai_learning_rag.document_index import IndexStatus, VectorIndexManifest
from ai_learning_rag.retrieval_annotation import (
    build_annotation_review_markdown,
    build_approved_evaluation_dataset,
)
from ai_learning_rag.retrieval_evaluation import (
    ChunkIdentity,
    CorpusInventory,
    CorpusSample,
    EvaluationCaseType,
    ReviewStatus,
    load_retrieval_evaluation_dataset,
)


def make_payload() -> dict[str, object]:
    """Return one minimal real-preview-shaped candidate payload."""

    return {
        "schema_version": "pdf-indexing-retrieval-preview-v1",
        "index_id": "idx_fixture",
        "model_id": "fixture-model",
        "model_revision": "revision-1",
        "queries": [
            {
                "query_id": "candidate-1",
                "query": "What is the evidence?",
                "expected_sample_id": "sample-a",
                "expected_pages": [2],
                "hits": [
                    {
                        "rank": 1,
                        "score": 0.75,
                        "sample_id": "sample-a",
                        "document_id": "document-a",
                        "chunk_fingerprint": "a" * 64,
                        "record_id": "record-a",
                        "page_start": 2,
                        "page_end": 2,
                        "content_type": "body_text",
                        "section_path": ["Section"],
                        "content": "Full evidence with a | separator.\nTAIL_MARKER",
                    }
                ],
            }
        ],
    }


def test_review_markdown_never_presents_candidates_as_gold() -> None:
    """AI-prepared suggestions must visibly require a human decision."""

    markdown = build_annotation_review_markdown(make_payload())

    assert "not approved relevance gold" in markdown
    assert "[ ] approve  [ ] revise  [ ] reject" in markdown
    assert "[ ] relevant [ ] not relevant" in markdown
    assert "`document-a`" in markdown
    assert f"`{'a' * 64}`" in markdown
    assert "\\| separator" in markdown
    assert "> TAIL_MARKER" in markdown


def test_review_markdown_rejects_unrelated_payload_schema() -> None:
    """A generation preview must not be silently treated as retrieval annotations."""

    payload = make_payload()
    payload["schema_version"] = "other-schema"

    with pytest.raises(ValueError, match="unsupported"):
        build_annotation_review_markdown(payload)


def test_annotation_manifest_has_fifteen_questions_and_three_diagnostics() -> None:
    """The formal candidate batch stays balanced and separate from approved gold."""

    manifest_path = (
        Path(__file__).parents[1] / "configs" / "pdf-retrieval-annotation-preview-v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    queries = manifest["queries"]

    assert manifest["schema_version"] == "pdf-indexing-preview-v1"
    assert manifest["corpus_id"] == "pdf-sample-corpus-v1"
    answerable = [query for query in queries if query["case_type"] == "answerable"]
    diagnostics = [query for query in queries if query["case_type"] != "answerable"]

    assert len(queries) == 18
    assert Counter(query["expected_sample_id"] for query in answerable) == {
        "native-001": 5,
        "layout-001": 5,
        "scan-001": 5,
    }
    assert Counter(query["case_type"] for query in diagnostics) == {
        "unanswerable": 2,
        "ambiguous": 1,
    }
    assert all("expected_sample_id" not in query for query in diagnostics)
    assert all("expected_pages" not in query for query in diagnostics)
    assert all(query["top_k"] == 5 for query in queries)
    assert all("relevant_chunks" not in query for query in queries)


def test_approved_decisions_are_promoted_with_catalog_audit_fields() -> None:
    """Humans approve identities while the program copies exact source context."""

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
    manifest = VectorIndexManifest.create(
        logical_name="fixture-index",
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
    )
    catalog_entry = {
        "sample_id": "sample-a",
        "path": "data/PDF/a.pdf",
        "chunk_id": "chunk-a",
        "document_id": identity.document_id,
        "chunk_fingerprint": identity.chunk_fingerprint,
        "chunk_index": 0,
        "content_type": "body_text",
        "section_path": ["Section"],
        "page_start": 2,
        "page_end": 2,
        "source_block_ids": ["block-a"],
        "content": "Complete approved evidence",
    }
    preview = {
        "schema_version": "pdf-indexing-retrieval-preview-v1",
        "corpus_inventory": inventory.to_payload(),
        "index_manifest": {
            "index_id": manifest.index_id,
            "status": manifest.status.value,
            "identity": manifest.identity_payload,
            "required_pgvector_version": manifest.required_pgvector_version,
            "audit_metadata": {},
        },
        "chunk_catalog": [catalog_entry],
        "queries": [
            {
                "query_id": "query-a",
                "query": "What is the approved evidence?",
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
                        "content": catalog_entry["content"],
                        "content_type": "body_text",
                        "section_path": ["Section"],
                        "page_start": 2,
                        "page_end": 2,
                        "embedding_input_hash": sha256(b"input").hexdigest(),
                        "embedding_input_token_count": 4,
                        "provenance": {"schema_version": 1, "source_regions": []},
                    }
                ],
            }
        ],
    }
    decisions = {
        "schema_version": "pdf-retrieval-annotation-decisions-v1",
        "dataset_id": "fixture-evaluation",
        "dataset_version": "1.0.0",
        "recall_ks": [1, 3, 5],
        "mrr_k": 5,
        "cases": [
            {
                "query_id": "query-a",
                "case_type": "answerable",
                "relevant_chunks": [identity.to_payload()],
                "notes": "User approved this chunk.",
            }
        ],
    }

    dataset = build_approved_evaluation_dataset(preview=preview, decisions=decisions)

    reference = dataset.cases[0].relevant_chunks[0]
    assert dataset.corpus == inventory.snapshot
    assert dataset.cases[0].review_status is ReviewStatus.APPROVED
    assert reference.identity == identity
    assert reference.page_start == 2
    assert reference.source_block_ids == ("block-a",)
    assert reference.content == "Complete approved evidence"


def test_committed_pilot_dataset_contains_only_approved_gold() -> None:
    """The first baseline input stays bound to the accepted pilot review."""

    dataset_path = Path(__file__).parents[1] / "configs" / "pdf-retrieval-evaluation-pilot-v1.json"

    dataset = load_retrieval_evaluation_dataset(dataset_path)

    assert dataset.dataset_id == "pdf-retrieval-pilot"
    assert dataset.dataset_version == "1.0.0"
    assert dataset.corpus.corpus_id == "pdf-sample-corpus-v1"
    assert dataset.corpus.chunk_count == 87
    assert len(dataset.cases) == 6
    assert all(case.review_status is ReviewStatus.APPROVED for case in dataset.cases)
    assert [len(case.relevant_chunks) for case in dataset.cases] == [1, 1, 5, 1, 2, 2]


def test_committed_formal_dataset_matches_all_human_decisions() -> None:
    """The formal baseline input preserves fifteen scored and three diagnostic cases."""

    root = Path(__file__).parents[1]
    dataset = load_retrieval_evaluation_dataset(
        root / "configs" / "pdf-retrieval-evaluation-v1.json"
    )
    decisions = json.loads(
        (root / "configs" / "pdf-retrieval-annotation-decisions-v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert dataset.dataset_id == "pdf-retrieval-evaluation"
    assert dataset.dataset_version == "1.0.0"
    assert dataset.corpus.corpus_id == "pdf-sample-corpus-v1"
    assert dataset.corpus.chunk_count == 87
    assert Counter(case.case_type for case in dataset.cases) == {
        EvaluationCaseType.ANSWERABLE: 15,
        EvaluationCaseType.UNANSWERABLE: 2,
        EvaluationCaseType.AMBIGUOUS: 1,
    }
    assert all(case.review_status is ReviewStatus.APPROVED for case in dataset.cases)
    assert [len(case.relevant_chunks) for case in dataset.cases] == [
        1,
        1,
        5,
        1,
        2,
        2,
        3,
        1,
        1,
        1,
        1,
        2,
        1,
        1,
        1,
        0,
        0,
        0,
    ]
    assert [case.query_id for case in dataset.cases] == [
        decision["query_id"] for decision in decisions["cases"]
    ]
