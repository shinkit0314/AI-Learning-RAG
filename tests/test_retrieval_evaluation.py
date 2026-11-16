"""Tests for deterministic, human-audited PDF retrieval evaluation."""

import json
from hashlib import sha256
from pathlib import Path

import pytest

from ai_learning_rag.document_index import (
    IndexStatus,
    SearchResult,
    VectorIndexManifest,
    VectorRecord,
)
from ai_learning_rag.retrieval_evaluation import (
    ChunkIdentity,
    CorpusCompatibilityError,
    CorpusInventory,
    CorpusSample,
    EvaluationCaseType,
    RelevantChunkReference,
    RetrievalEvaluationCase,
    RetrievalEvaluationDataset,
    ReviewStatus,
    evaluate_retrieval,
    load_retrieval_evaluation_dataset,
)


def make_manifest() -> VectorIndexManifest:
    """Create one active embedding manifest for report audit fields."""

    return VectorIndexManifest.create(
        logical_name="evaluation-fixture",
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


def make_identity(label: str, *, document_id: str = "document-a") -> ChunkIdentity:
    """Create a stable full-length chunk identity from a readable label."""

    return ChunkIdentity(document_id, sha256(label.encode()).hexdigest())


def make_inventory(*chunks: ChunkIdentity) -> CorpusInventory:
    """Build the exact small corpus shared by evaluator tests."""

    return CorpusInventory.create(
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
        chunks=chunks,
    )


def make_reference(
    identity: ChunkIdentity, *, content: str = "Relevant evidence"
) -> RelevantChunkReference:
    """Attach human-review context to a gold chunk key."""

    return RelevantChunkReference(
        identity=identity,
        sample_id="sample-a",
        page_start=1,
        page_end=1,
        content_type="body_text",
        section_path=("Section",),
        source_block_ids=(f"block-{identity.chunk_fingerprint[:8]}",),
        content=content,
    )


def make_case(
    query_id: str,
    *gold: ChunkIdentity,
    case_type: EvaluationCaseType = EvaluationCaseType.ANSWERABLE,
    review_status: ReviewStatus = ReviewStatus.APPROVED,
) -> RetrievalEvaluationCase:
    """Create one approved answerable or diagnostic question."""

    return RetrievalEvaluationCase(
        query_id=query_id,
        query=f"Question for {query_id}?",
        case_type=case_type,
        review_status=review_status,
        relevant_chunks=tuple(make_reference(identity) for identity in gold),
    )


def make_dataset(
    inventory: CorpusInventory,
    *cases: RetrievalEvaluationCase,
) -> RetrievalEvaluationDataset:
    """Bind fixed questions to the exact fixture corpus."""

    return RetrievalEvaluationDataset(
        dataset_id="fixture-evaluation",
        dataset_version="1.0.0",
        corpus=inventory.snapshot,
        recall_ks=(1, 3, 5),
        mrr_k=5,
        cases=cases,
    )


def make_result(
    identity: ChunkIdentity,
    *,
    score: float,
    content: str = "Retrieved evidence",
) -> SearchResult:
    """Create one fully traceable result without a model or database."""

    record_digest = sha256(
        f"{identity.document_id}-{identity.chunk_fingerprint}".encode()
    ).hexdigest()
    return SearchResult(
        record=VectorRecord(
            record_id=f"rec_{record_digest}",
            index_id=make_manifest().index_id,
            generation_id=f"gen_{'1' * 64}",
            document_id=identity.document_id,
            chunk_fingerprint=identity.chunk_fingerprint,
            chunk_index=0,
            content=content,
            content_type="body_text",
            section_path=("Section",),
            page_start=1,
            page_end=1,
            embedding_input_hash=sha256(b"embedding-input").hexdigest(),
            embedding_input_token_count=4,
            vector=(1.0, 0.0),
            provenance={"schema_version": 1, "source_regions": []},
        ),
        score=score,
    )


def test_corpus_snapshot_is_stable_across_input_order() -> None:
    """Compatibility must depend on chunk membership, not caller iteration order."""

    first = make_identity("first")
    second = make_identity("second")

    forward = make_inventory(first, second)
    reverse = make_inventory(second, first)

    assert forward.snapshot == reverse.snapshot
    assert forward.snapshot.chunk_count == 2


def test_question_types_enforce_relevance_cardinality() -> None:
    """Formal and diagnostic questions must not silently exchange semantics."""

    identity = make_identity("gold")

    with pytest.raises(ValueError, match="answerable questions require"):
        make_case("missing-gold")
    with pytest.raises(ValueError, match="diagnostic questions must have an empty"):
        make_case("unexpected-gold", identity, case_type=EvaluationCaseType.UNANSWERABLE)


def test_evaluator_computes_macro_recall_and_mrr() -> None:
    """Recall counts all gold chunks while MRR uses only the first relevant rank."""

    a = make_identity("a")
    b = make_identity("b")
    c = make_identity("c")
    distractor = make_identity("distractor")
    inventory = make_inventory(a, b, c, distractor)
    dataset = make_dataset(
        inventory,
        make_case("multi", a, b),
        make_case("single", c),
    )

    report = evaluate_retrieval(
        dataset=dataset,
        inventory=inventory,
        results_by_query={
            "multi": [
                make_result(distractor, score=0.9),
                make_result(a, score=0.8),
                make_result(b, score=0.7),
            ],
            "single": [make_result(c, score=0.95)],
        },
        index_manifest=make_manifest(),
    )

    assert dict(report.queries[0].recall_at_k) == {1: 0.0, 3: 1.0, 5: 1.0}
    assert report.queries[0].first_relevant_rank == 2
    assert report.queries[0].reciprocal_rank == 0.5
    assert dict(report.aggregate.recall_at_k) == {1: 0.5, 3: 1.0, 5: 1.0}
    assert report.aggregate.mrr == 0.75


def test_missing_relevant_hit_scores_zero_and_remains_auditable() -> None:
    """A real gold set with no Top-K hit is a retrieval miss, not an empty-gold case."""

    gold = make_identity("gold")
    distractor = make_identity("distractor")
    inventory = make_inventory(gold, distractor)
    dataset = make_dataset(inventory, make_case("miss", gold))

    report = evaluate_retrieval(
        dataset=dataset,
        inventory=inventory,
        results_by_query={"miss": [make_result(distractor, score=0.8)]},
        index_manifest=make_manifest(),
    )

    assert dict(report.queries[0].recall_at_k) == {1: 0.0, 3: 0.0, 5: 0.0}
    assert report.queries[0].reciprocal_rank == 0.0
    assert report.queries[0].missing_relevant == (gold,)


def test_diagnostic_cases_are_excluded_from_formal_metrics() -> None:
    """An empty gold set is diagnostic and must never become a fake zero or perfect score."""

    gold = make_identity("gold")
    distractor = make_identity("distractor")
    inventory = make_inventory(gold, distractor)
    dataset = make_dataset(
        inventory,
        make_case("answerable", gold),
        make_case("unsupported", case_type=EvaluationCaseType.UNANSWERABLE),
        make_case("ambiguous", case_type=EvaluationCaseType.AMBIGUOUS),
    )

    report = evaluate_retrieval(
        dataset=dataset,
        inventory=inventory,
        results_by_query={
            "answerable": [make_result(gold, score=0.9)],
            "unsupported": [make_result(distractor, score=0.8)],
            "ambiguous": [make_result(distractor, score=0.7)],
        },
        index_manifest=make_manifest(),
    )

    assert report.aggregate.scored_query_count == 1
    assert report.aggregate.diagnostic_query_count == 2
    assert report.aggregate.mrr == 1.0
    assert report.queries[1].reciprocal_rank is None
    assert report.queries[2].recall_at_k == ()


def test_evaluator_rejects_a_different_chunk_corpus() -> None:
    """Adding or removing a chunk must create a new corpus baseline instead of fake regression."""

    gold = make_identity("gold")
    original = make_inventory(gold)
    expanded = make_inventory(gold, make_identity("new-chunk"))
    dataset = make_dataset(original, make_case("query", gold))

    with pytest.raises(CorpusCompatibilityError, match="does not match"):
        evaluate_retrieval(
            dataset=dataset,
            inventory=expanded,
            results_by_query={"query": [make_result(gold, score=1.0)]},
            index_manifest=make_manifest(),
        )


def test_evaluator_rejects_gold_absent_from_the_compatible_corpus() -> None:
    """An invalid gold key must not be misreported as a retrieval miss."""

    available = make_identity("available")
    absent = make_identity("absent")
    inventory = make_inventory(available)
    dataset = make_dataset(inventory, make_case("query", absent))

    with pytest.raises(CorpusCompatibilityError, match="absent"):
        evaluate_retrieval(
            dataset=dataset,
            inventory=inventory,
            results_by_query={"query": []},
            index_manifest=make_manifest(),
        )


def test_evaluator_rejects_unapproved_questions() -> None:
    """AI-proposed drafts must never silently enter formal baseline metrics."""

    gold = make_identity("gold")
    inventory = make_inventory(gold)
    dataset = make_dataset(
        inventory,
        make_case("draft", gold, review_status=ReviewStatus.DRAFT),
    )

    with pytest.raises(ValueError, match="requires approved cases"):
        evaluate_retrieval(
            dataset=dataset,
            inventory=inventory,
            results_by_query={"draft": [make_result(gold, score=1.0)]},
            index_manifest=make_manifest(),
        )


def test_evaluator_rejects_duplicate_returned_chunk_identity() -> None:
    """One chunk cannot occupy two ranks and inflate or confuse retrieval metrics."""

    gold = make_identity("gold")
    inventory = make_inventory(gold)
    dataset = make_dataset(inventory, make_case("query", gold))
    repeated = make_result(gold, score=1.0)

    with pytest.raises(ValueError, match="duplicate chunk identity"):
        evaluate_retrieval(
            dataset=dataset,
            inventory=inventory,
            results_by_query={"query": [repeated, repeated]},
            index_manifest=make_manifest(),
        )


def test_evaluator_rejects_returned_chunk_outside_the_corpus() -> None:
    """A ranking from another corpus must not be scored as an ordinary distractor."""

    gold = make_identity("gold")
    outside = make_identity("outside")
    inventory = make_inventory(gold)
    dataset = make_dataset(inventory, make_case("query", gold))

    with pytest.raises(CorpusCompatibilityError, match="outside the compatible corpus"):
        evaluate_retrieval(
            dataset=dataset,
            inventory=inventory,
            results_by_query={"query": [make_result(outside, score=0.9)]},
            index_manifest=make_manifest(),
        )


def test_dataset_json_round_trip_preserves_the_review_contract(tmp_path: Path) -> None:
    """The committed JSON artifact must load into the same validated dataset."""

    gold = make_identity("gold")
    inventory = make_inventory(gold)
    expected = make_dataset(inventory, make_case("query", gold))
    path = tmp_path / "evaluation.json"
    path.write_text(
        json.dumps(expected.to_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    observed = load_retrieval_evaluation_dataset(path)

    assert observed == expected


def test_json_and_markdown_reports_preserve_complete_ranked_content() -> None:
    """A compact summary must not hide the exact evidence behind a score."""

    gold = make_identity("gold")
    inventory = make_inventory(gold)
    dataset = make_dataset(inventory, make_case("query", gold))
    content = f"First line\n{'x' * 220}\nTAIL_MARKER"
    report = evaluate_retrieval(
        dataset=dataset,
        inventory=inventory,
        results_by_query={"query": [make_result(gold, score=1.0, content=content)]},
        index_manifest=make_manifest(),
    )

    payload = json.loads(report.to_json())
    markdown = report.to_markdown()

    assert payload["metrics"]["recall_at_k"] == {"1": 1.0, "3": 1.0, "5": 1.0}
    assert payload["queries"][0]["hits"][0]["content"] == content
    assert "MRR@5: `1.000000`" in markdown
    assert f"> {'x' * 220}" in markdown
    assert "> TAIL_MARKER" in markdown
