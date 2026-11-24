from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from runpy import run_path

import pytest

from ai_learning_rag.document_index import SearchResult, VectorRecord
from ai_learning_rag.hybrid_ranking import fuse_rankings
from ai_learning_rag.retrieval_evaluation import ChunkIdentity


def candidate(record_id: str, score: float = 0.0) -> SearchResult:
    digest = sha256(record_id.encode()).hexdigest()
    record = VectorRecord(
        record_id=record_id,
        index_id="index",
        generation_id="generation",
        document_id="document",
        chunk_fingerprint=digest,
        chunk_index=0,
        content=f"Content for {record_id}",
        content_type="body_text",
        section_path=(),
        page_start=1,
        page_end=1,
        embedding_input_hash=digest,
        embedding_input_token_count=1,
        vector=(1.0, 0.0),
        provenance={"schema_version": 1},
    )
    return SearchResult(record=record, score=score)


def test_equal_weight_rrf_merges_duplicates_and_uses_record_id_for_ties() -> None:
    shared = candidate("shared", 0.8)
    vectors = [candidate("z-vector-only", 0.99), shared]
    keywords = [candidate("a-keywords-only", 20.0), shared]

    result = fuse_rankings(vectors, keywords, top_k=2)

    assert result.vector_candidates == tuple(vectors)
    assert result.keyword_candidates == tuple(keywords)
    assert [hit.record.record_id for hit in result.fused_candidates] == [
        "shared",
        "a-keywords-only",
        "z-vector-only",
    ]
    assert result.hits == result.fused_candidates[:2]
    assert result.hits[0].score == pytest.approx(2 / 62)
    assert result.hits[1].score == pytest.approx(1 / 61)
    assert result.hits[0].record.provenance == shared.record.provenance


@pytest.mark.parametrize("keyword_side", [False, True])
def test_single_lane_keeps_its_order_and_returns_at_most_five(keyword_side: bool) -> None:
    ranking = [candidate(str(index)) for index in range(10)]
    vectors, keywords = ([], ranking) if keyword_side else (ranking, [])

    result = fuse_rankings(vectors, keywords, top_k=10)

    assert [hit.record for hit in result.hits] == [hit.record for hit in ranking[:5]]
    assert len(result.fused_candidates) == 10


def test_no_candidates_returns_no_hits() -> None:
    assert fuse_rankings([], []).hits == ()


@pytest.mark.parametrize("leak_candidate", [False, True])
def test_filter_evaluation_checks_both_hybrid_candidate_lists(
    monkeypatch: pytest.MonkeyPatch, leak_candidate: bool
) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    evaluate_filter = run_path(str(scripts / "pdf_hybrid_experiment.py"))["filter_comparison"]
    inside = candidate("inside")
    outside = replace(inside, record=replace(inside.record, document_id="other"))
    identity = ChunkIdentity(inside.record.document_id, inside.record.chunk_fingerprint)
    unfiltered = [outside, inside]
    candidates = (outside,) if leak_candidate else (inside,)
    fused = fuse_rankings((inside,), candidates, top_k=1)
    rankings = {name: {"layout-figure-two-dimensions": unfiltered} for name in ("vector", "hybrid")}
    scoped = {"vector_scoped": [inside], "hybrid_scoped": fused}

    if leak_candidate:
        with pytest.raises(RuntimeError, match="hybrid candidate crossed"):
            evaluate_filter(rankings, scoped, identity.document_id, identity)
    else:
        result = evaluate_filter(rankings, scoped, identity.document_id, identity)
        assert result["hybrid"]["outside_before_after"] == [1, 0]
        assert result["hybrid"]["gold_rank_before_after"] == [2, 1]
