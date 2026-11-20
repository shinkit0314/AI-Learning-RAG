"""Regression checks for the replay rule used before comparing query prompts."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from runpy import run_path

import pytest
from test_document_index import activate, make_generation, make_manifest, make_record

from ai_learning_rag.document_index import IndexStatus, InMemoryVectorIndex, VectorSearchQuery

experiment = run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "pdf_embedding_experiment.py")
)
check_control = experiment["check_control"]
check_runtime_versions = experiment["check_runtime_versions"]
copy_index = experiment["copy_index"]


def sample_report() -> dict:
    return {
        "dataset": {"dataset_version": "1.0.0"},
        "corpus": {},
        "index": {},
        "metrics": {},
        "queries": [
            {
                "query_id": "sample-question",
                "query": "Where is the evidence?",
                "case_type": "answerable",
                "metrics": {"first_relevant_rank": 1},
                "hits": [
                    {
                        "document_id": "document",
                        "chunk_fingerprint": label,
                        "relevant": rank == 0,
                        "score": 1.0 - rank * 0.1,
                    }
                    for rank, label in enumerate(("evidence", "other-a", "other-b"))
                ],
            }
        ],
    }


def test_control_accepts_nonrelevant_rank_and_score_changes() -> None:
    original = sample_report()
    candidate = deepcopy(original)
    hits = candidate["queries"][0]["hits"]
    hits[0]["score"] = 0.99
    hits[1], hits[2] = hits[2], hits[1]

    check_control(candidate, original)


def test_control_rejects_different_evidence_even_when_metrics_match() -> None:
    original = sample_report()
    candidate = deepcopy(original)
    candidate["queries"][0]["hits"][0]["chunk_fingerprint"] = "other-evidence"

    with pytest.raises(RuntimeError, match="changed evidence for sample-question at 1"):
        check_control(candidate, original)


def test_runtime_matches_the_audit_key_to_the_distribution_name(monkeypatch) -> None:
    installed = {"torch": "2.14.1+cu126", "sentence-transformers": "5.7.0"}
    monkeypatch.setitem(check_runtime_versions.__globals__, "version", installed.__getitem__)

    check_runtime_versions({"torch": "2.14.1+cu126", "sentence_transformers": "5.7.0"})

    with pytest.raises(ValueError, match="runtime sentence-transformers differs"):
        check_runtime_versions({"torch": "2.14.1+cu126", "sentence_transformers": "5.6.0"})


def test_candidate_reuses_vectors_without_replacing_baseline() -> None:
    index = InMemoryVectorIndex()
    manifest = make_manifest()
    activate(index, manifest)
    baseline = replace(manifest, status=IndexStatus.ACTIVE)
    generation = make_generation(baseline)
    source = make_record(generation, record_id="original", vector=(0.6, 0.8))
    index.replace_document(generation, (source,), expected_revision=0)

    candidate, written = copy_index(index, baseline, (source,), "pdf-evidence")
    reused, repeated_writes = copy_index(index, baseline, (source,), "pdf-evidence")

    assert (candidate, written, repeated_writes) == (reused, 1, 0)
    assert index.get_manifest(baseline.index_id) == baseline
    assert candidate.index_id != baseline.index_id
    [hit] = index.search(VectorSearchQuery(index_id=candidate.index_id, vector=(1.0, 0.0)))
    assert hit.record.vector == source.vector
    assert hit.record.chunk_fingerprint == source.chunk_fingerprint
    assert hit.record.embedding_input_hash == source.embedding_input_hash
