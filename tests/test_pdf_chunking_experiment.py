"""Source coverage checks for the 5.2 chunking gate."""

from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import pytest


@pytest.fixture
def experiment(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return run_path(str(scripts / "pdf_chunking_experiment.py"))


def test_adjacent_and_overlapping_candidates_keep_original_span(experiment) -> None:
    uncovered_ranges = experiment["uncovered_ranges"]
    assert uncovered_ranges(10, 40, [(27, 50), (0, 25), (18, 27)]) == []


def test_rejected_source_text_keeps_exact_character_gaps(experiment) -> None:
    uncovered_ranges = experiment["uncovered_ranges"]
    assert uncovered_ranges(10, 40, [(15, 20), (25, 27), (100, 110)]) == [
        (10, 15),
        (20, 25),
        (27, 40),
    ]


def test_original_evidence_remaps_across_candidate_boundaries_and_true_ranks(experiment) -> None:
    evidence_rank = experiment["evidence_rank"]

    def chunk(document, start, end):
        return SimpleNamespace(
            id="example",
            document_id=document,
            source_fingerprint="example",
            content="x" * (end - start),
            source_regions=(
                SimpleNamespace(
                    page_number=3,
                    block_id="paragraph",
                    source_text_start=start,
                    source_text_end=end,
                    chunk_content_start=0,
                ),
            ),
        )

    original = chunk("source", 10, 32)
    assert evidence_rank(original, [chunk("different", 0, 38)]) is None
    assert evidence_rank(original, [chunk("source", 0, 20), None, chunk("source", 20, 40)]) == 3
