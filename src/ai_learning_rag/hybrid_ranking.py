"""Rank two candidate lists without comparing their retrieval scores."""

from collections.abc import Sequence
from dataclasses import dataclass

from ai_learning_rag.document_index import SearchResult

HYBRID_CANDIDATE_LIMIT = 10
HYBRID_OUTPUT_LIMIT = 5
RRF_RANK_CONSTANT = 60


@dataclass(frozen=True, slots=True)
class HybridSearchResult:
    """Both source rankings and the fused order used to select the final hits."""

    vector_candidates: tuple[SearchResult, ...]
    keyword_candidates: tuple[SearchResult, ...]
    fused_candidates: tuple[SearchResult, ...]
    top_k: int

    @property
    def hits(self) -> tuple[SearchResult, ...]:
        """Return the requested number of hits up to the stage output limit."""
        return self.fused_candidates[: min(self.top_k, HYBRID_OUTPUT_LIMIT)]


def fuse_rankings(
    vector_candidates: Sequence[SearchResult],
    keyword_candidates: Sequence[SearchResult],
    *,
    top_k: int = HYBRID_OUTPUT_LIMIT,
) -> HybridSearchResult:
    """Apply equal-weight RRF to two rankings of distinct active records."""

    by_record: dict[str, SearchResult] = {}
    for candidates in (vector_candidates, keyword_candidates):
        for rank, candidate in enumerate(candidates, start=1):
            record = candidate.record
            previous = by_record.get(record.record_id)
            score = 1 / (RRF_RANK_CONSTANT + rank)
            if previous is not None:
                score += previous.score
            by_record[record.record_id] = SearchResult(record=record, score=score)

    return HybridSearchResult(
        vector_candidates=tuple(vector_candidates),
        keyword_candidates=tuple(keyword_candidates),
        fused_candidates=tuple(
            sorted(by_record.values(), key=lambda hit: (-hit.score, hit.record.record_id))
        ),
        top_k=top_k,
    )
