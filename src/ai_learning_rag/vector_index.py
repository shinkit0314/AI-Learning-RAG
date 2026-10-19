"""A small exact-search vector index for learning the retrieval contract."""

from dataclasses import dataclass
from math import isfinite

from ai_learning_rag.vector_math import cosine_similarity

type MetadataValue = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """One embedded chunk and the payload needed to trace it to its source."""

    id: str
    content: str
    vector: tuple[float, ...]
    metadata: dict[str, MetadataValue]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One matching record together with its similarity score."""

    record: VectorRecord
    score: float


class InMemoryVectorIndex:
    """Store vectors in memory and rank every record with exact cosine search."""

    def __init__(self, *, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        self.dimension = dimension
        self._records: dict[str, VectorRecord] = {}

    def __len__(self) -> int:
        return len(self._records)

    def upsert(self, record: VectorRecord) -> None:
        """Insert a record, or replace the existing record with the same ID."""
        if not record.id.strip():
            raise ValueError("record id must not be blank")
        if not record.content.strip():
            raise ValueError("record content must not be blank")
        self._validate_vector(record.vector, label="record vector")
        self._records[record.id] = record

    def search(self, query_vector: tuple[float, ...], *, top_k: int = 5) -> list[SearchResult]:
        """Return the most similar records using exact cosine similarity."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self._validate_vector(query_vector, label="query vector")

        ranked_results = [
            SearchResult(
                record=record,
                score=cosine_similarity(query_vector, record.vector),
            )
            for record in self._records.values()
        ]
        ranked_results.sort(key=lambda result: (-result.score, result.record.id))
        return ranked_results[:top_k]

    def _validate_vector(self, vector: tuple[float, ...], *, label: str) -> None:
        if len(vector) != self.dimension:
            raise ValueError(f"{label} must have exactly {self.dimension} dimensions")
        if not all(isfinite(value) for value in vector):
            raise ValueError(f"{label} must contain only finite values")
        if not any(value != 0.0 for value in vector):
            raise ValueError(f"{label} must not be a zero vector")
