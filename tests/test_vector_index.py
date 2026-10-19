"""Tests for the exact-search learning index."""

import pytest

from ai_learning_rag.vector_index import InMemoryVectorIndex, VectorRecord


def make_record(record_id: str, content: str, vector: tuple[float, ...]) -> VectorRecord:
    """Create a traceable record for retrieval tests."""
    return VectorRecord(
        id=record_id,
        content=content,
        vector=vector,
        metadata={"source": f"manual://{record_id}", "chunk_index": 0},
    )


def test_search_returns_top_k_in_similarity_order() -> None:
    """The index should return A then B while excluding irrelevant C."""
    index = InMemoryVectorIndex(dimension=2)
    index.upsert(make_record("a", "Embedding 会把文本映射到向量空间。", (1.0, 0.0)))
    index.upsert(make_record("b", "RAG 会先检索资料，再生成回答。", (0.8, 0.2)))
    index.upsert(make_record("c", "uv 用于管理 Python 项目依赖。", (0.0, 1.0)))

    results = index.search((1.0, 0.0), top_k=2)

    assert [result.record.id for result in results] == ["a", "b"]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].record.metadata == {
        "source": "manual://a",
        "chunk_index": 0,
    }


def test_upsert_replaces_a_record_instead_of_duplicating_it() -> None:
    """Stable chunk IDs should make document re-indexing idempotent."""
    index = InMemoryVectorIndex(dimension=2)
    index.upsert(make_record("chunk-1", "旧内容", (1.0, 0.0)))

    index.upsert(make_record("chunk-1", "新内容", (0.0, 1.0)))

    assert len(index) == 1
    result = index.search((0.0, 1.0), top_k=1)[0]
    assert result.record.content == "新内容"


def test_search_returns_every_record_when_top_k_exceeds_index_size() -> None:
    """A small index should not invent results to fill top_k."""
    index = InMemoryVectorIndex(dimension=2)
    index.upsert(make_record("a", "A", (1.0, 0.0)))

    assert len(index.search((1.0, 0.0), top_k=5)) == 1


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ((1.0,), "exactly 2 dimensions"),
        ((0.0, 0.0), "must not be a zero vector"),
        ((float("nan"), 1.0), "only finite values"),
    ],
)
def test_upsert_rejects_invalid_vectors(
    vector: tuple[float, ...],
    message: str,
) -> None:
    """Invalid vectors must not be allowed to corrupt the index."""
    index = InMemoryVectorIndex(dimension=2)

    with pytest.raises(ValueError, match=message):
        index.upsert(make_record("bad", "invalid", vector))


def test_search_rejects_invalid_top_k() -> None:
    """A retrieval request must ask for at least one result."""
    index = InMemoryVectorIndex(dimension=2)

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        index.search((1.0, 0.0), top_k=0)
