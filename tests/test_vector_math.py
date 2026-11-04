"""Tests for vector operations used during retrieval."""

import pytest

from ai_learning_rag.vector_math import cosine_similarity


def test_cosine_similarity_is_one_for_the_same_direction() -> None:
    """Vectors pointing in the same direction should be maximally similar."""
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_is_zero_for_perpendicular_vectors() -> None:
    """Perpendicular vectors should have no directional similarity."""
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_ranks_the_relevant_chunk_higher() -> None:
    """The toy query vector should rank the RAG chunk above the uv chunk."""
    query = [0.9, 0.1]
    rag_chunk = [0.8, 0.2]
    uv_chunk = [0.1, 0.9]

    assert cosine_similarity(query, rag_chunk) > cosine_similarity(query, uv_chunk)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([1.0], [1.0, 0.0]),
        ([0.0, 0.0], [1.0, 0.0]),
        ([1.0, 0.0], [0.0, 0.0]),
    ],
)
def test_cosine_similarity_rejects_invalid_vectors(
    left: list[float],
    right: list[float],
) -> None:
    """Invalid vectors should fail instead of producing a misleading score."""
    with pytest.raises(ValueError):
        cosine_similarity(left, right)
