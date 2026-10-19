"""Small vector operations used to teach and test semantic retrieval."""

from collections.abc import Sequence
from math import fsum, sqrt


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return directional similarity for two non-zero vectors of equal length."""
    if not left or len(left) != len(right):
        raise ValueError("vectors must be non-empty and have the same dimensions")

    left_norm = sqrt(fsum(value * value for value in left))
    right_norm = sqrt(fsum(value * value for value in right))

    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")

    dot_product = fsum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )
    return dot_product / (left_norm * right_norm)
