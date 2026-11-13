"""Offline helper tests for the PostgreSQL vector index adapter."""

import pytest

from ai_learning_rag.postgres_vector_index import (
    _parse_vector_literal,
    _vector_literal,
    _version_at_least,
)


@pytest.mark.parametrize(
    ("actual", "required", "expected"),
    [
        ("0.8.7", "0.8.7", True),
        ("0.9.0", "0.8.7", True),
        ("0.8.6", "0.8.7", False),
        ("0.8.7-dev1", "0.8.7", True),
    ],
)
def test_pgvector_version_comparison(
    actual: str,
    required: str,
    expected: bool,
) -> None:
    """Extension checks should compare numeric versions instead of strings."""

    assert _version_at_least(actual, required) is expected


def test_vector_literal_round_trip_preserves_values() -> None:
    """Explicit casts avoid relying on global psycopg adapter registration."""

    vector = (1.0, -0.25, 0.3333333333333333)

    assert _parse_vector_literal(_vector_literal(vector)) == vector


@pytest.mark.parametrize("value", [(), (float("nan"),), (float("inf"),)])
def test_vector_literal_rejects_invalid_values(value: tuple[float, ...]) -> None:
    """Invalid data must fail before reaching PostgreSQL."""

    with pytest.raises(ValueError, match="finite values"):
        _vector_literal(value)


def test_vector_parser_rejects_non_vector_text() -> None:
    """Unexpected driver output should be visible instead of silently coerced."""

    with pytest.raises(ValueError, match="invalid pgvector"):
        _parse_vector_literal("1,2")
