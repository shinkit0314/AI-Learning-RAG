"""Unit tests for plain-text normalization and chunking."""

import pytest

from ai_learning_rag.text_processing import build_document, split_document


def test_build_document_normalizes_text_and_has_stable_id() -> None:
    """Equivalent input should produce clean text and a repeatable identifier."""
    first = build_document(
        title="  RAG basics  ",
        source="  manual://rag-basics  ",
        text="Retrieval\n\n augmented   generation",
    )
    second = build_document(
        title="RAG basics",
        source="manual://rag-basics",
        text="Retrieval augmented generation",
    )

    assert first.text == "Retrieval augmented generation"
    assert first.title == "RAG basics"
    assert first.source == "manual://rag-basics"
    assert first.id == second.id


def test_split_document_keeps_overlap_and_character_offsets() -> None:
    """Every chunk should be traceable to an exact document slice."""
    document = build_document(
        title="Alphabet",
        source="manual://alphabet",
        text="abcdefghijklmnop",
    )

    chunks = split_document(document, chunk_size_chars=6, chunk_overlap_chars=2)

    assert [(chunk.text, chunk.start_char, chunk.end_char) for chunk in chunks] == [
        ("abcdef", 0, 6),
        ("efghij", 4, 10),
        ("ijklmn", 8, 14),
        ("mnop", 12, 16),
    ]
    assert all(chunk.document_id == document.id for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3]
    assert len({chunk.id for chunk in chunks}) == len(chunks)


@pytest.mark.parametrize(
    ("chunk_size_chars", "chunk_overlap_chars"),
    [(0, 0), (5, -1), (5, 5), (5, 6)],
)
def test_split_document_rejects_invalid_windows(
    chunk_size_chars: int,
    chunk_overlap_chars: int,
) -> None:
    """Invalid windows should fail clearly instead of looping or losing data."""
    document = build_document(title="Title", source="manual://source", text="content")

    with pytest.raises(ValueError):
        split_document(
            document,
            chunk_size_chars=chunk_size_chars,
            chunk_overlap_chars=chunk_overlap_chars,
        )
