"""API tests for plain-text document previews."""

from fastapi.testclient import TestClient

from ai_learning_rag.main import create_app


def test_preview_document_returns_normalized_traceable_chunks() -> None:
    """The preview endpoint should expose chunks without storing the document."""
    client = TestClient(create_app())

    response = client.post(
        "/documents/preview",
        json={
            "title": "  Demo  ",
            "source": "manual://demo",
            "text": "abcdefghij",
            "chunk_size_chars": 6,
            "chunk_overlap_chars": 2,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"].startswith("doc_")
    assert body["normalized_text"] == "abcdefghij"
    assert body["chunk_count"] == 2
    assert [
        {
            "content": chunk["content"],
            "metadata": chunk["metadata"],
        }
        for chunk in body["chunks"]
    ] == [
        {
            "content": "abcdef",
            "metadata": {
                "document_id": body["document_id"],
                "title": "Demo",
                "source": "manual://demo",
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 6,
            },
        },
        {
            "content": "efghij",
            "metadata": {
                "document_id": body["document_id"],
                "title": "Demo",
                "source": "manual://demo",
                "chunk_index": 1,
                "start_char": 4,
                "end_char": 10,
            },
        },
    ]


def test_preview_document_rejects_overlap_that_cannot_advance() -> None:
    """The API should explain invalid chunk settings as a validation error."""
    client = TestClient(create_app())

    response = client.post(
        "/documents/preview",
        json={
            "title": "Demo",
            "source": "manual://demo",
            "text": "content",
            "chunk_size_chars": 5,
            "chunk_overlap_chars": 5,
        },
    )

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(
        "chunk_overlap_chars must be smaller than chunk_size_chars" in error["msg"]
        for error in errors
    )
