"""Tests for the chunk-to-retrieval orchestration."""

from collections.abc import Sequence

import pytest

from ai_learning_rag.retrieval import RetrievalPipeline
from ai_learning_rag.text_processing import build_document
from ai_learning_rag.vector_index import InMemoryVectorIndex


class FakeEmbedder:
    """Deterministic semantic stand-in for fast retrieval tests."""

    def __init__(self, *, dimension: int = 2) -> None:
        self.dimension = dimension
        self.document_vectors = {
            "Embedding 会把文本映射到可比较的向量空间。": [1.0, 0.0],
            "RAG 会先检索资料，再根据资料生成回答。": [0.8, 0.2],
            "uv 用于管理 Python 项目的依赖。": [0.0, 1.0],
        }

    def embed_query(self, query: str) -> list[float]:
        assert query == "RAG 为什么需要把文本转换成向量？"
        return [1.0, 0.0]

    def embed_documents(self, documents: Sequence[str]) -> list[list[float]]:
        return [self.document_vectors[document] for document in documents]


def make_learning_documents():
    """Build the A, B, C documents from the learner's ranking exercise."""
    return [
        build_document(
            title="Embedding",
            source="manual://a",
            text="Embedding 会把文本映射到可比较的向量空间。",
        ),
        build_document(
            title="RAG",
            source="manual://b",
            text="RAG 会先检索资料，再根据资料生成回答。",
        ),
        build_document(
            title="uv",
            source="manual://c",
            text="uv 用于管理 Python 项目的依赖。",
        ),
    ]


def test_pipeline_connects_chunking_embedding_indexing_and_top_k() -> None:
    """The complete retrieval path should return A then B with source metadata."""
    index = InMemoryVectorIndex(dimension=2)
    pipeline = RetrievalPipeline(embedder=FakeEmbedder(), index=index)

    indexed_ids = pipeline.index_documents(
        make_learning_documents(),
        chunk_size_chars=100,
        chunk_overlap_chars=0,
    )
    results = pipeline.search("RAG 为什么需要把文本转换成向量？", top_k=2)

    assert len(indexed_ids) == 3
    assert len(index) == 3
    assert [result.record.metadata["source"] for result in results] == [
        "manual://a",
        "manual://b",
    ]
    assert results[0].record.metadata["title"] == "Embedding"
    assert results[0].record.metadata["start_char"] == 0


def test_reindexing_the_same_documents_is_idempotent() -> None:
    """Stable chunk IDs plus upsert should prevent duplicate records."""
    index = InMemoryVectorIndex(dimension=2)
    pipeline = RetrievalPipeline(embedder=FakeEmbedder(), index=index)
    documents = make_learning_documents()

    pipeline.index_documents(documents, chunk_size_chars=100, chunk_overlap_chars=0)
    pipeline.index_documents(documents, chunk_size_chars=100, chunk_overlap_chars=0)

    assert len(index) == 3


def test_pipeline_rejects_embedding_and_index_dimension_mismatch() -> None:
    """A mismatch must fail rather than padding or truncating vectors silently."""
    with pytest.raises(
        ValueError,
        match="embedding dimension does not match index dimension: 768 != 1024",
    ):
        RetrievalPipeline(
            embedder=FakeEmbedder(dimension=768),
            index=InMemoryVectorIndex(dimension=1024),
        )
