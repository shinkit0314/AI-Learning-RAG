"""Tests for the natural-language PDF retrieval adapter."""

from dataclasses import replace

import pytest

from ai_learning_rag.document_index import (
    IndexStatus,
    InMemoryVectorIndex,
    VectorIndexManifest,
)
from ai_learning_rag.embedding import EmbeddingConfig
from ai_learning_rag.pdf_retrieval import PdfRetriever, QueryTooLongError


def make_manifest(*, status: IndexStatus = IndexStatus.ACTIVE) -> VectorIndexManifest:
    """Create the fixed embedding-space identity used by adapter tests."""

    config = EmbeddingConfig(dimension=32, max_sequence_length=8, device="cpu")
    return VectorIndexManifest.create(
        logical_name="pdf-test-index",
        embedding_dimension=config.dimension,
        model_id=config.model_id,
        model_revision=config.model_revision,
        tokenizer_id=config.tokenizer_id,
        tokenizer_revision=config.tokenizer_revision,
        normalized=config.normalize_embeddings,
        pooling=config.pooling,
        dtype=config.dtype,
        query_profile=config.query_profile,
        document_input_profile="section-path-prefix-v1",
        max_input_tokens=config.max_sequence_length,
        oversize_policy="reject-v1",
        adapter_version=config.adapter_version,
        required_pgvector_version="0.8.7",
        status=status,
    )


class FakeEmbedder:
    """Return a controlled vector while recording query work."""

    def __init__(self, config: EmbeddingConfig | None = None, *, token_count: int = 3) -> None:
        self.config = config or EmbeddingConfig(
            dimension=32,
            max_sequence_length=8,
            device="cpu",
        )
        self.token_count = token_count
        self.counted: list[str] = []
        self.embedded: list[str] = []

    @property
    def dimension(self) -> int:
        return self.config.dimension

    def count_tokens(self, text: str) -> int:
        self.counted.append(text)
        return self.token_count

    def embed_query(self, query: str) -> list[float]:
        self.embedded.append(query)
        return [1.0, *([0.0] * (self.dimension - 1))]


def active_index(manifest: VectorIndexManifest) -> InMemoryVectorIndex:
    """Publish one manifest through the same lifecycle as production."""

    index = InMemoryVectorIndex()
    building = replace(manifest, status=IndexStatus.BUILDING)
    index.ensure_index(building)
    index.ensure_index(replace(building, status=IndexStatus.READY))
    index.ensure_index(replace(building, status=IndexStatus.ACTIVE))
    return index


def test_retriever_builds_document_search_query_from_natural_language() -> None:
    manifest = make_manifest()
    embedder = FakeEmbedder()
    retriever = PdfRetriever(
        index=active_index(manifest),
        manifest=manifest,
        embedder=embedder,
    )

    results = retriever.search("什么是可维护性？", top_k=3)

    assert results == []
    assert embedder.counted == ["什么是可维护性？"]
    assert embedder.embedded == ["什么是可维护性？"]


@pytest.mark.parametrize("query", ["", "   "])
def test_retriever_rejects_blank_query_before_model_or_index_work(query: str) -> None:
    manifest = make_manifest()
    embedder = FakeEmbedder()
    retriever = PdfRetriever(
        index=active_index(manifest),
        manifest=manifest,
        embedder=embedder,
    )

    with pytest.raises(ValueError, match="query must not be blank"):
        retriever.search(query)

    assert embedder.counted == []
    assert embedder.embedded == []


def test_retriever_rejects_oversize_query_before_embedding() -> None:
    manifest = make_manifest()
    embedder = FakeEmbedder(token_count=9)
    retriever = PdfRetriever(
        index=active_index(manifest),
        manifest=manifest,
        embedder=embedder,
    )

    with pytest.raises(QueryTooLongError, match="9 tokens"):
        retriever.search("过长问题")

    assert embedder.counted == ["过长问题"]
    assert embedder.embedded == []


def test_retriever_rejects_incompatible_embedder_during_construction() -> None:
    manifest = make_manifest()
    incompatible = FakeEmbedder(
        EmbeddingConfig(
            dimension=64,
            max_sequence_length=8,
            device="cpu",
        )
    )

    with pytest.raises(ValueError, match="embedding_dimension"):
        PdfRetriever(
            index=active_index(manifest),
            manifest=manifest,
            embedder=incompatible,
        )


def test_retriever_requires_the_bound_index_to_be_active() -> None:
    manifest = make_manifest(status=IndexStatus.BUILDING)
    index = InMemoryVectorIndex()
    index.ensure_index(manifest)
    embedder = FakeEmbedder()
    retriever = PdfRetriever(index=index, manifest=manifest, embedder=embedder)

    with pytest.raises(ValueError, match="index must be active"):
        retriever.search("问题")

    assert embedder.counted == []
    assert embedder.embedded == []
