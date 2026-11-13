"""Unit tests for the local embedding boundary."""

from collections import UserDict
from typing import Any

import pytest

from ai_learning_rag.embedding import EmbeddingConfig, Qwen3Embedder


class FakeEncoderModel:
    """Predictable stand-in that avoids downloading a real model in tests."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.max_seq_length = 0
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.tokenizer = FakeTokenizer()

    def encode(self, inputs: list[str], **kwargs: Any) -> list[list[float]]:
        self.calls.append((inputs, kwargs))
        return [[float(index) for index in range(self.dimension)] for _ in inputs]


class FakeTokenizer:
    """Expose untruncated token IDs without any external model dependency."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool, bool]] = []

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> dict[str, list[int]]:
        self.calls.append((text, add_special_tokens, truncation))
        return {"input_ids": [101, *range(len(text.split())), 102]}


class BatchEncodingLikeTokenizer(FakeTokenizer):
    """Mimic Transformers' mapping-based BatchEncoding return type."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> UserDict[str, list[int]]:
        self.calls.append((text, add_special_tokens, truncation))
        return UserDict({"input_ids": [101, *range(len(text.split())), 102]})


def test_query_uses_retrieval_prompt_and_normalized_vectors() -> None:
    """Queries need Qwen's query instruction while documents do not."""
    config = EmbeddingConfig(dimension=32, max_sequence_length=512, batch_size=4)
    model = FakeEncoderModel(dimension=config.dimension)
    embedder = Qwen3Embedder(config, model=model)

    vector = embedder.embed_query("如何提高 RAG 的召回率？")

    assert len(vector) == 32
    assert model.max_seq_length == 512
    assert model.calls == [
        (
            ["如何提高 RAG 的召回率？"],
            {
                "batch_size": 4,
                "show_progress_bar": False,
                "normalize_embeddings": True,
                "truncate_dim": 32,
                "prompt_name": "query",
            },
        )
    ]


def test_documents_are_encoded_as_a_batch_without_query_prompt() -> None:
    """Chunk ingestion should batch work without pretending chunks are queries."""
    config = EmbeddingConfig(dimension=32)
    model = FakeEncoderModel(dimension=config.dimension)
    embedder = Qwen3Embedder(config, model=model)

    vectors = embedder.embed_documents(["RAG 先检索资料。", "切分会影响召回率。"])

    assert len(vectors) == 2
    assert all(len(vector) == 32 for vector in vectors)
    assert "prompt_name" not in model.calls[0][1]
    assert model.calls[0][1]["normalize_embeddings"] is True


def test_empty_document_batch_needs_no_model_call() -> None:
    """An empty ingestion batch should finish without unnecessary model work."""
    config = EmbeddingConfig(dimension=32)
    model = FakeEncoderModel(dimension=config.dimension)
    embedder = Qwen3Embedder(config, model=model)

    assert embedder.embed_documents([]) == []
    assert model.calls == []


def test_token_count_disables_truncation_before_index_gate() -> None:
    """The indexing gate needs the real full length, not an already-clipped count."""

    config = EmbeddingConfig(dimension=32, max_sequence_length=3)
    model = FakeEncoderModel(dimension=config.dimension)
    embedder = Qwen3Embedder(config, model=model)

    count = embedder.count_tokens("one two three four")

    assert count == 6
    assert model.tokenizer.calls == [("one two three four", True, False)]


def test_token_count_accepts_transformers_mapping_payload() -> None:
    """Transformers returns BatchEncoding, a Mapping that is not a plain dict."""

    config = EmbeddingConfig(dimension=32)
    model = FakeEncoderModel(dimension=config.dimension)
    model.tokenizer = BatchEncodingLikeTokenizer()
    embedder = Qwen3Embedder(config, model=model)

    assert embedder.count_tokens("one two") == 4


def test_embedding_identity_settings_are_explicit_and_pinned() -> None:
    """Production manifests must not infer revision or precision from runtime state."""

    config = EmbeddingConfig()

    assert len(config.model_revision) == 40
    assert config.tokenizer_revision == config.model_revision
    assert config.dtype == "float16"
    assert config.normalize_embeddings is True
    assert config.query_profile == "qwen-query-prompt-v1"


def test_unsupported_embedding_dtype_is_rejected() -> None:
    """A misspelled precision must fail before model loading changes vector values."""

    with pytest.raises(ValueError, match="dtype must be"):
        EmbeddingConfig(dtype="fp16")


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_blank_text_is_rejected(text: str) -> None:
    """Blank input should fail before spending GPU time."""
    config = EmbeddingConfig(dimension=32)
    embedder = Qwen3Embedder(config, model=FakeEncoderModel(config.dimension))

    with pytest.raises(ValueError, match="must not be blank"):
        embedder.embed_query(text)


def test_unexpected_vector_dimension_is_rejected() -> None:
    """A model/config mismatch must not silently corrupt a future vector index."""
    config = EmbeddingConfig(dimension=32)
    embedder = Qwen3Embedder(config, model=FakeEncoderModel(dimension=31))

    with pytest.raises(RuntimeError, match="unexpected vector dimension"):
        embedder.embed_query("RAG")
