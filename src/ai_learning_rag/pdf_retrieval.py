"""Natural-language query adapter for the document-level PDF vector index."""

from typing import Protocol

from ai_learning_rag.document_index import (
    IndexStatus,
    SearchResult,
    VectorIndex,
    VectorIndexManifest,
    VectorSearchQuery,
)
from ai_learning_rag.embedding import EmbeddingConfig


class QueryTooLongError(ValueError):
    """The complete query cannot be encoded without truncation."""


class PdfQueryEmbedder(Protocol):
    """Embedding capabilities required by the PDF query adapter."""

    config: EmbeddingConfig

    @property
    def dimension(self) -> int:
        """Return the configured output dimension."""

    def count_tokens(self, text: str) -> int:
        """Count the untruncated query tokens."""

    def embed_query(self, query: str) -> list[float]:
        """Encode one natural-language retrieval query."""


class PdfRetriever:
    """Turn natural-language questions into exact PDF vector-index queries."""

    def __init__(
        self,
        *,
        index: VectorIndex,
        manifest: VectorIndexManifest,
        embedder: PdfQueryEmbedder,
    ) -> None:
        self._index = index
        self._manifest = manifest
        self._embedder = embedder
        _validate_embedding_compatibility(manifest, embedder)

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Embed a complete question and search the configured active PDF index."""

        if not query.strip():
            raise ValueError("query must not be blank")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        stored_manifest = self._index.get_manifest(self._manifest.index_id)
        if stored_manifest is None:
            raise ValueError("query references an unknown index")
        if stored_manifest.identity_payload != self._manifest.identity_payload:
            raise ValueError("stored vector index identity does not match the query adapter")
        if stored_manifest.status is not IndexStatus.ACTIVE:
            raise ValueError("query index must be active")

        token_count = self._embedder.count_tokens(query)
        if token_count <= 0:
            raise RuntimeError("query token counter returned a non-positive token count")
        if token_count > self._manifest.max_input_tokens:
            raise QueryTooLongError(
                f"query has {token_count} tokens, exceeding the "
                f"{self._manifest.max_input_tokens}-token index limit"
            )

        vector = tuple(self._embedder.embed_query(query))
        return self._index.search(
            VectorSearchQuery(
                index_id=self._manifest.index_id,
                vector=vector,
                top_k=top_k,
            )
        )


def _validate_embedding_compatibility(
    manifest: VectorIndexManifest,
    embedder: PdfQueryEmbedder,
) -> None:
    config = embedder.config
    actual = {
        "adapter_version": config.adapter_version,
        "dtype": config.dtype,
        "embedding_dimension": embedder.dimension,
        "max_input_tokens": config.max_sequence_length,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "normalized": config.normalize_embeddings,
        "pooling": config.pooling,
        "query_profile": config.query_profile,
        "tokenizer_id": config.tokenizer_id,
        "tokenizer_revision": config.tokenizer_revision,
    }
    expected = {
        "adapter_version": manifest.adapter_version,
        "dtype": manifest.dtype,
        "embedding_dimension": manifest.embedding_dimension,
        "max_input_tokens": manifest.max_input_tokens,
        "model_id": manifest.model_id,
        "model_revision": manifest.model_revision,
        "normalized": manifest.normalized,
        "pooling": manifest.pooling,
        "query_profile": manifest.query_profile,
        "tokenizer_id": manifest.tokenizer_id,
        "tokenizer_revision": manifest.tokenizer_revision,
    }
    mismatches = sorted(key for key, value in expected.items() if actual[key] != value)
    if mismatches:
        raise ValueError(
            "query embedder is incompatible with the vector index: " + ", ".join(mismatches)
        )
