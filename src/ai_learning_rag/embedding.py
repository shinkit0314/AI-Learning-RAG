"""Embedding boundary for turning queries and document chunks into vectors."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


def default_model_cache_dir() -> Path:
    """Return the project-local Hugging Face cache directory."""
    return Path(__file__).resolve().parents[2] / ".cache" / "huggingface"


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Settings for the first local embedding model."""

    model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    device: str = "cuda"
    dimension: int = 1024
    max_sequence_length: int = 512
    batch_size: int = 4
    cache_dir: Path = field(default_factory=default_model_cache_dir)

    def __post_init__(self) -> None:
        """Reject settings outside the supported learning baseline."""
        if not self.model_id.strip():
            raise ValueError("model_id must not be blank")
        if not self.device.strip():
            raise ValueError("device must not be blank")
        if not 32 <= self.dimension <= 1024:
            raise ValueError("dimension must be between 32 and 1024")
        if not 1 <= self.max_sequence_length <= 32_768:
            raise ValueError("max_sequence_length must be between 1 and 32768")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")


class EncoderModel(Protocol):
    """Small part of SentenceTransformer used by the application."""

    max_seq_length: int

    def encode(self, inputs: list[str], **kwargs: Any) -> Sequence[Sequence[float]]:
        """Encode a batch of strings."""


class TextEmbedder(Protocol):
    """Model-independent contract used by the retrieval pipeline."""

    @property
    def dimension(self) -> int:
        """Return the number of values in every output vector."""

    def embed_query(self, query: str) -> list[float]:
        """Encode one retrieval query."""

    def embed_documents(self, documents: Sequence[str]) -> list[list[float]]:
        """Encode document chunks as a batch."""


class Qwen3Embedder:
    """Encode text with Qwen3 while hiding model-library details from the app."""

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        model: EncoderModel | None = None,
    ) -> None:
        self.config = config or EmbeddingConfig()
        self._model = model or self._load_model()
        self._model.max_seq_length = self.config.max_sequence_length

    @property
    def dimension(self) -> int:
        """Return the configured output dimension."""
        return self.config.dimension

    def embed_query(self, query: str) -> list[float]:
        """Encode one search query using Qwen's retrieval instruction."""
        self._validate_text(query, label="query")
        return self._encode([query], prompt_name="query")[0]

    def embed_documents(self, documents: Sequence[str]) -> list[list[float]]:
        """Encode document chunks without adding the query instruction."""
        document_batch = list(documents)
        for document in document_batch:
            self._validate_text(document, label="document")
        if not document_batch:
            return []
        return self._encode(document_batch)

    def _encode(self, texts: list[str], *, prompt_name: str | None = None) -> list[list[float]]:
        options: dict[str, Any] = {
            "batch_size": self.config.batch_size,
            "show_progress_bar": False,
            "normalize_embeddings": True,
            "truncate_dim": self.config.dimension,
        }
        if prompt_name is not None:
            options["prompt_name"] = prompt_name

        encoded = self._model.encode(texts, **options)
        vectors = [list(map(float, vector)) for vector in encoded]
        if len(vectors) != len(texts):
            raise RuntimeError("embedding model returned an unexpected batch size")
        if any(len(vector) != self.config.dimension for vector in vectors):
            raise RuntimeError("embedding model returned an unexpected vector dimension")
        return vectors

    def _load_model(self) -> EncoderModel:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Local embedding dependencies are missing. Run "
                "scripts/setup.ps1 -WithLocalEmbedding."
            ) from error

        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        return SentenceTransformer(
            self.config.model_id,
            device=self.config.device,
            cache_folder=str(self.config.cache_dir),
            model_kwargs={"torch_dtype": torch.float16},
            truncate_dim=self.config.dimension,
        )

    @staticmethod
    def _validate_text(text: str, *, label: str) -> None:
        if not text.strip():
            raise ValueError(f"{label} must not be blank")
