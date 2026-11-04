"""Plain-text normalization and chunking for the first RAG learning step."""

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True, slots=True)
class TextDocument:
    """A normalized source document with a stable identifier."""

    id: str
    title: str
    source: str
    text: str


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A traceable slice of a normalized document."""

    id: str
    document_id: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int


def normalize_text(text: str) -> str:
    """Collapse all runs of whitespace into a single space."""
    return " ".join(text.split())


def build_document(*, title: str, source: str, text: str) -> TextDocument:
    """Normalize source data and create a deterministic document identifier."""
    normalized_title = title.strip()
    normalized_source = source.strip()
    normalized_text = normalize_text(text)

    if not normalized_title:
        raise ValueError("title must not be blank")
    if not normalized_source:
        raise ValueError("source must not be blank")
    if not normalized_text:
        raise ValueError("text must not be blank")

    identity = f"{normalized_source}\0{normalized_title}\0{normalized_text}"
    document_id = f"doc_{sha256(identity.encode()).hexdigest()[:16]}"

    return TextDocument(
        id=document_id,
        title=normalized_title,
        source=normalized_source,
        text=normalized_text,
    )


def split_document(
    document: TextDocument,
    *,
    chunk_size_chars: int = 500,
    chunk_overlap_chars: int = 50,
) -> list[TextChunk]:
    """Split a document into fixed-size, overlapping character windows."""
    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be greater than zero")
    if chunk_overlap_chars < 0:
        raise ValueError("chunk_overlap_chars must not be negative")
    if chunk_overlap_chars >= chunk_size_chars:
        raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")

    chunks: list[TextChunk] = []
    start = 0

    while start < len(document.text):
        end = min(start + chunk_size_chars, len(document.text))
        identity = f"{document.id}:{start}:{end}"
        chunks.append(
            TextChunk(
                id=f"chunk_{sha256(identity.encode()).hexdigest()[:16]}",
                document_id=document.id,
                chunk_index=len(chunks),
                text=document.text[start:end],
                start_char=start,
                end_char=end,
            )
        )

        if end == len(document.text):
            break
        start = end - chunk_overlap_chars

    return chunks
