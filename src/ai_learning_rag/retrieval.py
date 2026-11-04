"""Orchestration that connects chunking, embedding, indexing, and search."""

from collections.abc import Sequence

from ai_learning_rag.embedding import TextEmbedder
from ai_learning_rag.text_processing import TextDocument, split_document
from ai_learning_rag.vector_index import InMemoryVectorIndex, SearchResult, VectorRecord


class RetrievalPipeline:
    """Build and query a traceable in-memory index."""

    def __init__(self, *, embedder: TextEmbedder, index: InMemoryVectorIndex) -> None:
        if embedder.dimension != index.dimension:
            raise ValueError(
                "embedding dimension does not match index dimension: "
                f"{embedder.dimension} != {index.dimension}"
            )
        self._embedder = embedder
        self._index = index

    def index_documents(
        self,
        documents: Sequence[TextDocument],
        *,
        chunk_size_chars: int = 500,
        chunk_overlap_chars: int = 50,
    ) -> list[str]:
        """Split and embed documents, then upsert every traceable chunk."""
        chunks_with_documents = [
            (chunk, document)
            for document in documents
            for chunk in split_document(
                document,
                chunk_size_chars=chunk_size_chars,
                chunk_overlap_chars=chunk_overlap_chars,
            )
        ]
        if not chunks_with_documents:
            return []

        vectors = self._embedder.embed_documents(
            [chunk.text for chunk, _document in chunks_with_documents]
        )
        if len(vectors) != len(chunks_with_documents):
            raise RuntimeError("embedder returned an unexpected number of vectors")

        indexed_ids: list[str] = []
        for (chunk, document), vector in zip(chunks_with_documents, vectors, strict=True):
            self._index.upsert(
                VectorRecord(
                    id=chunk.id,
                    content=chunk.text,
                    vector=tuple(vector),
                    metadata={
                        "document_id": document.id,
                        "title": document.title,
                        "source": document.source,
                        "chunk_index": chunk.chunk_index,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                    },
                )
            )
            indexed_ids.append(chunk.id)
        return indexed_ids

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Embed one query and return the highest-scoring indexed chunks."""
        query_vector = tuple(self._embedder.embed_query(query))
        return self._index.search(query_vector, top_k=top_k)
