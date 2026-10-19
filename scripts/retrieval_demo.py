"""Run the first real local-model Top-K retrieval experiment."""

from ai_learning_rag.embedding import Qwen3Embedder
from ai_learning_rag.retrieval import RetrievalPipeline
from ai_learning_rag.text_processing import build_document
from ai_learning_rag.vector_index import InMemoryVectorIndex


def main() -> None:
    """Index three small documents and print the two most relevant results."""
    embedder = Qwen3Embedder()
    index = InMemoryVectorIndex(dimension=embedder.dimension)
    pipeline = RetrievalPipeline(embedder=embedder, index=index)
    documents = [
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

    indexed_ids = pipeline.index_documents(
        documents,
        chunk_size_chars=100,
        chunk_overlap_chars=0,
    )
    query = "RAG 为什么需要把文本转换成向量？"
    results = pipeline.search(query, top_k=2)

    print(f"indexed_chunks = {len(indexed_ids)}")
    print(f"query = {query}")
    for rank, result in enumerate(results, start=1):
        print(
            f"rank={rank} score={result.score:.4f} "
            f"source={result.record.metadata['source']} "
            f"content={result.record.content}"
        )


if __name__ == "__main__":
    main()
