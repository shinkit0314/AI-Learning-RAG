"""Run the first complete RAG loop with local retrieval and OpenAI generation."""

from ai_learning_rag.embedding import Qwen3Embedder
from ai_learning_rag.generation import GroundedAnswerService
from ai_learning_rag.openai_generation import OpenAITextGenerator
from ai_learning_rag.retrieval import RetrievalPipeline
from ai_learning_rag.text_processing import build_document
from ai_learning_rag.vector_index import InMemoryVectorIndex


def main() -> None:
    """Index learning documents, retrieve evidence, and generate one cited answer."""
    embedder = Qwen3Embedder()
    index = InMemoryVectorIndex(dimension=embedder.dimension)
    retrieval = RetrievalPipeline(embedder=embedder, index=index)
    retrieval.index_documents(
        [
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
        ],
        chunk_size_chars=100,
        chunk_overlap_chars=0,
    )

    service = GroundedAnswerService(
        retriever=retrieval,
        generator=OpenAITextGenerator(),
    )
    response = service.answer("RAG 为什么需要把文本转换成向量？", top_k=2)

    print(f"answer = {response.answer}")
    print("citations:")
    for citation in response.citations:
        print(
            f"[{citation.marker}] source={citation.source} "
            f"chars={citation.start_char}-{citation.end_char} "
            f"retrieval_score={citation.score:.4f}"
        )


if __name__ == "__main__":
    main()
