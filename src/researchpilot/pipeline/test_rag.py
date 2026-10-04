import pytest

from researchpilot.generation.llm import GeminiLLM
from researchpilot.pipeline.rag import RAGPipeline
from researchpilot.retrieval.bm25 import BM25Retriever
from researchpilot.retrieval.embedder import Embedder
from researchpilot.retrieval.reranker import Reranker
from researchpilot.retrieval.reranked import RerankedRetriever
from researchpilot.retrieval.vector_store import VectorStore


@pytest.mark.integration
def test_rag_end_to_end():

    # --------------------------------------------------
    # 1. Load existing indexed chunks
    # --------------------------------------------------

    vector_store = VectorStore()

    chunks = vector_store.get_all_chunks()
    print("Chunks found in ChromaDB: ", chunks)

    assert chunks, (
        "No chunks found in ChromaDB. "
        "Run the ingestion pipeline first."
    )

    # --------------------------------------------------
    # 2. Build retrieval components
    # --------------------------------------------------

    embedder = Embedder()

    bm25_retriever = BM25Retriever(
        chunks=chunks
    )

    reranker = Reranker()

    retriever = RerankedRetriever(
        vector_store=vector_store,
        embedder=embedder,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
    )

    # --------------------------------------------------
    # 3. Build Gemini LLM
    # --------------------------------------------------

    llm = GeminiLLM()

    # --------------------------------------------------
    # 4. Build complete RAG pipeline
    # --------------------------------------------------

    rag = RAGPipeline(
        retriever=retriever,
        llm=llm,
    )

    # --------------------------------------------------
    # 5. Ask a question
    # --------------------------------------------------

    question = (
        "What is the main finding of the research "
        "discussed in the provided documents?"
    )

    result = rag.query(
        question=question,
        top_k=5,
        candidate_k=20,
    )

    # --------------------------------------------------
    # 6. Validate response
    # --------------------------------------------------

    assert result is not None

    assert result.question == question

    assert isinstance(
        result.answer,
        str,
    )

    assert result.answer.strip()

    # --------------------------------------------------
    # 7. Validate citations
    # --------------------------------------------------

    assert result.citations

    assert len(result.citations) <= 5

    for citation in result.citations:

        assert citation.chunk_id

        assert citation.document_id

        assert citation.text

        assert isinstance(
            citation.metadata,
            dict,
        )

    # --------------------------------------------------
    # 8. Print result for manual inspection
    # --------------------------------------------------

    print("\n")
    print("=" * 80)
    print("RAG RESULT")
    print("=" * 80)

    print("\nQUESTION:")
    print(result.question)

    print("\nANSWER:")
    print(result.answer)

    print("\nCITATIONS:")
    print("-" * 80)

    for index, citation in enumerate(
        result.citations,
        start=1,
    ):

        print(
            f"\n[{index}] "
            f"{citation.document_id}"
        )

        print(
            f"Chunk: {citation.chunk_id}"
        )

        print(
            f"Score: {citation.score}"
        )

        print(
            f"Metadata: {citation.metadata}"
        )

        print(
            f"Text: {citation.text[:500]}"
        )

    print("\n")
    print("=" * 80)
