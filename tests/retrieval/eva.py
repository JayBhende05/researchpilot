import json
from pathlib import Path

from researchpilot.retrieval.embedder import Embedder
from researchpilot.retrieval.vector_store import VectorStore
from researchpilot.retrieval.bm25 import BM25Retriever
from researchpilot.retrieval.reranker import Reranker
from researchpilot.retrieval.reranked import RerankedRetriever



def evidence_retrieved(
    retrieved_metadata,
    expected_source,
    gold_pages,
    k,
):
    return any(
        metadata["source"] == expected_source
        and metadata["page"] in gold_pages
        for metadata in retrieved_metadata[:k]
    )


def get_evidence_rank(
    retrieved_metadata,
    expected_source,
    gold_pages,
):
    for rank, metadata in enumerate(
        retrieved_metadata,
        start=1,
    ):
        if (
            metadata["source"] == expected_source
            and metadata["page"] in gold_pages
        ):
            return rank

    return None

def evaluate():

    # ------------------------------------------------------
    # Single hardcoded question
    # ------------------------------------------------------

    question = "What is transformer"

    # ------------------------------------------------------
    # Initialize components
    # ------------------------------------------------------

    embedder = Embedder()

    vector_store = VectorStore()

    chunks = vector_store.get_all_chunks()

    print(
        f"\nBuilding BM25 index over "
        f"{len(chunks)} chunks..."
    )

    bm25_retriever = BM25Retriever(
        chunks=chunks
    )

    print("\nLoading reranker...")

    reranker = Reranker()

    # ------------------------------------------------------
    # Retrieval pipeline
    # ------------------------------------------------------

    retriever = RerankedRetriever(
        vector_store=vector_store,
        embedder=embedder,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
    )

    # ------------------------------------------------------
    # Run one question
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("SINGLE QUESTION EVALUATION")
    print("=" * 60)

    print(f"\nQuestion: {question}")

    results = retriever.search(
        query=question,
        top_k=5,
        candidate_k=20,
    )

    # ------------------------------------------------------
    # Final results
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TOP-5 RESULTS")
    print("=" * 60)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        metadata = result.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
            "unknown",
        )

        page = metadata.get(
            "page",
            "unknown",
        )

        text = (
            result.get("text")
            or result.get("content")
            or result.get("document")
            or ""
        )

        text = " ".join(
            str(text).split()
        )

        if len(text) > 550:
            text = text[:550] + "..."

        print(
            f"\n#{rank}"
        )

        print(
            f"Source : {source}"
        )

        print(
            f"Page   : {page}"
        )

        print(
            f"Chunk  : {text}"
        )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()


if __name__ == "__main__":
    evaluate()