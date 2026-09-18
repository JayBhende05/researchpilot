import json
from pathlib import Path

from researchpilot.retrieval.embedder import Embedder
from researchpilot.retrieval.vector_store import VectorStore
from researchpilot.retrieval.bm25 import BM25Retriever
from researchpilot.retrieval.reranker import Reranker
from researchpilot.retrieval.reranked import RerankedRetriever


QUESTIONS_FILE = Path(
    "tests/retrieval/benchmark.json"
)


def load_questions():
    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


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

    questions = load_questions()

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

    reranked_retriever = RerankedRetriever(
        vector_store=vector_store,
        embedder=embedder,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
    )

    # ------------------------------------------------------
    # Counters
    # ------------------------------------------------------

    top_1_document = 0
    top_3_document = 0
    top_5_document = 0

    top_1_evidence = 0
    top_3_evidence = 0
    top_5_evidence = 0

    evidence_rank_counts = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        "not_retrieved": 0,
    }

    # ------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------

    print("\n" + "=" * 50)
    print("ResearchPilot — E5 Reranked Retrieval")
    print("=" * 50)

    for item in questions:

        question = item["question"]

        expected_source = (
            item["expected_source"]
        )

        gold_pages = {
            evidence["page"]
            for evidence in item["gold_evidence"]
        }

        results = reranked_retriever.search(
            query=question,
            top_k=5,
            candidate_k=20,
        )

        metadatas = [
            result["metadata"]
            for result in results
        ]

        sources = [
            metadata["source"]
            for metadata in metadatas
        ]

        # --------------------------------------------------
        # Document evaluation
        # --------------------------------------------------

        if expected_source in sources[:1]:
            top_1_document += 1

        if expected_source in sources[:3]:
            top_3_document += 1

        if expected_source in sources[:5]:
            top_5_document += 1

        # --------------------------------------------------
        # Evidence evaluation
        # --------------------------------------------------

        if evidence_retrieved(
            metadatas,
            expected_source,
            gold_pages,
            1,
        ):
            top_1_evidence += 1

        if evidence_retrieved(
            metadatas,
            expected_source,
            gold_pages,
            3,
        ):
            top_3_evidence += 1

        if evidence_retrieved(
            metadatas,
            expected_source,
            gold_pages,
            5,
        ):
            top_5_evidence += 1

        # --------------------------------------------------
        # Evidence rank
        # --------------------------------------------------

        evidence_rank = get_evidence_rank(
            metadatas,
            expected_source,
            gold_pages,
        )

        if evidence_rank is None:

            evidence_rank_counts[
                "not_retrieved"
            ] += 1

        elif evidence_rank <= 5:

            evidence_rank_counts[
                evidence_rank
            ] += 1

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    total = len(questions)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    print(
        f"\nQuestions evaluated: {total}"
    )

    print("\nDocument Retrieval")

    print(
        f"Top-1: "
        f"{top_1_document / total:.2%}"
    )

    print(
        f"Top-3: "
        f"{top_3_document / total:.2%}"
    )

    print(
        f"Top-5: "
        f"{top_5_document / total:.2%}"
    )

    print("\nEvidence Retrieval")

    print(
        f"Top-1: "
        f"{top_1_evidence / total:.2%}"
    )

    print(
        f"Top-3: "
        f"{top_3_evidence / total:.2%}"
    )

    print(
        f"Top-5: "
        f"{top_5_evidence / total:.2%}"
    )

    # ------------------------------------------------------
    # Rank distribution
    # ------------------------------------------------------

    print("\n" + "=" * 50)
    print("EVIDENCE RANK DISTRIBUTION")
    print("=" * 50)

    for rank in range(1, 6):

        count = evidence_rank_counts[rank]

        print(
            f"Gold evidence at #{rank}: "
            f"{count}/{total} "
            f"({count / total:.2%})"
        )

    count = evidence_rank_counts[
        "not_retrieved"
    ]

    print(
        f"Gold evidence NOT in Top-5: "
        f"{count}/{total} "
        f"({count / total:.2%})"
    )


if __name__ == "__main__":
    evaluate()