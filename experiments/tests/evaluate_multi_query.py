
import json
from pathlib import Path

from researchpilot.retrieval.embedder import Embedder
from researchpilot.retrieval.vector_store import VectorStore
from researchpilot.retrieval.multi_query import (
    generate_query_variants,
    reciprocal_rank_fusion,
)


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

    embedder = Embedder()
    vector_store = VectorStore()

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

    print("\n" + "=" * 50)
    print("ResearchPilot — E3 Multi-Query Retrieval")
    print("=" * 50)

    for item in questions:

        question = item["question"]
        expected_source = item["expected_source"]

        gold_pages = {
            evidence["page"]
            for evidence in item["gold_evidence"]
        }

        # --------------------------------------------------
        # Generate query variants
        # --------------------------------------------------

        query_variants = generate_query_variants(
            question
        )

        print(f"\nQuestion: {question}")

        print("\nQueries:")
        for index, query in enumerate(
            query_variants,
            start=1,
        ):
            print(f"  Q{index}: {query}")

        # --------------------------------------------------
        # Retrieve for each query
        # --------------------------------------------------

        ranked_results = []

        for query in query_variants:

            query_embedding = (
                embedder.embed_query(query)
            )

            results = vector_store.search_ranked(
                query_embedding=query_embedding,
                top_k=5,
            )

            ranked_results.append(results)

        # --------------------------------------------------
        # Fuse results
        # --------------------------------------------------

        fused_results = reciprocal_rank_fusion(
            ranked_results,
            top_k=5,
        )

        metadatas = [
            result["metadata"]
            for result in fused_results
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
        else:
            evidence_rank_counts[
                evidence_rank
            ] += 1

        # --------------------------------------------------
        # Results
        # --------------------------------------------------

        print("\nFused Results:")

        for rank, result in enumerate(
            fused_results,
            start=1,
        ):
            metadata = result["metadata"]

            is_gold = (
                metadata["source"]
                == expected_source
                and metadata["page"] in gold_pages
            )

            marker = (
                " <-- GOLD EVIDENCE"
                if is_gold
                else ""
            )

            print(
                f"  {rank}. "
                f"{metadata['source']} | "
                f"page {metadata['page']} | "
                f"chunk {metadata['chunk']} "
                f"| RRF={result['rrf_score']:.5f}"
                f"{marker}"
            )

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

