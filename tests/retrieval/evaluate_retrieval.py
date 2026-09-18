
import json
from pathlib import Path

from researchpilot.retrieval.embedder import Embedder
from researchpilot.retrieval.vector_store import VectorStore
from researchpilot.retrieval.query_transformer import transform_query


QUESTIONS_FILE = Path("tests/retrieval/benchmark.json")


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
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
    """
    Return the rank of the first gold evidence chunk.

    Returns:
        1-based rank if gold evidence is retrieved.
        None if gold evidence is not in the retrieved results.
    """
    for rank, metadata in enumerate(retrieved_metadata, start=1):
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

    # -------------------------
    # Overall counters
    # -------------------------

    top_1_document = 0
    top_3_document = 0
    top_5_document = 0

    top_1_evidence = 0
    top_3_evidence = 0
    top_5_evidence = 0

    # -------------------------
    # Failure analysis
    # -------------------------

    evidence_rank_counts = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        "not_retrieved": 0,
    }

    failed_questions = []

    print("\n" + "=" * 50)
    print("ResearchPilot Retrieval Evaluation")
    print("=" * 50)

    for item in questions:
        question = item["question"]
        expected_source = item["expected_source"]

        gold_evidence = item["gold_evidence"]

        gold_pages = {
            evidence["page"]
            for evidence in gold_evidence
        }

        transformed_query = transform_query(question)

        query_embedding = embedder.embed_query(
            transformed_query
        )

        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=5,
        )

        print(f"Original query: {question}")
        print(f"Transformed query: {transformed_query}")
        
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        sources = [
            metadata["source"]
            for metadata in metadatas
        ]

        # -------------------------
        # Document-level evaluation
        # -------------------------

        if expected_source in sources[:1]:
            top_1_document += 1

        if expected_source in sources[:3]:
            top_3_document += 1

        if expected_source in sources[:5]:
            top_5_document += 1

        # -------------------------
        # Evidence-level evaluation
        # -------------------------

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

        # -------------------------
        # Evidence rank analysis
        # -------------------------

        evidence_rank = get_evidence_rank(
            metadatas,
            expected_source,
            gold_pages,
        )

        if evidence_rank is None:
            evidence_rank_counts["not_retrieved"] += 1

            failed_questions.append(
                {
                    "question": question,
                    "expected_source": expected_source,
                    "gold_pages": sorted(gold_pages),
                    "retrieved": [
                        {
                            "rank": rank,
                            "source": metadata["source"],
                            "page": metadata["page"],
                            "chunk": metadata["chunk"],
                        }
                        for rank, metadata in enumerate(
                            metadatas,
                            start=1,
                        )
                    ],
                }
            )

        else:
            evidence_rank_counts[evidence_rank] += 1

        # -------------------------
        # Per-question output
        # -------------------------

        print(f"\nQuestion: {question}")
        print(f"Expected source: {expected_source}")
        print(f"Gold evidence pages: {sorted(gold_pages)}")

        if evidence_rank is None:
            print("Evidence rank: NOT IN TOP-5")
        else:
            print(f"Evidence rank: #{evidence_rank}")

        print("\nRetrieved:")

        for rank, (document, metadata) in enumerate(
            zip(documents, metadatas),
            start=1,
        ):
            is_evidence = (
                metadata["source"] == expected_source
                and metadata["page"] in gold_pages
            )

            marker = " <-- GOLD EVIDENCE" if is_evidence else ""

            print(
                f"{rank}. "
                f"{metadata['source']} | "
                f"page {metadata['page']} | "
                f"chunk {metadata['chunk']}"
                f"{marker}"
            )

            preview = document[:250].replace("\n", " ")
            print(f"   {preview}")

    # -------------------------
    # Summary
    # -------------------------

    total = len(questions)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    print(f"\nQuestions evaluated: {total}")

    print("\nDocument Retrieval")
    print(f"Top-1: {top_1_document / total:.2%}")
    print(f"Top-3: {top_3_document / total:.2%}")
    print(f"Top-5: {top_5_document / total:.2%}")

    print("\nEvidence Retrieval")
    print(f"Top-1: {top_1_evidence / total:.2%}")
    print(f"Top-3: {top_3_evidence / total:.2%}")
    print(f"Top-5: {top_5_evidence / total:.2%}")

    # -------------------------
    # Evidence rank distribution
    # -------------------------

    print("\n" + "=" * 50)
    print("EVIDENCE RANK DISTRIBUTION")
    print("=" * 50)

    for rank in range(1, 6):
        count = evidence_rank_counts[rank]
        percentage = count / total

        print(
            f"Gold evidence at #{rank}: "
            f"{count}/{total} ({percentage:.2%})"
        )

    not_retrieved = evidence_rank_counts["not_retrieved"]

    print(
        f"Gold evidence NOT in Top-5: "
        f"{not_retrieved}/{total} "
        f"({not_retrieved / total:.2%})"
    )

    # -------------------------
    # Failed evidence queries
    # -------------------------

    print("\n" + "=" * 50)
    print("FAILED EVIDENCE QUERIES")
    print("=" * 50)

    if not failed_questions:
        print("\nAll gold evidence was retrieved in Top-5.")

    else:
        for index, failure in enumerate(
            failed_questions,
            start=1,
        ):
            print(f"\n{index}. {failure['question']}")

            print(
                f"   Gold: "
                f"{failure['expected_source']} "
                f"| pages {failure['gold_pages']}"
            )

            print("   Retrieved:")

            for result in failure["retrieved"]:
                print(
                    f"      {result['rank']}. "
                    f"{result['source']} | "
                    f"page {result['page']} | "
                    f"chunk {result['chunk']}"
                )


if __name__ == "__main__":
    evaluate()

