import json
from pathlib import Path

from researchpilot.retrieval.embedder import Embedder
from researchpilot.retrieval.vector_store import VectorStore


QUESTIONS_FILE = Path("tests/retrieval_questions.json")


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


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

    print("\nResearchPilot Retrieval Evaluation")
    print("=" * 80)

    for item in questions:
        question = item["question"]
        expected_source = item["expected_source"]
        expected_page = item["expected_page"]

        query_embedding = embedder.embed_query(question)

        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=5,
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        sources = [
            metadata["source"]
            for metadata in metadatas
        ]

        pages = [
            metadata["page"]
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

        expected_pages = set(expected_page)

        retrieved_evidence = [
            (
                metadata["source"],
                metadata["page"],
            )
            for metadata in metadatas
        ]


        def evidence_matches(metadata):
            return (
                metadata["source"] == expected_source
                and metadata["page"] in expected_pages
            )


        if any(
            evidence_matches(metadata)
            for metadata in metadatas[:1]
        ):
            top_1_evidence += 1

        if any(
            evidence_matches(metadata)
            for metadata in metadatas[:3]
        ):
            top_3_evidence += 1

        if any(
            evidence_matches(metadata)
            for metadata in metadatas[:5]
        ):
            top_5_evidence += 1

        print(f"\nQuestion: {question}")
        print(f"Expected source: {expected_source}")
        print(f"Expected page:   {expected_page}")

        print("\nRetrieved:")

        print("\nRetrieved:")

        for rank, (document, metadata) in enumerate(
            zip(documents, metadatas),
            start=1,
        ):
            print(
                f"{rank}. "
                f"{metadata['source']} | "
                f"page {metadata['page']} | "
                f"chunk {metadata['chunk']}"
            )

            print(f"   {document[:250].replace(chr(10), ' ')}") 
    total = len(questions)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nDocument Retrieval")
    print(
        f"Top-1: {top_1_document / total:.2%}"
    )
    print(
        f"Top-3: {top_3_document / total:.2%}"
    )
    print(
        f"Top-5: {top_5_document / total:.2%}"
    )

    print("\nEvidence Retrieval")
    print(
        f"Top-1: {top_1_evidence / total:.2%}"
    )
    print(
        f"Top-3: {top_3_evidence / total:.2%}"
    )
    print(
        f"Top-5: {top_5_evidence / total:.2%}"
    )


if __name__ == "__main__":
    evaluate()