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

    top_1_correct = 0
    top_3_correct = 0
    top_5_correct = 0

    print("\nResearchPilot Retrieval Evaluation")
    print("=" * 80)

    for item in questions:
        question = item["question"]
        expected_source = item["expected_source"]

        query_embedding = embedder.embed_query(question)

        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=5,
        )

        sources = [
            metadata["source"]
            for metadata in results["metadatas"][0]
        ]

        top_1 = expected_source in sources[:1]
        top_3 = expected_source in sources[:3]
        top_5 = expected_source in sources[:5]

        if top_1:
            top_1_correct += 1

        if top_3:
            top_3_correct += 1

        if top_5:
            top_5_correct += 1

        print(f"\nQuestion: {question}")
        print(f"Expected: {expected_source}")
        print(f"Top 5:    {sources}")
        print(
            f"Top-1: {'✓' if top_1 else '✗'} | "
            f"Top-3: {'✓' if top_3 else '✗'} | "
            f"Top-5: {'✓' if top_5 else '✗'}"
        )

    total = len(questions)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"Queries:          {total}")
    print(f"Top-1 Accuracy:   {top_1_correct / total:.2%}")
    print(f"Top-3 Accuracy:   {top_3_correct / total:.2%}")
    print(f"Top-5 Accuracy:   {top_5_correct / total:.2%}")


if __name__ == "__main__":
    evaluate()