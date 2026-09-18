
from collections import defaultdict


def generate_query_variants(question: str) -> list[str]:
    """
    Generate deterministic retrieval variants.

    E3 deliberately keeps the original question and
    adds alternative representations.
    """

    # --------------------------------------------------
    # Query 1: Original question
    # --------------------------------------------------

    original = question.strip()

    # --------------------------------------------------
    # Query 2: Keyword-oriented representation
    # --------------------------------------------------

    stop_words = {
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "which",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "for",
        "in",
        "on",
        "with",
        "from",
        "as",
        "by",
        "it",
        "this",
        "that",
    }

    words = (
        original
        .lower()
        .replace("?", "")
        .replace(",", "")
        .replace(".", "")
        .split()
    )

    keywords = [
        word
        for word in words
        if word not in stop_words
    ]

    keyword_query = " ".join(
        dict.fromkeys(keywords)
    )

    # --------------------------------------------------
    # Query 3: Question without interrogative structure
    # --------------------------------------------------

    question_words = {
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "which",
    }

    focused_words = [
        word
        for word in words
        if word not in question_words
    ]

    focused_query = " ".join(
        dict.fromkeys(focused_words)
    )

    # --------------------------------------------------
    # Return unique queries
    # --------------------------------------------------

    variants = [
        original,
        keyword_query,
        focused_query,
    ]

    return list(
        dict.fromkeys(
            query.strip()
            for query in variants
            if query.strip()
        )
    )


def reciprocal_rank_fusion(
    ranked_results: list[list[dict]],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Combine multiple ranked retrieval results using
    Reciprocal Rank Fusion (RRF).

    Each result must contain:

        {
            "id": ...,
            "document": ...,
            "metadata": ...
        }
    """

    scores = defaultdict(float)
    result_lookup = {}

    for results in ranked_results:

        for rank, result in enumerate(
            results,
            start=1,
        ):
            result_id = result["id"]

            scores[result_id] += (
                1 / (rrf_k + rank)
            )

            result_lookup[result_id] = result

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    final_results = []

    for result_id in ranked_ids[:top_k]:

        result = result_lookup[result_id].copy()

        result["rrf_score"] = scores[result_id]

        final_results.append(result)

    return final_results
