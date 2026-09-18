
import re


STOP_WORDS = {
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
    "their",
    "they",
    "them",
}


def transform_query(question: str) -> str:
    """
    Transform a natural-language question into a
    retrieval-oriented query.

    This is intentionally deterministic for E2 so
    that we can measure the effect of query
    transformation independently of an LLM.
    """

    # Normalize whitespace
    query = re.sub(r"\s+", " ", question).strip()

    # Remove punctuation
    query = re.sub(r"[^\w\s\-]", " ", query)

    # Tokenize
    words = query.lower().split()

    # Remove common question words
    keywords = [
        word
        for word in words
        if word not in STOP_WORDS
    ]

    # Remove duplicate terms while preserving order
    unique_keywords = list(dict.fromkeys(keywords))

    return " ".join(unique_keywords)

