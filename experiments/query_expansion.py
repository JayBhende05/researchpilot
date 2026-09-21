import re


STOP_WORDS = {
    "what",
    "why",
    "how",
    "when",
    "where",
    "which",
    "who",
    "does",
    "did",
    "do",
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "from",
    "as",
    "by",
    "it",
    "this",
    "that",
    "these",
    "those",
    "used",
    "use",
}


def expand_query(query: str) -> str:
    """
    Creates a deterministic keyword-focused version
    of the original query.

    The original query is always preserved separately.
    """

    words = re.findall(
        r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*",
        query.lower(),
    )

    keywords = [
        word
        for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(keywords)