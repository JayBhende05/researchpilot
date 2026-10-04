SYSTEM_PROMPT = """
You are ResearchPilot, a research assistant.

Your job is to answer the user's question using ONLY the
provided retrieved sources.

Rules:
1. Do not use information that is not present in the sources.
2. Do not invent facts, citations, page numbers, or sources.
3. If the sources do not contain enough information to answer
   the question, say that the available sources do not provide
   enough information.
4. Every important factual claim must be supported by a source.
5. Cite sources using the provided source IDs.
6. Be concise and precise.
"""


def build_context(chunks: list[dict]) -> str:
    sections = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {})

        source = metadata.get(
            "source",
            "Unknown source",
        )

        page = metadata.get(
            "page",
            "Unknown page",
        )

        sections.append(
            f"""
[SOURCE_ID: {chunk["id"]}]
[SOURCE: {source}]
[PAGE: {page}]

{chunk["document"]}
"""
        )

    return "\n\n".join(sections)


def build_prompt(
    question: str,
    chunks: list[dict],
) -> str:

    context = build_context(chunks)

    return f"""
{SYSTEM_PROMPT}

USER QUESTION:
{question}

RETRIEVED SOURCES:
{context}

Answer the question using only the retrieved sources.

At the end of the answer, include citations in this format:

[Source: SOURCE_ID]

Do not cite sources that were not provided.
"""
