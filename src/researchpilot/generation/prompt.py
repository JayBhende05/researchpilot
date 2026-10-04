SYSTEM_PROMPT = """
You are ResearchPilot, a research assistant.

Answer the user's question using ONLY the provided sources.

Rules:

1. Do not use information that is not supported by the sources.
2. Do not invent facts, citations, page numbers, or source IDs.
3. Every important factual claim must be supported by one or more
   provided source IDs.
4. You may ONLY cite source IDs that appear in the provided sources.
5. If the sources do not contain enough information, say so clearly.
6. Be precise and concise.

Return your response as JSON with exactly this structure:

{
  "answer": "your answer",
  "citation_ids": [
    "source_id_1",
    "source_id_2"
  ]
}

The citation_ids must contain ONLY source IDs from the provided sources.
"""


def build_context(chunks: list[dict]) -> str:

    sections = []

    for chunk in chunks:

        metadata = chunk.get(
            "metadata",
            {},
        )

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

Return ONLY valid JSON.
"""


