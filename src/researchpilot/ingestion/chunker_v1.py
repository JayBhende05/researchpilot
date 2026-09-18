from researchpilot.config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents: list[dict]) -> list[dict]:
    chunks = []

    for document in documents:
        text = document["text"]
        metadata = document["metadata"]

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            **metadata,
                            "chunk": chunk_index,
                        },
                    }
                )
            chunk_index += 1

            next_start = end - CHUNK_OVERLAP

            if next_start <= start:
                break

            start = next_start

    return chunks