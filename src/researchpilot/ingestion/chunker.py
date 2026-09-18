
from researchpilot.config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split page text into non-empty paragraphs.
    """
    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    return paragraphs


def chunk_paragraphs(
    paragraphs: list[str],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Build chunks from paragraphs while trying to preserve
    paragraph boundaries.

    If a paragraph itself is larger than chunk_size,
    it is split into smaller pieces.
    """

    chunks = []

    current_chunk = ""

    for paragraph in paragraphs:

        # --------------------------------------------------
        # Case 1: Paragraph fits into current chunk
        # --------------------------------------------------

        if len(current_chunk) + len(paragraph) + 1 <= chunk_size:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

            continue

        # --------------------------------------------------
        # Save current chunk
        # --------------------------------------------------

        if current_chunk:
            chunks.append(current_chunk.strip())

        # --------------------------------------------------
        # Case 2: Paragraph itself is too large
        # --------------------------------------------------

        if len(paragraph) > chunk_size:
            start = 0

            while start < len(paragraph):
                end = start + chunk_size

                piece = paragraph[start:end].strip()

                if piece:
                    chunks.append(piece)

                next_start = end - chunk_overlap

                if next_start <= start:
                    break

                start = next_start

            current_chunk = ""

        else:
            current_chunk = paragraph

    # ------------------------------------------------------
    # Add final chunk
    # ------------------------------------------------------

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Structure-aware document chunking.

    Documents are split into paragraphs first, then
    paragraphs are grouped into chunks.
    """

    chunks = []

    for document in documents:
        text = document["text"]
        metadata = document["metadata"]

        paragraphs = split_into_paragraphs(text)

        text_chunks = chunk_paragraphs(
            paragraphs,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        for chunk_index, chunk_text in enumerate(text_chunks):

            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk": chunk_index,
                    },
                }
            )

    return chunks
