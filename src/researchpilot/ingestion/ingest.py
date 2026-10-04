from pathlib import Path

from researchpilot.ingestion.loader import load_pdf
from researchpilot.ingestion.chunker import chunk_documents
from researchpilot.retrieval.embedder import Embedder
from researchpilot.retrieval.vector_store import VectorStore
from researchpilot.config.settings import PAPERS_DIR

query = "How does QLoRA reduce memory requirements?"

def ingest_papers():
    embedder = Embedder()
    vector_store = VectorStore(reset=True)

    pdf_files = list(PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    total_chunks = 0

    documents = load_pdf('D:/Study/researchpilot/data/papers/attentionisallyouneed.pdf')
    chunks = chunk_documents(documents)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed_documents(texts)
    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )
    total_chunks += len(chunks)
    print('chunks: ', chunks)
    print(f"  Pages: {len(documents)}")
    print(f"  Chunks: {len(chunks)}")

    # for pdf_path in pdf_files:
    #     print(f"Processing: {pdf_path.name}")



    print(f"\nTotal chunks indexed: {total_chunks}")


if __name__ == "__main__":
    ingest_papers()