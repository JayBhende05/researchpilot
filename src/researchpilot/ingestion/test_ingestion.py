

from researchpilot.ingestion.loader import load_pdf
from researchpilot.ingestion.chunker import chunk_documents
from researchpilot.config.settings import PAPERS_DIR


documents = load_pdf(PAPERS_DIR)
chunks = chunk_documents(documents)

for chunk in chunks[:3]:
    print("\n--- CHUNK ---")
    print(chunk["metadata"])
    print(chunk["text"][:500])

print(f"Number of documents loaded: {len(documents)}")
print(f"Number of chunks: {len(chunks)}")
