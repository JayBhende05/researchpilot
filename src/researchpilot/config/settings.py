from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
print("Base Directory " ,BASE_DIR)



PAPERS_DIR = BASE_DIR / "data" / "papers"



CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

COLLECTION_NAME = "researchpilot_papers"