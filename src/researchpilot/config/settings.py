from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[3]
print("Base Directory " ,BASE_DIR)



PAPERS_DIR = BASE_DIR / "data" / "papers"



CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

COLLECTION_NAME = "researchpilot_papers"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")