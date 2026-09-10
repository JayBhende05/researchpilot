from pathlib import Path

from pypdf import PdfReader



BASE_DIR = Path(__file__).resolve().parents[2]
pdf_path = BASE_DIR / "data" / "papers"


def load_pdf(pdf_path: Path) -> list[dict]:
    
    reader = PdfReader(pdf_path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            documents.append(
                {
                    "text": text,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_number,
                    },
                }
            )

    return documents