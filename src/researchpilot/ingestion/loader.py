from pathlib import Path

from pypdf import PdfReader

# from researchpilot.config.settings import PAPERS_DIR


def load_pdf(pdf_path: Path) -> list[dict]:
    pdf_path = Path(pdf_path)

    if pdf_path.is_dir():
        documents = []
        for file in sorted(pdf_path.glob("*.pdf")):
            documents.extend(load_pdf(file))
        return documents

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

# value   = load_pdf(PAPERS_DIR)

# print("The loaded value of the PDF is " , value)