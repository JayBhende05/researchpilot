from pathlib import Path

from pypdf import PdfReader



BASE_DIR = Path(__file__).resolve().parents[2]
pdf_path = BASE_DIR / "data" / "papers"


def load_pdf() -> list[dict]:
    
    
    reader = PdfReader("D:/Study/researchpilot/data/papers/Impact-of-Ipv6-Adoption-on-Internet-Infrastructure.pdf")

    documents = []

    print("Reader is " , reader)

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




value =  load_pdf()

print("VAlue is " , value)
