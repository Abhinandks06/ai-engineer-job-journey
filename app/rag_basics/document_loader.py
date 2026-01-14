from pypdf import PdfReader
from typing import List, Dict


class PDFLoader:
    def load(self, file_path: str) -> List[Dict]:
        """
        Loads a PDF and returns text with metadata.
        """
        reader = PdfReader(file_path)
        documents = []

        for page_number, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                documents.append({
                    "text": text,
                    "metadata": {
                        "source": file_path,
                        "page": page_number + 1
                    }
                })

        return documents
