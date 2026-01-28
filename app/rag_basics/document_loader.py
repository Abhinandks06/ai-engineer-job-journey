from pypdf import PdfReader
from typing import List, Dict
import os


class PDFLoader:
    def load(self, file_path: str) -> List[Dict]:
        """
        Loads a PDF and returns text with metadata.
        Source is derived deterministically from filename.
        """
        reader = PdfReader(file_path)
        documents = []

        # ✅ Derive source from filename (ethical & scalable)
        filename = os.path.basename(file_path)
        source = filename.lower().replace(".pdf", "")

        for page_number, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                documents.append({
                    "text": text,
                    "metadata": {
                        "source": source,         
                        "page": page_number + 1
                    }
                })

        return documents
