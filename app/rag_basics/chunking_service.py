from typing import List, Dict


class ChunkingService:
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Splits documents into overlapping chunks while preserving metadata.
        """
        chunks = []

        for doc in documents:
            text = doc["text"]
            metadata = doc["metadata"]

            start = 0
            text_length = len(text)

            while start < text_length:
                end = start + self.chunk_size
                chunk_text = text[start:end]

                chunks.append({
                    "text": chunk_text,
                    "metadata": metadata
                })

                start = end - self.overlap

        return chunks
