import faiss
import numpy as np
from typing import List, Dict


class FAISSVectorStore:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: List[Dict] = []

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict]):
        """
        Store embeddings and corresponding chunk objects (text + metadata).
        """
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Retrieve top_k most similar chunks with metadata.
        """
        scores, indices = self.index.search(query_embedding, top_k)
        return [self.chunks[i] for i in indices[0]]
