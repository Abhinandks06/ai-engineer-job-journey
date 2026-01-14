import faiss
import numpy as np
from typing import List


class FAISSVectorStore:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.text_chunks: List[str] = []

    def add_embeddings(self, embeddings: np.ndarray, texts: List[str]):
        """
        Store embeddings and corresponding text chunks.
        """
        self.index.add(embeddings)
        self.text_chunks.extend(texts)

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        """
        Retrieve top_k most similar chunks.
        """
        scores, indices = self.index.search(query_embedding, top_k)
        results = [self.text_chunks[i] for i in indices[0]]
        return results
