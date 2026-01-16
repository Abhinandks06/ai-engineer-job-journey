import faiss
import numpy as np
import json
import os
from typing import List


class FAISSVectorStore:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.text_chunks: List[dict] = []

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[dict]):
        self.index.add(embeddings)
        self.text_chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        scores, indices = self.index.search(query_embedding, top_k)
        return [self.text_chunks[i] for i in indices[0]]

    # 🔹 NEW: Save index + metadata
    def save(self, index_path: str, metadata_path: str):
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.text_chunks, f)

    # 🔹 NEW: Load index + metadata
    @classmethod
    def load(cls, index_path: str, metadata_path: str):
        index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        store = cls(index.d)
        store.index = index
        store.text_chunks = chunks
        return store
