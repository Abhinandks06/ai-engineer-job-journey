from typing import List, Dict
from app.api.v1.routes.rag import (
    get_user_vector_store,
    embedding_service,
    deduplicate_chunks,
    MIN_SIMILARITY_SCORE,
    TOP_K,
    llm_service,
)


def retrieve_chunks_adapter(question: str, user_id: str) -> List[Dict]:
    """
    Returns chunks in evaluation-friendly format:
    [
        {
            "content": "...",
            "source": "fastapi_docs"
        }
    ]
    """

    vector_store = get_user_vector_store(user_id)
    if vector_store is None:
        return []

    query_embedding = embedding_service.embed_query(question)

    retrieved = vector_store.search(query_embedding, top_k=TOP_K)
    retrieved = deduplicate_chunks(retrieved)

    filtered = [
        r for r in retrieved
        if r["score"] >= MIN_SIMILARITY_SCORE
    ]

    return [
        {
            "content": r["chunk"]["text"],
            "source": r["chunk"]["metadata"].get("source", ""),
        }
        for r in filtered
    ]


def generate_answer_adapter(question: str, chunks: List[Dict]) -> str:
    texts = [c["content"] for c in chunks]

    if not texts:
        return "I don't know based on the provided context."

    return llm_service.generate_answer(question, texts)
