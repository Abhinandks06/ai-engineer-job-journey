from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
import os
from typing import Optional
from threading import Lock

from rag_basics.document_loader import PDFLoader
from rag_basics.chunking_service import ChunkingService
from rag_basics.embeddings import EmbeddingService
from rag_basics.vector_store import FAISSVectorStore
from rag_basics.llm_service import LLMService


# =========================
# User-scoped storage utils
# =========================

DATA_ROOT = "data/users"


def get_user_dir(user_id: str) -> str:
    path = os.path.join(DATA_ROOT, user_id)
    os.makedirs(path, exist_ok=True)
    return path


def get_user_upload_dir(user_id: str) -> str:
    path = os.path.join(get_user_dir(user_id), "uploads")
    os.makedirs(path, exist_ok=True)
    return path


def get_user_vector_paths(user_id: str):
    user_dir = get_user_dir(user_id)
    return (
        os.path.join(user_dir, "faiss.index"),
        os.path.join(user_dir, "chunks.json"),
    )


# =========================
# FastAPI router
# =========================

router = APIRouter()


# =========================
# Services (shared, stateless)
# =========================

embedding_service = EmbeddingService()
chunker = ChunkingService()
llm_service = LLMService()


# =========================
# Per-user state
# =========================

vector_stores: dict[str, FAISSVectorStore] = {}
user_locks: dict[str, Lock] = {}


def get_user_lock(user_id: str) -> Lock:
    if user_id not in user_locks:
        user_locks[user_id] = Lock()
    return user_locks[user_id]


def get_user_vector_store(user_id: str) -> Optional[FAISSVectorStore]:
    if user_id in vector_stores:
        return vector_stores[user_id]

    index_path, metadata_path = get_user_vector_paths(user_id)

    if os.path.exists(index_path) and os.path.exists(metadata_path):
        vector_store = FAISSVectorStore.load(index_path, metadata_path)
        vector_stores[user_id] = vector_store
        return vector_store

    return None


# =========================
# Retrieval quality control
# =========================

MIN_SIMILARITY_SCORE = 0.15


def normalize_question(question: str) -> str:
    q = question.strip().lower()

    vague_summary_phrases = [
        "explain this pdf",
        "explain this document",
        "explain this",
        "summarize this pdf",
        "summarize this document",
        "what is this pdf about",
    ]

    if q in vague_summary_phrases:
        return "Provide a concise summary of the main topics covered in this document."

    return question


def is_refusal(answer: str) -> bool:
    normalized = answer.strip().lower()

    refusal_phrases = [
        "i don't know",
        "i do not know",
        "not present in the provided context",
        "cannot be determined from the context",
    ]

    return any(phrase in normalized for phrase in refusal_phrases)


# =========================
# Upload endpoint
# =========================

@router.post("/users/{user_id}/upload-pdf")
async def upload_pdf(
    user_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    upload_dir = get_user_upload_dir(user_id)
    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    loader = PDFLoader()
    documents = loader.load(file_path)

    chunks = chunker.chunk_documents(documents)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text found in PDF")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_service.embed_texts(texts)

    lock = get_user_lock(user_id)

    with lock:
        vector_store = get_user_vector_store(user_id)

        if vector_store is None:
            vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
            vector_stores[user_id] = vector_store

        vector_store.add_embeddings(embeddings, chunks)

        index_path, metadata_path = get_user_vector_paths(user_id)
        vector_store.save(index_path, metadata_path)

    background_tasks.add_task(ingest_pdf_for_user, user_id, file_path)

    return {
        "message": "PDF upload received. Indexing in progress.",
        "file": file.filename,
    }


# =========================
# Query endpoint
# =========================

@router.post("/users/{user_id}/ask")
async def ask_question(
    user_id: str,
    question: str,
    doc_id: Optional[str] = None,
):
    vector_store = get_user_vector_store(user_id)

    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet for this user",
        )

    query_embedding = embedding_service.embed_query(question)

    lock = get_user_lock(user_id)
    with lock:
        retrieved = vector_store.search(query_embedding, top_k=5)

    filtered = [
        r for r in retrieved
        if r["score"] >= MIN_SIMILARITY_SCORE
    ]

    if doc_id:
        filtered = [
            r for r in filtered
            if r["chunk"]["metadata"].get("doc_id") == doc_id
        ]

    if not filtered:
        return {
            "question": question,
            "answer": "I don't know based on the provided context.",
            "sources": [],
        }

    final_chunks = [r["chunk"] for r in filtered]

    normalized_question = normalize_question(question)

    answer = llm_service.generate_answer(
        normalized_question,
        [chunk["text"] for chunk in final_chunks],
    )

    # ✅ Confidence-aware refusal handling
    if is_refusal(answer):
        return {
            "question": question,
            "answer": answer,
            "sources": [],
        }

    unique_sources = {
        (chunk["metadata"]["source"], chunk["metadata"]["page"]): chunk["metadata"]
        for chunk in final_chunks
    }

    return {
        "question": question,
        "answer": answer,
        "sources": list(unique_sources.values()),
    }


# =========================
# Background ingestion
# =========================

def ingest_pdf_for_user(user_id: str, file_path: str):
    loader = PDFLoader()
    documents = loader.load(file_path)

    chunks = chunker.chunk_documents(documents)
    if not chunks:
        return

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_service.embed_texts(texts)

    lock = get_user_lock(user_id)

    with lock:
        vector_store = get_user_vector_store(user_id)

        if vector_store is None:
            vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
            vector_stores[user_id] = vector_store

        vector_store.add_embeddings(embeddings, chunks)

        index_path, metadata_path = get_user_vector_paths(user_id)
        vector_store.save(index_path, metadata_path)
