from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
import os
from typing import Optional
from threading import Lock

from app.rag_basics.document_loader import PDFLoader
from app.rag_basics.chunking_service import ChunkingService
from app.rag_basics.embeddings import EmbeddingService
from app.rag_basics.vector_store import FAISSVectorStore
from app.rag_basics.llm_service import LLMService

from app.auth import get_current_user
from app.policy import check_upload_quota, check_query_rate
from app.evaluation import log_retrieval_metrics, log_answer_outcome
import os

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE", 0.15))
TOP_K = int(os.getenv("TOP_K", 5))


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
# Services (stateless)
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
        store = FAISSVectorStore.load(index_path, metadata_path)
        vector_stores[user_id] = store
        return store

    return None





def is_refusal(answer: str) -> bool:
    return answer.strip().lower().startswith("i don't know")


def normalize_question(question: str) -> str:
    q = question.strip().lower()

    summary_triggers = [
        "what is the content",
        "what is the content in this pdf",
        "what is the content in the pdf",
        "what is this pdf about",
        "explain this pdf",
        "explain this document",
        "summarize this pdf",
        "summarize this document",
        "what is in this document",
    ]

    for trigger in summary_triggers:
        if trigger in q:
            return "Provide a concise summary of the main topics covered in this document."

    return question


def deduplicate_chunks(retrieved: list) -> list:
    """
    Keep only the highest-score chunk per (doc_id, page).
    """
    best_per_page = {}

    for r in retrieved:
        meta = r["chunk"]["metadata"]
        key = (meta.get("doc_id"), meta.get("page"))

        if key not in best_per_page or r["score"] > best_per_page[key]["score"]:
            best_per_page[key] = r

    return list(best_per_page.values())


# =========================
# Upload endpoint
# =========================

@router.post("/upload-pdf")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["username"]

    check_upload_quota(user_id)

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
# Ask endpoint
# =========================

@router.post("/ask")
async def ask_question(
    question: str,
    doc_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["username"]

    check_query_rate(user_id)

    vector_store = get_user_vector_store(user_id)
    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet for this user",
        )

    normalized_question = normalize_question(question)
    if normalized_question != question:
        print(f"[RAG INTENT NORMALIZED] '{question}' → '{normalized_question}'")

    query_embedding = embedding_service.embed_query(normalized_question)

    lock = get_user_lock(user_id)
    with lock:
        retrieved = vector_store.search(query_embedding, top_k=TOP_K)

    retrieved = deduplicate_chunks(retrieved)

    log_retrieval_metrics(
        user_id=user_id,
        question=question,
        retrieved=retrieved,
        threshold=MIN_SIMILARITY_SCORE,
    )

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
        answer = "I don't know based on the provided context."

        log_answer_outcome(user_id, question, answer)

        return {
            "question": question,
            "answer": answer,
            "sources": [],
        }

    final_chunks = [r["chunk"] for r in filtered]

    answer = llm_service.generate_answer(
        question,
        [chunk["text"] for chunk in final_chunks],
    )

    if is_refusal(answer):
        log_answer_outcome(user_id, question, answer)
        return {
            "question": question,
            "answer": answer,
            "sources": [],
        }

    unique_sources = {
        (chunk["metadata"]["source"], chunk["metadata"]["page"]): chunk["metadata"]
        for chunk in final_chunks
    }

    log_answer_outcome(user_id, question, answer)

    return {
        "question": question,
        "answer": answer,
        "sources": list(unique_sources.values()),
    }


# =========================
# Debug endpoint (dev-only)
# =========================

@router.post("/debug/ask")
async def debug_ask(
    question: str,
    doc_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    if not DEBUG_MODE:
        raise HTTPException(status_code=404, detail="Not found")

    user_id = current_user["username"]

    vector_store = get_user_vector_store(user_id)
    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet for this user",
        )

    normalized_question = normalize_question(question)
    if normalized_question != question:
        print(f"[RAG INTENT NORMALIZED] '{question}' → '{normalized_question}'")

    query_embedding = embedding_service.embed_query(normalized_question)

    lock = get_user_lock(user_id)
    with lock:
        retrieved = vector_store.search(query_embedding, top_k=TOP_K)

    retrieved = deduplicate_chunks(retrieved)

    if doc_id:
        retrieved = [
            r for r in retrieved
            if r["chunk"]["metadata"].get("doc_id") == doc_id
        ]

    debug_chunks = []
    for r in retrieved:
        debug_chunks.append({
            "score": r["score"],
            "passed_threshold": r["score"] >= MIN_SIMILARITY_SCORE,
            "text_preview": r["chunk"]["text"][:300],
            "metadata": r["chunk"]["metadata"],
        })

    return {
        "question": question,
        "threshold": MIN_SIMILARITY_SCORE,
        "top_k": 5,
        "retrieved": debug_chunks,
        "passed_any": any(c["passed_threshold"] for c in debug_chunks),
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
