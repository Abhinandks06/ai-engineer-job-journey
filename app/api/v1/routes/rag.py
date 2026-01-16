from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi import BackgroundTasks
import os
from typing import Optional
from threading import Lock
from rag_basics.document_loader import PDFLoader
from rag_basics.chunking_service import ChunkingService
from rag_basics.embeddings import EmbeddingService
from rag_basics.vector_store import FAISSVectorStore
from rag_basics.llm_service import LLMService

router = APIRouter()

# Persistence paths
INDEX_PATH = "data/faiss.index"
METADATA_PATH = "data/chunks.json"

# Upload directory
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Services
embedding_service = EmbeddingService()
chunker = ChunkingService()
llm_service = LLMService()

# Vector store
vector_store: Optional[FAISSVectorStore] = None
vector_store_lock = Lock()


# Retrieval quality control
MIN_SIMILARITY_SCORE = 0.15

# Load persisted index on startup
if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
    vector_store = FAISSVectorStore.load(INDEX_PATH, METADATA_PATH)
else:
    vector_store = None


@router.post("/upload-pdf")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a PDF, extract text, chunk it, embed it,
    and store embeddings in FAISS (persistent).
    """
    global vector_store

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Load PDF
    loader = PDFLoader()
    documents = loader.load(file_path)

    # Chunk documents
    chunks = chunker.chunk_documents(documents)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text found in PDF")

    # Embed chunks
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_service.embed_texts(texts)

    # Initialize or append to vector store
    with vector_store_lock:
        if vector_store is None:
            vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])

        vector_store.add_embeddings(embeddings, chunks)
        vector_store.save(INDEX_PATH, METADATA_PATH)
        
    background_tasks.add_task(ingest_pdf, file_path)

    return {
        "message": "PDF upload received. Indexing in progress.",
        "file": file.filename
    }



@router.post("/ask")
async def ask_question(question: str, doc_id: Optional[str] = None):
    """
    Ask a question against indexed PDFs using
    confidence-aware RAG with optional document filtering.
    """
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No documents indexed yet")

    # Embed query
    query_embedding = embedding_service.embed_query(question)

    # Retrieve chunks with similarity scores
    with vector_store_lock:
        retrieved = vector_store.search(query_embedding, top_k=5)


    # Apply similarity threshold
    filtered = [
        r for r in retrieved
        if r["score"] >= MIN_SIMILARITY_SCORE
    ]

    # Optional document-level filtering
    if doc_id:
        filtered = [
            r for r in filtered
            if r["chunk"]["metadata"].get("doc_id") == doc_id
        ]

    if not filtered:
        return {
            "question": question,
            "answer": "I don't know based on the provided context.",
            "sources": []
        }

    # Extract final chunks
    final_chunks = [r["chunk"] for r in filtered]

    # Generate grounded answer
    answer = llm_service.generate_answer(
        question,
        [chunk["text"] for chunk in final_chunks]
    )

    # Deduplicate sources
    unique_sources = {
        (chunk["metadata"]["source"], chunk["metadata"]["page"]): chunk["metadata"]
        for chunk in final_chunks
    }

    return {
        "question": question,
        "answer": answer,
        "sources": list(unique_sources.values())
    }
def ingest_pdf(file_path: str):
    global vector_store

    loader = PDFLoader()
    documents = loader.load(file_path)

    chunks = chunker.chunk_documents(documents)
    if not chunks:
        return

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_service.embed_texts(texts)

    with vector_store_lock:
        if vector_store is None:
            vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])

        vector_store.add_embeddings(embeddings, chunks)
        vector_store.save(INDEX_PATH, METADATA_PATH)
