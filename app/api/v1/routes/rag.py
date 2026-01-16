from fastapi import APIRouter, UploadFile, File, HTTPException
import os
from typing import Optional

from rag_basics.document_loader import PDFLoader
from rag_basics.chunking_service import ChunkingService
from rag_basics.embeddings import EmbeddingService
from rag_basics.vector_store import FAISSVectorStore
from rag_basics.llm_service import LLMService

router = APIRouter()

INDEX_PATH = "data/faiss.index"
METADATA_PATH = "data/chunks.json"

# Directory to store uploaded PDFs
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize shared services (in-memory RAG)
embedding_service = EmbeddingService()
chunker = ChunkingService()
llm_service = LLMService()

vector_store: Optional[FAISSVectorStore] = None

if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
    vector_store = FAISSVectorStore.load(INDEX_PATH, METADATA_PATH)
else:
    vector_store = None

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, extract text, chunk it, embed it,
    and store embeddings in FAISS.
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

    if vector_store is None:
        vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])

    vector_store.add_embeddings(embeddings, chunks)
    vector_store.save(INDEX_PATH, METADATA_PATH)


    return {
        "message": "PDF uploaded and indexed successfully",
        "chunks_indexed": len(chunks),
        "file": file.filename
    }


@router.post("/ask")
async def ask_question(question: str, doc_id: str | None = None):
    """
    Ask a question against the indexed PDFs using RAG.
    Optionally filter by document ID.
    """
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No documents indexed yet")

    # Embed query
    query_embedding = embedding_service.embed_query(question)

    # Retrieve relevant chunks
    retrieved_chunks = vector_store.search(query_embedding, top_k=5)

    # Optional document-level filtering
    if doc_id:
        retrieved_chunks = [
            chunk for chunk in retrieved_chunks
            if chunk["metadata"].get("doc_id") == doc_id
        ]

    if not retrieved_chunks:
        return {
            "question": question,
            "answer": "I don't know based on the provided context.",
            "sources": []
        }

    # Generate answer using retrieved context
    answer = llm_service.generate_answer(
        question,
        [chunk["text"] for chunk in retrieved_chunks]
    )

    # Deduplicate sources (by source + page)
    unique_sources = {
        (chunk["metadata"]["source"], chunk["metadata"]["page"]): chunk["metadata"]
        for chunk in retrieved_chunks
    }

    return {
        "question": question,
        "answer": answer,
        "sources": list(unique_sources.values())
    }

