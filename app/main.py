from fastapi import FastAPI
from api.v1.routes import rag

app = FastAPI(
    title="RAG API",
    description="PDF-based Retrieval Augmented Generation API",
    version="1.0.0"
)

app.include_router(rag.router, prefix="/api/v1/rag", tags=["RAG"])
