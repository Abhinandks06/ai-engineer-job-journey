from fastapi import APIRouter

from app.api.v1.routes import rag
from app.api.v1.routes.rag_eval import router as rag_eval_router

api_router = APIRouter()

api_router.include_router(rag.router)
api_router.include_router(rag_eval_router)
