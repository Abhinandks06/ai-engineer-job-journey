from fastapi import FastAPI
from app.api.chat import router as chat_router

app = FastAPI(title="AI Engineer Job Journey")

app.include_router(chat_router)

@app.get("/")
def root():
    return {"message": "FastAPI + Ollama is running"}
