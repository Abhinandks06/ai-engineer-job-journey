from fastapi import FastAPI
from pydantic import BaseModel
from ollama import chat
from fastapi import HTTPException

app = FastAPI(title="AI Engineer Job Journey")

SYSTEM_PROMPT = """
You are an AI assistant used inside a backend API.

Rules:
- Answer clearly and concisely
- If the question is unclear, ask for clarification
- Do not make up facts
- If you do not know something, say so
- Keep answers under 150 words unless asked otherwise
"""

# ✅ NEW: Request schema
class ChatRequest(BaseModel):
    prompt: str

# ✅ NEW: Response schema
class ChatResponse(BaseModel):
    prompt: str
    response: str

@app.get("/")
def root():
    return {"message": "FastAPI + Ollama is running"}

@app.post("/chat", response_model=ChatResponse)
def chat_with_llama(request: ChatRequest):
    try:
        result = chat(
            model="llama3",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.prompt}
            ]
        )

        return ChatResponse(
            prompt=request.prompt,
            response=result["message"]["content"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="AI service is currently unavailable"
        )