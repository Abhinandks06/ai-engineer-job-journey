from fastapi import APIRouter, HTTPException
from ollama import chat

from app.models.schemas import ChatRequest, ChatResponse
from app.core.prompts import SYSTEM_PROMPT

router = APIRouter(
    prefix="/api/v1",
    tags=["AI Chat"]
)


@router.post("/chat", response_model=ChatResponse)
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

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="AI service is currently unavailable"
        )
