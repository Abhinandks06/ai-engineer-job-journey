from fastapi import FastAPI
from api.v1.routes import rag
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta

from auth import authenticate_user, create_access_token


app = FastAPI(
    title="RAG API",
    description="PDF-based Retrieval Augmented Generation API",
    version="1.0.0"
)

app.include_router(rag.router, prefix="/api/v1/rag", tags=["RAG"])

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)

    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=60),
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }
