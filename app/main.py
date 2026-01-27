from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from dotenv import load_dotenv

from app.api.v1.routes import api_router
from app.auth import authenticate_user, create_access_token

load_dotenv()

app = FastAPI(
    title="RAG API",
    description="PDF-based Retrieval Augmented Generation API",
    version="1.0.0"
)

# ✅ SINGLE source of truth for routes
app.include_router(api_router, prefix="/api/v1")

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
