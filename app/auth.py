from datetime import datetime, timedelta
import os
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext


# =========================
# Config
# =========================

SECRET_KEY = os.getenv("SECRET_KEY", "unsafe-default")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


# =========================
# Password hashing (bcrypt)
# =========================

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


# =========================
# Fake user store (PRE-HASHED)
# =========================
# password for both users = "test123"

fake_users_db = {
    "user1": {
        "username": "user1",
        "hashed_password": "$2b$12$KIXQ4z3Zk6nKx9xRz0N4eO3lXzYpQwZk9K0Z7rXb6r0k9ZpYQw7yG",
    },
    "user2": {
        "username": "user2",
        "hashed_password": "$2b$12$KIXQ4z3Zk6nKx9xRz0N4eO3lXzYpQwZk9K0Z7rXb6r0k9ZpYQw7yG",
    },
}


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# =========================
# Auth helpers
# =========================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = fake_users_db.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = fake_users_db.get(username)
    if user is None:
        raise credentials_exception

    return user
