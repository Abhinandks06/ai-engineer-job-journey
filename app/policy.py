from datetime import datetime, timedelta
from collections import defaultdict
from fastapi import HTTPException, status


# =========================
# Quota configuration
# =========================

MAX_UPLOADS_PER_DAY = 5
MAX_QUERIES_PER_MINUTE = 10


# =========================
# In-memory stores
# =========================

user_uploads = defaultdict(list)   # user_id -> [timestamps]
user_queries = defaultdict(list)   # user_id -> [timestamps]


# =========================
# Helpers
# =========================

def _prune_old(entries: list, window: timedelta):
    now = datetime.utcnow()
    return [t for t in entries if now - t < window]


# =========================
# Policy checks
# =========================

def check_upload_quota(user_id: str):
    now = datetime.utcnow()
    window = timedelta(days=1)

    user_uploads[user_id] = _prune_old(user_uploads[user_id], window)

    if len(user_uploads[user_id]) >= MAX_UPLOADS_PER_DAY:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Daily upload quota exceeded",
        )

    user_uploads[user_id].append(now)


def check_query_rate(user_id: str):
    now = datetime.utcnow()
    window = timedelta(minutes=1)

    user_queries[user_id] = _prune_old(user_queries[user_id], window)

    if len(user_queries[user_id]) >= MAX_QUERIES_PER_MINUTE:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Query rate limit exceeded",
        )

    user_queries[user_id].append(now)
