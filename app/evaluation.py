from datetime import datetime
from typing import List, Dict


def log_retrieval_metrics(
    user_id: str,
    question: str,
    retrieved: List[Dict],
    threshold: float,
):
    scores = [r["score"] for r in retrieved]

    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "question": question,
        "retrieved_chunks": len(retrieved),
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "avg_score": sum(scores) / len(scores) if scores else None,
        "passed_threshold": any(s >= threshold for s in scores),
    }

    print("[RAG RETRIEVAL METRICS]", metrics)


def log_answer_outcome(
    user_id: str,
    question: str,
    answer: str,
):
    outcome = "refused" if answer.lower().startswith("i don't know") else "answered"

    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "question": question,
        "outcome": outcome,
    }

    print("[RAG ANSWER OUTCOME]", metrics)
