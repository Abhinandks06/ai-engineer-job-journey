import json
from typing import List, Dict


def load_eval_dataset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def keyword_coverage_score(text: str, expected_keywords: List[str]) -> float:
    """
    Returns a score between 0 and 1 indicating
    how many expected keywords appear in the text.
    """
    text_lower = text.lower()
    matched = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
    return matched / max(len(expected_keywords), 1)


def evaluate_retrieval(
    retrieved_chunks: List[Dict],
    expected_keywords: List[str],
    expected_source: str,
) -> Dict:
    """
    retrieved_chunks: [
        {
            "content": "...",
            "source": "rag" | "fastapi" | ...
        }
    ]
    """

    if not retrieved_chunks:
        return {
            "passed": False,
            "reason": "no_chunks_retrieved",
            "keyword_score": 0.0,
            "source_match": False,
        }

    combined_text = " ".join(chunk["content"] for chunk in retrieved_chunks)

    keyword_score = keyword_coverage_score(combined_text, expected_keywords)

    source_match = any(
        expected_source.lower() in chunk.get("source", "").lower()
        for chunk in retrieved_chunks
    )

    passed = keyword_score >= 0.5 and source_match

    return {
        "passed": passed,
        "keyword_score": round(keyword_score, 2),
        "source_match": source_match,
        "num_chunks": len(retrieved_chunks),
    }
