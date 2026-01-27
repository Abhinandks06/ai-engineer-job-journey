from typing import Dict


def evaluate_faithfulness(
    answer: str,
    retrieved_context: str,
) -> Dict:
    """
    Simple faithfulness check:
    - How much of the answer overlaps with retrieved context?
    """

    if not retrieved_context.strip():
        return {
            "faithful": False,
            "reason": "empty_context",
            "overlap_score": 0.0,
        }

    answer_words = set(answer.lower().split())
    context_words = set(retrieved_context.lower().split())

    if not answer_words:
        return {
            "faithful": False,
            "reason": "empty_answer",
            "overlap_score": 0.0,
        }

    overlap = answer_words.intersection(context_words)
    overlap_score = len(overlap) / max(len(answer_words), 1)

    faithful = overlap_score >= 0.3

    return {
        "faithful": faithful,
        "overlap_score": round(overlap_score, 2),
        "overlap_words_sample": list(overlap)[:10],
    }
