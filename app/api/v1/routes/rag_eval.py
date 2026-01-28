from fastapi import APIRouter, Depends
from app.auth import get_current_user

from app.rag_eval.retrieval_evaluator import (
    evaluate_retrieval,
    load_eval_dataset,
)
from app.rag_eval.faithfulness_evaluator import evaluate_faithfulness
from app.rag_eval.rag_adapters import (
    retrieve_chunks_adapter,
    generate_answer_adapter,
)


def user_has_required_sources(retrieved_chunks, required_sources):
    if not required_sources:
        return True

    present_sources = {
        c["source"].lower() for c in retrieved_chunks
    }

    for required in required_sources:
        required = required.lower()
        for present in present_sources:
            if required in present:
                return True

    return False



router = APIRouter(prefix="/rag/eval", tags=["RAG Evaluation"])


@router.post("/")
def evaluate_rag(
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["username"]

    dataset = load_eval_dataset("app/rag_eval/eval_dataset.json")
    results = []

    for item in dataset:
        question = item["question"]
        required_sources = item.get("required_sources", [])

        # 1️⃣ Retrieve chunks (REAL pipeline)
        retrieved_chunks = retrieve_chunks_adapter(
            question=question,
            user_id=user_id,
        )

        # 🚦 NEW: document-aware evaluation gate
        if not user_has_required_sources(
            retrieved_chunks,
            required_sources,
        ):
            results.append({
                "question_id": item["id"],
                "question": question,
                "skipped": True,
                "reason": "required_documents_not_present",
            })
            continue

        # 2️⃣ Evaluate retrieval quality
        retrieval_result = evaluate_retrieval(
            retrieved_chunks=retrieved_chunks,
            expected_keywords=item["expected_keywords"],
            expected_source=item["expected_source"],
        )

        # 3️⃣ Generate answer (REAL LLM)
        answer = generate_answer_adapter(
            question=question,
            chunks=retrieved_chunks,
        )

        # 4️⃣ Faithfulness check
        combined_context = " ".join(
            c["content"] for c in retrieved_chunks
        )

        faithfulness_result = evaluate_faithfulness(
            answer=answer,
            retrieved_context=combined_context,
        )

        results.append({
            "question_id": item["id"],
            "question": question,
            "answer": answer,
            "retrieval": retrieval_result,
            "faithfulness": faithfulness_result,
        })

    return {
        "summary": {
            "total": len(results),
            "passed": sum(
                1 for r in results
                if not r.get("skipped")
                and r["retrieval"]["passed"]
                and r["faithfulness"]["faithful"]
            ),
            "skipped": sum(
                1 for r in results
                if r.get("skipped")
            ),
        },
        "details": results,
    }
