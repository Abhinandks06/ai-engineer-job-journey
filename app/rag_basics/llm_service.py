import subprocess
from typing import List


class LLMService:
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name

    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        context = "\n\n".join(context_chunks)

        prompt = f"""
You are an AI assistant. Answer the question using ONLY the context below.
If the answer is not present in the context, say "I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
""".strip()

        result = subprocess.run(
            ["ollama", "run", self.model_name],
            input=prompt,
            text=True,
            capture_output=True
        )

        return result.stdout.strip()
