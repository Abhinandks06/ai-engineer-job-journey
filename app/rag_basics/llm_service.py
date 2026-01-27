import subprocess
from typing import List


class LLMService:
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name

    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        context = "\n\n".join(context_chunks)

        prompt = f"""
        Use ONLY the information provided in the context below.
        Do NOT mention yourself, the model, or the context.
        Do NOT say phrases like "as an AI assistant".

        If the answer is not present, say:
        "I don't know based on the provided context."

        If the user asks for a summary:
        - Write 2–3 concise professional sentences
        - Make it suitable for a recruiter
        If multiple documents mention different answers,
            list all distinct answers clearly instead of choosing only one.


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
                capture_output=True,
                encoding="utf-8",
                errors="replace",
            )

        return result.stdout.strip()
