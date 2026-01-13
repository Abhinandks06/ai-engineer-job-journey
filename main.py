from fastapi import FastAPI
from ollama import chat

app = FastAPI(title="AI Engineer Job Journey")

@app.get("/")
def root():
    return {"message": "FastAPI + Ollama is running"}

@app.post("/chat")
def chat_with_llama(prompt: str):
    response = chat(
        model="llama3",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return {
        "prompt": prompt,
        "response": response["message"]["content"]
    }
