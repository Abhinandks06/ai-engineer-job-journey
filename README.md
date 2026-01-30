# AI Engineer Job Journey

This repository documents my transition into an **AI Engineer / Python Backend Engineer (AI-focused)** role through hands-on projects.

The focus is on **applied AI engineering**, backend reliability, and building systems that behave predictably in real-world conditions.

---

## Project Overview

This project is a **production-oriented Retrieval Augmented Generation (RAG) backend** built using FastAPI, FAISS, and a local large language model.

The system allows authenticated users to upload PDF documents, retrieve semantically relevant content, and generate answers that are **strictly grounded in retrieved context**.  
Rather than optimizing for fluency alone, the system prioritizes **correctness, traceability, and safe failure behavior**.

To reduce hallucinations, the backend applies retrieval thresholds, confidence gating, and explicit refusal logic.

---

## Architecture Overview

The system follows a modular backend design:

- **FastAPI** for API routing, authentication, and request handling
- **PDF ingestion pipeline** with chunking and metadata tracking
- **Sentence-transformer embeddings** for semantic similarity
- **FAISS** as the vector database for retrieval
- **Local LLM (via Ollama, LLaMA 3)** for answer generation
- **Evaluation module** to assess retrieval quality and answer faithfulness

Each user operates on an **isolated document index**, preventing data leakage across users.

---

## Retrieval Augmented Generation Flow

1. A user uploads a PDF document.
2. The document is split into chunks and converted into embeddings.
3. Embeddings are stored in a per-user FAISS index.
4. When a question is submitted:
   - The question is embedded
   - Top-K similar chunks are retrieved
   - Chunks are deduplicated by document and page
5. A similarity threshold removes weak or irrelevant matches.
6. A confidence gate evaluates the average retrieval score.
7. The language model is invoked **only if retrieval confidence is sufficient**.
8. If retrieval is weak or unrelated, the system explicitly refuses to answer.

This flow ensures that generated answers remain grounded in user-provided documents.

---

## Hallucination Prevention and Safety

The system is designed to prefer **safe refusal over uncertain generation**.

Hallucination reduction is achieved through:
- Minimum similarity thresholds on retrieved chunks
- Confidence gating based on average retrieval scores
- Explicit refusal when context is insufficient
- Prompt constraints that restrict the model to provided context only

When these conditions are not met, the system responds with:

```

"I don't know based on the provided context."

````

---

## Evaluation Pipeline

An internal evaluation API is included to validate RAG behavior.

For each evaluation question:
- Retrieval quality is assessed using keyword overlap and source matching
- Answer faithfulness is measured by overlap with retrieved context
- Questions are skipped when required documents are not available

This evaluation layer helps verify that answers are both relevant and grounded.

---

## How to Run

### Local
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
````

### API Interface

* Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Upload PDFs, ask questions, and run evaluations directly from Swagger

### Docker

```bash
docker build -t rag-backend .
docker run -p 8000:8000 rag-backend
```

---

## Design Notes

* This project intentionally focuses on **backend behavior and evaluation**, not frontend UI.
* Swagger UI is used as the primary interaction interface.
* The system prioritizes **predictable behavior and refusal over confident but incorrect answers**.
* Evaluation is treated as a core part of the system, not an afterthought.

The goal is to demonstrate **realistic, production-style RAG behavior**, rather than a demo chatbot.

