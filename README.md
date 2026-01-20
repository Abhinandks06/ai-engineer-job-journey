# AI Engineer Job Journey 🚀

This repository documents my journey transitioning into an **AI Engineer / Python Backend Engineer (AI-focused)** role by building **production-grade AI systems** step by step.

The focus is on **practical applied AI**, not research ML — aligning with real-world startup and product engineering requirements.

---

## 🧠 What This Project Is

A **production-ready, multi-user Retrieval-Augmented Generation (RAG) backend** built using:

- FastAPI
- Local LLMs (Ollama – LLaMA 3)
- Vector databases (FAISS)
- Clean backend architecture
- Strong hallucination and data-isolation controls

This is **not a demo chatbot** — it is designed like a real backend service.

---

## ✨ Current Features

### 🔹 Backend & API
- FastAPI-based backend with clean, modular structure
- Versioned and user-scoped API endpoints
- Background tasks for non-blocking document ingestion

### 🔹 LLM Integration
- Offline LLM inference using **Ollama (LLaMA 3)**
- Strict prompt discipline (no hallucination, no self-reference)
- Confidence-aware responses

### 🔹 Retrieval-Augmented Generation (RAG)
- PDF document ingestion
- Text chunking with overlap
- Embedding generation
- FAISS vector store integration
- Source-aware answers with page-level attribution

### 🔹 Multi-User Support (Day 11)
- **Per-user document isolation**
- **Per-user FAISS vector stores**
- No shared global index
- User-scoped persistence under:

