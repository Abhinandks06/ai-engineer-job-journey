from rag_basics.embeddings import EmbeddingService
from rag_basics.vector_store import FAISSVectorStore
from rag_basics.llm_service import LLMService

documents = [
    "FastAPI is a modern Python web framework for building APIs.",
    "FastAPI supports async programming and is very fast.",
    "Django is a high-level Python web framework.",
    "Retrieval Augmented Generation reduces hallucination in LLMs."
]

# Initialize services
embedding_service = EmbeddingService()
llm_service = LLMService()

# Embed documents
doc_embeddings = embedding_service.embed_texts(documents)

# Vector store
vector_store = FAISSVectorStore(embedding_dim=doc_embeddings.shape[1])
vector_store.add_embeddings(doc_embeddings, documents)

# Query
query = "Why is FastAPI good for APIs?"
query_embedding = embedding_service.embed_query(query)

# Retrieve relevant chunks
retrieved_chunks = vector_store.search(query_embedding, top_k=2)

# Generate answer using RAG
answer = llm_service.generate_answer(query, retrieved_chunks)

print("Answer:")
print(answer)
