from rag_basics.document_loader import PDFLoader
from rag_basics.chunking_service import ChunkingService
from rag_basics.embeddings import EmbeddingService
from rag_basics.vector_store import FAISSVectorStore
from rag_basics.llm_service import LLMService

# Load PDF
loader = PDFLoader()
documents = loader.load("data/sample.pdf")

# Chunk documents
chunker = ChunkingService(chunk_size=500, overlap=100)
chunks = chunker.chunk_documents(documents)

# Embed chunks
embedding_service = EmbeddingService()
texts = [chunk["text"] for chunk in chunks]
embeddings = embedding_service.embed_texts(texts)

# Store in FAISS
vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
vector_store.add_embeddings(embeddings, chunks)

# Query
query = "What is the professional summary about?"
query_embedding = embedding_service.embed_query(query)

# Retrieve
retrieved_chunks = vector_store.search(query_embedding, top_k=3)

# Generate answer
llm_service = LLMService()
answer = llm_service.generate_answer(
    query,
    [chunk["text"] for chunk in retrieved_chunks]
)

print("Answer:")
print(answer)

print("\nSources:")
for chunk in retrieved_chunks:
    print(chunk["metadata"])
