from rag_basics.document_loader import PDFLoader
from rag_basics.chunking_service import ChunkingService

loader = PDFLoader()
documents = loader.load("data/sample.pdf")

chunker = ChunkingService(chunk_size=500, overlap=100)
chunks = chunker.chunk_documents(documents)

print(f"Total chunks: {len(chunks)}")
print(chunks[0]["metadata"])
print(chunks[0]["text"][:300])
