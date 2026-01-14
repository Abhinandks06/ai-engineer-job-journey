from rag_basics.document_loader import PDFLoader

loader = PDFLoader()
docs = loader.load("data/sample.pdf")

print(docs[0]["metadata"])
print(docs[0]["text"][:300])
