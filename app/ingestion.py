from dotenv import load_dotenv
from app.loaders.text_loader import documents
from app.embeddings.embedding_model import embeddings
from app.splitters.semantic_chunking import splitter
from app.vectorstore.faiss_store import FAISS
from rank_bm25 import BM25Okapi
import pickle

load_dotenv()

# -----------------------
# Split Documents
# -----------------------

docs = splitter.split_documents(documents)

# -----------------------
# Assign Unique IDs
# -----------------------

for i, doc in enumerate(docs):
    doc.metadata["doc_id"] = i

# -----------------------
# Create FAISS
# -----------------------

vectorstore = FAISS.from_documents(
    docs,
    embeddings
)
print(docs[0].metadata)
vectorstore.save_local("indexes/vector_index")

# -----------------------
# Create BM25
# -----------------------

corpus = [doc.page_content.split() for doc in docs]

bm25 = BM25Okapi(corpus)

# -----------------------
# Save BM25
# -----------------------

with open("indexes/bm25/bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)

# -----------------------
# Save Documents
# -----------------------

with open("indexes/bm25/docs.pkl", "wb") as f:
    pickle.dump(docs, f)

print(f"Indexed {len(docs)} chunks successfully.")