from dotenv import load_dotenv
from app.loaders.text_loader import documents
from app.embeddings.embedding_model import embeddings
from app.splitters.semantic_chunking import Semantic_splitter, recursive_splitter
from app.vectorstore.faiss_store import FAISS
from rank_bm25 import BM25Okapi
import pickle

load_dotenv()

# -----------------------
# Split Documents
# -----------------------

semantic_docs = Semantic_splitter.split_documents(documents)

final_docs = []

for doc in semantic_docs:
    if len(doc.page_content) > 1200:
        final_docs.extend(recursive_splitter.split_documents([doc]))
    else:
        final_docs.append(doc)

# -----------------------
# Assign Unique IDs
# -----------------------

for i, doc in enumerate(final_docs):
    doc.metadata["doc_id"] = i

# -----------------------
# Create FAISS
# -----------------------

vectorstore = FAISS.from_documents(
    final_docs,
    embeddings
)
print(final_docs[0].metadata)
vectorstore.save_local("indexes/vector_index")

# -----------------------
# Create BM25
# -----------------------

corpus = [doc.page_content.split() for doc in final_docs]

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
    pickle.dump(final_docs, f)

print(f"Indexed {len(final_docs)} chunks successfully.")