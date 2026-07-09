from app.vectorstore.faiss_store import FAISS
from app.embeddings.embedding_model import embeddings

# -----------------------
# Load FAISS
# -----------------------

faiss_db = FAISS.load_local("indexes/vector_index", embeddings, allow_dangerous_deserialization=True)

faiss_retriever = faiss_db.as_retriever(
    search_type = "similarity",
    search_kwargs = {
        "k": 30
    }
)

# -----------------------
# FAISS Search
# -----------------------

def faiss_search(query: str):
    return faiss_retriever.invoke(query)