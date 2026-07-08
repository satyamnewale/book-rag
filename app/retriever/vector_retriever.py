from app.vectorstore.faiss_store import FAISS
from app.embeddings.embedding_model import embeddings
from sentence_transformers import CrossEncoder
import re
import pickle
from collections import defaultdict
import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------
# load rerankers
# -----------------------

reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    token=os.getenv("HF_TOKEN")
)

# -----------------------
# Load FAISS
# -----------------------

db = FAISS.load_local("indexes/vector_index", embeddings, allow_dangerous_deserialization=True)


retriever = db.as_retriever(
    search_type = "similarity",
    search_kwargs = {
        "k": 30
    }
)

# -----------------------
# Load BM25
# -----------------------

with open("indexes/bm25/bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)

with open("indexes/bm25/docs.pkl", "rb") as f:
    docs = pickle.load(f)

# -----------------------
# unique id
# -----------------------

# naming
strategy = "(similarity + bm25) -> rrf -> reranker"

# -----------------------
# FAISS Search
# -----------------------

def faiss_search(query: str):
    return retriever.invoke(query)


# -----------------------
# BM25 Search
# -----------------------

def bm25_search(query: str):

    tokenized_query = query.split()

    return bm25.get_top_n(
        tokenized_query,
        docs,
        n=30
    )

# -----------------------
# RRF implement
# -----------------------

def reciprocal_rank_fusion(results, k = 60):
    scores = defaultdict(float)
    unique_docs = {}

    for ranking in results:
        for rank, doc in enumerate(ranking):
            doc_id = doc.metadata["doc_id"]
            scores[doc_id] += 1 / (k + rank + 1)
            unique_docs[doc_id] = doc

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        unique_docs[doc_id]
        for doc_id, _ in ranked
    ]

# -----------------------
# reranker function 
# -----------------------

def rerank(query: str, docs: list, top_k: int = 10):
    pairs = [
        (query, doc.page_content)
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key = lambda x: x[1],
        reverse = True
    )

    return [
        doc for doc, _ in ranked[:top_k]
    ]

# -----------------------
# Main
# -----------------------

while True:
    
    file = input("Enter the question name: ")
    if file == "exit":
        break
    else:
        faiss_result = faiss_search(file)
        bm25_results = bm25_search(file)

        hybrid_result = reciprocal_rank_fusion([faiss_result, bm25_results])
        result = rerank(file, hybrid_result[:20], top_k=10)

        safe_file = re.sub(r'[<>:"/\\|? *]', "_", file)

        with open(f"./output/{safe_file}.txt", "a", encoding="utf-8") as f:
            f.write("\n")
            f.write(f"Question: {file}, strategy: {strategy}\n")

            for doc in result:
                f.write(f"\n{doc.page_content}\n")
                f.write("-"*80)
                f.write("\n")
            f.write("\n")
            f.write("-end-"*20)