from sentence_transformers import CrossEncoder
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