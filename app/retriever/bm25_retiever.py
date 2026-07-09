import pickle

# -----------------------
# Load BM25
# -----------------------

with open("indexes/bm25/bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)

with open("indexes/bm25/docs.pkl", "rb") as f:
    docs = pickle.load(f)

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