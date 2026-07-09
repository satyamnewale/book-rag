from collections import defaultdict

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
