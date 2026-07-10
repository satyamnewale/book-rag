from app.retriever.retriever_model import decompose_chain as chain
from app.retriever.hybrid import hybridRetriever
from app.retriever.deduplicate import deduplicate
from app.retriever.reranker import rerank


hybrid_search = hybridRetriever()

async def decomposition(query):
    queries = chain.invoke({"question": query})
    queries = queries.query

    all_docs = []
    doc = hybrid_search.invoke(query)
    doc = rerank(query, doc, 30)
    all_docs.extend(doc)

    for query in queries:    
        docs = hybrid_search.invoke(query)
        docs = rerank(query, docs, 30)
        all_docs.extend(docs)

    unique_docs = deduplicate(all_docs)

    return rerank(query, unique_docs, 10)