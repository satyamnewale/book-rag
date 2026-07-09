from app.retriever.retriever_model import chain
from app.retriever.hybrid import hybridRetriever
from app.retriever.deduplicate import deduplicate
from app.retriever.reranker import rerank

hybid_search = hybridRetriever()

def multiQuery(query):
    queries = chain.invoke({"question": query})
    queries = queries.query

    all_docs = []

    for query in queries:
        
        all_docs.extend(hybid_search.invoke(query))

    unique_docs = deduplicate(all_docs)

    return rerank(query, unique_docs, 10)