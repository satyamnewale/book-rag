from app.retriever.hybrid import hybridRetriever
from app.retriever.reranker import rerank
from langchain_core.documents import Document

from app.retriever.decomposition import decomposition
from app.retriever.multi_model_retriever import multiQuery

from app.retriever.strategies_model import strategy_chain as chain
class retriever:

    def hybrid_approch(self , query : str) -> list[Document] :
        # hybrid approch
        hybrid_search = hybridRetriever()
        hybrid_result = hybrid_search.invoke(query)
        # rerank
        rerank_result = rerank(query, hybrid_result, 10)
        return rerank_result

    def multi_query(self, query : str) -> list[Document] :
        # multi query approch
        return multiQuery(query)
    
    def decompose(self , query : str) -> list[Document] :
        # decomposition approch
        return decomposition(query)

    def invoke(self , query : str) -> list[Document] :
        approch = chain.invoke({"question": query})
        approch = approch.strategy
        
        retrievers = {
            "hybrid" : self.hybrid_approch,
            "multi_query" : self.multi_query,
            "decomposition" : self.decompose
        }

        return retrievers[approch](query) 