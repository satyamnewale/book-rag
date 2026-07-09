from app.retriever.faiss_retriever import faiss_search
from app.retriever.bm25_retiever import bm25_search
from app.retriever.rrf import reciprocal_rank_fusion
from langchain_core.documents import Document

class hybridRetriever :

    def invoke(self , query : str) -> list[Document] :
        faiss_result = faiss_search(query)
        bm25_results = bm25_search(query)
        rrf_result = reciprocal_rank_fusion([faiss_result, bm25_results])
        return rrf_result[:20]