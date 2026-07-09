from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document
import logfire

class RankingService:
    """Wrapper service using local FlashRank Cross-Encoder models to re-rank search results on CPU."""
    
    def __init__(self):
        logfire.info("Initializing local FlashRank ranker...")
        try:
            # We use ms-marco-MiniLM-L-12-v2 as it offers excellent precision-to-speed ratio
            self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")
            logfire.info("✅ FlashRank initialized successfully (model: ms-marco-MiniLM-L-12-v2)")
        except Exception as e:
            logfire.warning("⚠️ FlashRank custom model failed: {error}. Initializing default ranker...", error=str(e))
            self.ranker = Ranker()
            logfire.info("✅ FlashRank initialized successfully (default model)")

    def rerank(self, query: str, documents: list[Document], top_n: int = 4) -> list[Document]:
        """
        Reranks LangChain Documents using FlashRank.
        
        Args:
            query (str): The search query.
            documents (list[Document]): Candidate retrieved documents.
            top_n (int): Number of final documents to return.
            
        Returns:
            list[Document]: Reranked documents.
        """
        if not documents:
            return []
            
        try:
            passages = []
            for idx, doc in enumerate(documents):
                passages.append({
                    "id": idx,
                    "text": doc.page_content,
                    "meta": doc.metadata
                })
                
            rerank_request = RerankRequest(query=query, passages=passages)
            results = self.ranker.rerank(rerank_request)
            
            reranked_docs = []
            # Take only the top_n results after reranking
            for item in results[:top_n]:
                original_idx = item["id"]
                original_doc = documents[original_idx]
                
                # Copy metadata and append the rerank score
                new_metadata = original_doc.metadata.copy()
                new_metadata["rerank_score"] = item["score"]
                
                reranked_docs.append(
                    Document(
                        page_content=original_doc.page_content,
                        metadata=new_metadata
                    )
                )
            logfire.info("🎯 Reranked {total} candidates to top {top} results.", total=len(documents), top=len(reranked_docs))
            return reranked_docs
        except Exception as e:
            logfire.error("❌ FlashRank reranking failed: {error}. Falling back to input order.", error=str(e))
            return documents[:top_n]
