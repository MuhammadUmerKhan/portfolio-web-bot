from qdrant_client import QdrantClient
from langchain_core.documents import Document
from app.core import get_settings, get_embeddings_model, get_logger

settings = get_settings()
logger = get_logger(__name__)

class QdrantRetrievalService:
    """Wrapper service using direct Qdrant SDK query_points method for dense search."""
    
    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant.endpoint,
            api_key=settings.qdrant.api_key.get_secret_value(),
            timeout=30.0
        )
        self.embeddings = get_embeddings_model()
        self.collection_name = settings.qdrant.collection_name
        logger.info("🔍 QdrantRetrievalService initialized for collection: %s", self.collection_name)

    def retrieve(self, query: str, limit: int = 10) -> list[Document]:
        """
        Retrieves top matches from Qdrant Cloud collection.
        
        Args:
            query (str): Search query string.
            limit (int): Number of results to return.
            
        Returns:
            list[Document]: LangChain Document objects.
        """
        try:
            # Generate search query vector embedding
            query_vector = self.embeddings.embed_query(query)
            
            # Query Qdrant Cloud using modern query_points interface
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True
            )
            
            documents = []
            for point in response.points:
                payload = point.payload or {}
                content = payload.get("content", "")
                
                # Reconstruct langchain Document representation
                doc = Document(
                    page_content=content,
                    metadata={
                        "source_type": payload.get("source_type"),
                        "source_path": payload.get("source_path"),
                        "project_name": payload.get("project_name"),
                        "project_description": payload.get("project_description"),
                        "project_url": payload.get("project_url"),
                        "project_language": payload.get("project_language"),
                        "project_topics": payload.get("project_topics", []),
                        "project_pushed_at": payload.get("project_pushed_at"),
                        "is_private": payload.get("is_private", False),
                        "relative_path": payload.get("relative_path"),
                        "page_number": payload.get("page_number"),
                        "start_index": payload.get("start_index"),
                        "headers": payload.get("headers", {}),
                        "score": point.score
                    }
                )
                documents.append(doc)
            return documents
        except Exception as e:
            logger.error("❌ Direct Qdrant query_points retrieval failed: %s", str(e))
            return []
