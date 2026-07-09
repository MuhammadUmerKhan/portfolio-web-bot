from qdrant_client import QdrantClient
from langchain_core.documents import Document
from app.core import get_settings, get_embeddings_model
from app.core.circuit_breaker import AsyncCircuitBreaker
import logfire

qdrant_circuit_breaker = AsyncCircuitBreaker("qdrant", failure_threshold=3, recovery_timeout=30.0)

settings = get_settings()

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
        logfire.info("🔍 QdrantRetrievalService initialized for collection: {collection}", collection=self.collection_name)

    @qdrant_circuit_breaker
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
            logfire.error("❌ Direct Qdrant query_points retrieval failed: {error}", error=str(e))
            return []

    @qdrant_circuit_breaker
    def fetch_all_chunks(self) -> list[dict]:
        """
        Scrolls all points in the collection from Qdrant Cloud.
        Handles auto-suspended free clusters with a robust retry-with-backoff loop.
        """
        import time
        max_retries = 5
        backoff_delay = 2.0
        
        for attempt in range(1, max_retries + 1):
            try:
                # Check collection exists
                if not self.client.collection_exists(self.collection_name):
                    logfire.error("❌ Collection '{collection}' does not exist in Qdrant.", collection=self.collection_name)
                    return []
                
                # Fetch all points using scroll
                logfire.info("📦 Fetching all chunks from Qdrant Cloud collection '{collection}' (Attempt {attempt}/{max_retries})...", collection=self.collection_name, attempt=attempt, max_retries=max_retries)                
                all_points = []
                next_offset = None
                
                while True:
                    response, next_offset = self.client.scroll(
                        collection_name=self.collection_name,
                        with_payload=True,
                        with_vectors=False,
                        limit=1000,
                        offset=next_offset
                    )
                    all_points.extend(response)
                    if next_offset is None:
                        break
                        
                logfire.info("✅ Successfully loaded {count} chunks from Qdrant Cloud.", count=len(all_points))
                
                # Reconstruct records in chunks.json format
                records = []
                for point in all_points:
                    payload = point.payload or {}
                    records.append({
                        "id": str(point.id),
                        "content": payload.get("content", ""),
                        "source_path": payload.get("source_path", ""),
                        "source_type": payload.get("source_type"),
                        "project_name": payload.get("project_name"),
                        "metadata": payload.get("metadata", {})
                    })
                return records
                
            except Exception as e:
                logfire.warning("⚠️ Qdrant scroll failed on attempt {attempt}: {error}", attempt=attempt, error=str(e))
                if attempt == max_retries:
                    logfire.error("❌ Max retries reached. Failing Qdrant scroll.")
                    raise RuntimeError(f"Failed to load knowledge base from Qdrant Cloud: {str(e)}") from e
                
                # Exponential backoff
                sleep_time = backoff_delay * (2 ** (attempt - 1))
                logfire.info("Sleeping for {sleep_time}s before retrying...", sleep_time=sleep_time)
                time.sleep(sleep_time)
        return []
