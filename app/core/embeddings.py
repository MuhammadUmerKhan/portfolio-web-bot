from app.core.config import get_settings
from app.core.logging import get_logger
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

logger = get_logger("embeddings_factory")

def get_embeddings_model() -> Embeddings:
    """
    Factory function that retrieves the primary embeddings model.
    Uses the highly-ranked BAAI/bge-base-en-v1.5 model from Hugging Face
    running locally on CPU. This provides state-of-the-art retrieval
    quality (768 dimensions) without API costs, tokens quota limits, or rate limits (429s).
    """
    settings = get_settings()
    model_name = settings.app.embedding_model
    
    logger.info("Initializing local HuggingFace embeddings model: %s", model_name)
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"}
        )
        logger.info("✅ Local embeddings model initialized successfully.")
        return embeddings
    except Exception as e:
        logger.error("❌ Critical: Failed to initialize local HuggingFace embeddings: %s", str(e))
        raise e
