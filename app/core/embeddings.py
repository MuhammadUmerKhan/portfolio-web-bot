from app.core.config import get_settings
from app.core.logging import get_logger
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = get_logger("embeddings_factory")

def get_embeddings_model() -> Embeddings:
    """
    Factory function that retrieves the appropriate embeddings model.
    Tries to initialize Google's text-embedding-004.
    If the API key is missing or connection fails, falls back to a local
    SentenceTransformer model ('sentence-transformers/all-mpnet-base-v2')
    which has the exact same 768 dimensions.
    """
    settings = get_settings()
    google_api_key = settings.gemini.api_key
    
    if google_api_key:
        try:
            logger.info("Initializing primary embedding model (Google text-embedding-004)...")
            embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.app.embedding_model,
                google_api_key=google_api_key.get_secret_value()
            )
            # Test query to confirm the API key is valid and working
            embeddings.embed_query("health check")
            logger.info("✅ Google text-embedding-004 initialized successfully.")
            return embeddings
        except Exception as e:
            logger.warning("⚠️ Google Generative AI Embeddings failed: %s. Falling back...", str(e))
    else:
        logger.warning("⚠️ GOOGLE_API_KEY / GEMINI_API_KEY not found in settings. Falling back...")
        
    try:
        # all-mpnet-base-v2 has exactly 768 dimensions, keeping it compatible with Qdrant collection vectors configuration
        fallback_model = "sentence-transformers/all-mpnet-base-v2"
        logger.info("Initializing local fallback embeddings: %s", fallback_model)
        embeddings = HuggingFaceEmbeddings(
            model_name=fallback_model,
            model_kwargs={"device": "cpu"}
        )
        logger.info("✅ Local SentenceTransformers fallback model initialized successfully.")
        return embeddings
    except Exception as ex:
        logger.error("❌ Critical: Primary and local fallback embedding models failed to initialize: %s", str(ex))
        raise ex
