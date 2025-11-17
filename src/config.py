import os
from dotenv import load_dotenv
from src.logger import logging

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
logger.info({"message": "📂 Loaded environment variables"})

# Configuration settings for the RAG application
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error({"message": "❌ Missing GROQ_API_KEY in .env"})
    raise ValueError("Missing GROQ_API_KEY in .env")

RESUME_PATH = os.getenv("RESUME_PATH", os.path.join("assets", "Muhammad_Umer_Khan_AI_Resume.pdf"))
MODEL_NAME = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "text-embedding-3-small"

# Validate resume path
if not os.path.exists(RESUME_PATH):
    logger.error({"message": f"❌ PDF not found at {RESUME_PATH}"})
    raise FileNotFoundError(f"PDF not found at {RESUME_PATH}")
logger.info({"message": f"✅ Validated resume path: {RESUME_PATH}"})