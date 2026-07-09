import os
from dotenv import load_dotenv

# 1. Load basic environment variables
load_dotenv()

# 2. Configure LangSmith Environment Variables IMMEDIATELY
if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# 3. Configure Logfire IMMEDIATELY to prevent poisoning
import logfire
if os.getenv("LOGFIRE_TOKEN"):
    logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), send_to_logfire=True)
else:
    logfire.configure(send_to_logfire=False)

# 4. Now safe to import the rest of the application
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.services.chatbot import CustomDocChatbot
from app.core import get_settings, setup_logging, instrument_app, get_logger

settings = get_settings()
os.environ["LANGCHAIN_PROJECT"] = settings.app.name

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Initialize FastAPI app with descriptive title
app = FastAPI(title="Muhammad Umer Khan's RAG Bot")

# Instrument FastAPI application with Logfire
instrument_app(app)

# Enable CORS for React frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.app.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot instance
try:
    chatbot = CustomDocChatbot()
    logger.info({"message": "🤖 Chatbot initialized successfully"})
except Exception as e:
    logger.critical({"message": f"❌ Failed to initialize chatbot: {str(e)}"})
    raise

# Define request model for /chat endpoint
class QueryRequest(BaseModel):
    """Pydantic model for validating chat query requests."""
    query: str

@app.get("/")
async def root():
    """Root endpoint returning a welcome message."""
    return {"message": "Hello, I am Muhammad Umer Khan's AI Bot! 🤖"}

@app.post("/chat")
async def chat(request: QueryRequest):
    """
    Handle chat queries with rate limiting and caching.
    
    Args:
        request (QueryRequest): JSON payload with the user's query.
    
    Returns:
        dict: Response containing the chatbot's reply.
    
    Raises:
        HTTPException: If the query is invalid or processing fails.
    """
    try:
        response = await chatbot.query(request.query)
        logger.info({"message": f"💬 Query processed: {request.query} | Response: {response}"})
        return {"reply": response}
    except Exception as e:
        logger.error({"message": f"❌ API error: {str(e)}"})
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify LLM and vector store status.
    
    Returns:
        dict: Status indicating if the chatbot is operational.
    
    Raises:
        HTTPException: If critical components are not initialized.
    """
    if hasattr(chatbot, 'qa_chain') and hasattr(chatbot, 'vector_db'):
        logger.info({"message": "✅ Health check passed"})
        return {"status": "healthy"}
    logger.error({"message": "❌ Health check failed"})
    raise HTTPException(status_code=503, detail="Service Unavailable")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    await chatbot.shutdown()
    logger.info({"message": "🛑 Application shutdown gracefully"})

if __name__ == "__main__":
    import uvicorn
    logger.info({"message": "🚀 Starting FastAPI server on port 8000"})
    uvicorn.run(app, host="0.0.0.0", port=8000)
