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
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.services.chatbot import CustomDocChatbot
from app.core import get_settings, setup_logging, instrument_app
from app.guardrails.rails import initialize_rails
import logfire

settings = get_settings()
os.environ["LANGCHAIN_PROJECT"] = settings.app.name

# Setup logging
setup_logging()

# Initialize FastAPI app with descriptive title
app = FastAPI(title="Muhammad Umer Khan's RAG Bot")

# Setup rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Instrument FastAPI application with Logfire
instrument_app(app)

@app.on_event("startup")
async def startup_event():
    """Initialize app-level singletons on boot."""
    logfire.info("Booting application and initializing guardrails...")
    initialize_rails()

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
    logfire.info("🤖 Chatbot initialized successfully")
except Exception as e:
    logfire.error("❌ Failed to initialize chatbot: {error}", error=str(e))
    raise

# Define request model for /query endpoint
class QueryRequest(BaseModel):
    """Pydantic model for validating chat query requests."""
    query: str
    thread_id: str = "default"

@app.get("/")
async def root():
    """Root endpoint returning a welcome message."""
    return {"message": "Hello, I am Muhammad Umer Khan's AI Bot! 🤖"}

@app.post("/query")
@limiter.limit("5/minute")
async def query_endpoint(request: Request, body: QueryRequest):
    """
    Handle chat queries with rate limiting (5 req/min per IP) and caching.
    
    Args:
        request (Request): FastAPI request object (required by slowapi).
        body (QueryRequest): JSON payload with the user's query and optional thread_id.
    
    Returns:
        dict: Response containing the chatbot's reply.
    
    Raises:
        HTTPException: If the query is invalid or processing fails.
    """
    try:
        response = await chatbot.query(body.query, thread_id=body.thread_id)
        logfire.info("💬 Query processed: {query} | Response: {response}", query=body.query, response=response)
        return {"reply": response}
    except Exception as e:
        logfire.error("❌ API error: {error}", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify LLM and vector store status.
    Also acts as a keep-alive ping for the Qdrant Cloud free-tier cluster.
    
    Returns:
        dict: Status indicating if the chatbot is operational.
    
    Raises:
        HTTPException: If critical components are not initialized.
    """
    if hasattr(chatbot, 'qa_chain') and hasattr(chatbot, 'vector_db'):
        try:
            # Perform a lightweight ping to Qdrant to keep the cluster awake
            chatbot.vector_db.client.get_collection(chatbot.vector_db.collection_name)
            logfire.info("✅ Health check passed (Qdrant pinged successfully)")
            return {"status": "healthy"}
        except Exception as e:
            logfire.error("❌ Health check failed (Qdrant unreachable): {error}", error=str(e))
            raise HTTPException(status_code=503, detail="Service Unavailable: Qdrant unreachable")
            
    logfire.error("❌ Health check failed (chatbot not fully initialized)")
    raise HTTPException(status_code=503, detail="Service Unavailable: Components missing")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    await chatbot.shutdown()
    logfire.info("🛑 Application shutdown gracefully")

if __name__ == "__main__":
    import uvicorn
    logfire.info("🚀 Starting FastAPI server on port 8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
