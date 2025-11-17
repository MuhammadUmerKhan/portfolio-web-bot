import logging, os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.rag_pipeline import CustomDocChatbot
from src.logger import logging

os.environ["LANGCHAIN_PROJECT"] = "PersonalAssistant"

logger = logging.getLogger(__name__)

# Initialize FastAPI app with descriptive title
app = FastAPI(title="Muhammad Umer Khan's RAG Bot")

# Enable CORS for React frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://muhammadumerkhaninfo.vercel.app"],
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
