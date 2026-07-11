import gradio as gr
from main import app as fastapi_app
import logfire

logfire.info("Initializing Hugging Face Gradio Space")

with gr.Blocks(title="Portfolio API Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Umer's Portfolio RAG Backend")
    gr.Markdown("""
    Welcome! This Hugging Face Space is actively serving the FastAPI backend for Muhammad Umer Khan's portfolio frontends.
    
    The backend uses a LangGraph ReAct Agent, NeMo Guardrails, and Qdrant Vector Search.
    
    **API Endpoints Active:**
    - `/chat` : Main conversational ReAct endpoint
    - `/health` : Health check endpoint
    
    *Note: The frontend chat is hosted externally on Vercel. This space just serves the high-performance API.*
    """)

# HF Spaces Gradio SDK looks for an instance named `app`.
# By mounting the Gradio demo onto our FastAPI app, we get both!
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
