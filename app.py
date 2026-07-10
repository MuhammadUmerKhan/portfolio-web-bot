import uvicorn
from app.main import app
import gradio as gr

# Hugging Face Spaces Gradio SDK expects a Gradio interface.
# We create a simple visual interface so recruiters see a clean UI, 
# while our existing robust FastAPI app is mounted in the background to serve /query.

def ui_placeholder(text):
    return "Hello! I am Muhammad Umer Khan's AI Portfolio Bot. My API is fully active behind this screen. You can send POST requests to `/query` to interact programmatically!"

demo = gr.Interface(
    fn=ui_placeholder,
    inputs=gr.Textbox(label="Message"),
    outputs=gr.Textbox(label="Status"),
    title="Muhammad Umer Khan - AI Agent",
    description="This is the frontend shell. The FastAPI backend is running live!"
)

# Mount the FastAPI app inside the Gradio interface
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    # Hugging Face dynamically assigns port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
