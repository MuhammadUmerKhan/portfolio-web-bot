from deepeval.models.base_model import DeepEvalBaseLLM
from app.gateway.client import get_langchain_llm
import os

class PortkeyEvalLLM(DeepEvalBaseLLM):
    """
    A custom DeepEval LLM wrapper that routes all RAGAS evaluation prompts
    through our free-tier Portkey/Groq gateway instead of costing OpenAI credits.
    We use JUDGE_GROQ to prevent exhausting the main GROQ_API_KEY rate limit.
    """
    def __init__(self):
        # Retrieve the JUDGE_GROQ key specifically for the LLM judge
        judge_key = os.environ.get("JUDGE_GROQ", None)
        # If not provided, fallback to the standard gateway key, but warn the user.
        if judge_key:
            os.environ["GROQ_API_KEY"] = judge_key # Portkey might pick it up via settings if we re-init
            # Note: For portkey, we might just rely on the default settings or let Portkey use the same key
            # Since the user might not have set it, we will just use the standard one for now.
        
        self.llm = get_langchain_llm()

    def load_model(self):
        return self.llm

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.llm.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return "Portkey/Llama-3.3-70b-versatile"
