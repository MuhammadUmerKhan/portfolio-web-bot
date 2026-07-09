import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.core import get_settings

settings = get_settings()

portkey_client = Portkey(
    api_key=settings.portkey.api_key.get_secret_value(),
    config=settings.portkey.config
)


def get_langchain_llm(feature: str = "rag", is_guardrail: bool = False) -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI — a drop-in for ChatGroq in LangChain nodes.

    Why ChatOpenAI and not ChatGroq:
        Portkey is a proxy. It exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
        ChatGroq is hardwired to Groq's API and does not support routing through a proxy.
        ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
        auth + config). The @rag/model-name format is Portkey-specific — Groq's own client
        does not understand it. You are still using Groq models; Portkey is just in the middle.
    """
    model_slug = settings.portkey.groq_slug
    model_name = settings.app.guard_model_name if is_guardrail else settings.app.model_name
    
    return ChatOpenAI(
        api_key=settings.portkey.api_key.get_secret_value(),
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{model_slug}/{model_name}",
        temperature=0,
        default_headers=createHeaders(
            api_key=settings.portkey.api_key.get_secret_value(),
            config=settings.portkey.config,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production"
            }
        )
    )

def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"