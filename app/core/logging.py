
import logfire
from typing import Any
from app.core.config import get_settings

def setup_logging() -> None:
    """
    Initializes Logfire configuration.
    Instruments standard libraries (like requests) for automatic tracing.
    """
    import sys
    if sys.platform.startswith("win"):
        try:
            import io
            if hasattr(sys.stdout, "buffer"):
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
            if hasattr(sys.stderr, "buffer"):
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
        except Exception:
            pass

    # Logfire is already configured in main.py, but for isolated script runs:
    settings = get_settings()
    logfire_token = settings.logfire.token
    
    if logfire_token:
        logfire.configure(
            token=logfire_token.get_secret_value(),
            send_to_logfire=True
        )
    else:
        logfire.configure(send_to_logfire=False)
        
    # Automatically instrument all requests.Session calls for tracing
    logfire.instrument_requests()

def instrument_app(app: Any) -> None:
    """
    Instruments a FastAPI application with Logfire.
    """
    logfire.instrument_fastapi(app)
