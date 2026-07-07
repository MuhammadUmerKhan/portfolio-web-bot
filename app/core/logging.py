import logging
import logfire
from typing import Any
from app.core.config import get_settings

def setup_logging() -> None:
    """
    Initializes Logfire configuration and integrates it with Python's standard logging.
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

    settings = get_settings()
    logfire_token = settings.logfire.token
    
    # Configure Logfire. If a token is provided, use it. Otherwise, fallback
    # to standard local configuration (which may use console output).
    if logfire_token:
        logfire.configure(
            token=logfire_token.get_secret_value(),
            send_to_logfire=True
        )
    else:
        logfire.configure(send_to_logfire=False)
        
    # Redirect standard logging to Logfire so that any third-party or standard
    # logging calls are captured as Logfire spans/logs.
    logging.basicConfig(
        handlers=[logfire.LogfireLoggingHandler()],
        level=logging.INFO,
        force=True  # Override any pre-existing basicConfig
    )
    
    # Automatically instrument all requests.Session calls for tracing
    logfire.instrument_requests()

def instrument_app(app: Any) -> None:
    """
    Instruments a FastAPI application with Logfire.
    """
    logfire.instrument_fastapi(app)

def get_logger(name: str) -> logging.Logger:
    """
    Returns a standard logging logger configured to route through Logfire.
    """
    return logging.getLogger(name)
