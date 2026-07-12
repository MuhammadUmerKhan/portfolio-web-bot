FROM python:3.12-slim AS builder

# Set environment variables to optimize Python and uv
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install uv securely
RUN pip install uv

# Install dependencies using uv sync --frozen
# We copy only pyproject.toml and uv.lock first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy the actual application source code
COPY . .

# Install the project itself (if applicable, though --no-install-project skips the package metadata install)
RUN uv sync --frozen --no-dev

# Pre-warm the embedding model at build time
# This ensures BAAI/bge-base-en-v1.5 is downloaded into the Docker image layers
# rather than downloading on the first startup (which causes cold-start timeouts)
RUN uv run python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')"

# Final stage - using the same base image to keep it slim
FROM python:3.12-slim

WORKDIR /app

# Copy the fully built virtual environment and app from the builder stage
COPY --from=builder /app /app
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Add the virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Run the FastAPI server via uvicorn directly
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
