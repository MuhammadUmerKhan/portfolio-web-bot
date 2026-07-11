# Use an official, lightweight Python 3.12 image
FROM python:3.12-slim

# Set environment variables for Python and cache directories
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/tmp/huggingface \
    FLASHRANK_CACHE=/tmp/flashrank \
    PORT=8000

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
# --no-dev: skips dev dependencies (e.g. pytest)
# --frozen: ensures we strictly use versions from uv.lock
RUN uv sync --no-dev --frozen

# Copy the rest of the application
COPY . .

# Pre-warm the HuggingFace embedding model at build time.
# This downloads the model weights so they are baked into the Docker image,
# avoiding a massive download and cold-start penalty on platforms like Render.
RUN uv run python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5', model_kwargs={'device': 'cpu'})"

# Pre-warm the FlashRank cross-encoder model at build time.
RUN uv run python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='/tmp/flashrank')"

# Expose the FastAPI port
EXPOSE 8000

# Start the application using uv run to automatically use the created virtual environment
CMD uv run uvicorn main:app --host 0.0.0.0 --port $PORT
