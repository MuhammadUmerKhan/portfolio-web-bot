# Use a slim Python 3.12 image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

# Install uv for blazingly fast dependency installation
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy pyproject.toml and uv.lock first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen

# Copy the rest of the application
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI server, reading PORT from environment (defaulting to 8000)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
