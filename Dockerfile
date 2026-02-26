# Build stage for smaller final image
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies required for OpenCV and Audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# Exclude heavy GPU packages for inference to keep image small (if deploying to free tier)
# For production GPU deployment, use the full requirements.
RUN pip install --no-cache-dir -r requirements.txt torchvision torchaudio

# Final Stage
FROM python:3.10-slim

WORKDIR /app

# Copy system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code and frontend
COPY src/ /app/src/
COPY public/ /app/public/

# Expose the API port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application using Gunicorn with Uvicorn workers for ASGI support
CMD ["gunicorn", "src.api:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120"]
