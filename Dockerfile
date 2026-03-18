# 1. Base Image
FROM python:3.10-slim

# 2. Set Directory
WORKDIR /app

# 3. Install System Dependencies for Video/Audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Python Libraries (Using the CPU-only URL to save memory)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 5. Copy EVERYTHING from D:\MULTIMODAL_EMOTION_DETECTION_01
# This replaces the broken COPY src/ and COPY public/ lines
COPY . .

# 6. Expose the Port (Streamlit uses 8501)
EXPOSE 8501

# 7. Start the Dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
