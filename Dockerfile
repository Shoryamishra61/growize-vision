# Growize Vision — Production Cloud Run Dockerfile
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT="8080"

WORKDIR /app

# Install system dependencies for OpenCV and FFmpeg video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy configuration and dependency specs
COPY pyproject.toml README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    gradio>=4.36.0 \
    opencv-python-headless>=4.8.0 \
    numpy>=1.24.0 \
    scikit-learn>=1.3.0 \
    pillow>=10.0.0 \
    pyyaml>=6.0.0 \
    imageio-ffmpeg>=0.4.9

# Copy package source code, models, and assets
COPY src/ ./src/
COPY config/ ./config/
COPY assets/ ./assets/
COPY data/demo/ ./data/demo/
COPY apps/ ./apps/

# Install the package in editable/local mode
RUN pip install --no-cache-dir -e .

EXPOSE 8080

# Healthcheck for Cloud Run
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

CMD ["python", "apps/gradio_app.py"]
