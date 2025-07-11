# Simple Dockerfile for Video Processing Pipeline
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY video_pipeline.py .
COPY test_video_pipeline.py .

# Create directories for input/output
RUN mkdir -p /app/input /app/output /app/logs

# Run as non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Default command shows help
ENTRYPOINT ["python", "video_pipeline.py"]
CMD ["--help"] 