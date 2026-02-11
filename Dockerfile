# === QM Multi Agent System ===
# Python 3.11 | FFmpeg for pydub audio processing

FROM python:3.11-slim AS base

# Install ffmpeg (required by pydub for audio duration detection)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and config
COPY src/ src/
COPY config/ config/

# Create data directories
RUN mkdir -p data/audio data/transcripts data/evaluations data/exports

# Set Python path so src/ imports work
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "src/main.py"]
# Usage:
#   docker run --env-file .env qm-system --folder data/audio
#   docker run --env-file .env qm-system --date-from 2025-02-01 --date-to 2025-02-11
#   docker run --env-file .env -v ./data:/app/data qm-system --folder data/audio
