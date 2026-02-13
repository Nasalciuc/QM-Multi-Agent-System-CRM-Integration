# === QM Multi Agent System ===
# Multi-stage build: test → production
# Production image runs as non-root user with minimal attack surface

# ── Stage 1: Base with dependencies ─────────────────────────────────
FROM python:3.11-slim AS base

# Install ffmpeg (required by pydub for audio duration detection)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install production dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Test runner (includes dev deps + tests) ────────────────
FROM base AS test

COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY src/ src/
COPY config/ config/
COPY tests/ tests/
COPY pyproject.toml .

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "pytest"]
CMD ["tests/", "-v", "--tb=short"]

# ── Stage 3: Production (no tests, no dev deps, non-root) ──────────
FROM base AS production

# Create non-root user
RUN useradd -m -s /bin/bash appuser

# Copy only production code
COPY src/ src/
COPY config/ config/

# Create data directories owned by appuser
RUN mkdir -p data/audio data/transcripts data/evaluations data/cache && \
    chown -R appuser:appuser data/

# Set Python path so src/ imports work
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# MED-NEW-10: Docker HEALTHCHECK — validates config and env vars
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD ["python", "src/main.py", "--check"]

# Run as non-root
USER appuser

ENTRYPOINT ["python", "src/main.py"]
# Usage:
#   docker build --target production -t qm-system .
#   docker build --target test -t qm-tests .
#   docker run --env-file .env qm-system --folder data/audio
#   docker run --env-file .env -v ./data:/app/data qm-system --folder data/audio
