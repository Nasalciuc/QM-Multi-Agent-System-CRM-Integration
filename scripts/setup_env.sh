#!/usr/bin/env bash
# Setup development environment for QM Multi Agent System
# Usage: bash scripts/setup_env.sh

set -euo pipefail

echo "=== QM Multi Agent System — Environment Setup ==="

# Check Python version
PYTHON=${PYTHON:-python3}
if ! command -v "$PYTHON" &>/dev/null; then
    PYTHON=python
fi

PYTHON_VERSION=$("$PYTHON" --version 2>&1 | awk '{print $2}')
MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

echo "Python: $PYTHON_VERSION"
if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 11 ]; }; then
    echo "ERROR: Python 3.11+ required (found $PYTHON_VERSION)"
    exit 1
fi

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate
echo "Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create data directories
echo "Creating data directories..."
mkdir -p data/audio data/transcripts data/evaluations data/cache

# Check .env
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "Created .env from .env.example — edit it with your API keys."
    else
        echo "WARNING: No .env or .env.example found."
    fi
else
    echo ".env already exists."
fi

# Check FFmpeg
if command -v ffmpeg &>/dev/null; then
    echo "FFmpeg: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "WARNING: FFmpeg not found. Audio duration detection will be unavailable."
    echo "  Install: sudo apt-get install ffmpeg  (or brew install ffmpeg)"
fi

echo ""
echo "=== Setup complete ==="
echo "  Activate:  source .venv/bin/activate"
echo "  Run:       python src/main.py --folder data/audio"
echo "  Tests:     python -m pytest tests/ -v"
