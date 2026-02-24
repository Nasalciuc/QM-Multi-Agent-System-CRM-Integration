# Setup Guide

## Prerequisites

- **Python** 3.11+
- **FFmpeg** (for pydub audio duration detection)
- **API Keys**: ElevenLabs + OpenRouter (minimum)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Nasalciuc/QM-Multi-Agent-System-Ring-Central.git
cd QM-Multi-Agent-System-Ring-Central
```

### 2. Create Python virtual environment

```bash
python -m venv .venv

# Activate:
.venv\Scripts\activate        # Windows (PowerShell)
.venv\Scripts\activate.bat    # Windows (CMD)
source .venv/bin/activate     # Linux / macOS
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg

FFmpeg is needed by `pydub` for reading audio file durations.

**Windows** (via Chocolatey):
```powershell
choco install ffmpeg
```

**macOS** (via Homebrew):
```bash
brew install ffmpeg
```

**Ubuntu/Debian**:
```bash
sudo apt-get install ffmpeg
```

### 5. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in the required API keys:

```env
# Required
ELEVENLABS_API_KEY=your_elevenlabs_key
OPENROUTER_API_KEY=your_openrouter_key

# Optional fallbacks
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# CRM API (needed for --date-from mode)
CRM_AI_TOKEN=your_crm_token

# Optional
WEBHOOK_URL=
```

## Configuration Files

| File | Purpose |
|------|---------|
| `config/agents.yaml` | Pipeline settings, ElevenLabs config |
| `config/qa_criteria.yaml` | 48 evaluation criteria definitions |
| `config/models.yaml` | LLM model definitions, pricing, fallback chain |
| `config/logging.yaml` | Logging configuration (handlers, levels) |

### Customizing QA Criteria

Edit `config/qa_criteria.yaml` to add, remove, or modify evaluation criteria.
Each criterion requires:

```yaml
criterion_key:
  description: "What the agent should do"
  category: "opening"          # opening | interview | psychological_framing | first_call_closing | second_call_opening | strategic_presentation | creating_certainty | second_call_objection_handling | commitment_closing | communication
  weight: 1.0                 # Score weight (default 1.0)
  call_applicability: both    # "first_only" | "second_only" | "both"
```

### Customizing LLM Models

Edit `config/models.yaml` to change the primary model or fallback chain:

```yaml
primary:
  provider: openrouter
  model: openai/gpt-4o-2024-11-20
  api_key_env: OPENROUTER_API_KEY
  base_url: https://openrouter.ai/api/v1
```

## Running

### Local audio files

```bash
# Single folder
python src/main.py --folder data/audio

# Specific files
python src/main.py --local data/audio/call1.mp3 data/audio/call2.mp3
```

### CRM API

```bash
python src/main.py --date-from 2026-02-01 --date-to 2026-02-10
```

### Docker

```bash
# Build
docker build -t qm-system .

# Run
docker run --env-file .env -v ./data:/app/data qm-system --folder data/audio

# Docker Compose
docker compose up
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Specific test file
python -m pytest tests/test_processing.py -v
```

## Output Files

Results are saved to `data/evaluations/`:

| File | Content |
|------|---------|
| `QM_YYYYMMDD_HHMMSS.xlsx` | Excel with Summary + Details sheets |
| `QM_YYYYMMDD_HHMMSS_summary.csv` | CSV summary (one row per call) |
| `QM_YYYYMMDD_HHMMSS_details.csv` | CSV details (one row per criterion) |
| `QM_YYYYMMDD_HHMMSS.json` | Full JSON evaluation data |

## Troubleshooting

### Common Issues

**`ModuleNotFoundError: No module named 'src'`**
Set `PYTHONPATH` to include the src directory:
```bash
export PYTHONPATH=./src  # Linux/Mac
set PYTHONPATH=./src     # Windows
```

**`pydub` can't read audio duration**
Install FFmpeg (see step 4 above). Duration becomes optional — pipeline continues without it.

**OpenRouter rate limits**
The system retries with exponential backoff. If persistent, reduce batch size or switch to a higher-rate model in `config/models.yaml`.

**ElevenLabs quota exceeded**
Batch processing stops early when quota is hit. Check your ElevenLabs dashboard for remaining credits.
