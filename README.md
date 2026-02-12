# 🎯 QM Multi Agent System — Call Center Quality Assurance

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-173%20passed-brightgreen.svg)](#tests)
[![License](https://img.shields.io/badge/License-Private-lightgrey.svg)](#)

Automated **4-agent pipeline** that evaluates call center recordings against **24 quality criteria** using LLM-powered analysis.

---

## Overview

| Agent | Purpose | Technology |
|-------|---------|------------|
| **Agent 1** — Audio | Download / find call recordings | RingCentral SDK or local files |
| **Agent 2** — Transcription | Speech-to-text with diarization | ElevenLabs Scribe v2 |
| **Agent 3** — Evaluation | Score transcript against 24 QA criteria | OpenRouter / OpenAI GPT-4o |
| **Agent 4** — Export | Generate Excel, CSV, JSON reports | pandas + openpyxl |

### Pipeline Flow

```
┌─────────────┐    ┌─────────────────┐    ┌───────────────┐    ┌──────────┐
│  Agent 1    │───▶│    Agent 2      │───▶│   Agent 3     │───▶│ Agent 4  │
│  Audio      │    │  Transcription  │    │  Evaluation   │    │ Export   │
│  Retrieval  │    │  (ElevenLabs)   │    │  (LLM + 24   │    │ Excel/   │
│             │    │                 │    │   criteria)   │    │ CSV/JSON │
└─────────────┘    └─────────────────┘    └───────────────┘    └──────────┘
```

## Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/Nasalciuc/QM-Multi-Agent-System-Ring-Central.git
cd QM-Multi-Agent-System-Ring-Central
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env    # Edit with your keys

# 3. Run
python src/main.py --folder data/audio
```

## Usage Modes

```bash
# Local audio files
python src/main.py --folder data/audio
python src/main.py --local data/audio/call1.mp3 data/audio/call2.mp3

# RingCentral API (requires RC credentials)
python src/main.py --date-from 2025-02-01 --date-to 2025-02-11
```

## Architecture

```
src/
├── agents/          # 4 pipeline agents
├── core/            # LLM abstraction (BaseLLM, ModelFactory with fallback chain)
├── processing/      # Transcript cleaning, PII redaction, token counting, chunking
├── inference/       # Response parsing, retry orchestration, caching
├── prompts/         # Prompt templates (system + user)
├── pipeline.py      # Pipeline orchestrator
├── main.py          # CLI entry point
└── utils.py         # Config, logging, env loading
```

### Key Features

- **LLM Fallback Chain** — OpenRouter (GPT-4o) → Claude Sonnet → Direct OpenAI
- **PII Redaction** — Phone, email, credit card, SSN masked before LLM calls
- **Smart Truncation** — Preserves beginning (60%) + end (40%) for greeting/closing criteria
- **Response Caching** — SHA256-keyed JSON cache avoids redundant API calls
- **Transcript Persistence** — Raw transcripts saved to `data/transcripts/`

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ELEVENLABS_API_KEY` | ✅ | ElevenLabs Scribe v2 STT |
| `OPENROUTER_API_KEY` | ✅ | OpenRouter LLM evaluation |
| `OPENAI_API_KEY` | — | Direct OpenAI fallback |
| `ANTHROPIC_API_KEY` | — | Claude fallback |
| `RC_APP_CLIENT_ID` | RC mode | RingCentral credentials |
| `RC_APP_CLIENT_SECRET` | RC mode | |
| `RC_SERVER_URL` | RC mode | |
| `RC_USER_JWT` | RC mode | |
| `WEBHOOK_URL` | — | Result notifications |

## Evaluation Criteria (24 total)

| Category | Count | Examples |
|----------|-------|---------|
| **Phone Skills** | 5 | Greeting, caller ID, hold procedure |
| **Sales Techniques** | 8 | Needs assessment, objection handling |
| **Urgency & Closing** | 3 | Creating urgency, closing attempt |
| **Soft Skills** | 8 | Active listening, empathy, professionalism |

Scoring: **YES** (100%) · **PARTIAL** (50%) · **NO** (0%) · **N/A** (excluded)

## Tests

```bash
python -m pytest tests/ -v            # 173 tests
python -m pytest tests/ -v --cov=src  # with coverage
```

| Test File | Count | Scope |
|-----------|-------|-------|
| `test_agent_01.py` | 7 | Audio file discovery |
| `test_agent_02.py` | 16 | ElevenLabs transcription (Scribe v2) |
| `test_agent_03.py` | 24 | QA evaluation + scoring |
| `test_agent_04.py` | 9 | Export (Excel/CSV/JSON/webhook) |
| `test_inference_engine.py` | 15 | Inference engine + caching |
| `test_integration.py` | 3 | End-to-end smoke test |
| `test_model_factory.py` | 12 | Model factory + fallback chain |
| `test_pipeline.py` | 13 | Pipeline orchestration |
| `test_processing.py` | 32 | Cleaner, counter, chunker, PII |
| `test_prompt_loader.py` | 10 | Prompt template loading |
| `test_response_parser.py` | 17 | Response parsing + validation |
| `test_scoring.py` | 12 | Parameterized scoring edge cases |
| **Total** | **173** | |

## Docker

```bash
# Build & run
docker build -t qm-system .
docker run --env-file .env -v ./data:/app/data qm-system --folder data/audio

# Docker Compose
docker compose up
docker compose run qm-tests   # run tests in container
```

## Output Files

Results saved to `data/evaluations/`:

| File | Content |
|------|---------|
| `QM_YYYYMMDD_HHMMSS.xlsx` | Excel with Summary + Details sheets |
| `QM_YYYYMMDD_HHMMSS_summary.csv` | One row per call |
| `QM_YYYYMMDD_HHMMSS_details.csv` | One row per criterion |
| `QM_YYYYMMDD_HHMMSS.json` | Full evaluation data |

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/SETUP.md](docs/SETUP.md) | Installation, configuration, troubleshooting |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture & module reference |

## License

Private / Internal use.
