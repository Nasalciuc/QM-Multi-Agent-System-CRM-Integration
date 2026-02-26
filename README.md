# 🎯 QM Multi Agent System — Call Center Quality Assurance

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-455%20passed-brightgreen.svg)](#tests)
[![License](https://img.shields.io/badge/License-Private-lightgrey.svg)](#)

Automated **4-agent pipeline** that evaluates call center recordings against **48 quality criteria** using LLM-powered analysis.

---

## Overview

| Agent | Purpose | Technology |
|-------|---------|------------|
| **Agent 1** — Audio | Download / find call recordings | CRM API or local files |
| **Agent 2** — Transcription | Speech-to-text with diarization | ElevenLabs Scribe v2 |
| **Agent 3** — Evaluation | Score transcript against 48 QA criteria | OpenRouter / OpenAI GPT-4o |
| **Agent 4** — Export | Generate Excel, CSV, JSON reports | pandas + openpyxl |

### Pipeline Flow

```
┌─────────────┐    ┌─────────────────┐    ┌───────────────┐    ┌──────────┐
│  Agent 1    │───▶│    Agent 2      │───▶│   Agent 3     │───▶│ Agent 4  │
│  Audio      │    │  Transcription  │    │  Evaluation   │    │ Export   │
│  Retrieval  │    │  (ElevenLabs)   │    │  (LLM + 48   │    │ Excel/   │
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

# CRM API (default — requires CRM_AI_TOKEN)
python src/main.py --date-from 2026-02-01 --date-to 2026-02-10

# CRM API with agent filter
python src/main.py --date-from 2026-02-01 --date-to 2026-02-10 --agent-id 120
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
| `CRM_AI_TOKEN` | CRM mode | CRM API bearer token |
| `OPENAI_API_KEY` | — | Direct OpenAI fallback |
| `ANTHROPIC_API_KEY` | — | Claude fallback |
| `WEBHOOK_URL` | — | Result notifications |

## Evaluation Criteria (48 total)

Grouped into 10 categories, filtered by call type (28 per call):

| Category | Count | Call Type | Key Criteria |
|----------|-------|-----------|-------------|
| Opening | 5 | First Call | Greeting, permission check, advisor positioning |
| Interview | 4 | First Call | Travel needs, dream outcome, buying motive |
| Psychological Framing | 4 | First Call | Urgency, experience vs utility, scarcity |
| First Call Closing | 7 | First Call | Research time, follow-up appointment, contact info |
| Second Call Opening | 4 | Follow-up | Punctuality, recap, reconnection |
| Strategic Presentation | 5 | Follow-up | Anchor/recommendation/value builder pattern |
| Creating Certainty | 4 | Follow-up | Social proof, guarantees, trial close |
| Objection Handling | 3 | Follow-up | Advanced objection techniques |
| Commitment & Closing | 4 | Follow-up | Booking commitment, payment, confirmation |
| Communication | 8 | Both | Active listening, tone, expertise, pace |

Each criterion scored: **YES** (100%), **PARTIAL** (50%), **NO** (0%), **N/A** (excluded).
First Call: 20 specific + 8 communication = 28 criteria.
Follow-up Call: 20 specific + 8 communication = 28 criteria.

Scoring: **YES** (100%) · **PARTIAL** (50%) · **NO** (0%) · **N/A** (excluded)

## Tests

```bash
python -m pytest tests/ -v            # 455 tests
python -m pytest tests/ -v --cov=src  # with coverage
```

| Test File | Count | Scope |
|-----------|-------|-------|
| `test_agent_01.py` | 7 | Audio file discovery |
| `test_agent_01_crm.py` | 26 | CRM agent + streaming + pagination |
| `test_agent_02.py` | 25 | ElevenLabs transcription (Scribe v2) |
| `test_agent_03.py` | 31 | QA evaluation + scoring + direction |
| `test_agent_04.py` | 12 | Export (Excel/CSV/JSON/webhook/jitter) |
| `test_caching.py` | 20 | Enhanced caching + summary |
| `test_inference_engine.py` | 21 | Inference engine + cache + write failures |
| `test_integration.py` | 6 | End-to-end smoke tests |
| `test_model_factory.py` | 15 | Model factory + fallback chain |
| `test_pipeline.py` | 15 | Pipeline orchestration + shutdown |
| `test_processing.py` | 38 | Cleaner, counter, chunker, PII |
| `test_prompt_loader.py` | 10 | Prompt template loading |
| `test_response_parser.py` | 17 | Response parsing + validation |
| `test_scoring.py` | 12 | Parameterized scoring edge cases |
| `test_stt_cache.py` | 29 | STT cache + TTL + LRU eviction |
| **Total** | **455** | |

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
