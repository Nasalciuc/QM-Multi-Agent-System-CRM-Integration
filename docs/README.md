# QM Multi Agent System — Call Center Quality Assurance

Automated 4-agent pipeline for evaluating call center recordings against 48 quality criteria.

## Overview

| Agent | Purpose | Technology |
|-------|---------|------------|
| **Agent 1** — Audio | Download / find call recordings | CRM API or local files |
| **Agent 2** — Transcription | Speech-to-text with diarization | ElevenLabs Scribe v2 |
| **Agent 3** — Evaluation | Score transcript against 48 QA criteria | Mistral EU / OpenAI GPT-4o |
| **Agent 4** — Export | Generate Excel, CSV, JSON reports | pandas + openpyxl |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Nasalciuc/QM-Multi-Agent-System-Ring-Central.git
cd QM-Multi-Agent-System-Ring-Central

# 2. Setup environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env with your API keys

# 4. Run
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

## Pipeline Flow

```
┌─────────────┐    ┌─────────────────┐    ┌───────────────┐    ┌──────────┐
│  Agent 1    │───▶│    Agent 2      │───▶│   Agent 3     │───▶│ Agent 4  │
│  Audio      │    │  Transcription  │    │  Evaluation   │    │ Export   │
│  Retrieval  │    │  (ElevenLabs)   │    │  (LLM + 48   │    │ Excel/   │
│             │    │                 │    │   criteria)   │    │ CSV/JSON │
└─────────────┘    └─────────────────┘    └───────────────┘    └──────────┘
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ELEVENLABS_API_KEY` | Yes | ElevenLabs API key for STT |
| `MISTRAL_API_KEY` | Yes | Mistral EU API key for LLM evaluation |
| `CRM_AI_TOKEN` | CRM mode | CRM API bearer token |
| `OPENAI_API_KEY` | No | Direct OpenAI fallback |
| `WEBHOOK_URL` | No | Webhook for result notifications |

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

## Running Tests

```bash
python -m pytest tests/ -v
```

## Docker

```bash
docker build -t qm-system .
docker run --env-file .env -v ./data:/app/data qm-system --folder data/audio
```

## Documentation

- [SETUP.md](SETUP.md) — Detailed installation and configuration guide
- [ARCHITECTURE.md](ARCHITECTURE.md) — System architecture and module reference

## License

Private / Internal use.
