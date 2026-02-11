# Call Center QA System — Multi Agent Pipeline

Automated 4-agent pipeline for call center quality assurance evaluation.

## Quick Start
1. Copy `.env.example` to `.env` and fill in API keys
2. `pip install -r requirements.txt`
3. `python src/main.py --folder data/audio`

## Agents
- **Agent 1:** Download / find call recordings (RingCentral SDK or local files)
- **Agent 2:** Speech-to-text with diarization (ElevenLabs Scribe v1)
- **Agent 3:** LLM evaluation against 24 QA criteria (OpenRouter GPT-4o with fallback)
- **Agent 4:** Export results to Excel, CSV, JSON (+ optional webhook)

## Pipeline Flow
```
Audio Files → ElevenLabs STT → LLM Evaluation → Excel/CSV/JSON
  (Agent 1)     (Agent 2)        (Agent 3)        (Agent 4)
```

## Tests
```bash
python -m pytest tests/ -v     # 90 tests
```

## Documentation
- [docs/README.md](docs/README.md) — Full project overview
- [docs/SETUP.md](docs/SETUP.md) — Installation & configuration guide
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System architecture & module reference
