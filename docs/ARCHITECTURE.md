# Architecture

## System Overview

The QM Multi Agent System is a 4-agent pipeline that processes call center recordings through audio retrieval, speech-to-text transcription, LLM-based quality evaluation, and report generation.

```
┌──────────────────────────────────────────────────────────────────────┐
│                           Pipeline                                  │
│                                                                      │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ Agent 1  │─▶│   Agent 2    │─▶│   Agent 3    │─▶│  Agent 4   │  │
│  │ Audio    │  │ Transcription│  │ Evaluation   │  │  Export    │  │
│  └──────────┘  └──────────────┘  └──────┬───────┘  └────────────┘  │
│                                         │                            │
│                          ┌──────────────┼──────────────┐            │
│                          ▼              ▼              ▼            │
│                    ┌──────────┐  ┌────────────┐  ┌──────────┐      │
│                    │  core/   │  │ processing/│  │inference/│      │
│                    │ LLM      │  │ Clean/PII  │  │ Parse/   │      │
│                    │ Factory  │  │ Chunk/Tok  │  │ Retry    │      │
│                    └──────────┘  └────────────┘  └──────────┘      │
└──────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
├── config/                     # Configuration files
│   ├── agents.yaml             # Pipeline & agent settings
│   ├── qa_criteria.yaml        # 24 evaluation criteria
│   ├── models.yaml             # LLM model definitions & fallback chain
│   └── logging.yaml            # Logging config (dictConfig format)
│
├── data/                       # Runtime data (gitignored)
│   ├── audio/                  # Input audio files
│   ├── transcripts/            # Persisted transcript .txt files
│   ├── evaluations/            # Output: Excel/CSV/JSON reports
│   └── cache/                  # LLM response cache (SHA256 keyed)
│
├── src/                        # Application source code
│   ├── main.py                 # CLI entry point
│   ├── pipeline.py             # Pipeline orchestrator
│   ├── utils.py                # Shared utilities (config, logging, env)
│   │
│   ├── agents/                 # 4 pipeline agents
│   │   ├── agent_01_audio.py          # AudioFileFinder + CRMAgent
│   │   ├── agent_02_transcription.py  # ElevenLabsSTTAgent
│   │   ├── agent_03_evaluation.py     # QualityManagementAgent
│   │   └── agent_04_export.py         # IntegrationAgent
│   │
│   ├── core/                   # LLM abstraction layer
│   │   ├── base_llm.py         # BaseLLM ABC + LLMResponse dataclass
│   │   ├── openai_client.py    # OpenAI SDK-compatible client
│   │   └── model_factory.py    # Factory with automatic fallback routing
│   │
│   ├── processing/             # Transcript processing pipeline
│   │   ├── transcript_cleaner.py  # Speaker label normalization, filler removal
│   │   ├── token_counter.py       # tiktoken-based token counting + cost estimation
│   │   ├── chunker.py             # Smart truncation (60% start, 40% end)
│   │   └── pii_redactor.py        # Phone/email/CC/SSN masking
│   │
│   ├── inference/              # LLM inference orchestration
│   │   ├── response_parser.py  # JSON extraction + criteria validation
│   │   └── inference_engine.py # Prompt building, retry, caching
│   │
│   └── prompts/                # Prompt templates
│       ├── templates.py        # PromptLoader with SafeDict rendering
│       ├── qa_system.txt       # System prompt template
│       └── qa_user.txt         # User prompt template
│
├── tests/                      # Test suite (pytest)
│   ├── conftest.py             # Shared fixtures
│   ├── fixtures/               # Test data
│   ├── test_agent_01.py        # Agent 1 tests (7)
│   ├── test_agent_02.py        # Agent 2 tests (9)
│   ├── test_agent_03.py        # Agent 3 tests (19)
│   ├── test_agent_04.py        # Agent 4 tests (9)
│   ├── test_processing.py      # Processing pipeline tests (22)
│   └── test_scoring.py         # Parameterized scoring tests (12)
│
├── docs/                       # Documentation
│   ├── README.md               # Project overview
│   ├── SETUP.md                # Installation & configuration guide
│   └── ARCHITECTURE.md         # This file
│
├── scripts/                    # Dev/CI scripts
│   ├── setup_env.sh            # Environment setup
│   └── run_tests.sh            # Test runner
│
├── Dockerfile                  # Container build
├── docker-compose.yml          # Container orchestration
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
└── .gitignore                  # Git exclusions
```

## Module Reference

### src/core/ — LLM Abstraction

**BaseLLM** (`base_llm.py`)
Abstract base class defining the provider contract:
- `chat(messages, temperature, max_tokens)` → `LLMResponse`
- `is_available()` → `bool`
- `calculate_cost(input_tokens, output_tokens)` → `float`

**LLMResponse** (`base_llm.py`)
Dataclass returned by all providers:
- `text`, `input_tokens`, `output_tokens`, `cost_usd`, `model`, `provider`, `elapsed_seconds`

**OpenAIClient** (`openai_client.py`)
OpenAI SDK-compatible client. Works for both OpenRouter and direct OpenAI by configuring `base_url`:
- OpenRouter: `https://openrouter.ai/api/v1`
- OpenAI direct: `https://api.openai.com/v1`

**ModelFactory** (`model_factory.py`)
Creates LLM clients from `config/models.yaml`. Provides:
- `primary` — first available provider
- `chat_with_fallback(messages)` — tries each provider in order
- Skips providers with missing API keys automatically

### src/processing/ — Transcript Pipeline

**TranscriptCleaner** — normalizes ElevenLabs `Speaker 0/1:` labels to `Agent:/Client:` based on call direction. Removes filler words (um, uh, etc.).

**TokenCounter** — estimates token counts using `tiktoken` (with word-based fallback). Used for cost estimation and truncation decisions.

**TranscriptChunker** — truncates long transcripts while preserving the beginning (60% — greeting criteria) and end (40% — closing criteria).

**PIIRedactor** — masks phone numbers, emails, credit card numbers, and SSNs before sending transcripts to external LLM APIs. Order: SSN → CC → email → phone (most specific first).

### src/inference/ — LLM Orchestration

**ResponseParser** — extracts JSON from LLM responses (handles markdown code blocks), validates that all expected criteria keys are present and scores are valid (YES/PARTIAL/NO/N/A).

**InferenceEngine** — full evaluation cycle:
1. Build prompts from templates
2. Call LLM via ModelFactory (with fallback)
3. Parse and validate response
4. Retry on validation failure
5. Cache results (SHA256 key from transcript + call_type + criteria_count)

### src/agents/ — Pipeline Agents

**Agent 1 — AudioFileFinder / CRMAgent**
- `AudioFileFinder`: scans local folder for audio files (.mp3, .wav, .m4a)
- `CRMAgent`: searches and downloads call recordings from CRM API

**Agent 2 — ElevenLabsSTTAgent**
- Transcribes audio using ElevenLabs Scribe v2 with speaker diarization
- Batch processing with progress tracking
- Persists transcripts to `data/transcripts/`
- Cost tracking (~$0.005/min)

**Agent 3 — QualityManagementAgent**
- Thin orchestration wrapper that delegates to core/processing/inference
- Pipeline: clean → redact PII → truncate → filter criteria → LLM eval
- Scoring: `calculate_score()` computes overall + per-category scores
- Call type detection from filename (first call vs follow-up)

**Agent 4 — IntegrationAgent**
- Exports evaluations to Excel (.xlsx), CSV (summary + details), JSON
- All export files share same timestamp (no drift)
- Optional webhook notification with 3× retry

## Data Flow

```
1. Audio files (.mp3/.wav)
   ↓
2. ElevenLabs Scribe v2 → raw transcript with word-level diarization
   ↓
3. TranscriptCleaner → Agent/Client labels, no fillers
   ↓
4. PIIRedactor → [PHONE], [EMAIL], [SSN], [CC_NUMBER] masks
   ↓
5. TranscriptChunker → truncated if >30K tokens
   ↓
6. InferenceEngine → LLM evaluation (24 criteria scored)
   ↓
7. ResponseParser → validated JSON with scores + evidence
   ↓
8. calculate_score() → overall 0-100%, per-category scores
   ↓
9. IntegrationAgent → Excel/CSV/JSON + optional webhook
```

## LLM Fallback Chain

Defined in `config/models.yaml`:

```
1. OpenRouter (GPT-4o) ──failed──▶ 2. OpenRouter (Claude Sonnet) ──failed──▶ 3. OpenAI Direct
```

Providers with missing API keys are skipped automatically.

## Scoring Formula

```
YES    = 1.0 × weight
PARTIAL = 0.5 × weight
NO     = 0.0 × weight
N/A    = excluded from total

overall_score = (total_points / total_weight) × 100
```

Category scores are computed independently for: phone_skills, sales_techniques, urgency_closing, soft_skills.

## Caching

LLM responses are cached in `data/cache/` using SHA256 hash of:
- transcript text
- call type
- criteria count

Cache files are JSON. Delete `data/cache/` to force re-evaluation.
