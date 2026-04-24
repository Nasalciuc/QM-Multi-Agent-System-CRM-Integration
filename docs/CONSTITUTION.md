# QM Multi-Agent System — CONSTITUTION

> **Version:** 2.0.0 | **Ratified:** 2026-03-16 | **Amended:** 2026-03-16
> **Repository:** github.com/Nasalciuc/QM-Multi-Agent-System-Ring-Central
> **Companion:** See `docs/ARCHITECTURE.md` for technical contracts (data flow, caching, scoring, module reference).

---

## §1. IDENTITY

Automated 4-agent CLI pipeline that evaluates BBC Sky Data call center recordings against 48 quality criteria using LLM analysis. Input: audio files or CRM API. Output: scored evaluations in Excel/CSV/JSON. Each sales call represents a €2,000–€15,000 business-class booking.

---

## §2. STACK & BOUNDARIES

| Layer | Locked To | Notes |
|-------|-----------|-------|
| Language | Python 3.11+ | Synchronous pipeline, thread pool for I/O |
| STT | ElevenLabs Scribe v2 | Speaker diarization, word timestamps |
| Primary LLM | Mistral EU (Paris) | `mistral-large-latest`, GDPR compliant |
| Fallback LLM | OpenAI Direct (GPT-4o) | US-based, DPA signed, fallback only |
| Audio | FFmpeg + pydub | Duration detection; future: pre-STT chunking |
| Tests | pytest ≥75% coverage | ruff (lint) + mypy (types) enforced in CI |
| Container | Docker multi-stage | test → production, non-root user |
| CI/CD | GitHub Actions | lint + typecheck + test + Docker build |

**FORBIDDEN — reject any code that introduces:**
LangChain, LlamaIndex, CrewAI, or any orchestration framework · Redis, Celery, or message queues · pandas outside Agent 4 export · asyncio in pipeline code · web server or REST API (this is a batch CLI tool)

---

## §3. ARTICLES

Every Article below is **NON-NEGOTIABLE**. Each includes a Gate Check that must pass before implementation proceeds. Violations require a documented exception approved by the architect.

### Article I — Pipeline Purity
Agents execute in strict sequence: Audio → STT → Evaluation → Export. Agents communicate through return values only — no shared state, no global variables, no side-channel writes. Adding a new agent or reordering requires a constitutional amendment.
> **Gate:** ☐ New code respects the sequential Agent 1→2→3→4 flow without bypass?

### Article II — Layer Isolation
Import hierarchy is one-directional and absolute:

| Module | May import from (within `src/`) |
|--------|-------------------------------|
| `agents/` | `core/`, `processing/`, `inference/`, `config_loader` |
| `inference/` | `core/`, `processing/` |
| `processing/` | own sub-modules only (no `core/`, `agents/`, `inference/`) |
| `core/` | own sub-modules only (no `processing/`, `agents/`, `inference/`) |
| `prompts/` | nothing in `src/` |

Circular dependencies or reverse imports = immediate rejection.
> **Gate:** ☐ Zero reverse imports detected? *(enforced: `test_no_circular_imports.py`)*

### Article III — Config Over Code
All values that may change between environments or runs belong in `config/*.yaml` or `.env`. Zero hardcoded URLs, API keys, model names, thresholds, or magic numbers in Python source.
> **Gate:** ☐ `grep -rn` finds no hardcoded API URLs, keys, or model strings in `src/`? *(enforced: `test_no_hardcoded_values.py`)*

### Article IV — Test-First Discipline
Tests are written before or alongside implementation, never deferred. CI pipeline (ruff + mypy + pytest) must pass. Coverage must not drop below 75%. No merge to main with failing CI.
> **Gate:** ☐ All tests green? Coverage ≥75%? *(enforced: CI/CD pipeline)*

### Article V — EU-First Data Residency
The primary LLM provider must process data within the EU. `config/models.yaml` primary section must point to an EU-based endpoint. Non-EU providers are permitted only as fallback, with a signed Data Processing Agreement.
> **Gate:** ☐ `models.yaml` primary provider is EU-based? *(enforced: `test_eu_primary.py`)*

### Article VI — Simplicity Over Cleverness
No new abstraction layer, wrapper class, or design pattern without documented justification. Use framework/library features directly. Start simple — add complexity only when proven necessary by a failing test or measured bottleneck.
> **Gate:** ☐ No unjustified new abstraction introduced?

### Article VII — Root Cause Elimination
When facing a complex failure mode, first question whether the problem itself can be removed. Do not enumerate edge cases when you can eliminate the category of error. Do not patch downstream symptoms when the upstream cause is fixable.
> **Gate:** ☐ Solution addresses root cause, not symptoms?

### Article VIII — Evaluation Integrity
LLM evaluation output must be validated: only YES/PARTIAL/NO/N/A scores accepted, every score must include transcript evidence, all expected criteria must be present in response. Before any change to evaluation model or prompt template, a benchmark against the golden dataset (≥20 manually-evaluated calls) must confirm no regression. Prompt template hashes are part of cache keys — changing a prompt invalidates cache automatically.
> **Gate:** ☐ If model or prompt changed — golden dataset benchmark run and passed?

---

## §4. PHASE -1 GATE CHECKLIST

Copy this checklist into every spec's `plan.md` before implementation begins. All gates must be checked or an exception documented.

```
## Pre-Implementation Gates
- [ ] Art. I   Pipeline Purity — sequential flow preserved?
- [ ] Art. II  Layer Isolation — no reverse imports? (test_no_circular_imports.py)
- [ ] Art. III Config Over Code — no hardcoded values? (test_no_hardcoded_values.py)
- [ ] Art. IV  Test-First — tests written? CI green? Coverage ≥75%?
- [ ] Art. V   EU-First — primary provider is EU? (test_eu_primary.py)
- [ ] Art. VI  Simplicity — no unjustified abstractions?
- [ ] Art. VII Root Cause — solving cause, not symptom?
- [ ] Art. VIII Eval Integrity — golden benchmark passed (if model/prompt changed)?
```

---

## §5. KNOWN DECISIONS

Decisions below are settled. Do not re-debate unless new evidence invalidates the rationale.

| Decision | Rationale | Date |
|----------|-----------|------|
| Pre-STT audio chunking via FFmpeg silence detection | Dual-provider STT reconciliation fails silently on long audio — both miss same section, reconciler reports OK | 2026-03 |
| Zero PII redaction anywhere in pipeline | All LLM providers on signed EU DPAs; US-based company serving US/CA/AU clients. Transcripts stored and exported complete and unredacted. | 2026-04 |
| Mistral EU Paris as primary LLM | EU data residency, direct API, no intermediary, GDPR compliant | 2026-03 |
| No orchestration frameworks | Hand-rolled pipeline gives full control over fallback, caching, cost tracking | 2026-01 |
| Synchronous pipeline with thread pool | Pipeline is batch-oriented; async adds complexity without throughput benefit for sequential Agent 1→2→3→4 flow | 2026-03 |
| Batch CLI tool, not a web service | QM evaluation runs periodically, not on-demand; no API needed | 2026-01 |

---

## §6. ANTI-PATTERNS

These are **prohibited** regardless of context:

- **DO NOT** add PII redaction before LLM evaluation (Debate VI, 2026-03)
- **DO NOT** cache `raw_response` — whitelist only via `_CACHE_SAFE_KEYS`
- **DO NOT** import agents from core, or inference from agents (Art. II)
- **DO NOT** hardcode model names, URLs, or thresholds in Python (Art. III)
- **DO NOT** assume STT dual-provider reconciliation catches all errors — chunk before STT
- **DO NOT** enumerate PII patterns endlessly — eliminate the problem at the root (Art. VII)
- **DO NOT** deploy with failing CI or coverage below 75% (Art. IV)
- **DO NOT** change evaluation model/prompt without golden dataset benchmark (Art. VIII)

---

## §7. GOVERNANCE

- **Amendment process:** Any article change requires documented rationale, architect review, and backward compatibility assessment. Update `Last Amended` date and increment version.
- **Versioning:** MAJOR = article removed or redefined. MINOR = new article or material expansion. PATCH = clarification or typo fix.
- **Sync obligation:** After any constitution change, verify `docs/ARCHITECTURE.md` and `specs/` templates remain consistent.
- **Supremacy:** This constitution supersedes all other project documentation where they conflict.
