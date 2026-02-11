# Call Center QA System - Minimal Version

Simple 4-agent pipeline for automated call quality assurance.

## Quick Start
1. Copy `.env.example` to `.env`
2. Fill in API credentials
3. Run: `python src/main.py --date-from 2025-02-01`

## Agents
- **Agent 1:** Download calls from RingCentral (JWT auth)
- **Agent 2:** Transcribe with ElevenLabs (diarization)
- **Agent 3:** QA evaluation via OpenRouter (24 criteria, Sonnet -> Gemini -> Llama)
- **Agent 4:** Send results to target system (webhook)

## Pipeline Flow
```
RingCentral -> ElevenLabs -> OpenRouter -> Webhook/CSV
   (audio)    (transcript)    (scores)     (delivery)
```
