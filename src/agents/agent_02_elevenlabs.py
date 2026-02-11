"""
Agent 2: ElevenLabs Transcription

Purpose: Transcribe audio files with speaker diarization

Dependencies:
    - httpx (for API calls)

API Details:
    - Endpoint: POST https://api.elevenlabs.io/v1/speech-to-text
    - Auth: xi-api-key header
    - Model: scribe_v1
    - Cost: $0.005 per minute of audio
    - Max file: 1GB
    - Formats: mp3, wav, m4a, ogg, flac

Response format:
    {
        "language_code": "eng",
        "language_probability": 0.98,
        "text": "full transcript...",
        "words": [
            {"text": "Hello", "start": 0.0, "end": 0.5,
             "type": "word", "speaker_id": "speaker_0"},
            ...
        ]
    }

TODO:
    - transcribe(audio_path) -> dict  (full result with transcript + segments)
    - _call_api(audio_path) -> dict   (raw API response)
    - _parse_diarization(words) -> list[dict]  (group words into speaker segments)
    - _assign_speakers(segments, direction) -> list[dict]  (label Agent vs Customer)
    - _format_transcript(segments) -> str  (readable format with timestamps)
    - _calculate_cost(duration_minutes) -> float
"""

import os
from pathlib import Path
from typing import Optional

# import httpx   # TODO: Uncomment when implementing


class ElevenLabsAgent:
    """
    Transcribes audio with speaker diarization via ElevenLabs.

    Usage:
        agent = ElevenLabsAgent(config)
        result = agent.transcribe("data/audio/call_001.mp3")
        print(result['formatted_transcript'])
        print(f"Cost: ${result['cost_usd']:.4f}")

    Config keys (from config/agents.yaml):
        elevenlabs.model            -> "scribe_v1"
        elevenlabs.diarization      -> True
        elevenlabs.cost_per_minute  -> 0.005

    Env vars (from .env):
        ELEVENLABS_API_KEY
    """

    API_URL = "https://api.elevenlabs.io/v1/speech-to-text"

    def __init__(self, config: dict):
        """
        TODO:
            - Read ELEVENLABS_API_KEY from os.environ
            - Store config (model, cost_per_minute)
        """
        # TODO: Implement
        pass

    def transcribe(self, audio_path: str, call_direction: str = "Outbound") -> dict:
        """
        Transcribe an audio file with diarization.

        TODO:
            1. Validate file exists
            2. Call ElevenLabs API -> raw_response
            3. Parse diarization: group words by speaker
            4. Assign speaker roles (Agent vs Customer)
            5. Format readable transcript
            6. Calculate cost
            7. Save transcript to data/transcripts/{call_id}.json
            8. Return result dict:
                {
                    'full_text': str,
                    'formatted_transcript': str,
                    'segments': [
                        {'speaker': 'Agent', 'text': '...', 'start': 0.0, 'end': 5.2},
                        {'speaker': 'Customer', 'text': '...', 'start': 5.3, 'end': 12.1},
                        ...
                    ],
                    'language': str,
                    'duration_minutes': float,
                    'cost_usd': float
                }
        """
        # TODO: Implement
        pass

    def _call_api(self, audio_path: str) -> dict:
        """
        Send audio to ElevenLabs Speech-to-Text API.

        TODO:
            Request:
                POST https://api.elevenlabs.io/v1/speech-to-text
                Headers: {"xi-api-key": api_key}
                Body (multipart/form-data):
                    file: audio file
                    model_id: "scribe_v1"
                    language_code: "eng"
                    diarize: "true"
                    tag_audio_events: "true"

            - Open file in binary mode
            - Send with httpx.post()
            - Check response.status_code == 200
            - Return response.json()
            - Raise on error (log details)
        """
        # TODO: Implement
        pass

    def _parse_diarization(self, words: list[dict]) -> list[dict]:
        """
        Group consecutive words by speaker into segments.

        TODO:
            Input:  [{"text":"Hi","speaker_id":"speaker_0","start":0.0,"end":0.3}, ...]
            Output: [{"speaker_id":"speaker_0","text":"Hi there","start":0.0,"end":1.2}, ...]

            Algorithm:
                - Initialize current_speaker, current_text, current_start
                - For each word:
                    - If same speaker: append to current_text
                    - If different speaker: save segment, start new one
                - Don't forget to save the last segment
        """
        # TODO: Implement
        pass

    def _assign_speakers(
        self, segments: list[dict], call_direction: str = "Outbound"
    ) -> list[dict]:
        """
        Label speaker_ids as "Agent" or "Customer".

        TODO:
            Heuristic:
                - Outbound call: first speaker = Agent (they initiated)
                - Inbound call: first speaker = Customer (they called in)
            - Map speaker_0 and speaker_1 to roles
            - Replace speaker_id with speaker role in each segment
            - Return updated segments
        """
        # TODO: Implement
        pass

    def _format_transcript(self, segments: list[dict]) -> str:
        """
        Create readable transcript with timestamps.

        TODO:
            Format:
                [00:00:02] Agent: Hello, thank you for calling...
                [00:00:08] Customer: Hi, I'm calling about...

            - For each segment: f"[{timestamp}] {speaker}: {text}"
            - Convert seconds to HH:MM:SS
            - Join with newlines
        """
        # TODO: Implement
        pass

    def _calculate_cost(self, duration_minutes: float) -> float:
        """
        TODO: return duration_minutes * self.cost_per_minute
        """
        # TODO: Implement
        pass
