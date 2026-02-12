"""
Tests for Agent 2: ElevenLabs Transcription

Tests ElevenLabsSTTAgent with mocked ElevenLabs client.
"""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.agent_02_transcription import ElevenLabsSTTAgent


# --- Fixtures ---

@pytest.fixture
def mock_client():
    """Mock ElevenLabs client."""
    client = MagicMock()
    result = MagicMock()
    result.text = "  Speaker 0: Hello, how can I help you today?\nSpeaker 1: I need help with my account.  "
    client.speech_to_text.convert.return_value = result
    return client


@pytest.fixture
def agent(mock_client, tmp_path):
    return ElevenLabsSTTAgent(
        client=mock_client,
        persist_transcripts=True,
        transcripts_folder=str(tmp_path / "transcripts"),
    )


@pytest.fixture
def agent_no_persist(mock_client):
    return ElevenLabsSTTAgent(client=mock_client, persist_transcripts=False)


@pytest.fixture
def fake_audio(tmp_path):
    """Create a fake audio file."""
    f = tmp_path / "call1.mp3"
    f.write_bytes(b"fake audio data")
    return f


# --- Tests: Single Transcription ---

class TestTranscribe:

    def test_transcribe_returns_text(self, agent, fake_audio):
        """transcribe() should return stripped text."""
        result = agent.transcribe(fake_audio)
        assert "Speaker 0:" in result
        assert not result.startswith(" ")  # should be stripped

    def test_transcribe_calls_api(self, agent, fake_audio, mock_client):
        """Should call speech_to_text.convert with correct model."""
        agent.transcribe(fake_audio)
        mock_client.speech_to_text.convert.assert_called_once()
        call_kwargs = mock_client.speech_to_text.convert.call_args
        assert call_kwargs[1]["model_id"] == "scribe_v1"


# --- Tests: Transcript Persistence ---

class TestPersistence:

    def test_save_transcript(self, agent, tmp_path):
        """_save_transcript should write .txt file."""
        path = agent._save_transcript("call1.mp3", "Hello world")
        assert path is not None
        assert path.exists()
        assert path.read_text() == "Hello world"
        assert path.name == "call1.txt"

    def test_save_transcript_disabled(self, agent_no_persist):
        """When persist=False, _save_transcript returns None."""
        result = agent_no_persist._save_transcript("call1.mp3", "text")
        assert result is None

    def test_save_transcript_redacts_pii(self, agent, tmp_path):
        """TEST-24: _save_transcript must redact PII before writing to disk."""
        transcript_with_pii = (
            "Customer: My number is 555-123-4567 and "
            "my email is john@example.com. "
            "My SSN is 123-45-6789."
        )
        path = agent._save_transcript("call_pii.mp3", transcript_with_pii)
        assert path is not None
        saved_text = path.read_text(encoding="utf-8")
        # Raw PII must NOT appear in saved file
        assert "555-123-4567" not in saved_text
        assert "john@example.com" not in saved_text
        assert "123-45-6789" not in saved_text
        # Redaction tokens must be present instead
        assert "[PHONE]" in saved_text
        assert "[EMAIL]" in saved_text
        assert "[SSN]" in saved_text


# --- Tests: Batch Transcription ---

class TestBatchTranscription:

    def test_batch_returns_all_files(self, agent, tmp_path):
        """transcribe_batch should return entry for each file."""
        files = []
        for name in ["a.mp3", "b.mp3", "c.mp3"]:
            f = tmp_path / name
            f.write_bytes(b"audio")
            files.append(f)

        results = agent.transcribe_batch(files)
        assert len(results) == 3
        assert all(name in results for name in ["a.mp3", "b.mp3", "c.mp3"])

    def test_batch_success_status(self, agent, fake_audio):
        """Successful transcription has 'Success' status."""
        results = agent.transcribe_batch([fake_audio])
        assert results["call1.mp3"]["status"] == "Success"
        assert "transcript" in results["call1.mp3"]

    def test_batch_includes_cost(self, agent, fake_audio):
        """Results include cost_usd field."""
        results = agent.transcribe_batch([fake_audio])
        assert "cost_usd" in results["call1.mp3"]

    def test_batch_transcript_path_when_persisted(self, agent, fake_audio):
        """When persistence enabled, transcript_path should be set."""
        results = agent.transcribe_batch([fake_audio])
        assert "transcript_path" in results["call1.mp3"]

    def test_batch_api_error_handled(self, agent, fake_audio, mock_client):
        """API errors should be caught, not crash the batch."""
        mock_client.speech_to_text.convert.side_effect = Exception("API Error")
        results = agent.transcribe_batch([fake_audio])
        assert "Error" in results["call1.mp3"]["status"]

    def test_batch_quota_exceeded_stops(self, agent, tmp_path, mock_client):
        """Quota exceeded error should stop the batch early."""
        files = [tmp_path / f"{i}.mp3" for i in range(5)]
        for f in files:
            f.write_bytes(b"audio")

        mock_client.speech_to_text.convert.side_effect = Exception("quota_exceeded")
        results = agent.transcribe_batch(files)
        # Should stop after first file
        assert len(results) == 1
