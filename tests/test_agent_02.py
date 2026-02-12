"""
Tests for Agent 2: ElevenLabs Transcription (Scribe v2)

Tests ElevenLabsSTTAgent with mocked ElevenLabs client.
Covers diarization, timeout, retry, batch processing.
"""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.agent_02_transcription import ElevenLabsSTTAgent, _speaker_label


# --- Helpers ---

def _make_word(text, speaker_id=None, wtype="word"):
    """Build a Scribe v2 word object (mock)."""
    w = MagicMock()
    w.text = text
    w.speaker_id = speaker_id
    w.type = wtype
    return w


def _make_scribe_v2_result(words=None, text=None, language_code="en"):
    """Build a mock Scribe v2 API result."""
    result = MagicMock()
    if words is None:
        words = [
            _make_word("Hello", "speaker_0"),
            _make_word(",", "speaker_0", "punctuation"),
            _make_word(" ", "speaker_0", "spacing"),
            _make_word("how", "speaker_0"),
            _make_word(" ", "speaker_0", "spacing"),
            _make_word("can", "speaker_0"),
            _make_word(" ", "speaker_0", "spacing"),
            _make_word("I", "speaker_0"),
            _make_word(" ", "speaker_0", "spacing"),
            _make_word("help", "speaker_0"),
            _make_word("?", "speaker_0", "punctuation"),
            _make_word("I", "speaker_1"),
            _make_word(" ", "speaker_1", "spacing"),
            _make_word("need", "speaker_1"),
            _make_word(" ", "speaker_1", "spacing"),
            _make_word("help", "speaker_1"),
            _make_word(".", "speaker_1", "punctuation"),
        ]
    result.words = words
    result.text = text or "Hello, how can I help? I need help."
    result.language_code = language_code
    return result


# --- Fixtures ---

@pytest.fixture
def mock_client():
    """Mock ElevenLabs client with Scribe v2 response."""
    client = MagicMock()
    client.speech_to_text.convert.return_value = _make_scribe_v2_result()
    return client


@pytest.fixture
def agent(mock_client, tmp_path):
    return ElevenLabsSTTAgent(
        client=mock_client,
        persist_transcripts=True,
        transcripts_folder=str(tmp_path / "transcripts"),
        model_id="scribe_v2",
        diarize=True,
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

    def test_transcribe_returns_dict(self, agent, fake_audio):
        """transcribe() should return a dict with text, raw_text, speakers_detected."""
        result = agent.transcribe(fake_audio)
        assert isinstance(result, dict)
        assert "text" in result
        assert "raw_text" in result
        assert "speakers_detected" in result
        assert "diarized" in result

    def test_transcribe_calls_api_with_scribe_v2(self, agent, fake_audio, mock_client):
        """Should call speech_to_text.convert with model_id=scribe_v2."""
        agent.transcribe(fake_audio)
        mock_client.speech_to_text.convert.assert_called_once()
        call_kwargs = mock_client.speech_to_text.convert.call_args[1]
        assert call_kwargs["model_id"] == "scribe_v2"

    def test_transcribe_detects_speakers(self, agent, fake_audio):
        """Diarization should detect 2 speakers."""
        result = agent.transcribe(fake_audio)
        assert result["speakers_detected"] == 2
        assert result["diarized"] is True

    def test_transcribe_diarized_text_has_labels(self, agent, fake_audio):
        """Diarized text should contain Speaker 0: / Speaker 1: labels."""
        result = agent.transcribe(fake_audio)
        assert "Speaker 0:" in result["text"]
        assert "Speaker 1:" in result["text"]


# --- Tests: Diarization ---

class TestDiarization:

    def test_build_diarized_transcript_basic(self):
        """_build_diarized_transcript should group words by speaker."""
        words = [
            {"text": "Hi", "speaker_id": "speaker_0", "type": "word"},
            {"text": " ", "speaker_id": "speaker_0", "type": "spacing"},
            {"text": "there", "speaker_id": "speaker_0", "type": "word"},
            {"text": "Hello", "speaker_id": "speaker_1", "type": "word"},
        ]
        text, speakers = ElevenLabsSTTAgent._build_diarized_transcript(words)
        assert "Speaker 0:" in text
        assert "Speaker 1:" in text
        assert len(speakers) == 2

    def test_build_diarized_transcript_empty(self):
        """Empty words list should return empty string."""
        text, speakers = ElevenLabsSTTAgent._build_diarized_transcript([])
        assert text == ""
        assert len(speakers) == 0

    def test_speaker_label_mapping(self):
        """_speaker_label should map speaker_0 → 'Speaker 0'."""
        assert _speaker_label("speaker_0") == "Speaker 0"
        assert _speaker_label("speaker_1") == "Speaker 1"
        assert _speaker_label(None) == "Speaker"

    def test_diarize_disabled(self, mock_client, tmp_path, fake_audio):
        """When diarize=False, should return raw text without speaker labels."""
        agent = ElevenLabsSTTAgent(
            client=mock_client,
            persist_transcripts=False,
            diarize=False,
        )
        result = agent.transcribe(fake_audio)
        assert result["diarized"] is False
        assert result["speakers_detected"] == 0


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
        assert "555-123-4567" not in saved_text
        assert "john@example.com" not in saved_text
        assert "123-45-6789" not in saved_text
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

    def test_batch_includes_v2_fields(self, agent, fake_audio):
        """Batch results include Scribe v2 fields."""
        results = agent.transcribe_batch([fake_audio])
        entry = results["call1.mp3"]
        assert "raw_text" in entry
        assert "speakers_detected" in entry
        assert "diarized" in entry
        assert "cost_usd" in entry

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
        assert len(results) == 1


# --- Tests: Timeout (CRIT-3) ---

class TestTimeout:

    def test_timeout_raises(self, mock_client, tmp_path):
        """Should raise TimeoutError when API call exceeds timeout."""
        import time as _time

        def slow_call(**kwargs):
            _time.sleep(5)
            return _make_scribe_v2_result()

        mock_client.speech_to_text.convert.side_effect = slow_call
        agent = ElevenLabsSTTAgent(
            client=mock_client,
            persist_transcripts=False,
            timeout_seconds=1,
        )
        f = tmp_path / "slow.mp3"
        f.write_bytes(b"audio")
        with pytest.raises(TimeoutError, match="timed out"):
            agent.transcribe(f)
