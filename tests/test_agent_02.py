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
        enable_stt_cache=False,
    )


@pytest.fixture
def agent_no_persist(mock_client):
    return ElevenLabsSTTAgent(client=mock_client, persist_transcripts=False,
                              enable_stt_cache=False)


@pytest.fixture
def fake_audio(tmp_path):
    """Create a fake audio file with MP3 magic number."""
    f = tmp_path / "call1.mp3"
    # ID3 header (MP3 with ID3 tag) followed by dummy data
    f.write_bytes(b"ID3" + b"\x00" * 100)
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
        text, speakers, merged, parsed_words = ElevenLabsSTTAgent._build_diarized_transcript(words)
        assert "Speaker 0:" in text
        assert "Speaker 1:" in text
        assert len(speakers) == 2
        assert merged is False

    def test_build_diarized_transcript_empty(self):
        """Empty words list should return empty string."""
        text, speakers, merged, parsed_words = ElevenLabsSTTAgent._build_diarized_transcript([])
        assert text == ""
        assert len(speakers) == 0
        assert merged is False

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

    def test_excess_speakers_merged_to_two(self):
        """SPEAKER-02: >2 speaker_ids should be merged into top-2 by word count."""
        # speaker_0: 4 words (major), speaker_1: 3 words (major)
        # speaker_2: 1 word (minor) → merge into nearest major
        # speaker_3: 1 word (minor) → merge into nearest major
        words = [
            {"text": "Hi", "speaker_id": "speaker_0", "type": "word"},
            {"text": " ", "speaker_id": "speaker_0", "type": "spacing"},
            {"text": "how", "speaker_id": "speaker_0", "type": "word"},
            {"text": " ", "speaker_id": "speaker_0", "type": "spacing"},
            {"text": "are", "speaker_id": "speaker_0", "type": "word"},
            {"text": " ", "speaker_id": "speaker_0", "type": "spacing"},
            {"text": "you", "speaker_id": "speaker_0", "type": "word"},
            {"text": "Good", "speaker_id": "speaker_1", "type": "word"},
            {"text": " ", "speaker_id": "speaker_1", "type": "spacing"},
            {"text": "thanks", "speaker_id": "speaker_1", "type": "word"},
            {"text": " ", "speaker_id": "speaker_1", "type": "spacing"},
            {"text": "bye", "speaker_id": "speaker_1", "type": "word"},
            {"text": "Wait", "speaker_id": "speaker_2", "type": "word"},
            {"text": "Also", "speaker_id": "speaker_3", "type": "word"},
        ]
        text, speakers, merged, parsed_words = ElevenLabsSTTAgent._build_diarized_transcript(words)
        assert merged is True
        # Only top-2 effective speaker IDs remain
        assert len(speakers) == 2
        assert "speaker_0" in speakers
        assert "speaker_1" in speakers
        # All lines should only reference Speaker 0 or Speaker 1
        for line in text.strip().split("\n"):
            assert line.startswith("Speaker 0:") or line.startswith("Speaker 1:"), f"Unexpected: {line}"

    def test_two_speakers_not_merged(self):
        """SPEAKER-02: Exactly 2 speakers should not trigger merging."""
        words = [
            {"text": "Hello", "speaker_id": "speaker_0", "type": "word"},
            {"text": "Hi", "speaker_id": "speaker_1", "type": "word"},
        ]
        text, speakers, merged, parsed_words = ElevenLabsSTTAgent._build_diarized_transcript(words)
        assert merged is False
        assert len(speakers) == 2


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
            f.write_bytes(b"ID3" + b"\x00" * 50)
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
            f.write_bytes(b"ID3" + b"\x00" * 50)

        mock_client.speech_to_text.convert.side_effect = Exception("quota_exceeded")
        results = agent.transcribe_batch(files)
        assert len(results) == 1

    def test_batch_rate_limiting(self, agent, tmp_path, mock_client):
        """CRIT-NEW-2: Batch should respect delay between API calls."""
        import time as _time

        files = [tmp_path / f"{i}.mp3" for i in range(3)]
        for f in files:
            f.write_bytes(b"ID3" + b"\x00" * 50)

        start = _time.time()
        agent.transcribe_batch(files, delay_between_calls=0.1)
        elapsed = _time.time() - start
        # With 3 files and 0.1s delay between calls, expect at least 0.2s total delay
        assert elapsed >= 0.15, f"Batch was too fast ({elapsed:.2f}s), rate limit may not be working"


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
        f.write_bytes(b"ID3" + b"\x00" * 50)
        with pytest.raises(TimeoutError, match="timed out"):
            agent.transcribe(f)


# --- Tests: Audio File Validation (CRIT-NEW-4) ---

class TestAudioValidation:

    def test_validate_missing_file(self, agent, tmp_path):
        """Should raise FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            agent.validate_audio_file(tmp_path / "nonexistent.mp3")

    def test_validate_empty_file(self, agent, tmp_path):
        """Should raise ValueError for 0-byte file."""
        empty = tmp_path / "empty.mp3"
        empty.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            agent.validate_audio_file(empty)

    def test_validate_oversized_file(self, agent, tmp_path):
        """Should raise ValueError for files exceeding size limit."""
        from agents.agent_02_transcription import _MAX_AUDIO_FILE_BYTES
        # Create a sparse-ish file that reports > limit
        big = tmp_path / "big.mp3"
        big.write_bytes(b"ID3" + b"\x00" * (_MAX_AUDIO_FILE_BYTES + 1))
        with pytest.raises(ValueError, match="too large"):
            agent.validate_audio_file(big)

    def test_validate_valid_mp3(self, agent, tmp_path):
        """Valid MP3 file (ID3 header) should pass validation."""
        mp3 = tmp_path / "valid.mp3"
        mp3.write_bytes(b"ID3\x04\x00\x00" + b"\x00" * 100)
        agent.validate_audio_file(mp3)  # Should not raise

    def test_validate_valid_wav(self, agent, tmp_path):
        """Valid WAV file (RIFF header) should pass validation."""
        wav = tmp_path / "valid.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)
        agent.validate_audio_file(wav)  # Should not raise

    def test_validate_unrecognised_format_warns(self, agent, tmp_path, caplog):
        """Unrecognised magic number should log a warning but not raise."""
        weird = tmp_path / "data.bin"
        weird.write_bytes(b"ZZZZ" + b"\x00" * 100)
        import logging
        with caplog.at_level(logging.WARNING):
            agent.validate_audio_file(weird)  # Should not raise
        assert "Unrecognised audio header" in caplog.text


# ═══════════════════════════════════════════════════════════════════
# MED-01: ElevenLabs Retry Verification
# ═══════════════════════════════════════════════════════════════════

class TestElevenLabsRetry:
    """Verify the existing retry/no-retry logic in transcribe()."""

    @pytest.fixture
    def agent(self, tmp_path):
        mock_client = MagicMock()
        a = ElevenLabsSTTAgent(
            client=mock_client,
            persist_transcripts=False,
            enable_stt_cache=False,
        )
        a.MAX_RETRIES = 2
        # Patch the retry backoff to avoid sleeping in tests
        a.RETRY_BACKOFF_BASE = 2  # Standard backoff value for patching
        return a

    @patch("agents.agent_02_transcription.time.sleep")
    def test_transcribe_retries_on_429(self, mock_sleep, agent, tmp_path):
        """429 (rate limit) should trigger retry, not immediate raise."""
        audio = tmp_path / "call.mp3"
        audio.write_bytes(b"ID3\x04\x00\x00" + b"\x00" * 100)

        rate_limit_err = Exception("rate_limit_exceeded: too many requests")
        good_result = _make_scribe_v2_result(text="Hello world")

        agent._call_api_with_timeout = MagicMock(
            side_effect=[rate_limit_err, good_result]
        )
        result = agent.transcribe(audio)
        assert agent._call_api_with_timeout.call_count == 2
        assert result["text"]  # got a valid transcript
        assert mock_sleep.called

    def test_transcribe_no_retry_on_auth_error(self, agent, tmp_path):
        """Auth errors (unauthorized) should raise immediately, no retry."""
        audio = tmp_path / "call.mp3"
        audio.write_bytes(b"ID3\x04\x00\x00" + b"\x00" * 100)

        auth_err = Exception("unauthorized: invalid API key")
        agent._call_api_with_timeout = MagicMock(side_effect=auth_err)

        with pytest.raises(Exception, match="unauthorized"):
            agent.transcribe(audio)
        # Should fail on first attempt without retrying
        assert agent._call_api_with_timeout.call_count == 1


# ═══════════════════════════════════════════════════════════════════
# COST-02: Skip short audio
# ═══════════════════════════════════════════════════════════════════

class TestSkipShortAudio:
    """COST-02: Audio shorter than min_audio_duration_sec should be skipped."""

    def test_short_audio_skipped(self, mock_client, tmp_path):
        """Audio < 5 seconds should return empty result with skipped_short=True."""
        agent = ElevenLabsSTTAgent(
            client=mock_client,
            persist_transcripts=False,
            enable_stt_cache=False,
            min_audio_duration_sec=5,
        )
        # Create a very short WAV file (~0.1 seconds of silence)
        from pydub import AudioSegment
        short_audio = AudioSegment.silent(duration=100)  # 100ms
        audio_path = tmp_path / "short.wav"
        short_audio.export(str(audio_path), format="wav")

        result = agent.transcribe(audio_path)
        assert result["skipped_short"] is True
        assert result["text"] == ""
        assert result["speakers_detected"] == 0
        # API should NOT have been called
        mock_client.speech_to_text.convert.assert_not_called()

    def test_normal_audio_not_skipped(self, mock_client, tmp_path):
        """Audio >= 5 seconds should NOT be skipped."""
        agent = ElevenLabsSTTAgent(
            client=mock_client,
            persist_transcripts=False,
            enable_stt_cache=False,
            min_audio_duration_sec=5,
        )
        from pydub import AudioSegment
        normal_audio = AudioSegment.silent(duration=6000)  # 6 seconds
        audio_path = tmp_path / "normal.wav"
        normal_audio.export(str(audio_path), format="wav")

        result = agent.transcribe(audio_path)
        assert result.get("skipped_short") is not True
        mock_client.speech_to_text.convert.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
# COST-01: Audio pre-processing
# ═══════════════════════════════════════════════════════════════════

class TestPreprocessAudio:
    """COST-01: Pre-processing trims silence and downsamples before API call."""

    def test_preprocess_trims_silence(self, mock_client, tmp_path):
        """Audio with large leading/trailing silence should be trimmed."""
        from pydub import AudioSegment
        from pydub.generators import Sine

        # Build: 5s silence + 2s tone + 5s silence = 12s total
        silence = AudioSegment.silent(duration=5000)
        tone = Sine(440).to_audio_segment(duration=2000).apply_gain(-20)
        audio = silence + tone + silence
        audio_path = tmp_path / "padded.wav"
        audio.export(str(audio_path), format="wav")

        agent = ElevenLabsSTTAgent(
            client=mock_client,
            persist_transcripts=False,
            enable_stt_cache=False,
            preprocess_audio=True,
        )
        preprocessed, stats = agent._preprocess_audio(audio_path)
        assert preprocessed is not None
        assert stats is not None
        assert stats["savings_pct"] > 10
        assert stats["action"] == "preprocessed"
        assert stats["processed_duration_ms"] < stats["original_duration_ms"]
        # Cleanup
        if preprocessed.exists():
            preprocessed.unlink()

    def test_preprocess_no_change_when_savings_small(self, mock_client, tmp_path):
        """Audio with negligible silence should not produce a temp file."""
        from pydub import AudioSegment
        from pydub.generators import Sine

        # Pure tone, no silence to trim
        tone = Sine(440).to_audio_segment(duration=6000).apply_gain(-20)
        audio_path = tmp_path / "clean.wav"
        tone.export(str(audio_path), format="wav")

        agent = ElevenLabsSTTAgent(
            client=mock_client,
            persist_transcripts=False,
            enable_stt_cache=False,
            preprocess_audio=True,
        )
        preprocessed, stats = agent._preprocess_audio(audio_path)
        assert preprocessed is None
        assert stats is not None
        assert stats["action"] == "no_change"

    def test_preprocess_all_silence(self, mock_client, tmp_path):
        """Audio that is entirely silence returns action=all_silence."""
        from pydub import AudioSegment

        silent = AudioSegment.silent(duration=3000)
        audio_path = tmp_path / "silent.wav"
        silent.export(str(audio_path), format="wav")

        agent = ElevenLabsSTTAgent(
            client=mock_client,
            persist_transcripts=False,
            enable_stt_cache=False,
            preprocess_audio=True,
        )
        preprocessed, stats = agent._preprocess_audio(audio_path)
        assert preprocessed is None
        assert stats is not None
        assert stats["action"] == "all_silence"
        assert stats["savings_pct"] == 100.0

    def test_preprocess_preserves_internal_gaps(self, mock_client, tmp_path):
        """SILENCE-FIX: Internal silence gaps must NOT be compressed."""
        from pydub import AudioSegment
        from pydub.generators import Sine

        # 2s tone + 10s silence + 2s tone = 14s
        tone = Sine(440).to_audio_segment(duration=2000).apply_gain(-20)
        gap = AudioSegment.silent(duration=10000)
        audio = tone + gap + tone
        audio_path = tmp_path / "internal_gap.wav"
        audio.export(str(audio_path), format="wav")

        agent = ElevenLabsSTTAgent(
            client=mock_client,
            persist_transcripts=False,
            enable_stt_cache=False,
            preprocess_audio=True,
        )
        preprocessed, stats = agent._preprocess_audio(audio_path)
        assert stats is not None
        # Internal gap should be preserved — only edge trim + downsample
        # Processed duration should be close to original (14s) minus minimal edge trim
        if preprocessed is not None:
            assert stats["processed_duration_ms"] > 12000  # Gap preserved
        else:
            # No preprocessing needed (savings < 5%) — that's fine too
            assert stats["action"] == "no_change"

    def test_preprocess_still_trims_edges(self, mock_client, tmp_path):
        """Edge silence (before Hello / after Goodbye) should still be trimmed."""
        from pydub import AudioSegment
        from pydub.generators import Sine

        # 10s silence + 2s tone + 10s silence = 22s
        silence = AudioSegment.silent(duration=10000)
        tone = Sine(440).to_audio_segment(duration=2000).apply_gain(-20)
        audio = silence + tone + silence
        audio_path = tmp_path / "edge_padded.wav"
        audio.export(str(audio_path), format="wav")

        agent = ElevenLabsSTTAgent(
            client=mock_client,
            persist_transcripts=False,
            enable_stt_cache=False,
            preprocess_audio=True,
        )
        preprocessed, stats = agent._preprocess_audio(audio_path)
        assert preprocessed is not None
        assert stats is not None
        assert stats["action"] == "preprocessed"
        # Should trim ~20s of edge silence, leaving ~3s (2s tone + 1s padding)
        assert stats["processed_duration_ms"] < 5000
        assert stats["savings_pct"] > 50
        if preprocessed.exists():
            preprocessed.unlink()

    def test_preprocess_long_audio_preserves_holds(self, mock_client, tmp_path):
        """A long call with hold periods should preserve ALL internal silence.

        Simulates the 1.5-hour call that exposed the COST-01/REAL-12 conflict:
        tone (agent talking) + 90s silence (hold) + tone (agent returns).
        The 90s hold MUST be preserved for REAL-12 to generate [1:30 silence].
        """
        from pydub import AudioSegment
        from pydub.generators import Sine

        tone = Sine(440).to_audio_segment(duration=3000).apply_gain(-20)
        hold = AudioSegment.silent(duration=90000)  # 90 seconds hold
        audio = tone + hold + tone  # 96s total
        audio_path = tmp_path / "long_with_hold.wav"
        audio.export(str(audio_path), format="wav")

        agent = ElevenLabsSTTAgent(
            client=mock_client,
            persist_transcripts=False,
            enable_stt_cache=False,
            preprocess_audio=True,
        )
        preprocessed, stats = agent._preprocess_audio(audio_path)

        # The 90s hold is INTERNAL — must be preserved
        # No edge silence to trim, so no change expected
        if preprocessed is None:
            assert stats is not None
            assert stats["action"] == "no_change"
        else:
            # Even if some small edge trim, hold must remain
            assert stats is not None
            assert stats["processed_duration_ms"] > 90000

        if preprocessed and preprocessed.exists():
            preprocessed.unlink()

    def test_preprocess_disabled(self, mock_client, tmp_path):
        """When preprocess_audio=False, transcribe() should not call _preprocess_audio."""
        from pydub import AudioSegment
        from pydub.generators import Sine

        tone = Sine(440).to_audio_segment(duration=6000).apply_gain(-20)
        audio_path = tmp_path / "normal.wav"
        tone.export(str(audio_path), format="wav")

        agent = ElevenLabsSTTAgent(
            client=mock_client,
            persist_transcripts=False,
            enable_stt_cache=False,
            preprocess_audio=False,
        )
        result = agent.transcribe(audio_path)
        assert result.get("preprocess") is None
        mock_client.speech_to_text.convert.assert_called_once()

    def test_preprocess_cleanup_on_success(self, mock_client, tmp_path):
        """Preprocessed temp file should be cleaned up after transcription."""
        from pydub import AudioSegment

        # 3s silence + 2s silence + 3s silence (all silent but let's use tone)
        from pydub.generators import Sine
        silence = AudioSegment.silent(duration=5000)
        tone = Sine(440).to_audio_segment(duration=2000).apply_gain(-20)
        audio = silence + tone + silence
        audio_path = tmp_path / "cleanup_test.wav"
        audio.export(str(audio_path), format="wav")

        agent = ElevenLabsSTTAgent(
            client=mock_client,
            persist_transcripts=False,
            enable_stt_cache=False,
            preprocess_audio=True,
        )
        result = agent.transcribe(audio_path)
        # The temp .processed.mp3 file should have been cleaned up
        processed_path = tmp_path / ".cleanup_test.processed.mp3"
        assert not processed_path.exists()
