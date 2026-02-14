"""
Agent 2: ElevenLabs Speech-to-Text

Purpose: Transcribe audio files using ElevenLabs Scribe v2 with speaker diarization
API: client.speech_to_text.convert(file=audio_file, model_id="scribe_v2", ...)
Cost: ~$0.005/min (or ~280 credits/min on ElevenLabs)

Returns a dict per file with text, raw_text, speakers_detected, diarized transcript.
Persists transcripts to data/transcripts/ after successful transcription.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Dict, List, Optional
import time
import logging

from inference.stt_cache import STTCache
from utils import safe_log_filename

logger = logging.getLogger("qa_system.agents")

CREDITS_PER_MINUTE = 280
COST_PER_MINUTE = 0.005

# Lazy import to avoid circular dependency
_PIIRedactor = None


def _get_redactor():
    """Lazy-load PIIRedactor for transcript persistence redaction."""
    global _PIIRedactor
    if _PIIRedactor is None:
        from processing.pii_redactor import PIIRedactor
        _PIIRedactor = PIIRedactor
    return _PIIRedactor()


# Audio file validation constants (CRIT-NEW-4)
_AUDIO_MAGIC_NUMBERS = {
    b'ID3': 'mp3',         # MP3 with ID3 tag
    b'\xff\xfb': 'mp3',   # MP3 frame sync
    b'\xff\xf3': 'mp3',   # MP3 frame sync (alt)
    b'\xff\xf2': 'mp3',   # MP3 frame sync (alt)
    b'RIFF': 'wav',        # WAV
    b'fLaC': 'flac',       # FLAC
    b'OggS': 'ogg',        # OGG
}
_MAX_AUDIO_FILE_MB = 500
_MAX_AUDIO_FILE_BYTES = _MAX_AUDIO_FILE_MB * 1024 * 1024


class ElevenLabsSTTAgent:
    """Agent: Speech-to-Text with ElevenLabs Scribe v2 + diarization."""

    # CRIT-3: Default timeout for STT API calls (seconds)
    DEFAULT_TIMEOUT_SECONDS = 300
    # MED-12: Retry config for transient ElevenLabs failures
    MAX_RETRIES = 2
    RETRY_BACKOFF_BASE = 2

    def __init__(
        self,
        client,
        persist_transcripts: bool = True,
        transcripts_folder: str = "data/transcripts",
        timeout_seconds: int = 300,
        model_id: str = "scribe_v2",
        diarize: bool = True,
        num_speakers: Optional[int] = None,
        diarization_threshold: Optional[float] = None,
        tag_audio_events: bool = False,
        language_code: Optional[str] = None,
        keyterms: Optional[List[str]] = None,
        delay_between_calls: float = 0.5,
        stt_cache_dir: Optional[str] = "data/stt_cache",
        enable_stt_cache: bool = True,
        stt_cache_ttl_days: int = 30,
    ):
        """
        Initialize with ElevenLabs client.

        Args:
            client: ElevenLabs client instance.
            persist_transcripts: Save transcripts to disk after transcription.
            transcripts_folder: Directory to save transcript files.
            timeout_seconds: Max seconds to wait for STT API call.
            model_id: ElevenLabs model ID (default: scribe_v2).
            diarize: Enable speaker diarization.
            num_speakers: Expected number of speakers (None = auto-detect).
            diarization_threshold: Confidence threshold for speaker change.
            tag_audio_events: Tag non-speech events (laughter, music, etc.).
            language_code: ISO language code (None = auto-detect).
            keyterms: Domain-specific terms to boost recognition accuracy.
            delay_between_calls: Seconds to wait between consecutive API calls (#6).
            stt_cache_dir: Directory for STT transcript cache (None = disabled).
            enable_stt_cache: Enable/disable STT transcript caching.
            stt_cache_ttl_days: Days to keep cached transcripts before expiry.
        """
        self.client = client
        self.persist_transcripts = persist_transcripts
        self.transcripts_folder = Path(transcripts_folder)
        self.timeout_seconds = timeout_seconds
        self.model_id = model_id
        self.diarize = diarize
        self.num_speakers = num_speakers
        self.diarization_threshold = diarization_threshold
        self.tag_audio_events = tag_audio_events
        self.language_code = language_code
        self.keyterms = keyterms or []
        self.delay_between_calls = delay_between_calls

        # STT transcript cache
        self._stt_cache = STTCache(
            cache_dir=stt_cache_dir or "data/stt_cache",
            enable=enable_stt_cache and stt_cache_dir is not None,
            ttl_seconds=stt_cache_ttl_days * 24 * 3600,
        )

        if self.persist_transcripts:
            self.transcripts_folder.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"ElevenLabsSTTAgent initialized | Model: {self.model_id} | "
            f"diarize={self.diarize} | stt_cache={'ON' if self._stt_cache.enabled else 'OFF'}"
        )

    @property
    def stt_cache(self) -> STTCache:
        """Expose STT cache for stats and management."""
        return self._stt_cache

    @staticmethod
    def validate_audio_file(audio_path: Path) -> None:
        """CRIT-NEW-4: Validate audio file before sending to API.

        Checks:
          - File exists
          - File is not empty (0 bytes)
          - File does not exceed size limit
          - File starts with a recognised audio magic number

        Raises:
            FileNotFoundError: File does not exist.
            ValueError: File is empty, too large, or not a recognised audio format.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        size = audio_path.stat().st_size
        if size == 0:
            raise ValueError(f"Audio file is empty (0 bytes): {audio_path.name}")
        if size > _MAX_AUDIO_FILE_BYTES:
            raise ValueError(
                f"Audio file too large: {audio_path.name} "
                f"({size / 1024 / 1024:.1f} MB > {_MAX_AUDIO_FILE_MB} MB limit)"
            )

        # Magic-number check (read first 4 bytes)
        with open(audio_path, "rb") as f:
            header = f.read(4)
        recognised = any(header.startswith(magic) for magic in _AUDIO_MAGIC_NUMBERS)
        if not recognised:
            logger.warning(
                f"Unrecognised audio header for {audio_path.name}: {header[:4]!r}. "
                f"File may not be a valid audio file."
            )

    def transcribe(self, audio_path: Path) -> Dict:
        """Transcribe a single audio file using ElevenLabs Scribe v2.

        MED-12: Retries transient failures with exponential backoff.
        CRIT-3: Uses ThreadPoolExecutor for cross-platform timeout.
        CRIT-NEW-4: Validates audio file before API call.

        Returns:
            Dict with keys: text, raw_text, speakers_detected,
            language_code, diarized (bool).
        """
        # CRIT-NEW-4: Validate before spending API credits
        self.validate_audio_file(audio_path)

        last_error: Optional[Exception] = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                result = self._call_api_with_timeout(audio_path)

                # Build structured result from Scribe v2 response
                raw_text = result.text.strip() if hasattr(result, 'text') else ""
                words = getattr(result, 'words', None) or []
                language = getattr(result, 'language_code', None) or self.language_code

                # Build diarized transcript from word-level speaker tags
                if self.diarize and words:
                    diarized_text, speakers = self._build_diarized_transcript(words)
                else:
                    diarized_text = raw_text
                    speakers = set()

                return {
                    "text": diarized_text,
                    "raw_text": raw_text,
                    "speakers_detected": len(speakers),
                    "language_code": language,
                    "diarized": bool(self.diarize and words),
                }

            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                # Don't retry quota/auth errors
                if any(kw in error_msg for kw in (
                    "quota_exceeded", "unauthorized", "forbidden", "invalid_api_key"
                )):
                    raise
                if attempt < self.MAX_RETRIES:
                    # #13: Respect Retry-After header if available
                    retry_after = getattr(e, 'retry_after', None)
                    if retry_after is None and hasattr(e, 'response'):
                        resp = getattr(e, 'response', None)
                        if resp is not None and hasattr(resp, 'headers'):
                            retry_after_str = resp.headers.get('Retry-After')
                            if retry_after_str:
                                try:
                                    retry_after = float(retry_after_str)
                                except (ValueError, TypeError):
                                    pass
                    wait = retry_after if retry_after else self.RETRY_BACKOFF_BASE ** (attempt + 1)
                    logger.warning(
                        f"Transient STT error (attempt {attempt + 1}/{self.MAX_RETRIES + 1}): {e}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
        raise last_error  # type: ignore[misc]

    def _call_api_with_timeout(self, audio_path: Path):
        """CRIT-3: Cross-platform timeout using ThreadPoolExecutor.

        signal.alarm() only works on Unix main thread; this approach
        works on Windows, Linux, and inside threads.
        """
        def _do_call():
            with open(audio_path, "rb") as audio_file:
                kwargs = {
                    "file": audio_file,
                    "model_id": self.model_id,
                }
                if self.diarize:
                    kwargs["diarize"] = True
                if self.num_speakers is not None:
                    kwargs["num_speakers"] = self.num_speakers
                if self.tag_audio_events:
                    kwargs["tag_audio_events"] = True
                if self.language_code:
                    kwargs["language_code"] = self.language_code
                return self.client.speech_to_text.convert(**kwargs)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_do_call)
            try:
                return future.result(timeout=self.timeout_seconds)
            except FuturesTimeoutError:
                raise TimeoutError(
                    f"STT API call timed out after {self.timeout_seconds}s "
                    f"for {audio_path.name}"
                )

    @staticmethod
    def _build_diarized_transcript(words: list) -> tuple:
        """Build speaker-labeled transcript from Scribe v2 word-level data.

        Scribe v2 returns a list of word objects, each with:
          - text: str (the word)
          - speaker_id: str | None (e.g. "speaker_0", "speaker_1")
          - type: str ("word", "punctuation", "spacing")

        Returns:
            (diarized_text, speakers_set)
        """
        if not words:
            return "", set()

        lines = []
        current_speaker = None
        current_line_words = []
        speakers: set = set()

        for word_obj in words:
            # Support both dict and object attribute access
            if isinstance(word_obj, dict):
                text = word_obj.get("text", "")
                speaker = word_obj.get("speaker_id")
                wtype = word_obj.get("type", "word")
            else:
                text = getattr(word_obj, "text", "")
                speaker = getattr(word_obj, "speaker_id", None)
                wtype = getattr(word_obj, "type", "word")

            if speaker:
                speakers.add(speaker)

            # Speaker changed — flush current line
            if speaker and speaker != current_speaker:
                if current_line_words:
                    label = _speaker_label(current_speaker)
                    lines.append(f"{label}: {''.join(current_line_words).strip()}")
                current_line_words = []
                current_speaker = speaker

            if wtype == "spacing":
                current_line_words.append(" ")
            else:
                current_line_words.append(text)

        # Flush remaining words
        if current_line_words:
            label = _speaker_label(current_speaker)
            lines.append(f"{label}: {''.join(current_line_words).strip()}")

        return "\n".join(lines), speakers

    def _save_transcript(self, filename: str, transcript: str) -> Optional[Path]:
        """Save transcript to disk as .txt file."""
        if not self.persist_transcripts:
            return None
        try:
            # Redact PII before saving (HIGH-7)
            redactor = _get_redactor()
            redacted = redactor.redact(transcript)
            safe_text = redacted["text"]

            stem = Path(filename).stem
            txt_path = self.transcripts_folder / f"{stem}.txt"
            txt_path.write_text(safe_text, encoding="utf-8")
            if redacted["total_redactions"] > 0:
                logger.info(f"Transcript saved (PII redacted): {txt_path} | {redacted['pii_found']}")
            else:
                logger.debug(f"Transcript saved: {txt_path}")
            return txt_path
        except Exception as e:
            logger.warning(f"Failed to save transcript for {safe_log_filename(filename)}: {e}")
            return None

    def transcribe_batch(
        self,
        audio_files: List[Path],
        delay_between_calls: Optional[float] = None,
    ) -> Dict[str, Dict]:
        """Transcribe multiple files, track progress and costs.

        #6: Uses instance delay_between_calls by default, overridable per-call.

        Args:
            audio_files: List of audio file paths.
            delay_between_calls: Seconds to wait between consecutive API calls.
                                 Defaults to self.delay_between_calls.

        Returns:
            dict: {filename: {transcript, duration, credits_used, status, ...}}
        """
        if delay_between_calls is None:
            delay_between_calls = self.delay_between_calls
        transcripts = {}
        total = len(audio_files)

        for i, audio_path in enumerate(audio_files, 1):
            filename = audio_path.name
            print(f"  Transcribing {i} of {total}: {filename}...", end="\r")
            logger.info(f"Transcribing {i}/{total}: {safe_log_filename(filename)}")

            # Get duration for cost estimate
            duration = self._get_duration(audio_path)
            estimated_credits = int(duration * CREDITS_PER_MINUTE) if duration else 0
            estimated_cost = duration * COST_PER_MINUTE if duration else 0

            # ── STT Cache check ──────────────────────────────────────
            cache_key = STTCache.cache_key(
                audio_path,
                model_id=self.model_id,
                diarize=self.diarize,
                num_speakers=self.num_speakers,
                language_code=self.language_code,
            )
            cached = self._stt_cache.load(cache_key)
            if cached is not None:
                transcript_text = cached["text"]
                # Persist transcript to disk (even from cache)
                saved_path = self._save_transcript(filename, transcript_text)
                transcripts[filename] = {
                    "path": audio_path,
                    "transcript": transcript_text,
                    "raw_text": cached.get("raw_text", transcript_text),
                    "speakers_detected": cached.get("speakers_detected", 0),
                    "diarized": cached.get("diarized", False),
                    "language_code": cached.get("language_code"),
                    "duration": duration,
                    "process_time": 0.0,
                    "credits_used": 0,
                    "cost_usd": 0.0,
                    "status": "Success",
                    "cached": True,
                    "transcript_path": str(saved_path) if saved_path else None,
                }
                logger.info(f"  CACHED: {safe_log_filename(filename)} (skipped API call, saved ${estimated_cost:.4f})")
                continue
            # ── End cache check ──────────────────────────────────────

            start = time.time()
            try:
                result = self.transcribe(audio_path)
                elapsed = time.time() - start

                # The full text for downstream consumers
                transcript_text = result["text"]

                # Persist transcript to disk
                saved_path = self._save_transcript(filename, transcript_text)

                # ── Save to STT cache ────────────────────────────────
                self._stt_cache.save(cache_key, {
                    "text": transcript_text,
                    "raw_text": result.get("raw_text", transcript_text),
                    "speakers_detected": result.get("speakers_detected", 0),
                    "diarized": result.get("diarized", False),
                    "language_code": result.get("language_code"),
                })
                # ── End cache save ───────────────────────────────────

                transcripts[filename] = {
                    "path": audio_path,
                    "transcript": transcript_text,
                    "raw_text": result.get("raw_text", transcript_text),
                    "speakers_detected": result.get("speakers_detected", 0),
                    "diarized": result.get("diarized", False),
                    "language_code": result.get("language_code"),
                    "duration": duration,
                    "process_time": round(elapsed, 2),
                    "credits_used": estimated_credits,
                    "cost_usd": round(estimated_cost, 4),
                    "status": "Success",
                    "cached": False,
                    "transcript_path": str(saved_path) if saved_path else None,
                }
                logger.info(f"  OK: {safe_log_filename(filename)} ({duration or 0:.1f} min, {elapsed:.1f}s)")

                # CRIT-NEW-2: Rate-limit between consecutive API calls
                if i < total and delay_between_calls > 0:
                    time.sleep(delay_between_calls)

            except Exception as e:
                elapsed = time.time() - start
                error_msg = str(e)

                if "quota_exceeded" in error_msg.lower():
                    logger.error(f"Quota exceeded! Stopping batch at {i}/{total}")
                    transcripts[filename] = {
                        "path": audio_path,
                        "status": "Error: Quota exceeded",
                        "duration": duration
                    }
                    break

                transcripts[filename] = {
                    "path": audio_path,
                    "status": f"Error: {e}",
                    "duration": duration,
                    "process_time": round(elapsed, 2)
                }
                logger.error(f"  FAIL: {safe_log_filename(filename)} - {e}")

        # Summary
        success = sum(1 for v in transcripts.values() if v["status"] == "Success")
        total_cost = sum(v.get("cost_usd", 0) for v in transcripts.values())
        cached_count = sum(1 for v in transcripts.values() if v.get("cached"))
        api_count = success - cached_count
        cache_savings = sum(
            (v.get("duration", 0) or 0) * COST_PER_MINUTE
            for v in transcripts.values() if v.get("cached")
        )
        cache_status = f" | Cache: {cached_count} hits, {api_count} API calls, saved ${cache_savings:.4f}" if cached_count else ""
        print(f"\n  Transcription complete: {success}/{total} files | Cost: ${total_cost:.4f}{cache_status}")

        return transcripts

    def _get_duration(self, file_path: Path) -> Optional[float]:
        """Get audio duration in minutes using pydub."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(file_path))
            return round(len(audio) / 1000 / 60, 2)
        except ImportError:
            logger.warning(f"pydub not installed — cannot read duration for {file_path.name}")
            return None
        except Exception as e:
            # CRIT-6: Log actual error instead of swallowing silently
            logger.warning(f"Cannot read duration for {file_path.name}: {type(e).__name__}: {e}")
            return None


def _speaker_label(speaker_id: Optional[str]) -> str:
    """Map Scribe v2 speaker_id to human-readable label.

    speaker_0 → Speaker 0, speaker_1 → Speaker 1, etc.
    TranscriptCleaner will later remap these to Agent:/Client:.
    """
    if not speaker_id:
        return "Speaker"
    # "speaker_0" → "Speaker 0"
    parts = speaker_id.split("_")
    if len(parts) == 2 and parts[1].isdigit():
        return f"Speaker {parts[1]}"
    return speaker_id.replace("_", " ").title()
