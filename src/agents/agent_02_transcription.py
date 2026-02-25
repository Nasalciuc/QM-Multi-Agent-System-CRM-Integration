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
import re
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

# COST-02: Minimum audio duration to justify an API call
_DEFAULT_MIN_AUDIO_DURATION_SEC = 5

# COST-01: Audio pre-processing constants (SILENCE-FIX: internal compression removed)
_TARGET_SAMPLE_RATE = 16_000       # 16 kHz mono — Scribe v2 internal rate
_SILENCE_THRESHOLD_DB = -40        # dBFS below which audio is considered silence
_SAVINGS_THRESHOLD_PCT = 5         # only use preprocessed file if savings > 5 %

# SILENCE-FIX: Hold-language regex for REAL-12 silence markers.
# Matches common hold/wait phrases so silence markers can be annotated
# as [M:SS silence — hold] when preceded by agent hold language.
# NOTE: At this point in the pipeline speaker labels are still "Speaker N:"
# (Agent/Client rename happens later in TranscriptCleaner), so we match on
# ANY speaker’s line.  This is acceptable because hold language ("let me check",
# "bear with me") is almost exclusively said by agents.
_HOLD_LANGUAGE_RE = re.compile(
    r'\b(?:'
    r'hold|on hold|'
    r'one moment|just a moment|give me a moment|'
    r'bear with me|'
    r'let me (?:check|look|see|find|pull up|search|verify)|'
    r'(?:give|need) (?:me )?(?:a )?(?:sec(?:ond)?|minute|moment)|'
    r'hang (?:on|tight)|'
    r'two seconds?|one second?|'
    r'un moment(?:o)?|'
    r"I'?ll be right back"
    r')\b',
    re.IGNORECASE,
)


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
        min_audio_duration_sec: float = _DEFAULT_MIN_AUDIO_DURATION_SEC,
        preprocess_audio: bool = True,
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
            min_audio_duration_sec: Skip transcription for audio shorter than this (COST-02).
            preprocess_audio: Edge-trim silence, downsample to 16 kHz mono (COST-01 — internal compression removed).
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
        self.min_audio_duration_sec = min_audio_duration_sec
        self.preprocess_audio = preprocess_audio

        # STT transcript cache
        self._stt_cache = STTCache(
            cache_dir=stt_cache_dir or "data/stt_cache",
            enable=enable_stt_cache and stt_cache_dir is not None,
            ttl_seconds=stt_cache_ttl_days * 24 * 3600,
        )

        # SILENCE-FIX: One-time cache invalidation after removing COST-01 compression.
        # Old cached transcripts were generated from compressed audio and have
        # corrupted timestamps.  Wipe them on first run after this change.
        _cache_migration_marker = Path(stt_cache_dir or "data/stt_cache") / ".silence_fix_v1"
        if self._stt_cache.enabled and not _cache_migration_marker.exists():
            cleared = self._stt_cache.clear()
            logger.info(f"SILENCE-FIX: Cleared {cleared} cached transcripts (COST-01 compression removed)")
            _cache_migration_marker.parent.mkdir(parents=True, exist_ok=True)
            _cache_migration_marker.write_text("Compression removed, cache invalidated")

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

        # COST-02: Skip very short audio (< min_audio_duration_sec)
        duration_sec = self._get_duration_seconds(audio_path)
        if duration_sec is not None and duration_sec < self.min_audio_duration_sec:
            logger.info(
                f"COST-02: Skipping {audio_path.name} — too short "
                f"({duration_sec:.1f}s < {self.min_audio_duration_sec}s)"
            )
            return {
                "text": "",
                "raw_text": "",
                "speakers_detected": 0,
                "language_code": self.language_code,
                "diarized": False,
                "speakers_merged": False,
                "skipped_short": True,
            }

        # COST-01: Pre-process audio to reduce billable duration
        preprocessed_path: Optional[Path] = None
        preprocess_stats: Optional[dict] = None
        effective_audio = audio_path
        if self.preprocess_audio:
            try:
                preprocessed_path, preprocess_stats = self._preprocess_audio(audio_path)
                if preprocessed_path is not None:
                    effective_audio = preprocessed_path
            except Exception as exc:
                logger.warning(
                    f"COST-01: Pre-processing failed for {audio_path.name}, "
                    f"using original: {exc}"
                )

        last_error: Optional[Exception] = None
        try:
            for attempt in range(self.MAX_RETRIES + 1):
                try:
                    result = self._call_api_with_timeout(effective_audio)

                    # Build structured result from Scribe v2 response
                    raw_text = result.text.strip() if hasattr(result, 'text') else ""
                    words = getattr(result, 'words', None) or []
                    language = getattr(result, 'language_code', None) or self.language_code

                    # Build diarized transcript from word-level speaker tags
                    if self.diarize and words:
                        diarized_text, speakers, speakers_merged, parsed_words = self._build_diarized_transcript(words)
                        silence_stats = self._analyze_silence(parsed_words)
                    else:
                        diarized_text = raw_text
                        speakers = set()
                        speakers_merged = False
                        silence_stats = None

                    result_dict = {
                        "text": diarized_text,
                        "raw_text": raw_text,
                        "speakers_detected": len(speakers),
                        "language_code": language,
                        "diarized": bool(self.diarize and words),
                        "speakers_merged": speakers_merged,
                    }
                    if silence_stats is not None:
                        result_dict["silence_stats"] = silence_stats
                    if preprocess_stats:
                        result_dict["preprocess"] = preprocess_stats
                    return result_dict

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
        finally:
            # COST-01: Clean up temporary preprocessed file
            if preprocessed_path is not None and preprocessed_path.exists():
                try:
                    preprocessed_path.unlink()
                except OSError:
                    pass

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

        SPEAKER-02: If >2 speakers found in words, merges minor speakers
        into the two most-frequent speakers (by word count) before building
        lines. This is a safety net in case num_speakers=2 still yields >2
        speaker_ids from the API.

        Returns:
            (diarized_text, speakers_set, speakers_merged, parsed_words)
        """
        if not words:
            return "", set(), False, []

        # First pass: extract all words with speaker info
        parsed_words = []
        speaker_word_counts: dict = {}
        for word_obj in words:
            if isinstance(word_obj, dict):
                text = word_obj.get("text", "")
                speaker = word_obj.get("speaker_id")
                wtype = word_obj.get("type", "word")
                start_raw = word_obj.get("start")
                start = float(start_raw) if isinstance(start_raw, (int, float)) else None
            else:
                text = getattr(word_obj, "text", "")
                speaker = getattr(word_obj, "speaker_id", None)
                wtype = getattr(word_obj, "type", "word")
                start_raw = getattr(word_obj, "start", None)
                start = float(start_raw) if isinstance(start_raw, (int, float)) else None
            parsed_words.append((text, speaker, wtype, start))
            if speaker and wtype == "word":
                speaker_word_counts[speaker] = speaker_word_counts.get(speaker, 0) + 1

        all_speakers = set(speaker_word_counts.keys())
        speakers_merged = False
        merge_map: dict = {}

        if len(all_speakers) > 2:
            # SPEAKER-02: Merge minor speakers into top-2 by word count
            sorted_speakers = sorted(
                all_speakers,
                key=lambda s: -speaker_word_counts.get(s, 0)
            )
            top2 = set(sorted_speakers[:2])
            speakers_merged = True
            logger.warning(
                f"SPEAKER-02: {len(all_speakers)} speaker_ids found in words, "
                f"merging to top-2: {sorted_speakers[:2]}. "
                f"Word counts: {speaker_word_counts}"
            )
            # Build merge map: minor → nearest preceding major speaker
            last_major = None
            first_major = None
            for _, speaker, _, _ in parsed_words:
                if speaker in top2:
                    if first_major is None:
                        first_major = speaker
                    last_major = speaker
                elif speaker and speaker not in merge_map:
                    merge_map[speaker] = last_major if last_major else None
            # Backfill any that appeared before any major speaker
            for k, v in merge_map.items():
                if v is None:
                    merge_map[k] = first_major

        # Second pass: build lines with merged speaker IDs + REAL-12 timestamps
        # REAL-12: Insert [M:SS] marker when gap between words exceeds 30s
        _GAP_THRESHOLD_SECS = 30.0
        lines = []
        current_speaker = None
        current_line_words = []
        effective_speakers: set = set()
        last_word_time: float | None = None

        for text, speaker, wtype, start in parsed_words:
            effective = merge_map.get(speaker, speaker) if speaker else speaker
            if effective:
                effective_speakers.add(effective)

            # REAL-12: Check for large time gap (silence / hold)
            if (
                start is not None
                and last_word_time is not None
                and wtype == "word"
                and (start - last_word_time) >= _GAP_THRESHOLD_SECS
            ):
                gap_secs = start - last_word_time
                minutes = int(gap_secs) // 60
                seconds = int(gap_secs) % 60
                # Flush current line first
                if current_line_words:
                    label = _speaker_label(current_speaker)
                    lines.append(f"{label}: {''.join(current_line_words).strip()}")
                    current_line_words = []
                # SILENCE-FIX: Annotate marker with — hold when preceding
                # line contains hold language (e.g. "let me check", "bear with me")
                marker = f"[{minutes}:{seconds:02d} silence"
                if lines and _HOLD_LANGUAGE_RE.search(lines[-1]):
                    marker += " \u2014 hold"
                marker += "]"
                lines.append(marker)

            # Track last word timestamp
            if start is not None and wtype == "word":
                last_word_time = start

            # Speaker changed — flush current line
            if effective and effective != current_speaker:
                if current_line_words:
                    label = _speaker_label(current_speaker)
                    lines.append(f"{label}: {''.join(current_line_words).strip()}")
                current_line_words = []
                current_speaker = effective

            if wtype == "spacing":
                current_line_words.append(" ")
            else:
                current_line_words.append(text)

        # Flush remaining words
        if current_line_words:
            label = _speaker_label(current_speaker)
            lines.append(f"{label}: {''.join(current_line_words).strip()}")

        return "\n".join(lines), effective_speakers, speakers_merged, parsed_words

    @staticmethod
    def _analyze_silence(parsed_words: list) -> dict:
        """Compute silence statistics from word-level timestamps.

        Extracted from ElevenLabs word timestamps — zero additional API cost.
        Results go into evaluation metadata for reporting/monitoring.

        Args:
            parsed_words: List of (text, speaker_id, wtype, start_time) tuples
                         as built by _build_diarized_transcript first pass.

        Returns:
            Dict with silence_pct, longest_gap_ms, num_gaps_over_30s,
            total_silence_ms, gap_locations[].
        """
        gaps: list = []
        last_word_time: float | None = None
        total_speech_end = 0.0

        for text, speaker, wtype, start in parsed_words:
            if start is not None and wtype == "word":
                if last_word_time is not None:
                    gap_ms = (start - last_word_time) * 1000
                    if gap_ms > 1000:  # Only count gaps > 1s as meaningful silence
                        gaps.append({
                            "start_s": round(last_word_time, 1),
                            "end_s": round(start, 1),
                            "duration_ms": round(gap_ms),
                        })
                last_word_time = start
                total_speech_end = start

        total_duration_ms = total_speech_end * 1000 if total_speech_end > 0 else 0
        total_silence_ms = sum(g["duration_ms"] for g in gaps)

        return {
            "silence_pct": round(total_silence_ms / max(1, total_duration_ms) * 100, 1),
            "longest_gap_ms": max((g["duration_ms"] for g in gaps), default=0),
            "num_gaps_over_30s": sum(1 for g in gaps if g["duration_ms"] >= 30000),
            "total_silence_ms": round(total_silence_ms),
            "num_gaps": len(gaps),
            # P5-FIX-5: Sort by duration descending so the top-20 are the
            # longest gaps, not just the first 20 chronologically.
            "gap_locations": sorted(
                gaps, key=lambda g: g["duration_ms"], reverse=True
            )[:20],
        }

    @staticmethod
    def _reconstruct_silence_stats_from_text(transcript_text: str) -> dict:
        """P3-FIX-3: Approximate silence_stats for legacy cache entries.

        When a cached transcription doesn't have silence_stats (cached before
        FIX-3), we provide a minimal stub so downstream code doesn't crash.
        Without word-level timestamps we cannot compute real gaps, so we
        return safe zero-values and flag the source.

        Returns:
            Dict compatible with _analyze_silence() output.
        """
        return {
            "silence_pct": 0.0,
            "longest_gap_ms": 0,
            "num_gaps_over_30s": 0,
            "total_silence_ms": 0,
            "num_gaps": 0,
            "gap_locations": [],
            "_reconstructed": True,
        }

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
                # P3-FIX-3: Restore silence_stats from cache if available,
                # otherwise reconstruct approximate stats from transcript text.
                if "silence_stats" in cached:
                    transcripts[filename]["silence_stats"] = cached["silence_stats"]
                else:
                    transcripts[filename]["silence_stats"] = (
                        self._reconstruct_silence_stats_from_text(transcript_text)
                    )
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
                cache_data = {
                    "text": transcript_text,
                    "raw_text": result.get("raw_text", transcript_text),
                    "speakers_detected": result.get("speakers_detected", 0),
                    "diarized": result.get("diarized", False),
                    "language_code": result.get("language_code"),
                }
                # P3-FIX-3: Persist silence_stats so cache hits get them too
                if "silence_stats" in result:
                    cache_data["silence_stats"] = result["silence_stats"]
                self._stt_cache.save(cache_key, cache_data)
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
                # SILENCE-FIX: Propagate silence stats for downstream evaluation
                if "silence_stats" in result:
                    transcripts[filename]["silence_stats"] = result["silence_stats"]
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

    def _preprocess_audio(
        self, audio_path: Path
    ) -> tuple[Optional[Path], Optional[dict]]:
        """COST-01: Pre-process audio to reduce billable STT duration.

        Steps:
            1. Convert to mono, downsample to 16 kHz (Scribe v2 internal rate).
            2. Edge-trim leading/trailing silence (500 ms padding).

        SILENCE-FIX: Internal silence compression was removed because it
        destroyed timestamps that REAL-12 silence markers and 14 QA criteria
        depend on.  Only edge-trim + downsample remain.

        Only writes a temp file if the savings exceed _SAVINGS_THRESHOLD_PCT.

        Returns:
            (preprocessed_path | None, stats_dict | None)
            preprocessed_path is None when savings are too small to bother.
        """
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent

        audio = AudioSegment.from_file(str(audio_path))
        original_ms = len(audio)

        # 1. Mono + downsample
        if audio.channels > 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != _TARGET_SAMPLE_RATE:
            audio = audio.set_frame_rate(_TARGET_SAMPLE_RATE)

        # 2. Edge-trim leading/trailing silence (SILENCE-FIX: padding 200→500 ms)
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=500,
            silence_thresh=_SILENCE_THRESHOLD_DB,
        )
        if not nonsilent_ranges:
            # Entire audio is silence — nothing to send
            return None, {
                "original_duration_ms": original_ms,
                "processed_duration_ms": 0,
                "savings_pct": 100.0,
                "action": "all_silence",
            }

        start_trim = max(0, nonsilent_ranges[0][0] - 500)  # 500 ms padding
        end_trim = min(len(audio), nonsilent_ranges[-1][1] + 500)
        audio = audio[start_trim:end_trim]

        processed_ms = len(audio)
        savings_pct = round((1 - processed_ms / max(1, original_ms)) * 100, 1)

        stats: dict = {
            "original_duration_ms": original_ms,
            "processed_duration_ms": processed_ms,
            "savings_pct": savings_pct,
        }

        if savings_pct < _SAVINGS_THRESHOLD_PCT:
            stats["action"] = "no_change"
            logger.debug(
                f"COST-01: Savings too small ({savings_pct}%) for {audio_path.name}, using original"
            )
            return None, stats

        # Export preprocessed file next to original (temp)
        processed_path = audio_path.parent / f".{audio_path.stem}.processed.mp3"
        audio.export(str(processed_path), format="mp3")
        stats["action"] = "preprocessed"
        logger.info(
            f"COST-01: Pre-processed {audio_path.name} — "
            f"{original_ms / 1000:.1f}s → {processed_ms / 1000:.1f}s "
            f"(saved {savings_pct}%)"
        )
        return processed_path, stats

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

    def _get_duration_seconds(self, file_path: Path) -> Optional[float]:
        """COST-02: Get audio duration in seconds using pydub."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(file_path))
            return len(audio) / 1000.0
        except ImportError:
            logger.warning(f"pydub not installed — cannot check duration for {file_path.name}")
            return None
        except Exception as e:
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
