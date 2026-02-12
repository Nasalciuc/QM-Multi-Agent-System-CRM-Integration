"""
Agent 2: ElevenLabs Speech-to-Text

Purpose: Transcribe audio files using ElevenLabs Scribe v1
API: client.speech_to_text.convert(file=audio_file, model_id="scribe_v1")
Cost: ~$0.005/min (or ~280 credits/min on ElevenLabs)

Now persists transcripts to data/transcripts/ after successful transcription.
"""

from pathlib import Path
from typing import Dict, List, Optional
import time
import logging

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


class ElevenLabsSTTAgent:
    """Agent: Speech-to-Text with ElevenLabs"""

    def __init__(self, client, persist_transcripts: bool = True,
                 transcripts_folder: str = "data/transcripts"):
        """
        Initialize with ElevenLabs client.

        Args:
            client: ElevenLabs client instance.
            persist_transcripts: Save transcripts to disk after transcription.
            transcripts_folder: Directory to save transcript files.

        Usage:
            from elevenlabs import ElevenLabs
            client = ElevenLabs(api_key=os.environ['ELEVENLABS_API_KEY'])
            agent_stt = ElevenLabsSTTAgent(client)
        """
        self.client = client
        self.persist_transcripts = persist_transcripts
        self.transcripts_folder = Path(transcripts_folder)
        if self.persist_transcripts:
            self.transcripts_folder.mkdir(parents=True, exist_ok=True)
        logger.info("ElevenLabsSTTAgent initialized | Model: scribe_v1")

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe a single audio file using ElevenLabs Scribe v1."""
        with open(audio_path, "rb") as audio_file:
            result = self.client.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v1"
            )
        return result.text.strip()

    def _save_transcript(self, filename: str, transcript: str) -> Optional[Path]:
        """Save transcript to disk as .txt file.

        HIGH-7: Redacts PII before persisting to prevent raw customer data
        from being stored on disk.
        """
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
            logger.warning(f"Failed to save transcript for {filename}: {e}")
            return None

    def transcribe_batch(self, audio_files: List[Path]) -> Dict[str, Dict]:
        """
        Transcribe multiple files, track progress and costs.
        Returns dict: {filename: {transcript, duration, credits_used, status, ...}}
        """
        transcripts = {}
        total = len(audio_files)

        for i, audio_path in enumerate(audio_files, 1):
            filename = audio_path.name
            print(f"  Transcribing {i} of {total}: {filename}...", end="\r")
            logger.info(f"Transcribing {i}/{total}: {filename}")

            # Get duration for cost estimate
            duration = self._get_duration(audio_path)
            estimated_credits = int(duration * CREDITS_PER_MINUTE) if duration else 0
            estimated_cost = duration * COST_PER_MINUTE if duration else 0

            start = time.time()
            try:
                transcript = self.transcribe(audio_path)
                elapsed = time.time() - start

                # Persist transcript to disk
                saved_path = self._save_transcript(filename, transcript)

                transcripts[filename] = {
                    "path": audio_path,
                    "transcript": transcript,
                    "duration": duration,
                    "process_time": round(elapsed, 2),
                    "credits_used": estimated_credits,
                    "cost_usd": round(estimated_cost, 4),
                    "status": "Success",
                    "transcript_path": str(saved_path) if saved_path else None,
                }
                logger.info(f"  OK: {filename} ({duration or 0:.1f} min, {elapsed:.1f}s)")

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
                logger.error(f"  FAIL: {filename} - {e}")

        # Summary
        success = sum(1 for v in transcripts.values() if v["status"] == "Success")
        total_cost = sum(v.get("cost_usd", 0) for v in transcripts.values())
        print(f"\n  Transcription complete: {success}/{total} files | Cost: ${total_cost:.4f}")

        return transcripts

    def _get_duration(self, file_path: Path) -> Optional[float]:
        """Get audio duration in minutes using pydub."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(file_path))
            return round(len(audio) / 1000 / 60, 2)
        except Exception:
            return None
