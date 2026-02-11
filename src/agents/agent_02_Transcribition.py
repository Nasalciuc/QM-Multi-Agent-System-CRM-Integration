"""
Agent 2: ElevenLabs Speech-to-Text

Purpose: Transcribe audio files using ElevenLabs Scribe v1
API: client.speech_to_text.convert(file=audio_file, model_id="scribe_v1")
Cost: ~$0.005/min (or ~280 credits/min on ElevenLabs)
"""

from pathlib import Path
from typing import Dict, List, Optional
import time
import logging

logger = logging.getLogger("qa_system")

CREDITS_PER_MINUTE = 280
COST_PER_MINUTE = 0.005


class ElevenLabsSTTAgent:
    """Agent: Speech-to-Text cu ElevenLabs"""

    def __init__(self, client):
        """
        Initialize with ElevenLabs client.

        Usage:
            from elevenlabs import ElevenLabs
            client = ElevenLabs(api_key=os.environ['ELEVENLABS_API_KEY'])
            agent_stt = ElevenLabsSTTAgent(client)
        """
        self.client = client
        logger.info("ElevenLabsSTTAgent initialized | Model: scribe_v1")

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe a single audio file using ElevenLabs Scribe v1."""
        with open(audio_path, "rb") as audio_file:
            result = self.client.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v1"
            )
        return result.text.strip()

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

                transcripts[filename] = {
                    "path": audio_path,
                    "transcript": transcript,
                    "duration": duration,
                    "process_time": round(elapsed, 2),
                    "credits_used": estimated_credits,
                    "cost_usd": round(estimated_cost, 4),
                    "status": "Success"
                }
                logger.info(f"  OK: {filename} ({duration:.1f} min, {elapsed:.1f}s)")

            except Exception as e:
                elapsed = time.time() - start
                error_msg = str(e)

                if "quota_exceeded" in error_msg.lower():
                    logger.error(f"Quota exceeded! Stopping batch at {i}/{total}")
                    transcripts[filename] = {
                        "path": audio_path,
                        "status": f"Error: Quota exceeded",
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
