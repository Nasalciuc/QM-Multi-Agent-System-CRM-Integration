"""
Transcript Cleaner

Normalizes ElevenLabs speaker labels, removes STT artifacts,
and cleans up whitespace for consistent downstream processing.

Fixes the broken calculate_listening_ratio() which expects Agent:/Client:
prefixes that ElevenLabs Scribe doesn't produce.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger("qa_system.processing")

# Common STT filler words / artifacts to remove
# MED-4: Filler words — only standalone interjections, not legitimate words.
# "like," and "you know," require trailing comma to disambiguate from normal usage.
FILLER_WORDS = {
    "um", "uh", "umm", "uhh", "hmm", "hmmm",
    "er", "err", "ah", "ahh",
}

# Regex for standalone fillers (safe — won't match real words)
FILLER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in FILLER_WORDS) + r")\b[,]?\s*",
    re.IGNORECASE,
)


class TranscriptCleaner:
    """Normalize and clean transcripts from ElevenLabs Scribe.

    Handles:
      - Speaker label normalization (Speaker 0/1 → Agent/Client)
      - Filler word removal
      - Whitespace cleanup
      - Timestamp stripping (optional)

    Usage:
        cleaner = TranscriptCleaner(direction="outbound")
        cleaned = cleaner.clean(raw_transcript)
    """

    _VALID_DIRECTIONS = {"inbound", "outbound"}

    def __init__(self, direction: str = "outbound", remove_fillers: bool = True):
        """
        Args:
            direction: "outbound" = first speaker is Agent,
                       "inbound" = first speaker is Client.
            remove_fillers: Strip filler words from transcript.

        Raises:
            ValueError: If direction is not "inbound" or "outbound".
        """
        direction_lower = direction.lower()
        if direction_lower not in self._VALID_DIRECTIONS:
            raise ValueError(
                f"Invalid direction '{direction}': must be one of {self._VALID_DIRECTIONS}"
            )
        self.direction = direction_lower
        self.remove_fillers = remove_fillers

    def clean(self, transcript: str) -> str:
        """Full cleaning pipeline.

        Args:
            transcript: Raw transcript from ElevenLabs.

        Returns:
            Cleaned transcript with normalized labels.
        """
        if not transcript or not transcript.strip():
            return ""

        text = transcript

        # 1. Normalize speaker labels
        text = self._normalize_speakers(text)

        # 2. Remove filler words
        if self.remove_fillers:
            text = self._remove_fillers(text)

        # 3. Clean whitespace
        text = self._clean_whitespace(text)

        return text

    def _normalize_speakers(self, text: str) -> str:
        """Convert 'Speaker N:' labels to 'Agent:' / 'Client:'.

        Auto-detects which speaker number maps to Agent based on
        call direction and which speaker appears first.
        """
        lines = text.split("\n")
        first_speaker = self._detect_first_speaker(lines)

        if first_speaker is None:
            # No Speaker N: labels found — check for existing Agent/Client labels
            return text

        # Determine mapping based on direction
        if self.direction == "outbound":
            # Outbound: first speaker = Agent
            agent_speaker = first_speaker
        else:
            # Inbound: first speaker = Client
            agent_speaker = None  # will be assigned to the other speaker

        result_lines = []
        speakers_seen = set()

        for line in lines:
            match = re.match(r"^(Speaker\s*(\d+))\s*:\s*", line, re.IGNORECASE)
            if match:
                speaker_id = match.group(1).lower()
                speakers_seen.add(speaker_id)

                if self.direction == "outbound":
                    label = "Agent" if speaker_id == first_speaker else "Client"
                else:
                    label = "Client" if speaker_id == first_speaker else "Agent"

                line = re.sub(r"^Speaker\s*\d+\s*:\s*", f"{label}: ", line, flags=re.IGNORECASE)

            result_lines.append(line)

        return "\n".join(result_lines)

    @staticmethod
    def _detect_first_speaker(lines: list) -> Optional[str]:
        """Find the first 'Speaker N:' label in the transcript."""
        for line in lines:
            match = re.match(r"^(Speaker\s*\d+)\s*:", line, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        return None

    @staticmethod
    def _remove_fillers(text: str) -> str:
        """Remove common filler words from transcript.

        HIGH-11: Only removes standalone fillers (um, uh, etc.).
        Context-dependent fillers (like, you know) are preserved to avoid
        removing legitimate words.
        """
        text = FILLER_PATTERN.sub("", text)
        return text

    @staticmethod
    def _clean_whitespace(text: str) -> str:
        """Normalize whitespace: collapse multiple spaces, strip blank lines."""
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            line = re.sub(r"  +", " ", line).strip()
            if line:
                cleaned.append(line)
        return "\n".join(cleaned)
