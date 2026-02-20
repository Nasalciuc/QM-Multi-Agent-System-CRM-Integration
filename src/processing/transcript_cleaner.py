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

        SPEAKER-03: If >2 speakers detected, identifies the top 2 by line count
        and merges all minor speakers into the closest major speaker.
        """
        lines = text.split("\n")
        first_speaker = self._detect_first_speaker(lines)

        if first_speaker is None:
            # No Speaker N: labels found — check for existing Agent/Client labels
            return text

        # Count lines per speaker to identify top-2
        speaker_line_counts: dict = {}
        for line in lines:
            match = re.match(r"^(Speaker\s*(\d+))\s*:\s*", line, re.IGNORECASE)
            if match:
                sid = match.group(1).lower()
                speaker_line_counts[sid] = speaker_line_counts.get(sid, 0) + 1

        unique_speakers = set(speaker_line_counts.keys())

        if len(unique_speakers) > 2:
            logger.warning(
                f"SPEAKER-03: {len(unique_speakers)} speakers detected, "
                f"merging to 2. Counts: {speaker_line_counts}"
            )
            # Top 2 speakers by line count (tie-break by first appearance)
            appearance_order = []
            seen = set()
            for line in lines:
                m = re.match(r"^(Speaker\s*(\d+))\s*:\s*", line, re.IGNORECASE)
                if m:
                    sid = m.group(1).lower()
                    if sid not in seen:
                        appearance_order.append(sid)
                        seen.add(sid)
            top2 = sorted(
                unique_speakers,
                key=lambda s: (-speaker_line_counts[s], appearance_order.index(s) if s in appearance_order else 999)
            )[:2]
            # Ensure first_speaker is in top2 (preserve first-appearance semantics)
            if first_speaker not in top2:
                top2[1] = first_speaker
            major_a, major_b = top2[0], top2[1]
            # Build merge map: minor speaker → nearest major speaker
            # "nearest" = preceding major speaker in the transcript
            merge_map = self._build_merge_map(lines, {major_a, major_b})
        else:
            merge_map = {}

        # Determine mapping based on direction
        if self.direction == "outbound":
            # Outbound: first speaker = Agent
            agent_speaker = first_speaker
        else:
            # Inbound: first speaker = Client
            agent_speaker = first_speaker

        result_lines = []
        speakers_seen = set()

        for line in lines:
            match = re.match(r"^(Speaker\s*(\d+))\s*:\s*", line, re.IGNORECASE)
            if match:
                speaker_id = match.group(1).lower()
                speakers_seen.add(speaker_id)

                # Remap minor speaker to its merge target
                effective_id = merge_map.get(speaker_id, speaker_id)

                if self.direction == "outbound":
                    label = "Agent" if effective_id == agent_speaker else "Client"
                else:
                    label = "Client" if effective_id == agent_speaker else "Agent"

                line = re.sub(r"^Speaker\s*\d+\s*:\s*", f"{label}: ", line, flags=re.IGNORECASE)

            result_lines.append(line)

        return "\n".join(result_lines)

    @staticmethod
    def _build_merge_map(lines: list, major_speakers: set) -> dict:
        """Map each minor speaker to the nearest preceding major speaker.

        If a minor speaker appears before any major speaker, it maps to
        the first major speaker encountered later.
        """
        merge_map: dict = {}
        last_major = None

        # First pass: assign minor speakers that appear AFTER a major speaker
        for line in lines:
            match = re.match(r"^(Speaker\s*(\d+))\s*:\s*", line, re.IGNORECASE)
            if match:
                sid = match.group(1).lower()
                if sid in major_speakers:
                    last_major = sid
                elif sid not in merge_map and last_major is not None:
                    merge_map[sid] = last_major

        # Second pass: any minor speaker not yet assigned → first major speaker
        if merge_map or last_major:
            first_major = None
            for line in lines:
                match = re.match(r"^(Speaker\s*(\d+))\s*:\s*", line, re.IGNORECASE)
                if match:
                    sid = match.group(1).lower()
                    if sid in major_speakers:
                        first_major = sid
                        break
            if first_major:
                for sid in set(sid.lower() for line in lines
                               for match in [re.match(r"^(Speaker\s*(\d+))\s*:\s*", line, re.IGNORECASE)]
                               if match for sid in [match.group(1).lower()]):
                    if sid not in major_speakers and sid not in merge_map:
                        merge_map[sid] = first_major

        return merge_map

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
