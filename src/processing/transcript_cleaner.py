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

# REAL-01 / REAL-11: Patterns that indicate a speaker is an AGENT
# (self-introduction, calling from company, role declaration)
_AGENT_INTRO_PATTERNS = [
    re.compile(r"\bmy name is \w+", re.IGNORECASE),
    re.compile(r"\bthis is \w+\s+(?:from|calling|with)\b", re.IGNORECASE),
    re.compile(r"\bI'?m calling (?:from|on behalf)\b", re.IGNORECASE),
    re.compile(r"\bI'?ll be your\b.+?(?:advisor|expert|agent|consultant|specialist)", re.IGNORECASE),
    re.compile(r"\bcalling from\b", re.IGNORECASE),
    # TASK-2: Additional agent-content patterns
    re.compile(r"\bsenior travel\b", re.IGNORECASE),
    re.compile(r"\bdedicated expert\b", re.IGNORECASE),
    re.compile(r"\bI'?m (?:\w+\s+)?(?:from|with)\s+(?:buy\s*business|the company)\b", re.IGNORECASE),
    re.compile(r"\blet me (?:check|look|pull up|find)\b.+?(?:fares?|options?|flights?|availability)", re.IGNORECASE),
]

# P8-FIX-1: Inbound agent patterns — agent answers the phone on inbound calls
_INBOUND_AGENT_PATTERNS = [
    re.compile(r"\bthank you for (?:calling|holding|contacting)\b", re.IGNORECASE),
    re.compile(r"\byou(?:'re| are) speaking (?:to|with)\b", re.IGNORECASE),
    re.compile(r"\bhow (?:can|may) I (?:help|assist)\b", re.IGNORECASE),
    re.compile(r"\bwhat can I (?:do|help)\b", re.IGNORECASE),
    re.compile(r"\bwelcome to\b.+?(?:travel|support|service|desk|line)", re.IGNORECASE),
]

# P8-FIX-1: IVR / automated system patterns (not a real speaker)
_IVR_PATTERNS = [
    re.compile(r"\bthis call (?:is being|may be|will be) recorded\b", re.IGNORECASE),
    re.compile(r"\bfor quality (?:assurance|and training)\b", re.IGNORECASE),
    re.compile(r"\bpress \d+ (?:for|to)\b", re.IGNORECASE),
    re.compile(r"\bplease hold\b", re.IGNORECASE),
    re.compile(r"\byour call is important\b", re.IGNORECASE),
]

# AUD-01: Company name strings for BBC agent detection.
# When a speaker mentions the company AND shows agent behavior, they are the agent.
# This OVERRIDES "my name is" on other speakers (fixes 3309622803008 inversion).
_COMPANY_NAMES = [
    "business class",
    "buy business",
    "buybusinessclass",
    "buybusiness class",
]

# AUD-01: Agent behavior signals — used WITH company name for high-confidence detection.
# Also used standalone as additional inbound-agent indicators.
_AGENT_BEHAVIOR_PATTERNS = [
    re.compile(r"\bhow (?:can|may) I (?:help|assist)\b", re.IGNORECASE),
    re.compile(r"\bwhat can I (?:do|help)\b", re.IGNORECASE),
    re.compile(r"\byou (?:are|'re) (?:talking|speaking) (?:to|with)\b", re.IGNORECASE),
    re.compile(r"\bhave you ever contacted\b", re.IGNORECASE),
    re.compile(r"\bbefore we proceed\b", re.IGNORECASE),
    re.compile(r"\bmay I (?:have|know|get) your\b", re.IGNORECASE),
    re.compile(r"\bI(?:'ll| will) help you\b", re.IGNORECASE),
    re.compile(r"\blet me (?:check|find|look|pull up|search)\b", re.IGNORECASE),
]

# AUD-04: IVR content phrases — lines that are automated recordings, not real speakers.
# These should be skipped during agent detection to avoid wasting scan budget.
_IVR_CONTENT_PHRASES = [
    "this call is being recorded",
    "call may be monitored",
    "call is being recorded",
    "for quality assurance",
    "for quality and training",
    "your call is important",
    "please hold",
]

# REAL-01 / TASK-2: Patterns that indicate a speaker is a CLIENT
_CLIENT_ANSWER_PATTERNS = [
    # Bare "hello?" / "yes?" with question mark — client picking up the phone
    re.compile(r"^(?:hello|hi|hey|yes|yeah)\s*\?\s*$", re.IGNORECASE),
    re.compile(r"\bwho(?:'s| is) (?:this|calling)\b", re.IGNORECASE),
    re.compile(r"\bhow did you get (?:my|this) number\b", re.IGNORECASE),
    # TASK-2: Additional client-content patterns
    re.compile(r"\bI(?:'m| am) (?:looking|interested|trying)\b", re.IGNORECASE),
    re.compile(r"\bI (?:need|want|would like)\s+(?:a |to )?\b(?:fly|flight|ticket|book)", re.IGNORECASE),
    re.compile(r"\bmy budget is\b", re.IGNORECASE),
    re.compile(r"\bwe(?:'re| are) (?:traveling|flying|going)\b", re.IGNORECASE),
    re.compile(r"\bI (?:saw|found|got)\s+(?:your|an?|the)\s+(?:ad|number|email|website)", re.IGNORECASE),
]


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

        REAL-01: Detects agent by self-introduction patterns (\"my name is\",
        \"calling from\") rather than assuming first speaker = agent.
        Falls back to direction-based heuristic when no intro is found.

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

        # AUD-03: Priority-ordered agent detection:
        # Priority 0: Company name + agent behavior (highest confidence — AUD-02)
        # Priority 1: Agent self-introduction patterns (high confidence)
        # Priority 2: Client answer patterns (medium confidence — the OTHER speaker is agent)
        # Priority 3: Direction-based position heuristic (lowest confidence)
        agent_speaker = self._detect_agent_by_company(lines, merge_map)
        detection_method = "company"

        if agent_speaker is None:
            agent_speaker = self._detect_agent_by_intro(lines, merge_map)
            detection_method = "intro"

        if agent_speaker is None:
            # TASK-2: Try client-content detection — if we find the client,
            # the OTHER speaker must be the agent.
            client_speaker = self._detect_client_by_content(lines, merge_map)
            if client_speaker is not None:
                # Find the "other" major speaker
                second = self._detect_second_speaker(lines, first_speaker)
                all_speakers = {first_speaker}
                if second:
                    all_speakers.add(second)
                other = all_speakers - {client_speaker}
                if other:
                    agent_speaker = other.pop()
                    detection_method = "client_content"
                    logger.debug(
                        f"TASK-2: Agent inferred from client-content detection: "
                        f"client={client_speaker}, agent={agent_speaker}"
                    )

        if agent_speaker is None:
            # Fallback: direction-based heuristic (old behaviour)
            # For BOTH directions, the person who answers is NOT the agent.
            # Outbound: agent calls → client picks up first → agent is second.
            # Inbound: client calls → client speaks first → agent is second.
            second_speaker = self._detect_second_speaker(lines, first_speaker)
            agent_speaker = second_speaker if second_speaker else first_speaker
            detection_method = "position"
            logger.debug(
                f"REAL-01: No agent intro found, falling back to position "
                f"heuristic ({self.direction}): agent={agent_speaker}"
            )

        result_lines = []
        speakers_seen = set()

        for line in lines:
            match = re.match(r"^(Speaker\s*(\d+))\s*:\s*", line, re.IGNORECASE)
            if match:
                speaker_id = match.group(1).lower()
                speakers_seen.add(speaker_id)

                # Remap minor speaker to its merge target
                effective_id = merge_map.get(speaker_id, speaker_id)

                # REAL-01: Unified logic — agent_speaker is always the agent
                label = "Agent" if effective_id == agent_speaker else "Client"

                line = re.sub(r"^Speaker\s*\d+\s*:\s*", f"{label}: ", line, flags=re.IGNORECASE)

            result_lines.append(line)

        labeled_text = "\n".join(result_lines)

        # REAL-11: Post-labeling validation — warn if Client line has agent intro
        self._validate_speaker_labels(result_lines, detection_method)

        return labeled_text

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
    def _detect_second_speaker(lines: list, first_speaker: str) -> Optional[str]:
        """Find the second distinct 'Speaker N:' label in the transcript."""
        for line in lines:
            match = re.match(r"^(Speaker\s*\d+)\s*:", line, re.IGNORECASE)
            if match:
                sid = match.group(1).lower()
                if sid != first_speaker:
                    return sid
        return None

    @staticmethod
    def _detect_agent_by_company(
        lines: list, merge_map: dict, scan_lines: int = 20,
    ) -> Optional[str]:
        """AUD-02: Identify agent by BBC company name + agent behavior.

        Priority 0 detection — runs BEFORE intro pattern detection.
        When a speaker mentions the company name ("Business Class", "Buy Business")
        AND shows agent behavior ("how can I help", "have you ever contacted"),
        that speaker is definitively the agent.

        This fixes the 3309622803008 inversion where the customer's
        "my name is Matt" overrode Emma's "Thank you for calling Business Class."

        Also matches company name alone if combined with inbound-agent
        patterns (thank you for calling + company name = definitive agent).

        Returns the (effective) speaker_id of the agent, or None.
        """
        scanned = 0
        for line in lines:
            match = re.match(r"^(Speaker\s*(\d+))\s*:\s*(.+)", line, re.IGNORECASE)
            if not match:
                continue
            scanned += 1
            if scanned > scan_lines:
                break
            speaker_id = match.group(1).lower()
            content = match.group(3)
            content_lower = content.lower()
            effective = merge_map.get(speaker_id, speaker_id)

            # Skip IVR lines — "This call is being recorded" is not an agent
            if any(phrase in content_lower for phrase in _IVR_CONTENT_PHRASES):
                continue

            # Check: does this line mention a BBC company name?
            has_company = any(name in content_lower for name in _COMPANY_NAMES)

            if has_company:
                # Company name found — check for agent behavior on SAME line
                has_behavior = any(
                    pattern.search(content) for pattern in _AGENT_BEHAVIOR_PATTERNS
                )
                # Also check for inbound agent patterns (thank you for calling)
                has_inbound = any(
                    pattern.search(content) for pattern in _INBOUND_AGENT_PATTERNS
                )

                if has_behavior or has_inbound:
                    logger.info(
                        f"AUD-02: Agent detected by company name + behavior on "
                        f"{speaker_id}: {content[:80]!r}"
                    )
                    return effective

                # Company name WITHOUT behavior — still strong signal, but
                # only if another speaker does NOT also mention company name.
                # (Avoid false positive when client says "I found Business Class online")
                # Log and continue scanning — don't return yet.
                logger.debug(
                    f"AUD-02: Company name found on {speaker_id} without behavior: "
                    f"{content[:80]!r}"
                )

        return None

    @staticmethod
    def _detect_agent_by_intro(
        lines: list, merge_map: dict, scan_lines: int = 25,   # AUD-05: was 15
    ) -> Optional[str]:
        """REAL-01 + P8-FIX-1: Identify the agent by self-introduction patterns.

        Scans the first *scan_lines* speaker lines for phrases like
        "my name is …", "this is … calling from", etc.

        P8-FIX-1: Also checks for inbound agent patterns ("thank you for
        calling", "how can I help") and IVR patterns. If Speaker 0 matches
        IVR and Speaker 1 matches inbound-agent, Speaker 1 is the agent.

        Returns the (effective) speaker_id of the agent, or None if
        no introduction pattern was found.
        """
        scanned = 0
        ivr_speaker: Optional[str] = None
        inbound_agent_speaker: Optional[str] = None

        for line in lines:
            match = re.match(r"^(Speaker\s*(\d+))\s*:\s*(.+)", line, re.IGNORECASE)
            if not match:
                continue
            scanned += 1
            if scanned > scan_lines:
                break
            speaker_id = match.group(1).lower()
            content = match.group(3)
            effective = merge_map.get(speaker_id, speaker_id)

            # AUD-04: Skip IVR content lines — they waste scan budget
            # and should not trigger intro patterns
            content_lower = content.lower()
            if any(phrase in content_lower for phrase in _IVR_CONTENT_PHRASES):
                scanned -= 1  # Don't count IVR lines toward scan limit
                # Still track IVR speaker for P8-FIX-1 logic below
                if ivr_speaker is None:
                    ivr_speaker = effective
                    logger.debug(f"AUD-04: IVR detected on {speaker_id} (skipping for intro scan)")
                continue

            # Priority 1: Standard agent intro patterns (outbound calls)
            for pattern in _AGENT_INTRO_PATTERNS:
                if pattern.search(content):
                    logger.debug(
                        f"REAL-01: Agent detected by intro pattern on {speaker_id}: "
                        f"{pattern.pattern!r}"
                    )
                    return effective

            # P8-FIX-1: Track IVR and inbound-agent speakers
            if ivr_speaker is None:
                for pattern in _IVR_PATTERNS:
                    if pattern.search(content):
                        ivr_speaker = effective
                        logger.debug(
                            f"P8-FIX-1: IVR detected on {speaker_id}: "
                            f"{pattern.pattern!r}"
                        )
                        break

            if inbound_agent_speaker is None:
                for pattern in _INBOUND_AGENT_PATTERNS:
                    if pattern.search(content):
                        inbound_agent_speaker = effective
                        logger.debug(
                            f"P8-FIX-1: Inbound agent detected on {speaker_id}: "
                            f"{pattern.pattern!r}"
                        )
                        break

        # P8-FIX-1: If IVR was found on one speaker and inbound-agent on
        # another, the inbound-agent speaker is the real agent.
        if inbound_agent_speaker is not None:
            if ivr_speaker is not None and ivr_speaker != inbound_agent_speaker:
                logger.debug(
                    f"P8-FIX-1: IVR={ivr_speaker}, inbound agent={inbound_agent_speaker}"
                )
            return inbound_agent_speaker

        # If only IVR detected (no inbound agent pattern), IVR is not the agent.
        # Return None so the caller falls through to other heuristics.
        return None

    @staticmethod
    def _detect_client_by_content(
        lines: list, merge_map: dict, scan_lines: int = 25,   # AUD-05: was 15
    ) -> Optional[str]:
        """TASK-2: Identify the client by answer / inquiry patterns.

        Scans the first *scan_lines* speaker lines for phrases like
        "who is this?", "I'm looking for ...", "I need a flight", etc.

        Returns the (effective) speaker_id of the client, or None if
        no client-answer pattern was found.
        """
        scanned = 0
        for line in lines:
            match = re.match(r"^(Speaker\s*(\d+))\s*:\s*(.+)", line, re.IGNORECASE)
            if not match:
                continue
            scanned += 1
            if scanned > scan_lines:
                break
            speaker_id = match.group(1).lower()
            content = match.group(3)
            effective = merge_map.get(speaker_id, speaker_id)
            for pattern in _CLIENT_ANSWER_PATTERNS:
                if pattern.search(content):
                    logger.debug(
                        f"TASK-2: Client detected by content pattern on {speaker_id}: "
                        f"{pattern.pattern!r}"
                    )
                    return effective
        return None

    @staticmethod
    def _validate_speaker_labels(
        labeled_lines: list, detection_method: str,
    ) -> None:
        """REAL-11: Post-labeling sanity check.

        Scans Client:-labeled lines for agent intro patterns.
        If found, it means the agent was likely on the wrong speaker channel
        (e.g. Shiva introduced herself on Speaker 0 which was labeled Client).

        Only logs a warning — does NOT re-assign labels — so human reviewers
        can flag the call for manual inspection.
        """
        for line in labeled_lines:
            if not line.lower().startswith("client:"):
                continue
            content = line.split(":", 1)[1] if ":" in line else ""
            for pattern in _AGENT_INTRO_PATTERNS:
                if pattern.search(content):
                    logger.warning(
                        f"REAL-11: Agent intro detected on Client-labeled line "
                        f"(detection={detection_method}): {line[:120]!r}"
                    )
                    return  # One warning per call is enough

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
