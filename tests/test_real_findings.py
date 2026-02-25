"""
Tests for REAL-01 through REAL-12 findings.

These tests verify fixes for issues discovered by analysing a real
production transcript (Jerome ↔ Danny, 6 speakers detected).
"""
import re
import logging
from unittest.mock import MagicMock

import pytest

from processing.transcript_cleaner import TranscriptCleaner
from processing.pii_redactor import PIIRedactor
from processing.chunker import TranscriptChunker, TRUNCATION_MARKER
from agents.agent_02_transcription import ElevenLabsSTTAgent
from agents.agent_03_evaluation import QualityManagementAgent


# ═══════════════════════════════════════════════════════════════════
# REAL-01: Direction-agnostic agent detection via intro patterns
# ═══════════════════════════════════════════════════════════════════

class TestReal01AgentDetection:
    """REAL-01: Agent detected by self-introduction, not speaker position."""

    def test_agent_intro_my_name_outbound(self):
        """Agent who says 'my name is X' should be Agent regardless of position."""
        cleaner = TranscriptCleaner(direction="outbound")
        raw = (
            "Speaker 0: Hello?\n"
            "Speaker 1: Hi, my name is Jerome and I'm calling from Buy Business Class.\n"
            "Speaker 0: Oh hi Jerome"
        )
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")
        assert lines[0].startswith("Client:")
        assert lines[1].startswith("Agent:")
        assert "Jerome" in lines[1]

    def test_agent_intro_this_is_from(self):
        """'this is X from Y' pattern identifies the agent."""
        cleaner = TranscriptCleaner(direction="inbound")
        raw = (
            "Speaker 0: Yeah hello\n"
            "Speaker 1: This is Shiva from Travel Corp.\n"
            "Speaker 0: What do you want?"
        )
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")
        assert lines[0].startswith("Client:")
        assert lines[1].startswith("Agent:")

    def test_agent_intro_calling_from(self):
        """`calling from` pattern identifies the agent."""
        cleaner = TranscriptCleaner(direction="outbound")
        raw = (
            "Speaker 0: Hello\n"
            "Speaker 1: Hey, I'm calling from Buy Business Class\n"
            "Speaker 0: Go ahead"
        )
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")
        assert lines[1].startswith("Agent:")

    def test_fallback_second_speaker_is_agent(self):
        """Without intro patterns, second speaker = agent (both directions)."""
        for direction in ("outbound", "inbound"):
            cleaner = TranscriptCleaner(direction=direction)
            raw = "Speaker 0: Hello\nSpeaker 1: Hi\nSpeaker 0: Bye"
            result = cleaner.clean(raw)
            lines = result.strip().split("\n")
            assert lines[0].startswith("Client:"), f"Failed for {direction}"
            assert lines[1].startswith("Agent:"), f"Failed for {direction}"

    def test_agent_on_speaker_0_when_intro_present(self):
        """If Speaker 0 introduces themselves, Speaker 0 = Agent."""
        cleaner = TranscriptCleaner(direction="outbound")
        raw = (
            "Speaker 0: Hi my name is Maria calling from the agency\n"
            "Speaker 1: Oh hello"
        )
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")
        assert lines[0].startswith("Agent:")
        assert lines[1].startswith("Client:")


# ═══════════════════════════════════════════════════════════════════
# REAL-02: Multi-agent detection
# ═══════════════════════════════════════════════════════════════════

class TestReal02MultiAgent:
    """REAL-02: Detect multiple agents in a transcript."""

    def test_single_agent_detected(self):
        transcript = (
            "Agent: Hi, my name is Jerome from Buy Business Class.\n"
            "Client: Hello Jerome\n"
            "Agent: I'm calling about your trip"
        )
        agents = QualityManagementAgent.detect_agents_in_transcript(transcript)
        assert agents == ["Jerome"]

    def test_two_agents_detected(self):
        transcript = (
            "Agent: Hi, my name is Jerome from Buy Business Class.\n"
            "Client: Hello\n"
            "Agent: Let me transfer you.\n"
            "Agent: Hi this is Shiva from the same team.\n"
            "Client: Oh okay"
        )
        agents = QualityManagementAgent.detect_agents_in_transcript(transcript)
        assert len(agents) == 2
        names_lower = {a.lower() for a in agents}
        assert "jerome" in names_lower
        assert "shiva" in names_lower

    def test_no_agents_detected_no_intro(self):
        transcript = (
            "Agent: Hello how are you?\n"
            "Client: Fine thanks"
        )
        agents = QualityManagementAgent.detect_agents_in_transcript(transcript)
        assert agents == []

    def test_same_agent_not_duplicated(self):
        transcript = (
            "Agent: My name is Jerome.\n"
            "Agent: As I said, my name is Jerome.\n"
        )
        agents = QualityManagementAgent.detect_agents_in_transcript(transcript)
        assert len(agents) == 1


# ═══════════════════════════════════════════════════════════════════
# REAL-03: Listening ratio sanity checks
# ═══════════════════════════════════════════════════════════════════

class TestReal03ListeningRatio:
    """REAL-03: Listening ratio returns warnings for impossible ratios."""

    @pytest.fixture
    def agent(self):
        factory = MagicMock()
        factory.primary.model_name = "test"
        factory.primary.provider_name = "test"
        factory.primary_pricing = {"input_per_1m": 0, "output_per_1m": 0}
        factory.token_limits = {"max_input_tokens": 30000}
        return QualityManagementAgent(factory)

    def test_zero_words_returns_warning(self, agent):
        result = agent.calculate_listening_ratio("")
        assert result["ratio_warning"] == "no_labeled_lines"
        assert result["total_words"] == 0

    def test_agent_zero_words_warning(self, agent):
        transcript = "Client: Hello there\nClient: How are you"
        result = agent.calculate_listening_ratio(transcript)
        assert result["ratio_warning"] == "agent_zero_words"

    def test_client_zero_words_warning(self, agent):
        transcript = "Agent: Hello there\nAgent: Let me check"
        result = agent.calculate_listening_ratio(transcript)
        assert result["ratio_warning"] == "client_zero_words"

    def test_normal_ratio_no_warning(self, agent):
        transcript = "Agent: Hello there how are you\nClient: I'm fine thanks"
        result = agent.calculate_listening_ratio(transcript)
        assert "ratio_warning" not in result
        assert result["agent_percentage"] > 0
        assert result["client_percentage"] > 0


# ═══════════════════════════════════════════════════════════════════
# REAL-04: NATO alphabet / phonetic spelling PII
# ═══════════════════════════════════════════════════════════════════

class TestReal04NatoSpelling:
    """REAL-04: NATO alphabet spelling caught by PII redactor."""

    def test_nato_as_in_pattern(self):
        redactor = PIIRedactor()
        text = "D as in Denver, A as in Alpha, N as in Nancy, N as in Nancy, Y as in Yankee"
        result = redactor.redact(text)
        assert "[SPELLED_PII]" in result["text"]
        assert result["pii_found"]["nato_spelled"] >= 1

    def test_letter_enumeration_pattern(self):
        redactor = PIIRedactor()
        text = "The name is D, A, N, N, Y"
        result = redactor.redact(text)
        assert "[SPELLED_PII]" in result["text"]
        assert result["pii_found"]["nato_spelled"] >= 1

    def test_short_enumeration_not_caught(self):
        """3 or fewer letters should NOT trigger NATO pattern."""
        redactor = PIIRedactor()
        text = "That's A, B, C"
        result = redactor.redact(text)
        # Short enumeration (3 letters) should not be redacted
        assert "A, B, C" in result["text"]

    def test_nato_disabled(self):
        redactor = PIIRedactor(redact_nato_spelled=False)
        text = "D as in Denver, A as in Alpha, N as in Nancy, N as in Nancy, Y as in Yankee"
        result = redactor.redact(text)
        assert "Denver" in result["text"]


# ═══════════════════════════════════════════════════════════════════
# REAL-05: Multi-line spoken phone numbers
# ═══════════════════════════════════════════════════════════════════

class TestReal05MultiLinePhone:
    """REAL-05: Phone numbers spoken across multiple lines by same speaker."""

    def test_seven_digits_across_lines(self):
        redactor = PIIRedactor()
        text = (
            "Agent: What's your number?\n"
            "Client: five five five\n"
            "Client: one two three\n"
            "Client: four"
        )
        result = redactor.redact(text)
        assert result["pii_found"]["multiline_phone"] >= 1
        # Digit words should be redacted
        assert "five" not in result["text"].lower() or "[PHONE]" in result["text"]

    def test_few_digit_words_not_caught(self):
        """Fewer than 7 digit words should not trigger multiline phone."""
        redactor = PIIRedactor()
        text = (
            "Client: I have two cats\n"
            "Client: and three dogs"
        )
        result = redactor.redact(text)
        assert result["pii_found"]["multiline_phone"] == 0

    def test_multiline_disabled(self):
        redactor = PIIRedactor(redact_multiline_phone=False)
        text = (
            "Client: five five five\n"
            "Client: one two three\n"
            "Client: four"
        )
        result = redactor.redact(text)
        assert "five" in result["text"].lower()


# ═══════════════════════════════════════════════════════════════════
# REAL-06: Evidence fidelity (fillers preserved)
# ═══════════════════════════════════════════════════════════════════

class TestReal06EvidenceFidelity:
    """REAL-06: Fillers preserved for verbatim evidence quoting."""

    def test_cleaner_preserves_fillers_when_disabled(self):
        cleaner = TranscriptCleaner(direction="outbound", remove_fillers=False)
        raw = "Agent: um so uh the flight is ready"
        result = cleaner.clean(raw)
        assert "um" in result
        assert "uh" in result


# ═══════════════════════════════════════════════════════════════════
# REAL-07: 30/40/30 truncation strategy
# ═══════════════════════════════════════════════════════════════════

class TestReal07Truncation:
    """REAL-07: Truncation keeps start (30%) + middle (40%) + end (30%)."""

    def test_three_section_truncation(self):
        """Truncated output should have TWO truncation markers (3 sections)."""
        chunker = TranscriptChunker(max_tokens=80)
        lines = [f"Line {i}: content words here now" for i in range(300)]
        long_text = "\n".join(lines)
        result = chunker.truncate(long_text)
        assert result["truncated"] is True
        # Should have 2 truncation markers (start...middle...end)
        marker_count = result["text"].count("transcript truncated")
        assert marker_count == 2, f"Expected 2 markers, got {marker_count}"

    def test_preserves_start_middle_end(self):
        """All three sections of the call should be represented."""
        chunker = TranscriptChunker(max_tokens=120)
        lines = [f"Line {i:03d}: content here" for i in range(500)]
        long_text = "\n".join(lines)
        result = chunker.truncate(long_text)
        text = result["text"]
        # Start lines present
        assert "Line 000:" in text
        # End lines present
        assert "Line 499:" in text
        # Some middle content present (around line 250)
        has_middle = any(f"Line {i:03d}:" in text for i in range(200, 350))
        assert has_middle, "Middle section should contain lines from around the center"

    def test_ratios_configurable(self):
        chunker = TranscriptChunker(
            max_tokens=80,
            keep_start_ratio=0.5,
            keep_middle_ratio=0.2,
            keep_end_ratio=0.3,
        )
        assert chunker.keep_start_ratio == 0.5
        assert chunker.keep_middle_ratio == 0.2

    def test_short_text_not_truncated(self):
        chunker = TranscriptChunker(max_tokens=1000)
        result = chunker.truncate("Short text.")
        assert result["truncated"] is False
        assert "transcript truncated" not in result["text"]


# ═══════════════════════════════════════════════════════════════════
# REAL-08: PNR false positives (aviation codes exempted)
# ═══════════════════════════════════════════════════════════════════

class TestReal08PnrFalsePositives:
    """REAL-08: Aviation codes like EVA123 exempted from PNR redaction."""

    def test_aviation_code_not_redacted(self):
        redactor = PIIRedactor()
        result = redactor.redact("Your flight is EVA123.")
        assert "EVA123" in result["text"]
        assert result["pii_found"]["pnr"] == 0

    def test_real_pnr_still_redacted(self):
        redactor = PIIRedactor()
        result = redactor.redact("Booking reference: XY3Z4K")
        assert "[BOOKING_REF]" in result["text"]
        assert "XY3Z4K" not in result["text"]

    def test_multiple_aviation_codes_preserved(self):
        redactor = PIIRedactor()
        text = "Flights: ANA456, JAL789, SIA012"
        result = redactor.redact(text)
        assert "ANA456" in result["text"]
        assert "JAL789" in result["text"]
        assert "SIA012" in result["text"]


# ═══════════════════════════════════════════════════════════════════
# REAL-09: Prompt speaker guidance (indirect test)
# ═══════════════════════════════════════════════════════════════════

class TestReal09PromptSpeakerGuidance:
    """REAL-09: qa_system.txt contains speaker label instructions."""

    def test_prompt_has_speaker_labels_section(self):
        from pathlib import Path
        prompt_path = Path(__file__).parent.parent / "src" / "prompts" / "qa_system.txt"
        content = prompt_path.read_text(encoding="utf-8")
        assert "SPEAKER LABELS:" in content
        assert "Agent" in content
        assert "Client" in content
        assert "multiple different agents" in content.lower() or "multiple" in content.lower()

    def test_prompt_verbatim_evidence(self):
        from pathlib import Path
        prompt_path = Path(__file__).parent.parent / "src" / "prompts" / "qa_system.txt"
        content = prompt_path.read_text(encoding="utf-8")
        assert "VERBATIM" in content

    def test_prompt_has_silence_interpretation(self):
        """SILENCE-FIX: qa_system.txt contains silence marker instructions."""
        from pathlib import Path
        prompt_path = Path(__file__).parent.parent / "src" / "prompts" / "qa_system.txt"
        content = prompt_path.read_text(encoding="utf-8")
        assert "SILENCE MARKERS:" in content
        assert "hold" in content.lower()
        assert "continuous_feedback" in content or "trial_close" in content


# ═══════════════════════════════════════════════════════════════════
# REAL-10: Price redaction (opt-in)
# ═══════════════════════════════════════════════════════════════════

class TestReal10PriceRedaction:
    """REAL-10: Dollar amounts redacted when opt-in enabled."""

    def test_prices_not_redacted_by_default(self):
        redactor = PIIRedactor()
        result = redactor.redact("The price is $1,234.")
        assert "$1,234" in result["text"]

    def test_prices_redacted_when_enabled(self):
        redactor = PIIRedactor(redact_prices=True)
        result = redactor.redact("The price is $1,234.")
        assert "[PRICE]" in result["text"]
        assert "$1,234" not in result["text"]

    def test_verbal_price_redacted(self):
        redactor = PIIRedactor(redact_prices=True)
        result = redactor.redact("That's four thousand seven hundred dollars")
        assert "[PRICE]" in result["text"]

    def test_price_with_cents_redacted(self):
        redactor = PIIRedactor(redact_prices=True)
        result = redactor.redact("Total is $2,499.99 per person")
        assert "[PRICE]" in result["text"]
        assert "$2,499.99" not in result["text"]


# ═══════════════════════════════════════════════════════════════════
# REAL-11: Agent intro on Client label warning
# ═══════════════════════════════════════════════════════════════════

class TestReal11SpeakerLabelValidation:
    """REAL-11: Warning logged when agent intro found on Client-labeled line."""

    def test_warns_on_intro_on_client_line(self, caplog):
        """If agent intro lands on Client label, log REAL-11 warning."""
        cleaner = TranscriptCleaner(direction="outbound")
        # Both speakers introduce themselves. Intro scan finds Speaker 0 first
        # → Speaker 0 = Agent. But Speaker 1 also has intro → labeled Client
        # with agent-like content → REAL-11 warning.
        raw = (
            "Speaker 0: Hello my name is Jerome from Buy Business Class\n"
            "Speaker 1: Hi this is Danny from the support team\n"
            "Speaker 0: I was calling about your trip"
        )
        with caplog.at_level(logging.WARNING, logger="qa_system.processing"):
            result = cleaner.clean(raw)

        assert any("REAL-11" in rec.message for rec in caplog.records), \
            f"Expected REAL-11 warning, got: {[r.message for r in caplog.records]}"

    def test_no_warning_when_correct(self, caplog):
        """No warning when intro is on Agent-labeled speaker."""
        cleaner = TranscriptCleaner(direction="outbound")
        raw = (
            "Speaker 0: Hello?\n"
            "Speaker 1: Hi my name is Jerome from Buy Business Class\n"
            "Speaker 0: Hi Jerome"
        )
        with caplog.at_level(logging.WARNING, logger="qa_system.processing"):
            cleaner.clean(raw)
        real11_warnings = [r for r in caplog.records if "REAL-11" in r.message]
        assert len(real11_warnings) == 0


# ═══════════════════════════════════════════════════════════════════
# REAL-12: Timestamp markers in diarized transcript
# ═══════════════════════════════════════════════════════════════════

class TestReal12TimestampMarkers:
    """REAL-12: [M:SS silence] markers inserted for large gaps."""

    @staticmethod
    def _make_word(text, speaker_id, wtype="word", start=None):
        """Create a dict-based word object for _build_diarized_transcript."""
        d = {"text": text, "speaker_id": speaker_id, "type": wtype}
        if start is not None:
            d["start"] = start
        return d

    def test_gap_produces_silence_marker(self):
        """A 60-second gap between words should insert [1:00 silence]."""
        words = [
            self._make_word("Hello", "speaker_0", start=0.0),
            self._make_word(" ", "speaker_0", wtype="spacing", start=0.5),
            self._make_word("there", "speaker_0", start=1.0),
            # 60-second gap
            self._make_word("OK", "speaker_0", start=61.0),
            self._make_word(" ", "speaker_0", wtype="spacing", start=61.2),
            self._make_word("bye", "speaker_0", start=61.5),
        ]
        text, speakers, merged, parsed_words = ElevenLabsSTTAgent._build_diarized_transcript(words)
        assert "[1:00 silence]" in text

    def test_no_marker_for_short_gap(self):
        """A 10-second gap should NOT produce a silence marker."""
        words = [
            self._make_word("Hello", "speaker_0", start=0.0),
            self._make_word("there", "speaker_0", start=10.0),
        ]
        text, _, _, _ = ElevenLabsSTTAgent._build_diarized_transcript(words)
        assert "silence" not in text

    def test_no_timestamps_still_works(self):
        """Words without start times should not crash."""
        words = [
            self._make_word("Hello", "speaker_0"),
            self._make_word("there", "speaker_0"),
        ]
        text, _, _, _ = ElevenLabsSTTAgent._build_diarized_transcript(words)
        assert "Hello" in text
        assert "silence" not in text

    def test_multiple_gaps(self):
        """Multiple silences should produce multiple markers."""
        words = [
            self._make_word("Hi", "speaker_0", start=0.0),
            self._make_word("pause1", "speaker_0", start=45.0),
            self._make_word("pause2", "speaker_0", start=120.0),
        ]
        text, _, _, _ = ElevenLabsSTTAgent._build_diarized_transcript(words)
        assert text.count("silence") == 2

    def test_gap_formats_correctly(self):
        """90-second gap should format as [1:30 silence]."""
        words = [
            self._make_word("Hello", "speaker_0", start=0.0),
            self._make_word("bye", "speaker_0", start=90.0),
        ]
        text, _, _, _ = ElevenLabsSTTAgent._build_diarized_transcript(words)
        assert "[1:30 silence]" in text

    # ── SILENCE-FIX: Hold-language detection tests ───────────────

    def test_hold_language_produces_hold_marker(self):
        """SILENCE-FIX: Hold language before silence → [M:SS silence — hold]."""
        words = [
            self._make_word("Let", "speaker_0", start=0.0),
            self._make_word(" ", "speaker_0", wtype="spacing", start=0.2),
            self._make_word("me", "speaker_0", start=0.3),
            self._make_word(" ", "speaker_0", wtype="spacing", start=0.5),
            self._make_word("check", "speaker_0", start=0.6),
            # 60-second gap (hold)
            self._make_word("OK", "speaker_0", start=60.6),
        ]
        text, _, _, _ = ElevenLabsSTTAgent._build_diarized_transcript(words)
        assert "[1:00 silence \u2014 hold]" in text

    def test_no_hold_language_produces_bare_marker(self):
        """Regular silence without hold language → bare [M:SS silence]."""
        words = [
            self._make_word("The", "speaker_0", start=0.0),
            self._make_word(" ", "speaker_0", wtype="spacing", start=0.2),
            self._make_word("price", "speaker_0", start=0.3),
            self._make_word(" ", "speaker_0", wtype="spacing", start=0.5),
            self._make_word("is", "speaker_0", start=0.6),
            # 45-second gap (client thinking about price — NOT a hold)
            self._make_word("OK", "speaker_1", start=45.6),
        ]
        text, _, _, _ = ElevenLabsSTTAgent._build_diarized_transcript(words)
        assert "silence]" in text
        assert "hold" not in text

    def test_hold_variations_detected(self):
        """Various hold phrases should all produce hold markers."""
        hold_phrases = [
            "bear with me",
            "one moment please",
            "give me a second",
            "hang tight",
            "I'll be right back",
            "let me look that up",
        ]
        for phrase in hold_phrases:
            phrase_words = phrase.split()
            words = []
            t = 0.0
            for w in phrase_words:
                words.append(self._make_word(w, "speaker_0", start=t))
                t += 0.3
                words.append(self._make_word(" ", "speaker_0", wtype="spacing", start=t))
                t += 0.1
            # Add 60s gap + response
            words.append(self._make_word("OK", "speaker_0", start=t + 60))
            text, _, _, _ = ElevenLabsSTTAgent._build_diarized_transcript(words)
            assert "hold" in text, f"Hold phrase not detected: {phrase!r}"

    def test_hold_language_case_insensitive(self):
        """Hold language detection should be case-insensitive."""
        words = [
            self._make_word("PLEASE", "speaker_0", start=0.0),
            self._make_word(" ", "speaker_0", wtype="spacing", start=0.2),
            self._make_word("HOLD", "speaker_0", start=0.3),
            self._make_word("OK", "speaker_0", start=60.3),
        ]
        text, _, _, _ = ElevenLabsSTTAgent._build_diarized_transcript(words)
        assert "hold" in text.lower()


# ═══════════════════════════════════════════════════════════════════
# SILENCE-FIX: Silence statistics from word timestamps
# ═══════════════════════════════════════════════════════════════════

class TestSilenceAnalysis:
    """SILENCE-FIX: Silence statistics extracted from word timestamps."""

    def test_analyze_silence_basic(self):
        parsed_words = [
            ("Hello", "speaker_0", "word", 0.0),
            (" ", "speaker_0", "spacing", 0.5),
            ("there", "speaker_0", "word", 1.0),
            # 45-second gap
            ("OK", "speaker_0", "word", 46.0),
            (" ", "speaker_0", "spacing", 46.5),
            ("bye", "speaker_0", "word", 47.0),
        ]
        stats = ElevenLabsSTTAgent._analyze_silence(parsed_words)
        assert stats["num_gaps_over_30s"] == 1
        assert stats["longest_gap_ms"] >= 44000
        assert stats["num_gaps"] >= 1
        assert stats["silence_pct"] > 50

    def test_analyze_silence_no_gaps(self):
        parsed_words = [
            ("Hello", "speaker_0", "word", 0.0),
            ("there", "speaker_0", "word", 0.5),
        ]
        stats = ElevenLabsSTTAgent._analyze_silence(parsed_words)
        assert stats["num_gaps_over_30s"] == 0
        assert stats["longest_gap_ms"] == 0
        assert stats["silence_pct"] == 0

    def test_analyze_silence_no_timestamps(self):
        parsed_words = [
            ("Hello", "speaker_0", "word", None),
            ("there", "speaker_0", "word", None),
        ]
        stats = ElevenLabsSTTAgent._analyze_silence(parsed_words)
        assert stats["num_gaps"] == 0

    def test_analyze_silence_empty_words(self):
        """Empty word list should return zeroed stats."""
        stats = ElevenLabsSTTAgent._analyze_silence([])
        assert stats["num_gaps"] == 0
        assert stats["silence_pct"] == 0
        assert stats["longest_gap_ms"] == 0

    def test_analyze_silence_single_long_gap(self):
        """A single 60s gap should be flagged as >30s."""
        parsed_words = [
            ("Hello", "s0", "word", 0.0),
            ("bye", "s0", "word", 60.0),
        ]
        stats = ElevenLabsSTTAgent._analyze_silence(parsed_words)
        assert stats["num_gaps"] == 1
        assert stats["num_gaps_over_30s"] == 1
        assert stats["longest_gap_ms"] >= 59000

    def test_analyze_silence_anomaly_flag(self):
        """Calls with >50% silence should be flagged as anomalous."""
        parsed_words = [
            ("Hi", "s0", "word", 0.0),
            ("there", "s0", "word", 1.0),
            ("bye", "s0", "word", 100.0),  # 99s gap in 100s call
        ]
        stats = ElevenLabsSTTAgent._analyze_silence(parsed_words)
        assert stats["silence_pct"] > 50

    def test_analyze_silence_gap_locations_capped(self):
        """Gap locations list should be capped at 20 entries."""
        # Create 25 gaps > 1s
        words = []
        t = 0.0
        for i in range(26):
            words.append(("word", "s0", "word", t))
            t += 2.0  # 2s gap between each (> 1s threshold)
        stats = ElevenLabsSTTAgent._analyze_silence(words)
        assert len(stats["gap_locations"]) <= 20
