"""
Tests for src/processing/ modules:
  - TranscriptCleaner
  - TokenCounter
  - TranscriptChunker
  - PIIRedactor
"""
import sys
import os

import pytest

from processing.transcript_cleaner import TranscriptCleaner
from processing.token_counter import TokenCounter
from processing.chunker import TranscriptChunker
from processing.pii_redactor import PIIRedactor


# ═══════════════════════════════════════════════════════════════════
# TranscriptCleaner
# ═══════════════════════════════════════════════════════════════════

class TestTranscriptCleaner:

    # --- Speaker Normalisation ---

    def test_outbound_first_speaker_is_agent(self):
        cleaner = TranscriptCleaner(direction="outbound")
        raw = "Speaker 0: Hi there\nSpeaker 1: Hello"
        result = cleaner.clean(raw)
        assert result.startswith("Agent:")
        assert "Client:" in result

    def test_inbound_first_speaker_is_client(self):
        cleaner = TranscriptCleaner(direction="inbound")
        raw = "Speaker 0: Hello I need help\nSpeaker 1: Sure thing"
        result = cleaner.clean(raw)
        assert result.startswith("Client:")
        assert "Agent:" in result

    def test_existing_labels_not_changed(self):
        cleaner = TranscriptCleaner()
        raw = "Agent: Hello\nClient: Hi"
        result = cleaner.clean(raw)
        assert "Agent:" in result
        assert "Client:" in result

    def test_multiple_speakers(self):
        cleaner = TranscriptCleaner(direction="outbound")
        raw = "Speaker 0: A\nSpeaker 1: B\nSpeaker 0: C\nSpeaker 1: D"
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")
        assert lines[0].startswith("Agent:")
        assert lines[1].startswith("Client:")
        assert lines[2].startswith("Agent:")
        assert lines[3].startswith("Client:")

    # --- Filler Removal ---

    def test_fillers_removed(self):
        cleaner = TranscriptCleaner(remove_fillers=True)
        raw = "Agent: um so uh I wanted to say hello"
        result = cleaner.clean(raw)
        assert "um " not in result.lower()
        assert "uh " not in result.lower()

    def test_fillers_kept_when_disabled(self):
        cleaner = TranscriptCleaner(remove_fillers=False)
        raw = "Agent: um hello"
        result = cleaner.clean(raw)
        assert "um" in result.lower()

    # --- Edge Cases ---

    def test_empty_string(self):
        cleaner = TranscriptCleaner()
        assert cleaner.clean("") == ""
        assert cleaner.clean("   ") == ""

    def test_whitespace_cleanup(self):
        cleaner = TranscriptCleaner(remove_fillers=False)
        raw = "Agent:   too   many   spaces\n\n\n\nClient: hi"
        result = cleaner.clean(raw)
        assert "   " not in result  # no triple spaces
        assert "\n\n" not in result  # no blank lines

    # --- Direction Validation (MED-13) ---

    def test_invalid_direction_raises(self):
        """MED-13: Invalid direction should raise ValueError, not silently default."""
        with pytest.raises(ValueError, match="Invalid direction"):
            TranscriptCleaner(direction="sideways")

    def test_valid_directions_accepted(self):
        """Both 'inbound' and 'outbound' should work (case-insensitive)."""
        TranscriptCleaner(direction="OUTBOUND")
        TranscriptCleaner(direction="Inbound")

    # --- Filler Pattern Context-Awareness (HIGH-11) ---

    def test_filler_like_comma_not_removed_after_modal(self):
        """HIGH-11: 'I'd like,' should NOT be stripped as a filler."""
        cleaner = TranscriptCleaner(remove_fillers=True)
        raw = "Agent: I'd like, maybe a different plan"
        result = cleaner.clean(raw)
        assert "like" in result.lower()


# ═══════════════════════════════════════════════════════════════════
# TokenCounter
# ═══════════════════════════════════════════════════════════════════

class TestTokenCounter:

    def test_count_empty(self):
        counter = TokenCounter()
        assert counter.count_tokens("") == 0

    def test_count_positive(self):
        counter = TokenCounter()
        tokens = counter.count_tokens("Hello, how are you doing today?")
        assert tokens > 0

    def test_analyze_under_limit(self):
        counter = TokenCounter()
        result = counter.analyze("Short text", max_tokens=1000)
        assert result["needs_truncation"] is False
        assert result["excess_tokens"] == 0

    def test_analyze_over_limit(self):
        counter = TokenCounter()
        long_text = "word " * 50000
        result = counter.analyze(long_text, max_tokens=100)
        assert result["needs_truncation"] is True
        assert result["excess_tokens"] > 0

    def test_analyze_has_cost_estimate(self):
        counter = TokenCounter()
        result = counter.analyze("Some text")
        assert "estimated_cost_usd" in result
        assert result["estimated_cost_usd"] >= 0

    def test_estimate_cost(self):
        counter = TokenCounter(pricing={"input_per_1m": 2.50, "output_per_1m": 10.00})
        cost = counter.estimate_cost(1_000_000, 0)
        assert cost == 2.50

    def test_method_reported(self):
        counter = TokenCounter()
        result = counter.analyze("text")
        assert result["method"] in ("tiktoken", "estimate")


# ═══════════════════════════════════════════════════════════════════
# TranscriptChunker
# ═══════════════════════════════════════════════════════════════════

class TestTranscriptChunker:

    def test_short_not_truncated(self):
        chunker = TranscriptChunker(max_tokens=1000)
        result = chunker.truncate("Short transcript text.")
        assert result["truncated"] is False
        assert result["text"] == "Short transcript text."

    def test_long_is_truncated(self):
        chunker = TranscriptChunker(max_tokens=50)
        long_text = "\n".join(f"Line {i}: " + "word " * 10 for i in range(100))
        result = chunker.truncate(long_text)
        assert result["truncated"] is True
        assert result["removed_tokens"] > 0

    def test_truncated_has_marker(self):
        chunker = TranscriptChunker(max_tokens=50)
        long_text = "\n".join(f"Line {i}: " + "word " * 10 for i in range(100))
        result = chunker.truncate(long_text)
        assert "transcript truncated" in result["text"].lower()

    def test_preserves_beginning_and_end(self):
        chunker = TranscriptChunker(max_tokens=80)
        lines = [f"Line {i}: content here" for i in range(200)]
        long_text = "\n".join(lines)
        result = chunker.truncate(long_text)
        # Beginning should be preserved
        assert "Line 0:" in result["text"]
        # End should be preserved
        assert "Line 199:" in result["text"]

    def test_token_counts_returned(self):
        chunker = TranscriptChunker(max_tokens=50)
        result = chunker.truncate("word " * 100)
        assert "original_tokens" in result
        assert "final_tokens" in result
        assert "removed_tokens" in result


# ═══════════════════════════════════════════════════════════════════
# PIIRedactor
# ═══════════════════════════════════════════════════════════════════

class TestPIIRedactor:

    def test_redact_phone(self):
        redactor = PIIRedactor()
        result = redactor.redact("Call me at 555-123-4567 please.")
        assert "[PHONE]" in result["text"]
        assert "555-123-4567" not in result["text"]
        assert result["pii_found"]["phone"] >= 1

    def test_redact_phone_with_parens(self):
        redactor = PIIRedactor()
        result = redactor.redact("Number is (555) 123-4567.")
        assert "[PHONE]" in result["text"]

    def test_redact_email(self):
        redactor = PIIRedactor()
        result = redactor.redact("Send to john@example.com thanks.")
        assert "[EMAIL]" in result["text"]
        assert "john@example.com" not in result["text"]
        assert result["pii_found"]["email"] == 1

    def test_redact_credit_card(self):
        redactor = PIIRedactor()
        result = redactor.redact("Card number is 4111-1111-1111-1111.")
        assert "[CC_NUMBER]" in result["text"]
        assert "4111" not in result["text"]
        assert result["pii_found"]["credit_card"] >= 1

    def test_redact_ssn(self):
        redactor = PIIRedactor()
        result = redactor.redact("SSN is 123-45-6789.")
        assert "[SSN]" in result["text"]
        assert "123-45-6789" not in result["text"]
        assert result["pii_found"]["ssn"] == 1

    def test_no_pii(self):
        redactor = PIIRedactor()
        result = redactor.redact("Hello, how are you?")
        assert result["text"] == "Hello, how are you?"
        assert result["total_redactions"] == 0

    def test_empty_text(self):
        redactor = PIIRedactor()
        result = redactor.redact("")
        assert result["text"] == ""
        assert result["total_redactions"] == 0

    def test_multiple_pii_types(self):
        redactor = PIIRedactor()
        text = "Email john@test.com, call 555-111-2222, SSN 321-54-9876."
        result = redactor.redact(text)
        assert result["total_redactions"] >= 3
        assert "[EMAIL]" in result["text"]
        assert "[PHONE]" in result["text"]
        assert "[SSN]" in result["text"]

    def test_selective_redaction(self):
        """Only redact phones when other types disabled."""
        redactor = PIIRedactor(
            redact_phones=True,
            redact_emails=False,
            redact_credit_cards=False,
            redact_ssn=False,
        )
        text = "Call 555-111-2222, email me@test.com"
        result = redactor.redact(text)
        assert "[PHONE]" in result["text"]
        assert "me@test.com" in result["text"]  # email NOT redacted

    def test_redact_phone_unseparated(self):
        """CRIT-5: Unseparated 10-digit phone numbers must be redacted."""
        redactor = PIIRedactor()
        result = redactor.redact("Call me at 5551234567 please.")
        assert "[PHONE]" in result["text"]
        assert "5551234567" not in result["text"]
        assert result["pii_found"]["phone"] >= 1
