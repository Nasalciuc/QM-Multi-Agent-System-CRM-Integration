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
from utils import safe_log_filename


# ═══════════════════════════════════════════════════════════════════
# TranscriptCleaner
# ═══════════════════════════════════════════════════════════════════

class TestTranscriptCleaner:

    # --- Speaker Normalisation ---

    def test_outbound_second_speaker_is_agent(self):
        """REAL-01: Without intro patterns, second speaker = agent (both directions)."""
        cleaner = TranscriptCleaner(direction="outbound")
        raw = "Speaker 0: Hi there\nSpeaker 1: Hello"
        result = cleaner.clean(raw)
        # REAL-01: No intro found → second speaker (Speaker 1) = Agent
        assert result.startswith("Client:")
        assert "Agent:" in result
        lines = result.strip().split("\n")
        assert lines[0].startswith("Client:")
        assert lines[1].startswith("Agent:")

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
        """REAL-01: Without intro, second speaker = Agent."""
        cleaner = TranscriptCleaner(direction="outbound")
        raw = "Speaker 0: A\nSpeaker 1: B\nSpeaker 0: C\nSpeaker 1: D"
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")
        # REAL-01: Speaker 0 = Client (first), Speaker 1 = Agent (second)
        assert lines[0].startswith("Client:")
        assert lines[1].startswith("Agent:")
        assert lines[2].startswith("Client:")
        assert lines[3].startswith("Agent:")

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

    # --- Filler Pattern — like/you know preserved (HIGH-11) ---

    def test_filler_like_comma_preserved(self):
        """HIGH-11: 'like,' is now preserved — too risky to auto-remove."""
        cleaner = TranscriptCleaner(remove_fillers=True)
        raw = "Agent: I'd like, maybe a different plan"
        result = cleaner.clean(raw)
        assert "like" in result.lower()

    # --- SPEAKER-03: >2 speakers collapsed to 2 roles ---

    def test_five_speakers_collapsed_to_two_roles(self):
        """SPEAKER-03: Transcript with 5 speakers should collapse to Agent/Client only."""
        cleaner = TranscriptCleaner(direction="outbound", remove_fillers=False)
        raw = (
            "Speaker 0: Hi there\n"       # Minor → should merge
            "Speaker 1: Welcome\n"         # Major speaker A (most lines)
            "Speaker 0: Thanks\n"
            "Speaker 1: How can I help\n"
            "Speaker 2: I need help\n"     # Major speaker B (second most lines)
            "Speaker 1: Sure\n"
            "Speaker 2: With my account\n"
            "Speaker 3: Also this\n"       # Minor
            "Speaker 1: Let me check\n"
            "Speaker 2: Thanks\n"
            "Speaker 4: One more thing\n"  # Minor
        )
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")
        # Every line must be either Agent: or Client:
        for line in lines:
            assert line.startswith("Agent:") or line.startswith("Client:"), f"Unexpected label: {line}"
        # Must have both roles
        labels = {l.split(":")[0] for l in lines}
        assert labels == {"Agent", "Client"}

    def test_three_speakers_minor_merges_to_nearest(self):
        """SPEAKER-03 + REAL-01: Minor speaker merges into nearest major speaker."""
        cleaner = TranscriptCleaner(direction="outbound", remove_fillers=False)
        # REAL-01: Speaker 1 introduces self as agent → Agent
        # Speaker 0 = Client (first), Speaker 2 = minor → merges to nearest preceding major
        raw = (
            "Speaker 0: Hello\n"
            "Speaker 1: Hi my name is Alex calling from the travel desk\n"
            "Speaker 0: Sure thing\n"
            "Speaker 1: Let me check\n"
            "Speaker 1: Checking fares now\n"
            "Speaker 2: Also want hotels\n"
            "Speaker 0: Got it\n"
        )
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")
        # Every line must be either Agent: or Client:
        for line in lines:
            assert line.startswith("Agent:") or line.startswith("Client:"), f"Unexpected label: {line}"
        # "Also want hotels" merges to nearest major = Speaker 1 = Agent
        hotel_line = [l for l in lines if "Also want hotels" in l][0]
        assert hotel_line.startswith("Agent:")

    def test_two_speakers_unchanged(self):
        """SPEAKER-03: Exactly 2 speakers should not trigger merging.
        REAL-01: Second speaker = Agent when no intro pattern detected."""
        cleaner = TranscriptCleaner(direction="outbound", remove_fillers=False)
        raw = "Speaker 0: Hi\nSpeaker 1: Hello\nSpeaker 0: How are you"
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")
        # REAL-01: Speaker 0 = Client (first), Speaker 1 = Agent (second)
        assert lines[0].startswith("Client:")
        assert lines[1].startswith("Agent:")
        assert lines[2].startswith("Client:")

    # --- P8-FIX-1: IVR + Inbound Agent Pattern Detection ---

    def test_ivr_then_inbound_agent_labels_correctly(self):
        """P8-FIX-1: IVR on Speaker 0, inbound agent on Speaker 1 → Speaker 1 = Agent."""
        cleaner = TranscriptCleaner(direction="inbound", remove_fillers=False)
        raw = (
            "Speaker 0: This call is being recorded for quality assurance.\n"
            "Speaker 1: Thank you for calling, how can I help you today?\n"
            "Speaker 0: Yes hi, I need to book a flight.\n"
            "Speaker 1: Of course, let me check some options for you.\n"
        )
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")
        # Speaker 0 (IVR/client) = Client, Speaker 1 (agent) = Agent
        assert lines[0].startswith("Client:"), f"Expected Client:, got {lines[0]}"
        assert lines[1].startswith("Agent:"), f"Expected Agent:, got {lines[1]}"
        assert lines[2].startswith("Client:")
        assert lines[3].startswith("Agent:")

    def test_inbound_agent_without_ivr(self):
        """P8-FIX-1: Inbound agent pattern alone should label agent correctly."""
        cleaner = TranscriptCleaner(direction="inbound", remove_fillers=False)
        raw = (
            "Speaker 0: Hello?\n"
            "Speaker 1: Thank you for calling, you're speaking to Alex.\n"
            "Speaker 0: Hi Alex, I need help.\n"
        )
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")
        # Speaker 1 has inbound agent pattern → Agent
        assert lines[1].startswith("Agent:")

    def test_ivr_speaker_is_not_agent(self):
        """P8-FIX-1 + R-10: IVR alone should NOT make that speaker the agent.
        When Speaker 0 = IVR and Speaker 1 = client content ("I'm looking for"),
        client-content detection identifies Speaker 1 as client, making
        Speaker 0 the agent (by elimination)."""
        cleaner = TranscriptCleaner(direction="inbound", remove_fillers=False)
        raw = (
            "Speaker 0: This call is being recorded for quality assurance.\n"
            "Speaker 1: Hello, I'm looking for a flight to London.\n"
        )
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")

        # R-10: Strong assertions — verify exact label assignment
        assert len(lines) == 2

        # Both labels must be present and different
        labels = [l.split(":")[0] for l in lines]
        assert set(labels) == {"Agent", "Client"}, f"Expected both labels, got {labels}"

        # Speaker 1 has client content ("I'm looking for") → Client
        # Speaker 0 (IVR/elimination) → Agent
        if "looking for" in lines[1]:
            assert lines[1].startswith("Client:"), \
                f"Speaker 1 (client content) should be Client, got: {lines[1]}"
            assert lines[0].startswith("Agent:"), \
                f"Speaker 0 (IVR/elimination) should be Agent, got: {lines[0]}"

    def test_outbound_patterns_still_work(self):
        """P8-FIX-1: Standard outbound intro patterns should still function."""
        cleaner = TranscriptCleaner(direction="outbound", remove_fillers=False)
        raw = (
            "Speaker 0: Hello?\n"
            "Speaker 1: Hi, my name is Sarah calling from Buy Business Travel.\n"
            "Speaker 0: Oh hi Sarah.\n"
        )
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")
        assert lines[0].startswith("Client:")
        assert lines[1].startswith("Agent:")

    def test_how_may_i_help_detects_agent(self):
        """P8-FIX-1: 'How may I help you' should detect inbound agent."""
        cleaner = TranscriptCleaner(direction="inbound", remove_fillers=False)
        raw = (
            "Speaker 0: Hi, I'm calling about my booking.\n"
            "Speaker 1: Of course, how may I assist you today?\n"
            "Speaker 0: I need to change my flight.\n"
        )
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")
        assert lines[1].startswith("Agent:")


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

    def test_redact_dob(self):
        """#31: Date of birth should be redacted."""
        redactor = PIIRedactor()
        result = redactor.redact("My date of birth is 01/15/1990.")
        assert "[DOB]" in result["text"]
        assert "01/15/1990" not in result["text"]
        assert result["pii_found"]["dob"] >= 1

    def test_redact_dob_european_format(self):
        """#31: European DOB format (DD-MM-YYYY) should also be redacted."""
        redactor = PIIRedactor()
        result = redactor.redact("Born on 15-01-1990 in London.")
        assert "[DOB]" in result["text"]
        assert "15-01-1990" not in result["text"]

    def test_redact_passport(self):
        """#31: Passport numbers should be redacted (unified pattern — NEW-11)."""
        redactor = PIIRedactor()
        result = redactor.redact("My passport is A12345678.")
        assert "[PASSPORT]" in result["text"]
        assert "A12345678" not in result["text"]
        assert result["pii_found"]["passport"] >= 1

    def test_redact_address(self):
        """#31: Street addresses should be redacted."""
        redactor = PIIRedactor()
        result = redactor.redact("I live at 123 Main Street in Springfield.")
        assert "[ADDRESS]" in result["text"]
        assert "123 Main Street" not in result["text"]
        assert result["pii_found"]["address"] >= 1

    def test_redact_address_with_avenue(self):
        """#31: Address with avenue suffix should be redacted."""
        redactor = PIIRedactor()
        result = redactor.redact("Ship to 456 Oak Avenue please.")
        assert "[ADDRESS]" in result["text"]
        assert "456 Oak Avenue" not in result["text"]

    def test_redact_pnr(self):
        """Booking reference / PNR should be redacted."""
        redactor = PIIRedactor()
        result = redactor.redact("Your booking reference is AB3C4D.")
        assert "[BOOKING_REF]" in result["text"]
        assert "AB3C4D" not in result["text"]
        assert result["pii_found"]["pnr"] >= 1

    def test_pnr_no_false_positive_on_words(self):
        """Pure letter words should NOT be redacted as PNR."""
        redactor = PIIRedactor()
        result = redactor.redact("LONDON is a great city.")
        assert "LONDON" in result["text"]  # NOT redacted

    def test_order_number_not_redacted(self):
        """NEW-05: 10-digit order numbers should NOT be redacted as phone."""
        redactor = PIIRedactor()
        result = redactor.redact("Your order number 1234567890 has shipped")
        assert "1234567890" in result["text"]
        assert result["pii_found"]["phone"] == 0

    def test_account_id_not_redacted(self):
        """NEW-05: 10-digit account IDs should NOT be redacted as phone."""
        redactor = PIIRedactor()
        result = redactor.redact("Account 1234567890 is active")
        assert "1234567890" in result["text"]
        assert result["pii_found"]["phone"] == 0

    def test_actual_phone_still_redacted(self):
        """NEW-05: Real 10-digit phones (no context word) still redacted."""
        redactor = PIIRedactor()
        result = redactor.redact("Call me at 5551234567 please")
        assert "[PHONE]" in result["text"]
        assert "5551234567" not in result["text"]
        assert result["pii_found"]["phone"] >= 1


# ═══════════════════════════════════════════════════════════════════
# safe_log_filename (Fix #1)
# ═══════════════════════════════════════════════════════════════════

class TestSafeLogFilename:

    def test_basic_filename(self):
        result = safe_log_filename("call_recording.mp3")
        assert result.endswith(".mp3")
        assert "call_recording" in result

    def test_strips_directory_path(self):
        result = safe_log_filename("/home/user/data/audio/secret.mp3")
        assert "home" not in result
        assert "user" not in result
        assert "secret" in result

    def test_empty_returns_unknown(self):
        assert safe_log_filename("") == "unknown"

    def test_special_chars_replaced(self):
        result = safe_log_filename("John Doe (call #1).mp3")
        assert " " not in result
        assert "(" not in result
        assert "#" not in result

    def test_deterministic_hash(self):
        """Same input should always produce the same output."""
        a = safe_log_filename("test.mp3")
        b = safe_log_filename("test.mp3")
        assert a == b

    def test_different_files_different_hashes(self):
        a = safe_log_filename("call_a.mp3")
        b = safe_log_filename("call_b.mp3")
        assert a != b

    def test_preserves_extension(self):
        result = safe_log_filename("recording.wav")
        assert result.endswith(".wav")

    def test_no_consecutive_underscores(self):
        result = safe_log_filename("a   b---c.mp3")
        assert "__" not in result


# ═══════════════════════════════════════════════════════════════════
# HIGH-07: International PII patterns
# ═══════════════════════════════════════════════════════════════════

class TestPIIInternationalPatterns:
    """Verify international PII redaction added in HIGH-07."""

    def test_uk_phone_number_redacted(self):
        redactor = PIIRedactor()
        result = redactor.redact("Call me on +44 20 7946 0958 please")
        assert "[PHONE]" in result["text"]
        assert "+44" not in result["text"]
        assert result["pii_found"]["intl_phone"] >= 1

    def test_eu_phone_number_redacted(self):
        redactor = PIIRedactor()
        result = redactor.redact("Mein Nummer ist +49 30 1234567")
        assert "[PHONE]" in result["text"]
        assert "+49" not in result["text"]
        assert result["pii_found"]["intl_phone"] >= 1

    def test_passport_two_letter_prefix_redacted(self):
        """NEW-11: Unified passport pattern handles 1-2 letter prefixes."""
        redactor = PIIRedactor()
        result = redactor.redact("My passport is AB1234567")
        assert "[PASSPORT]" in result["text"]
        assert "AB1234567" not in result["text"]
        assert result["pii_found"]["passport"] >= 1

    def test_passport_no_double_count(self):
        """NEW-11: Single passport should not be counted twice after pattern merge."""
        redactor = PIIRedactor()
        result = redactor.redact("Passport: A1234567")
        assert result["text"].count("[PASSPORT]") == 1
        assert result["pii_found"]["passport"] == 1

    def test_loyalty_number_redacted(self):
        redactor = PIIRedactor()
        result = redactor.redact("My frequent flyer number is FF123456789012")
        assert "[LOYALTY_ID]" in result["text"]
        assert "FF123456789012" not in result["text"]
        assert result["pii_found"]["loyalty"] >= 1

    def test_short_codes_preserved(self):
        """Short numeric codes (< 6 digits) should NOT be redacted."""
        redactor = PIIRedactor()
        result = redactor.redact("Please press 1 for sales or enter code 4523")
        assert "1" in result["text"]
        assert "4523" in result["text"]


# ═══════════════════════════════════════════════════════════════════
# B3-FIX-1: Multiline Phone False Positive Prevention
# ═══════════════════════════════════════════════════════════════════

class TestB3Fix1MultilinePhoneFalsePositives:
    """B3-FIX-1: Travel sales patterns must NOT be redacted as phone numbers.

    Production evidence: 538 false [PHONE] tags across 20 calls.
    Root cause: _redact_multiline_spoken_phone() flags digit-words
    like 'one', 'two', 'five', 'seven' which appear constantly in
    fare pricing, stop counts, flight times, and baggage discussions.
    """

    def test_multiline_phone_defaults_to_false(self):
        """After B3-FIX-1, multiline phone detection is OFF by default."""
        redactor = PIIRedactor()
        assert redactor.redact_multiline_phone is False

    def test_multiline_phone_can_be_enabled(self):
        """Multiline phone CAN still be enabled via constructor parameter."""
        redactor = PIIRedactor(redact_multiline_phone=True)
        assert redactor.redact_multiline_phone is True

    def test_pricing_not_redacted(self):
        """Fare discussions must NOT produce [PHONE] tags."""
        redactor = PIIRedactor()
        text = (
            "Agent: The market rate is thirty-five, thirty-six hundred.\n"
            "Client: Okay.\n"
            "Agent: But the deal I found is twenty-seven hundred per passenger.\n"
            "Client: For two passengers?\n"
            "Agent: Yes, so fifty-four hundred total for two.\n"
        )
        result = redactor.redact(text)
        assert "[PHONE]" not in result["text"]
        assert "thirty-five" in result["text"]
        assert "twenty-seven" in result["text"]

    def test_stops_not_redacted(self):
        """Stop counts and routing must NOT produce [PHONE] tags."""
        redactor = PIIRedactor()
        text = (
            "Agent: If we take two stops, it becomes about three thousand.\n"
            "Agent: One stop would be one in the US and the other one in London.\n"
            "Client: How about nonstop?\n"
            "Agent: Nonstop is five thousand eight hundred.\n"
        )
        result = redactor.redact(text)
        assert "[PHONE]" not in result["text"]
        assert "two stops" in result["text"]

    def test_flight_times_not_redacted(self):
        """Flight times must NOT produce [PHONE] tags."""
        redactor = PIIRedactor()
        text = (
            "Agent: Departure is at four PM, four oh five PM to be exact.\n"
            "Agent: It has just one stop in Frankfurt.\n"
            "Agent: Arriving Barcelona at eleven fifty-five AM the next day.\n"
        )
        result = redactor.redact(text)
        assert "[PHONE]" not in result["text"]

    def test_baggage_not_redacted(self):
        """Baggage counts must NOT produce [PHONE] tags."""
        redactor = PIIRedactor()
        text = (
            "Agent: You get two checked bags, each seventy pounds.\n"
            "Agent: Plus one carry-on and one personal item.\n"
        )
        result = redactor.redact(text)
        assert "[PHONE]" not in result["text"]
        assert "two checked bags" in result["text"]

    def test_give_me_one_moment_not_redacted(self):
        """Common phrases with digit words must NOT produce [PHONE] tags."""
        redactor = PIIRedactor()
        text = (
            "Agent: Just give me one moment.\n"
            "Agent: Give me one second to check on that.\n"
            "Agent: Bear with me for just two minutes, please.\n"
        )
        result = redactor.redact(text)
        assert "[PHONE]" not in result["text"]
        assert "one moment" in result["text"]

    def test_real_phone_still_caught_single_line(self):
        """Real phone numbers on a single line must STILL be redacted."""
        redactor = PIIRedactor()
        text = "Agent: My direct number is 708-323-2815.\n"
        result = redactor.redact(text)
        assert "[PHONE]" in result["text"]
        assert "708" not in result["text"]

    def test_real_spelled_phone_still_caught(self):
        """Spelled-out phone (6+ digit words on one line) must STILL be caught."""
        redactor = PIIRedactor()
        text = "Client: My number is five five five one two three four five six seven.\n"
        result = redactor.redact(text)
        assert "[SPELLED_PII]" in result["text"] or "[PHONE]" in result["text"]


# ═══════════════════════════════════════════════════════════════════
# B3-FIX-3: Redaction Tag Concatenation Prevention
# ═══════════════════════════════════════════════════════════════════

class TestB3Fix3TagConcatenation:
    """B3-FIX-3: Redaction tags must have space before following text.

    Production evidence: '[SPELLED_PII]Veer Campbell' — tag directly
    concatenated with the customer name in exported evaluation.
    """

    def test_no_tag_letter_concatenation(self):
        """Redaction tags followed by letters must have a space."""
        redactor = PIIRedactor()
        # Force a phone number right before a word with no space
        text = "Call 555-123-4567immediately please"
        result = redactor.redact(text)
        # The regex ]([A-Za-z]) → ] \1 should fix this
        import re
        # No ] immediately followed by a letter
        assert not re.search(r'\\][A-Za-z]', result["text"]), \
            f"Tag concatenation found in: {result['text']}"

    def test_tag_space_after_phone(self):
        """[PHONE] followed by text should have a space."""
        redactor = PIIRedactor()
        text = "Number: 555-123-4567is my cell"
        result = redactor.redact(text)
        assert "] is" in result["text"] or "]  " in result["text"] or "[PHONE] is" in result["text"]

    def test_tag_space_preserves_existing_spaces(self):
        """Existing spaces after tags should not be doubled."""
        redactor = PIIRedactor()
        text = "Call me at 555-123-4567 anytime"
        result = redactor.redact(text)
        assert "[PHONE] anytime" in result["text"]
        # Should NOT have double space
        assert "[PHONE]  anytime" not in result["text"]


# ═══════════════════════════════════════════════════════════════════
# AUD-01/02/03: Company-Name Agent Detection + Inversion Prevention
# ═══════════════════════════════════════════════════════════════════

class TestAUD01CompanyNameAgentDetection:
    """AUD-01/02/03: Detect BBC agent by company name + behavior.

    Production evidence: Call 3309622803008 had INVERTED labels.
    Emma (BBC agent) was labeled "Client", Matt (customer) labeled "Agent".
    Root cause: Matt's "my name is Matt Kern" triggered intro pattern
    BEFORE Emma's "Thank you for calling Business Class" could be resolved.
    """

    def test_inbound_emma_matt_inversion_fixed(self):
        """GOLDEN TEST: Exact reproduction of 3309622803008 inversion.
        Emma (Speaker 1) says 'Thank you for calling Business Class' → must be Agent.
        Matt (Speaker 0) says 'my name is Matt Kern' → must be Client."""
        cleaner = TranscriptCleaner(direction="inbound")
        raw = (
            "Speaker 0: This call is being recorded.\n"
            "Speaker 1: Thank you for calling by Business Class. "
            "You are talking to Emma. How can I help you?\n"
            "Speaker 0: Yeah, my name is Matt Kern. I'm looking for flights "
            "from Atlanta, Pensacola or Atlanta to Barcelona.\n"
            "Speaker 1: Sure. Before we proceed further, just a quick question. "
            "Have you ever contacted us before?\n"
            "Speaker 0: No.\n"
        )
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")

        # Speaker 1 (Emma) MUST be Agent — she has company name + "how can I help"
        emma_line = [l for l in lines if "Emma" in l][0]
        assert emma_line.startswith("Agent:"), \
            f"Emma (BBC agent) should be Agent, got: {emma_line[:80]}"

        # Speaker 0 (Matt) MUST be Client — he's the customer
        matt_line = [l for l in lines if "Matt" in l][0]
        assert matt_line.startswith("Client:"), \
            f"Matt (customer) should be Client, got: {matt_line[:80]}"

    def test_company_name_plus_how_can_i_help(self):
        """Company name + 'how can I help' = definitive agent."""
        cleaner = TranscriptCleaner(direction="inbound")
        raw = (
            "Speaker 0: Hello?\n"
            "Speaker 1: Thank you for calling Buy Business Class. "
            "This is Allen. How can I help?\n"
            "Speaker 0: Hi, I am trying to book a ticket.\n"
        )
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")
        assert lines[1].startswith("Agent:"), f"Expected Agent:, got {lines[1][:60]}"
        assert "Allen" in lines[1]
        assert lines[2].startswith("Client:")

    def test_company_name_plus_have_you_contacted(self):
        """Company name on one line + 'have you ever contacted' on same speaker."""
        cleaner = TranscriptCleaner(direction="inbound")
        raw = (
            "Speaker 0: Hi there.\n"
            "Speaker 1: Hello, calling from Buy Business Class. "
            "Have you ever contacted us before?\n"
            "Speaker 0: No, first time.\n"
        )
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")
        assert lines[1].startswith("Agent:")

    def test_my_business_class_variation(self):
        """STT variation: 'calling My Business Class' (from production call 3324033166008)."""
        cleaner = TranscriptCleaner(direction="inbound")
        raw = (
            "Speaker 0: Hello?\n"
            "Speaker 1: Hello, thank you for calling My Business Class. "
            "Am I speaking to Mr. Ramiro?\n"
            "Speaker 0: This is Ramiro speaking.\n"
        )
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")
        # "My Business Class" contains "business class" → match
        assert lines[1].startswith("Agent:"), f"Expected Agent:, got {lines[1][:60]}"

    def test_client_mentions_company_no_false_positive(self):
        """Client saying 'I found Business Class online' must NOT become Agent."""
        cleaner = TranscriptCleaner(direction="inbound")
        raw = (
            "Speaker 0: I found Buy Business Class on Google.\n"
            "Speaker 1: Great! How can I help you today?\n"
            "Speaker 0: I need a flight to Paris.\n"
        )
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")
        # Speaker 0 has company name but NO agent behavior → NOT agent
        # Speaker 1 has "how can I help" → Agent
        assert lines[1].startswith("Agent:"), \
            f"Speaker 1 (agent behavior) should be Agent, got {lines[1][:60]}"
        assert lines[0].startswith("Client:"), \
            f"Speaker 0 (customer) should be Client, got {lines[0][:60]}"

    def test_ivr_with_company_not_agent(self):
        """IVR recording mentioning company should NOT make IVR the agent."""
        cleaner = TranscriptCleaner(direction="inbound")
        raw = (
            "Speaker 0: This call is being recorded. Thank you for calling "
            "Buy Business Class.\n"
            "Speaker 1: Hello, how can I help you?\n"
            "Speaker 0: I need a flight.\n"
        )
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")
        # Speaker 0 has IVR content → should be skipped by AUD-04
        # Speaker 1 has agent behavior → Agent
        assert lines[1].startswith("Agent:"), \
            f"Speaker 1 should be Agent, got {lines[1][:60]}"

    def test_outbound_calls_still_work(self):
        """Existing outbound intro detection must NOT be broken."""
        cleaner = TranscriptCleaner(direction="outbound")
        raw = (
            "Speaker 0: Hello?\n"
            "Speaker 1: Hi, my name is Jerome from Buy Business Class.\n"
            "Speaker 0: Yes, hi Jerome.\n"
        )
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")
        assert lines[0].startswith("Client:")
        assert lines[1].startswith("Agent:")
        assert "Jerome" in lines[1]

    def test_no_company_name_fallback_still_works(self):
        """When no company name is present, existing detection chain runs."""
        cleaner = TranscriptCleaner(direction="outbound")
        raw = (
            "Speaker 0: Hello?\n"
            "Speaker 1: Hi, my name is Sarah. I'm calling about your trip.\n"
            "Speaker 0: Oh, okay.\n"
        )
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")
        # "my name is Sarah" triggers existing intro pattern → Agent
        assert lines[1].startswith("Agent:")

    def test_dan_from_business_class_followup(self):
        """Production call 3394666620008: 'it's me, Dan, from business class.'"""
        cleaner = TranscriptCleaner(direction="outbound")
        raw = (
            "Speaker 0: Hello?\n"
            "Speaker 1: Sir, it's me, Dan, from business class.\n"
            "Speaker 0: Yes. Hi.\n"
        )
        result = cleaner.clean(raw)
        lines = result.strip().split("\n")
        # "from business class" → company name + "calling from" pattern → Agent
        assert lines[1].startswith("Agent:"), f"Expected Agent:, got {lines[1][:60]}"


# ═══════════════════════════════════════════════════════════════════
# AUD-08: Passport Pattern Precision Monitoring (P1)
# ═══════════════════════════════════════════════════════════════════

class TestAUD08PassportPatternMonitoring:
    """P1: Passport pattern [A-Z]{1,2}\\d{6,9} may false-positive on airline codes.

    These tests document KNOWN false positives. They are marked as xfail
    because the fix (airline code exemption) is deferred to Phase 3.
    When the fix ships, remove the xfail markers and these tests MUST pass.
    """

    @pytest.mark.xfail(reason="P1: Passport pattern matches airline codes — fix deferred to Phase 3")
    def test_united_airlines_code_not_redacted(self):
        """UA12345678 is a United Airlines reference, NOT a passport."""
        redactor = PIIRedactor()
        result = redactor.redact("Your United confirmation is UA12345678.")
        assert "UA12345678" in result["text"]

    @pytest.mark.xfail(reason="P1: Passport pattern matches airline codes — fix deferred to Phase 3")
    def test_british_airways_code_not_redacted(self):
        """BA1234567 is a British Airways reference, NOT a passport."""
        redactor = PIIRedactor()
        result = redactor.redact("British Airways ref: BA1234567.")
        assert "BA1234567" in result["text"]

    def test_real_passport_still_caught(self):
        """Real passport number must still be redacted."""
        redactor = PIIRedactor()
        result = redactor.redact("My passport number is C12345678.")
        assert "[PASSPORT]" in result["text"]
        assert "C12345678" not in result["text"]
