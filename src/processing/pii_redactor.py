"""
PII Redactor

Detects and masks personally identifiable information in transcripts
before sending them to third-party LLM APIs.

Call recordings contain sensitive customer data — sending raw PII
to external APIs is a privacy/compliance risk.

Patterns detected:
  - Phone numbers (US/international)
  - Email addresses
  - Credit card numbers (Visa, Mastercard, Amex, Discover)
  - Social Security Numbers (SSN)
  - Spelled-out emails (letter-dash-letter sequences with "at"/"dot")
  - Spelled-out phone numbers (digit words like "five five five...")

KNOWN LIMITATIONS (CRIT-3):
  - Spelled-out names (e.g. "J-O-H-N") without email context are NOT redacted.
    For robust name detection, consider integrating spaCy NER or Microsoft Presidio.
  - International phone formats beyond +1 (US/Canada) may not be caught.
  - Custom account/ID numbers read aloud are not detected.
  - PII embedded in URLs or complex strings may be missed.
"""

import re
import logging
from typing import Dict

logger = logging.getLogger("qa_system.processing")

# ── Regex patterns ──────────────────────────────────────────────────

# Phone: (123) 456-7890, 123-456-7890, +1-123-456-7890, 1234567890
# MED-2: Require separator between groups to reduce false positives on order numbers
_PHONE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:\+?1[-.\s]?)?"
    r"(?:\(?\d{3}\)?[-.\s])"              # area code MUST have separator
    r"\d{3}[-.\s]?\d{4}"
    r"(?!\d)"
)

# Email: user@domain.tld
_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# Credit card: 4 groups of 4 digits (Visa/MC/Discover) or Amex pattern
_CC_PATTERN = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b"  # 16-digit cards
    r"|\b\d{4}[-\s]?\d{6}[-\s]?\d{5}\b"  # 15-digit Amex
)

# SSN: 123-45-6789 or 123 45 6789
_SSN_PATTERN = re.compile(
    r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"
)

# Spelled-out email: J-O-H-N dot S-M-I-T-H at email dot com
# Matches sequences of single letters separated by dashes near email-like words
_SPELLED_EMAIL_PATTERN = re.compile(
    r"(?:[A-Za-z][\s-]){2,}[A-Za-z]"           # spelled-out part (e.g. J-O-H-N)
    r"(?:\s+(?:dot|at|period|underscore)\s+"     # connector words
    r"(?:[A-Za-z][\s-])*[A-Za-z])*"             # more spelled-out parts
    r"(?:\s+(?:dot|at|period)\s+\w+)+",          # domain part
    re.IGNORECASE,
)

# Spelled-out phone: five five five one two three four five six seven
_SPELLED_PHONE_WORDS = {
    "zero", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "oh",
}
_SPELLED_PHONE_PATTERN = re.compile(
    r"\b(?:(?:" + "|".join(_SPELLED_PHONE_WORDS) + r")[\s,.-]+){6,}\b"
    r"(?:" + "|".join(_SPELLED_PHONE_WORDS) + r")\b",
    re.IGNORECASE,
)

# Replacement tokens
_REPLACEMENTS = {
    "phone": "[PHONE]",
    "email": "[EMAIL]",
    "credit_card": "[CC_NUMBER]",
    "ssn": "[SSN]",
    "spelled_pii": "[SPELLED_PII]",
}


class PIIRedactor:
    """Detect and mask PII in transcript text.

    Usage:
        redactor = PIIRedactor()
        result = redactor.redact("Call me at 555-123-4567")
        print(result["text"])      # "Call me at [PHONE]"
        print(result["pii_found"]) # {"phone": 1, "email": 0, ...}
    """

    def __init__(
        self,
        redact_phones: bool = True,
        redact_emails: bool = True,
        redact_credit_cards: bool = True,
        redact_ssn: bool = True,
        redact_spelled_pii: bool = True,
    ):
        self.redact_phones = redact_phones
        self.redact_emails = redact_emails
        self.redact_credit_cards = redact_credit_cards
        self.redact_ssn = redact_ssn
        self.redact_spelled_pii = redact_spelled_pii

    def redact(self, text: str) -> Dict:
        """Redact all configured PII types from text.

        Args:
            text: Input text (transcript).

        Returns:
            Dict with:
              - text: Redacted text
              - pii_found: Count of each PII type found
              - total_redactions: Total number of redactions made
        """
        if not text:
            return {"text": "", "pii_found": {}, "total_redactions": 0}

        counts = {"phone": 0, "email": 0, "credit_card": 0, "ssn": 0, "spelled_pii": 0}

        # Order matters: SSN before phone (SSN is more specific)
        if self.redact_ssn:
            text, n = _SSN_PATTERN.subn(_REPLACEMENTS["ssn"], text)
            counts["ssn"] = n

        if self.redact_credit_cards:
            text, n = _CC_PATTERN.subn(_REPLACEMENTS["credit_card"], text)
            counts["credit_card"] = n

        if self.redact_emails:
            text, n = _EMAIL_PATTERN.subn(_REPLACEMENTS["email"], text)
            counts["email"] = n

        if self.redact_phones:
            text, n = _PHONE_PATTERN.subn(_REPLACEMENTS["phone"], text)
            counts["phone"] = n

        # CRIT-3: Spelled-out PII (common in call center recordings)
        if self.redact_spelled_pii:
            text, n = _SPELLED_EMAIL_PATTERN.subn(_REPLACEMENTS["spelled_pii"], text)
            counts["spelled_pii"] += n
            text, n = _SPELLED_PHONE_PATTERN.subn(_REPLACEMENTS["spelled_pii"], text)
            counts["spelled_pii"] += n

        total = sum(counts.values())
        if total > 0:
            logger.info(f"PII redacted: {counts} (total: {total})")

        return {
            "text": text,
            "pii_found": counts,
            "total_redactions": total,
        }
