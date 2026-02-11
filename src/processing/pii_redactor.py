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
"""

import re
import logging
from typing import Dict

logger = logging.getLogger("qa_system.processing")

# ── Regex patterns ──────────────────────────────────────────────────

# Phone: (123) 456-7890, 123-456-7890, +1-123-456-7890, 1234567890
_PHONE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:\+?1[-.\s]?)?"
    r"(?:\(?\d{3}\)?[-.\s]?)"
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

# Replacement tokens
_REPLACEMENTS = {
    "phone": "[PHONE]",
    "email": "[EMAIL]",
    "credit_card": "[CC_NUMBER]",
    "ssn": "[SSN]",
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
    ):
        self.redact_phones = redact_phones
        self.redact_emails = redact_emails
        self.redact_credit_cards = redact_credit_cards
        self.redact_ssn = redact_ssn

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

        counts = {"phone": 0, "email": 0, "credit_card": 0, "ssn": 0}

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

        total = sum(counts.values())
        if total > 0:
            logger.info(f"PII redacted: {counts} (total: {total})")

        return {
            "text": text,
            "pii_found": counts,
            "total_redactions": total,
        }
