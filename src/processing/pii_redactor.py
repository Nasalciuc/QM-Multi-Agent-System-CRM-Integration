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

# Phone: (123) 456-7890, 123-456-7890, +1-123-456-7890
_PHONE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:\+?1[-.\ s]?)?"
    r"\(?\d{3}\)?[-.\ s]\d{3}[-.\ s]?\d{4}"
    r"(?!\d)"
)

# CRIT-5: Unseparated 10-digit phone numbers (e.g., 5551234567)
_PHONE_NOSEP_PATTERN = re.compile(
    r"(?<!\d)\d{10}(?!\d)"
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

# #4: Date of birth patterns (MM/DD/YYYY, DD-MM-YYYY, etc.)
_DOB_PATTERN = re.compile(
    r"\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b"
    r"|\b(?:0[1-9]|[12]\d|3[01])[/\-](?:0[1-9]|1[0-2])[/\-](?:19|20)\d{2}\b"
)

# #4: Passport number (letter followed by 6-9 digits)
_PASSPORT_PATTERN = re.compile(
    r"\b[A-Z]\d{6,9}\b"
)

# #4: Street address (number + street name + suffix)
_ADDRESS_PATTERN = re.compile(
    r"\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+"
    r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Drive|Dr|Road|Rd|Lane|Ln|Way|Court|Ct|Place|Pl)\b",
    re.IGNORECASE,
)

# PNR / airline booking reference (6 uppercase alphanumeric, must contain at least 1 digit)
# Common format: exactly 6 characters, mix of uppercase letters and digits
# Requires at least one digit to avoid false positives on common words (LONDON, PLEASE, etc.)
_PNR_PATTERN = re.compile(
    r"\b(?=[A-Z0-9]*[0-9])[A-Z0-9]{6}\b"
)

# HIGH-07: International phone numbers: +XX XXXX XXXXXX (various formats)
_INTL_PHONE_PATTERN = re.compile(
    r'\+\d{1,3}[\s.-]?\(?\d{1,4}\)?[\s.-]?\d{2,4}[\s.-]?\d{2,4}[\s.-]?\d{0,4}'
)

# HIGH-07: Passport numbers with 1-2 letter prefix + 6-8 digits
_INTL_PASSPORT_PATTERN = re.compile(
    r'\b[A-Z]{1,2}\d{6,8}\b'
)

# HIGH-07: Frequent flyer / loyalty program: 2-3 letters + 8-12 digits
_LOYALTY_PATTERN = re.compile(
    r'\b[A-Z]{2,3}[-]?\d{8,12}\b'
)

# Replacement tokens
_REPLACEMENTS = {
    "phone": "[PHONE]",
    "email": "[EMAIL]",
    "credit_card": "[CC_NUMBER]",
    "ssn": "[SSN]",
    "spelled_pii": "[SPELLED_PII]",
    "dob": "[DOB]",
    "passport": "[PASSPORT]",
    "address": "[ADDRESS]",
    "pnr": "[BOOKING_REF]",
    "intl_phone": "[PHONE]",
    "intl_passport": "[PASSPORT]",
    "loyalty": "[LOYALTY_ID]",
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
        redact_dob: bool = True,
        redact_passport: bool = True,
        redact_address: bool = True,
        redact_pnr: bool = True,
        redact_intl_phone: bool = True,
        redact_intl_passport: bool = True,
        redact_loyalty: bool = True,
    ):
        self.redact_phones = redact_phones
        self.redact_emails = redact_emails
        self.redact_credit_cards = redact_credit_cards
        self.redact_ssn = redact_ssn
        self.redact_spelled_pii = redact_spelled_pii
        self.redact_dob = redact_dob
        self.redact_passport = redact_passport
        self.redact_address = redact_address
        self.redact_pnr = redact_pnr
        self.redact_intl_phone = redact_intl_phone
        self.redact_intl_passport = redact_intl_passport
        self.redact_loyalty = redact_loyalty

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

        counts = {"phone": 0, "email": 0, "credit_card": 0, "ssn": 0,
                  "spelled_pii": 0, "dob": 0, "passport": 0, "address": 0, "pnr": 0,
                  "intl_phone": 0, "intl_passport": 0, "loyalty": 0}

        # Order matters: SSN before phone (SSN is more specific)
        if self.redact_ssn:
            text, n = _SSN_PATTERN.subn(_REPLACEMENTS["ssn"], text)
            counts["ssn"] = n

        if self.redact_credit_cards:
            text, n = _CC_PATTERN.subn(_REPLACEMENTS["credit_card"], text)
            counts["credit_card"] = n

        # #4: DOB before phone (date patterns can overlap with phone)
        if self.redact_dob:
            text, n = _DOB_PATTERN.subn(_REPLACEMENTS["dob"], text)
            counts["dob"] = n

        if self.redact_emails:
            text, n = _EMAIL_PATTERN.subn(_REPLACEMENTS["email"], text)
            counts["email"] = n

        # HIGH-07: International phone BEFORE domestic phone (more specific)
        if self.redact_intl_phone:
            text, n = _INTL_PHONE_PATTERN.subn(_REPLACEMENTS["intl_phone"], text)
            counts["intl_phone"] = n

        if self.redact_phones:
            text, n = _PHONE_PATTERN.subn(_REPLACEMENTS["phone"], text)
            counts["phone"] = n
            # CRIT-5: Also catch unseparated 10-digit numbers
            text, n2 = _PHONE_NOSEP_PATTERN.subn(_REPLACEMENTS["phone"], text)
            counts["phone"] += n2

        # HIGH-07: International passport (1-2 letter prefix)
        if self.redact_intl_passport:
            text, n = _INTL_PASSPORT_PATTERN.subn(_REPLACEMENTS["intl_passport"], text)
            counts["intl_passport"] = n

        # #4: Passport numbers
        if self.redact_passport:
            text, n = _PASSPORT_PATTERN.subn(_REPLACEMENTS["passport"], text)
            counts["passport"] = n

        # HIGH-07: Loyalty / frequent flyer numbers
        if self.redact_loyalty:
            text, n = _LOYALTY_PATTERN.subn(_REPLACEMENTS["loyalty"], text)
            counts["loyalty"] = n

        # #4: Street addresses
        if self.redact_address:
            text, n = _ADDRESS_PATTERN.subn(_REPLACEMENTS["address"], text)
            counts["address"] = n

        # PNR / booking references (after address, before spelled_pii)
        if self.redact_pnr:
            text, n = _PNR_PATTERN.subn(_REPLACEMENTS["pnr"], text)
            counts["pnr"] = n

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
