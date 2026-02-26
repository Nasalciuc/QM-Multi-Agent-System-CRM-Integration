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
# NEW-05: Negative lookbehinds reject order/account/confirmation/booking/reference/id
#         contexts to reduce false positives on numeric identifiers.
_PHONE_NOSEP_PATTERN = re.compile(
    r"(?<!order )(?<!account )(?<!confirmation )(?<!booking )"
    r"(?<!reference )(?<!number )(?<!id )(?<!#)"
    r"(?<!\d)\d{10}(?!\d)",
    re.IGNORECASE,
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

# #4: Passport number — NEW-11: Merged with _INTL_PASSPORT_PATTERN
# Covers 1-2 letter prefix + 6-9 digits (handles both domestic & international)
_PASSPORT_PATTERN = re.compile(
    r'\b[A-Z]{1,2}\d{6,9}\b'
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

# REAL-08: Common airline/airport codes to exempt from PNR redaction
_AVIATION_CODES = {
    # Major airlines
    "EVA", "ANA", "JAL", "SIA", "CPA", "KAL", "AAL", "DAL", "UAL",
    # Major airports (IATA 3-letter codes are only 3 chars, won't match PNR,
    # but flight numbers like "EVA123" can)
}

# REAL-04: NATO / phonetic alphabet spelling — "D as in Denver, A, Alpha, N Nancy"
_NATO_SPELLED_PATTERN = re.compile(
    r"(?:[A-Za-z]\s*(?:,\s*)?(?:as in|for|like)\s+\w+[,.\s]*){4,}",
    re.IGNORECASE,
)

# REAL-04: Simple letter enumeration — "D, A, N, N, Y" (4+ single uppercase letters)
_LETTER_ENUM_PATTERN = re.compile(
    r"(?<!\w)(?:[A-Z]\s*,\s*){4,}[A-Z](?!\w)"
)

# HIGH-07: International phone numbers: +XX XXXX XXXXXX (various formats)
_INTL_PHONE_PATTERN = re.compile(
    r'\+\d{1,3}[\s.-]?\(?\d{1,4}\)?[\s.-]?\d{2,4}[\s.-]?\d{2,4}[\s.-]?\d{0,4}'
)

# HIGH-07: Frequent flyer / loyalty program: 2-3 letters + 8-12 digits
_LOYALTY_PATTERN = re.compile(
    r'\b[A-Z]{2,3}[-]?\d{8,12}\b'
)

# REAL-05: Spoken digit words for multi-line phone reconstruction
_DIGIT_WORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(_SPELLED_PHONE_WORDS) + r")\b",
    re.IGNORECASE,
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
    "loyalty": "[LOYALTY_ID]",
    "nato_spelled": "[SPELLED_PII]",
    "multiline_phone": "[PHONE]",
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
        redact_loyalty: bool = True,
        redact_nato_spelled: bool = True,
        redact_multiline_phone: bool = False,
        redact_prices: bool = False,
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
        self.redact_loyalty = redact_loyalty
        self.redact_nato_spelled = redact_nato_spelled
        self.redact_multiline_phone = redact_multiline_phone
        self.redact_prices = redact_prices

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
                  "intl_phone": 0, "loyalty": 0, "nato_spelled": 0,
                  "multiline_phone": 0}

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

        # #4: Passport numbers — NEW-11: unified pattern handles both domestic & intl
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
        # REAL-08: Exempt known aviation codes
        if self.redact_pnr:
            text, n = self._redact_pnr(text)
            counts["pnr"] = n

        # CRIT-3: Spelled-out PII (common in call center recordings)
        if self.redact_spelled_pii:
            text, n = _SPELLED_EMAIL_PATTERN.subn(_REPLACEMENTS["spelled_pii"], text)
            counts["spelled_pii"] += n
            text, n = _SPELLED_PHONE_PATTERN.subn(_REPLACEMENTS["spelled_pii"], text)
            counts["spelled_pii"] += n

        # REAL-04: NATO / phonetic alphabet spelling (e.g. "D as in Denver, A Alpha")
        if self.redact_nato_spelled:
            text, n = _NATO_SPELLED_PATTERN.subn(_REPLACEMENTS["nato_spelled"], text)
            counts["nato_spelled"] += n
            text, n = _LETTER_ENUM_PATTERN.subn(_REPLACEMENTS["nato_spelled"], text)
            counts["nato_spelled"] += n

        # REAL-05: Multi-line spoken phone numbers
        if self.redact_multiline_phone:
            text, n = self._redact_multiline_spoken_phone(text)
            counts["multiline_phone"] = n

        # REAL-10: Price / monetary amount redaction (off by default)
        if self.redact_prices:
            text, n = self._redact_prices(text)
            counts["price"] = n

        # B3-FIX-3: Ensure redaction tags don't concatenate with following text
        text = re.sub(r'\]([A-Za-z])', '] \\1', text)

        total = sum(counts.values())
        if total > 0:
            logger.info(f"PII redacted: {counts} (total: {total})")

        return {
            "text": text,
            "pii_found": counts,
            "total_redactions": total,
        }

    @staticmethod
    def _redact_pnr(text: str) -> tuple:
        """REAL-08: Redact PNR codes but exempt known aviation prefixes.

        Returns (redacted_text, count).
        """
        count = 0

        def _pnr_replacer(m: re.Match) -> str:
            nonlocal count
            token = m.group(0)
            # Check if the first 3 chars are a known aviation code
            prefix3 = token[:3].upper()
            if prefix3 in _AVIATION_CODES:
                return token  # exempt
            count += 1
            return _REPLACEMENTS["pnr"]

        text = _PNR_PATTERN.sub(_pnr_replacer, text)
        return text, count

    @staticmethod
    def _redact_multiline_spoken_phone(text: str) -> tuple:
        """REAL-05: Detect phone numbers spoken across multiple lines by same speaker.

        Scans for sequences of 7-10+ digit-words within 5 consecutive lines
        of the same speaker and redacts the digit words.

        Returns (redacted_text, count).
        """
        lines = text.split("\n")
        speaker_re = re.compile(r"^((?:Agent|Client|Speaker\s*\d+)\s*:)\s*(.+)", re.IGNORECASE)
        redacted_count = 0

        # Build list of (speaker, content, line_index) for speaker lines
        speaker_lines: list = []
        for idx, line in enumerate(lines):
            m = speaker_re.match(line)
            if m:
                speaker_lines.append((m.group(1).lower().split(":")[0].strip(), m.group(2), idx))

        # Sliding window: scan consecutive lines of same speaker for digit words
        i = 0
        lines_to_redact: set = set()
        while i < len(speaker_lines):
            speaker = speaker_lines[i][0]
            # Collect consecutive window (up to 8 lines, same speaker, within 5 original lines)
            window = [speaker_lines[i]]
            j = i + 1
            while j < len(speaker_lines) and j - i < 8:
                if speaker_lines[j][2] - speaker_lines[i][2] > 8:
                    break
                if speaker_lines[j][0] == speaker:
                    window.append(speaker_lines[j])
                j += 1

            # Count digit words across window
            digit_word_count = 0
            window_indices = []
            for spk, content, line_idx in window:
                dw = _DIGIT_WORD_PATTERN.findall(content)
                if dw:
                    digit_word_count += len(dw)
                    window_indices.append(line_idx)

            if digit_word_count >= 7:
                lines_to_redact.update(window_indices)
            i += 1

        # Redact digit words in flagged lines
        if lines_to_redact:
            for idx in lines_to_redact:
                original = lines[idx]
                redacted = _DIGIT_WORD_PATTERN.sub("[PHONE]", original)
                if redacted != original:
                    lines[idx] = redacted
                    redacted_count += 1
            text = "\n".join(lines)

        return text, redacted_count

    @staticmethod
    def _redact_prices(text: str) -> tuple:
        """REAL-10: Redact monetary amounts (opt-in, off by default).

        Catches: $1,234  /  $1,234.56  /  "four thousand seven hundred"

        Returns (redacted_text, count).
        """
        count = 0
        # Numeric dollar amounts: $1,234 or $1,234.56
        dollar_pattern = re.compile(r"\$[\d,]+(?:\.\d{2})?")
        text, n = dollar_pattern.subn("[PRICE]", text)
        count += n

        # Verbal amounts: "X thousand Y hundred" patterns
        verbal_pattern = re.compile(
            r"\b\w+\s+thousand(?:\s+\w+\s+hundred)?(?:\s+(?:and\s+)?\w+)?\s*(?:dollars?)?\b",
            re.IGNORECASE,
        )
        text, n = verbal_pattern.subn("[PRICE]", text)
        count += n

        return text, count
