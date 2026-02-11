"""
Response Parser

Extracts and validates JSON from LLM response text.
Handles both clean JSON and markdown code blocks.
Validates criteria keys and score values.

Extracted from the inline parsing in agent_03.evaluate_call().
"""

import json
import re
import logging
from typing import Dict, Set, Optional

logger = logging.getLogger("qa_system.inference")

VALID_SCORES = {"YES", "PARTIAL", "NO", "N/A"}


class ValidationError(Exception):
    """Raised when LLM response fails validation."""

    def __init__(self, message: str, missing_keys: list = None, invalid_keys: list = None):
        super().__init__(message)
        self.missing_keys = missing_keys or []
        self.invalid_keys = invalid_keys or []


class ResponseParser:
    """Parse and validate LLM evaluation responses.

    Usage:
        parser = ResponseParser(expected_keys={"greeting_prepared", "contact_info", ...})
        result = parser.parse(llm_response_text)
        # result is a validated dict with normalized scores
    """

    def __init__(self, expected_keys: Optional[Set[str]] = None):
        """
        Args:
            expected_keys: Set of criteria keys that must be present in the response.
                           If None, only JSON structure is validated.
        """
        self.expected_keys = expected_keys or set()

    def parse(self, text: str) -> Dict:
        """Extract JSON from LLM text, validate, and normalize.

        Args:
            text: Raw LLM response text.

        Returns:
            Validated evaluation dict with normalized scores.

        Raises:
            ValidationError: If JSON parsing or validation fails.
        """
        # 1. Extract JSON
        data = self._extract_json(text)

        # 2. Validate structure
        self._validate_structure(data)

        # 3. Validate criteria keys
        criteria = data.get("criteria", {})
        if self.expected_keys:
            self._validate_keys(criteria)

        # 4. Validate and normalize scores
        self._validate_scores(criteria)

        return data

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from text, handling markdown code blocks."""
        text = text.strip()

        # Try direct JSON parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        patterns = [
            r"```json\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
            r"\{.*\}",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    candidate = match.group(1) if match.lastindex else match.group(0)
                    return json.loads(candidate)
                except (json.JSONDecodeError, IndexError):
                    continue

        raise ValidationError(f"Could not extract valid JSON from response ({len(text)} chars)")

    @staticmethod
    def _validate_structure(data: Dict) -> None:
        """Validate top-level structure of evaluation response."""
        if not isinstance(data, dict):
            raise ValidationError(f"Expected dict, got {type(data).__name__}")

        required_fields = ["criteria", "overall_assessment", "strengths", "improvements"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValidationError(f"Missing required fields: {missing}")

        if not isinstance(data["criteria"], dict):
            raise ValidationError("'criteria' must be a dict")

    def _validate_keys(self, criteria: Dict) -> None:
        """Validate that all expected criteria keys are present."""
        present = set(criteria.keys())
        missing = list(self.expected_keys - present)
        unexpected = list(present - self.expected_keys)

        if missing:
            raise ValidationError(
                f"Missing {len(missing)} criteria keys: {missing[:5]}{'...' if len(missing) > 5 else ''}",
                missing_keys=missing,
            )

        if unexpected:
            logger.warning(f"Unexpected criteria keys in response: {unexpected}")

    @staticmethod
    def _validate_scores(criteria: Dict) -> None:
        """Validate all scores are in valid set and normalize to uppercase."""
        invalid = []
        for key, value in criteria.items():
            if not isinstance(value, dict):
                invalid.append(key)
                continue
            score = value.get("score", "")
            if score.upper() not in VALID_SCORES:
                invalid.append(key)
            else:
                # Normalize to uppercase
                value["score"] = score.upper()

        if invalid:
            raise ValidationError(
                f"Invalid scores for criteria: {invalid}",
                invalid_keys=invalid,
            )
