"""
Tests for src/inference/response_parser.py

Tests JSON extraction, validation, and score normalization.
"""
import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference.response_parser import ResponseParser, ValidationError


# --- Fixtures ---

@pytest.fixture
def parser():
    return ResponseParser(expected_keys={"greeting", "contact_info", "tone"})


@pytest.fixture
def parser_no_keys():
    return ResponseParser()


def _valid_response(criteria=None):
    """Build a valid evaluation response."""
    if criteria is None:
        criteria = {
            "greeting": {"score": "YES", "evidence": "Good greeting."},
            "contact_info": {"score": "PARTIAL", "evidence": "Email only."},
            "tone": {"score": "NO", "evidence": "Rude tone."},
        }
    return json.dumps({
        "criteria": criteria,
        "overall_assessment": "Average call.",
        "strengths": ["greeting"],
        "improvements": ["tone"],
    })


# --- Tests: JSON Extraction ---

class TestJsonExtraction:

    def test_parse_clean_json(self, parser):
        result = parser.parse(_valid_response())
        assert "criteria" in result
        assert len(result["criteria"]) == 3

    def test_parse_markdown_code_block(self, parser):
        text = "```json\n" + _valid_response() + "\n```"
        result = parser.parse(text)
        assert "criteria" in result

    def test_parse_markdown_no_lang(self, parser):
        text = "Here's the evaluation:\n```\n" + _valid_response() + "\n```"
        result = parser.parse(text)
        assert "criteria" in result

    def test_parse_invalid_json_raises(self, parser):
        with pytest.raises(ValidationError, match="Could not extract"):
            parser.parse("This is not JSON at all")

    def test_parse_empty_raises(self, parser):
        with pytest.raises(ValidationError):
            parser.parse("")

    def test_parse_nested_json_in_text(self, parser):
        text = "Some preamble\n" + _valid_response() + "\nSome epilogue"
        result = parser.parse(text)
        assert "criteria" in result


# --- Tests: Structure Validation ---

class TestStructureValidation:

    def test_missing_criteria_field(self, parser):
        text = json.dumps({"overall_assessment": "OK", "strengths": [], "improvements": []})
        with pytest.raises(ValidationError, match="Missing required"):
            parser.parse(text)

    def test_missing_assessment(self, parser):
        text = json.dumps({
            "criteria": {"greeting": {"score": "YES", "evidence": "ok"}},
            "strengths": [], "improvements": [],
        })
        with pytest.raises(ValidationError, match="Missing required"):
            parser.parse(text)

    def test_criteria_not_dict(self, parser):
        text = json.dumps({
            "criteria": ["list", "not", "dict"],
            "overall_assessment": "OK",
            "strengths": [], "improvements": [],
        })
        with pytest.raises(ValidationError, match="must be a dict"):
            parser.parse(text)


# --- Tests: Key Validation ---

class TestKeyValidation:

    def test_missing_expected_keys_raises(self, parser):
        incomplete = json.dumps({
            "criteria": {"greeting": {"score": "YES", "evidence": "ok"}},
            "overall_assessment": "OK", "strengths": [], "improvements": [],
        })
        with pytest.raises(ValidationError, match="Missing.*criteria"):
            parser.parse(incomplete)

    def test_all_keys_present_passes(self, parser):
        result = parser.parse(_valid_response())
        assert len(result["criteria"]) == 3

    def test_no_expected_keys_skips_validation(self, parser_no_keys):
        """Without expected_keys, any keys are accepted."""
        text = json.dumps({
            "criteria": {"random_key": {"score": "YES", "evidence": "ok"}},
            "overall_assessment": "OK", "strengths": [], "improvements": [],
        })
        result = parser_no_keys.parse(text)
        assert "random_key" in result["criteria"]


# --- Tests: Score Validation ---

class TestScoreValidation:

    def test_valid_scores_normalized_to_upper(self, parser):
        criteria = {
            "greeting": {"score": "yes", "evidence": "ok"},
            "contact_info": {"score": "partial", "evidence": "ok"},
            "tone": {"score": "n/a", "evidence": "ok"},
        }
        result = parser.parse(_valid_response(criteria))
        assert result["criteria"]["greeting"]["score"] == "YES"
        assert result["criteria"]["contact_info"]["score"] == "PARTIAL"
        assert result["criteria"]["tone"]["score"] == "N/A"

    def test_invalid_score_raises(self, parser):
        criteria = {
            "greeting": {"score": "MAYBE", "evidence": "ok"},
            "contact_info": {"score": "YES", "evidence": "ok"},
            "tone": {"score": "YES", "evidence": "ok"},
        }
        with pytest.raises(ValidationError, match="Invalid scores"):
            parser.parse(_valid_response(criteria))

    def test_criterion_not_dict_raises(self, parser):
        criteria = {
            "greeting": "just_a_string",
            "contact_info": {"score": "YES", "evidence": "ok"},
            "tone": {"score": "YES", "evidence": "ok"},
        }
        with pytest.raises(ValidationError, match="Invalid scores"):
            parser.parse(_valid_response(criteria))


# --- Tests: ValidationError Properties ---

class TestValidationError:

    def test_has_missing_keys(self):
        e = ValidationError("test", missing_keys=["a", "b"])
        assert e.missing_keys == ["a", "b"]
        assert e.invalid_keys == []

    def test_has_invalid_keys(self):
        e = ValidationError("test", invalid_keys=["c"])
        assert e.invalid_keys == ["c"]
        assert e.missing_keys == []
