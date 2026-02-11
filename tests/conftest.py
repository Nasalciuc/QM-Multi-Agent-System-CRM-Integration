"""
Shared pytest fixtures for the QA System test suite.

Provides:
  - Path fixtures (project root, config paths)
  - Mock clients (OpenAI, ElevenLabs)
  - Sample data loaders (transcript, evaluation)
  - Temporary output directories
  - QA agent with mocked dependencies
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src/ is on the import path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ── Path fixtures ───────────────────────────────────────────────────

@pytest.fixture
def project_root():
    """Absolute path to the project root."""
    return PROJECT_ROOT


@pytest.fixture
def criteria_path():
    """Path to qa_criteria.yaml config."""
    return str(PROJECT_ROOT / "config" / "qa_criteria.yaml")


@pytest.fixture
def fixtures_dir():
    """Path to tests/fixtures/ directory."""
    return PROJECT_ROOT / "tests" / "fixtures"


# ── Sample data ─────────────────────────────────────────────────────

@pytest.fixture
def sample_transcript(fixtures_dir):
    """Load the sample transcript from fixtures."""
    path = fixtures_dir / "sample_transcript.txt"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def sample_evaluation(fixtures_dir):
    """Load the sample evaluation JSON from fixtures."""
    path = fixtures_dir / "sample_evaluation.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Mock clients ────────────────────────────────────────────────────

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client that returns a canned completion."""
    client = MagicMock()
    return client


def _make_completion_response(text: str, input_tokens: int = 5000, output_tokens: int = 1500):
    """Helper: build a mock OpenAI ChatCompletion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.usage = MagicMock()
    response.usage.prompt_tokens = input_tokens
    response.usage.completion_tokens = output_tokens
    return response


@pytest.fixture
def mock_elevenlabs_client():
    """Mock ElevenLabs client."""
    client = MagicMock()
    result = MagicMock()
    result.text = "Speaker 0: Hello, how can I help?\nSpeaker 1: I need assistance."
    client.speech_to_text.convert.return_value = result
    return client


# ── Criteria fixture ────────────────────────────────────────────────

@pytest.fixture
def criteria_dict(criteria_path):
    """Load real criteria from YAML."""
    from utils import load_criteria
    return load_criteria(criteria_path)


# ── Temporary output ────────────────────────────────────────────────

@pytest.fixture
def tmp_output(tmp_path):
    """Temporary output directory for export tests."""
    return str(tmp_path / "exports")


# ── Full evaluation result for pipeline/export tests ────────────────

@pytest.fixture
def full_evaluation_list():
    """A list with one complete evaluation dict, as produced by the pipeline."""
    return [
        {
            "filename": "call1.mp3",
            "transcript": "Agent: Hello\nClient: Hi",
            "duration_min": 5.0,
            "call_type": "First Call",
            "overall_score": 75.0,
            "score_data": {
                "overall_score": 75.0,
                "category_scores": {
                    "phone_skills": {"score": 80.0, "count": 5},
                    "sales_techniques": {"score": 70.0, "count": 8},
                    "urgency_closing": {"score": 60.0, "count": 3},
                    "soft_skills": {"score": 85.0, "count": 8},
                },
                "score_breakdown": {
                    "yes_count": 10, "partial_count": 8,
                    "no_count": 4, "na_count": 2,
                },
            },
            "criteria": {
                "greeting_prepared": {"score": "YES", "evidence": "Good greeting."},
                "contact_info": {"score": "PARTIAL", "evidence": "Email only."},
            },
            "overall_assessment": "Decent call.",
            "strengths": ["Good tone", "Professional", "Product knowledge"],
            "improvements": ["Ask budget", "Create urgency", "Close better"],
            "critical_gaps": [],
            "model_used": "gpt-4o-2024-11-20",
            "tokens_used": {"input": 5000, "output": 1500},
            "cost_usd": 0.0275,
            "status": "Success",
        }
    ]


@pytest.fixture
def criteria_ref():
    """Minimal criteria reference dict for export tests."""
    return {
        "greeting_prepared": {"category": "phone_skills", "weight": 1.0},
        "contact_info": {"category": "phone_skills", "weight": 1.0},
    }
