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
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


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
    """Mock ElevenLabs client with Scribe v2 word-level diarization."""
    client = MagicMock()
    result = MagicMock()
    result.text = "Hello, how can I help? I need assistance."
    result.language_code = "en"
    # Scribe v2 word-level diarization format
    def w0(t, s="speaker_0", tp="word"):
        return MagicMock(text=t, speaker_id=s, type=tp)

    def w1(t, s="speaker_1", tp="word"):
        return MagicMock(text=t, speaker_id=s, type=tp)

    def sp0():
        return MagicMock(text=" ", speaker_id="speaker_0", type="spacing")

    def sp1():
        return MagicMock(text=" ", speaker_id="speaker_1", type="spacing")
    result.words = [
        w0("Hello"), w0(",", tp="punctuation"), sp0(),
        w0("how"), sp0(), w0("can"), sp0(), w0("I"), sp0(), w0("help"),
        w0("?", tp="punctuation"),
        w1("I"), sp1(), w1("need"), sp1(), w1("assistance"),
        w1(".", tp="punctuation"),
    ]
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
                    "opening": {"score": 80.0, "count": 5},
                    "interview": {"score": 70.0, "count": 8},
                    "psychological_framing": {"score": 60.0, "count": 3},
                    "communication": {"score": 85.0, "count": 8},
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
        "greeting_prepared": {"category": "opening", "weight": 1.0},
        "contact_info": {"category": "opening", "weight": 1.0},
    }
