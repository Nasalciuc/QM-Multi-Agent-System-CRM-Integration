"""
Tests for Agent 3: Quality Management Evaluation

Rewritten for the new architecture:
  - QualityManagementAgent now takes ModelFactory instead of raw OpenAI client.
  - Heavy mocking of ModelFactory + InferenceEngine for unit tests.
  - Score calculation and call-type detection are pure-logic tests (no mocks).
"""
import sys
import os
import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# --- Fixtures ---

@pytest.fixture
def criteria_path():
    return os.path.join(os.path.dirname(__file__), '..', 'config', 'qa_criteria.yaml')


@pytest.fixture
def mock_model_factory():
    """Create a fully-mocked ModelFactory."""
    factory = MagicMock()
    factory.primary.model_name = "gpt-4o-2024-11-20"
    factory.primary.provider_name = "openrouter"
    factory.token_limits = {
        "max_input_tokens": 30000,
        "max_output_tokens": 4096,
        "cost_warning_threshold_usd": 0.50,
    }
    factory.primary_pricing = {
        "input_per_1m": 2.50,
        "output_per_1m": 10.00,
    }
    return factory


@pytest.fixture
def agent(mock_model_factory, criteria_path, tmp_path):
    """Build QualityManagementAgent with mocked factory and temp cache dir."""
    with patch("agents.agent_03_evaluation.ModelFactory", autospec=True):
        from agents.agent_03_evaluation import QualityManagementAgent
        return QualityManagementAgent(
            model_factory=mock_model_factory,
            criteria_path=criteria_path,
            cache_dir=str(tmp_path / "cache"),
            enable_cache=False,
        )


@pytest.fixture
def all_yes_evaluation(agent):
    """Evaluation dict with all 24 criteria scored YES."""
    return {
        "criteria": {
            key: {"score": "YES", "evidence": "Agent did this well."}
            for key in agent.EVALUATION_CRITERIA
        }
    }


@pytest.fixture
def all_no_evaluation(agent):
    """Evaluation dict with all 24 criteria scored NO."""
    return {
        "criteria": {
            key: {"score": "NO", "evidence": "Not observed."}
            for key in agent.EVALUATION_CRITERIA
        }
    }


@pytest.fixture
def mixed_evaluation(agent):
    """Evaluation with rotating YES/PARTIAL/NO pattern."""
    evaluation = {"criteria": {}}
    for i, key in enumerate(agent.EVALUATION_CRITERIA):
        if i % 3 == 0:
            evaluation["criteria"][key] = {"score": "YES", "evidence": "Done."}
        elif i % 3 == 1:
            evaluation["criteria"][key] = {"score": "PARTIAL", "evidence": "Half."}
        else:
            evaluation["criteria"][key] = {"score": "NO", "evidence": "Missing."}
    return evaluation


# --- Tests: Criteria Loading ---

class TestCriteriaLoading:

    def test_load_24_criteria(self, agent):
        """All 24 criteria should be loaded from YAML."""
        assert len(agent.EVALUATION_CRITERIA) == 24

    def test_criteria_have_required_fields(self, agent):
        """Each criterion must have description, category, weight, first_call_only."""
        for key, crit in agent.EVALUATION_CRITERIA.items():
            assert "description" in crit, f"Missing description for {key}"
            assert "category" in crit, f"Missing category for {key}"
            assert "weight" in crit, f"Missing weight for {key}"
            assert "first_call_only" in crit, f"Missing first_call_only for {key}"

    def test_criteria_categories(self, agent):
        """Should have exactly 4 categories."""
        categories = set(c["category"] for c in agent.EVALUATION_CRITERIA.values())
        assert categories == {"phone_skills", "sales_techniques", "urgency_closing", "soft_skills"}

    def test_criteria_category_counts(self, agent):
        """Phone: 5, Sales: 8, Urgency: 3, Soft: 8."""
        counts = {}
        for crit in agent.EVALUATION_CRITERIA.values():
            cat = crit["category"]
            counts[cat] = counts.get(cat, 0) + 1
        assert counts["phone_skills"] == 5
        assert counts["sales_techniques"] == 8
        assert counts["urgency_closing"] == 3
        assert counts["soft_skills"] == 8


# --- Tests: Call Type Detection ---

class TestCallTypeDetection:

    def test_first_call(self, agent):
        is_followup, call_type = agent.detect_call_type("customer_call_jan15.mp3")
        assert not is_followup
        assert call_type == "First Call"

    def test_followup_call_2nd(self, agent):
        is_followup, call_type = agent.detect_call_type("customer_2nd_call.mp3")
        assert is_followup
        assert call_type == "Follow-up Call"

    def test_followup_call_follow(self, agent):
        is_followup, call_type = agent.detect_call_type("john_follow-up.wav")
        assert is_followup
        assert call_type == "Follow-up Call"

    def test_followup_case_insensitive(self, agent):
        is_followup, call_type = agent.detect_call_type("FOLLOW_UP_call.mp3")
        assert is_followup

    def test_regular_filename(self, agent):
        is_followup, call_type = agent.detect_call_type("recording_20240115.mp3")
        assert not is_followup
        assert call_type == "First Call"

    def test_metadata_followup(self, agent):
        """HIGH-NEW-5: Metadata with 'follow-up' result should detect follow-up."""
        is_followup, call_type = agent.detect_call_type(
            "12345_20250201.mp3",
            metadata={"result": "follow-up"},
        )
        assert is_followup
        assert call_type == "Follow-up Call"

    def test_metadata_callback(self, agent):
        """HIGH-NEW-5: Metadata with 'callback' result should detect follow-up."""
        is_followup, call_type = agent.detect_call_type(
            "auto_generated_id.mp3",
            metadata={"result": "callback"},
        )
        assert is_followup
        assert call_type == "Follow-up Call"

    def test_metadata_no_followup_falls_to_filename(self, agent):
        """Non-followup metadata should still fall through to filename heuristic."""
        is_followup, call_type = agent.detect_call_type(
            "2nd_call.mp3",
            metadata={"result": "completed"},
        )
        assert is_followup  # filename heuristic picks it up

    def test_metadata_none_uses_filename(self, agent):
        """When metadata is None, filename heuristic should be used."""
        is_followup, call_type = agent.detect_call_type("followup_call.mp3", metadata=None)
        assert is_followup


# --- Tests: Score Calculation ---

class TestScoreCalculation:

    def test_all_yes_scores_100(self, agent, all_yes_evaluation):
        result = agent.calculate_score(all_yes_evaluation)
        assert result["overall_score"] == 100.0

    def test_all_no_scores_0(self, agent, all_no_evaluation):
        result = agent.calculate_score(all_no_evaluation)
        assert result["overall_score"] == 0.0

    def test_all_partial_scores_50(self, agent):
        evaluation = {
            "criteria": {
                key: {"score": "PARTIAL", "evidence": "Half."}
                for key in agent.EVALUATION_CRITERIA
            }
        }
        result = agent.calculate_score(evaluation)
        assert result["overall_score"] == 50.0

    def test_na_excluded_from_score(self, agent):
        keys = list(agent.EVALUATION_CRITERIA.keys())
        evaluation = {
            "criteria": {
                keys[0]: {"score": "YES", "evidence": "Done."},
                **{k: {"score": "N/A", "evidence": "N/A."} for k in keys[1:]},
            }
        }
        result = agent.calculate_score(evaluation)
        assert result["overall_score"] == 100.0
        assert result["score_breakdown"]["na_count"] == 23

    def test_category_scores_present(self, agent, mixed_evaluation):
        result = agent.calculate_score(mixed_evaluation)
        assert "category_scores" in result
        for cat in ["phone_skills", "sales_techniques", "urgency_closing", "soft_skills"]:
            assert cat in result["category_scores"]

    def test_score_breakdown_sums_to_24(self, agent, mixed_evaluation):
        result = agent.calculate_score(mixed_evaluation)
        bd = result["score_breakdown"]
        total = bd["yes_count"] + bd["partial_count"] + bd["no_count"] + bd["na_count"]
        assert total == 24

    def test_empty_criteria_returns_zero(self, agent):
        result = agent.calculate_score({"criteria": {}})
        assert result["overall_score"] == 0

    def test_missing_criteria_key_returns_zero(self, agent):
        result = agent.calculate_score({})
        assert result["overall_score"] == 0


# --- Tests: Listening Ratio ---

class TestListeningRatio:

    def test_labeled_transcript(self, agent):
        transcript = "Agent: Hello how are you today\nClient: I am fine thanks"
        result = agent.calculate_listening_ratio(transcript)
        assert result["agent_percentage"] > 0
        assert result["client_percentage"] > 0
        assert abs(result["agent_percentage"] + result["client_percentage"] - 100.0) < 0.1

    def test_speaker_format(self, agent):
        """After #15, raw Speaker N: format is no longer recognized —
        use cleaned Agent:/Client: labels."""
        transcript = "Agent: Hello\nClient: Hi there"
        result = agent.calculate_listening_ratio(transcript)
        assert result["total_words"] > 0

    def test_empty_transcript(self, agent):
        result = agent.calculate_listening_ratio("")
        assert result["agent_percentage"] == 50.0
        assert result["client_percentage"] == 50.0
        assert result["total_words"] == 0

    def test_only_agent_speaks(self, agent):
        transcript = "Agent: I will talk the entire time here is more text"
        result = agent.calculate_listening_ratio(transcript)
        assert result["agent_percentage"] == 100.0
        assert result["client_percentage"] == 0.0

    def test_elevenlabs_speaker_format(self, agent):
        """MED-15: After cleaning, Speaker 0/1 → Agent/Client.
        Listening ratio now only accepts cleaned labels."""
        transcript = "Agent: Hello this is John from the company\nClient: Hi I need help with my booking"
        result = agent.calculate_listening_ratio(transcript)
        assert result["agent_percentage"] > 0
        assert result["client_percentage"] > 0
        assert result["total_words"] > 0

    def test_empty_transcript_listening_ratio(self, agent):
        """Duplicate-renamed: empty transcript returns 50/50 and 0 words."""
        result = agent.calculate_listening_ratio("")
        assert result["agent_percentage"] == 50.0
        assert result["client_percentage"] == 50.0
        assert result["total_words"] == 0

    def test_unlabeled_lines_not_counted_as_agent(self, agent):
        """Unlabeled lines should NOT default to agent anymore."""
        transcript = "Some random text without labels\nAnother line"
        result = agent.calculate_listening_ratio(transcript)
        assert result["total_words"] == 0


# --- Tests: MED-11 — evaluate_call with direction metadata ---

class TestEvaluateCallDirection:
    """MED-11: TranscriptCleaner direction from metadata."""

    def test_evaluate_call_accepts_metadata(self, agent):
        """evaluate_call should accept optional metadata kwarg."""
        import inspect
        sig = inspect.signature(agent.evaluate_call)
        assert "metadata" in sig.parameters

    def test_evaluate_call_inbound_direction(self, agent):
        """MED-11: When metadata contains direction='inbound',
        the per-call TranscriptCleaner should use that direction."""
        # Patch inference engine to avoid real LLM call
        agent._engine.evaluate = MagicMock(return_value={
            "criteria": {"greeting_prepared": {"score": "YES", "evidence": "OK"}},
            "overall_assessment": "Good call",
            "strengths": [], "improvements": [], "critical_gaps": [],
            "call_type": "First Call", "model_used": "test", "provider_used": "test",
            "tokens_used": {"input": 0, "output": 0}, "cost_usd": 0.0,
            "eval_time_seconds": 0.1,
        })

        transcript = "Speaker 0: Hello I need help\nSpeaker 1: Sure thing"
        result = agent.evaluate_call(
            transcript, "call1.mp3",
            metadata={"direction": "inbound"},
        )
        # Should succeed without error
        assert "error" not in result or result.get("status") == "TOO_SHORT"

    def test_evaluate_call_outbound_default(self, agent):
        """Without metadata direction, default 'outbound' is used."""
        agent._engine.evaluate = MagicMock(return_value={
            "criteria": {"greeting_prepared": {"score": "YES", "evidence": "OK"}},
            "overall_assessment": "Good", "strengths": [], "improvements": [],
            "critical_gaps": [], "call_type": "First Call", "model_used": "test",
            "provider_used": "test", "tokens_used": {"input": 0, "output": 0},
            "cost_usd": 0.0, "eval_time_seconds": 0.1,
        })

        # A long enough transcript
        transcript = "Speaker 0: " + " ".join(["hello"] * 30) + "\nSpeaker 1: hi there"
        result = agent.evaluate_call(transcript, "call1.mp3")
        assert "error" not in result


# --- Tests: Fix #6 — TranscriptCleaner direction cache ---

class TestCleanerCache:

    def test_cleaner_cache_has_two_entries(self, agent):
        """Fix #6: Agent should have pre-built cleaners for both directions."""
        assert "outbound" in agent._cleaners
        assert "inbound" in agent._cleaners
        assert len(agent._cleaners) == 2

    def test_cleaner_cache_reused(self, agent):
        """Fix #6: Same cleaner instance should be reused across calls."""
        cleaner_inbound = agent._cleaners["inbound"]
        cleaner_outbound = agent._cleaners["outbound"]

        # Call evaluate_call with inbound metadata
        agent._engine.evaluate = MagicMock(return_value={
            "criteria": {"greeting_prepared": {"score": "YES", "evidence": "OK"}},
            "overall_assessment": "Good", "strengths": [], "improvements": [],
            "critical_gaps": [], "call_type": "First Call", "model_used": "test",
            "provider_used": "test", "tokens_used": {"input": 0, "output": 0},
            "cost_usd": 0.0, "eval_time_seconds": 0.1,
        })

        transcript = "Speaker 0: " + " ".join(["hello"] * 30) + "\nSpeaker 1: hi there"
        agent.evaluate_call(transcript, "call1.mp3", metadata={"direction": "inbound"})
        agent.evaluate_call(transcript, "call2.mp3", metadata={"direction": "outbound"})

        # Cleaners should still be the same instances (not recreated)
        assert agent._cleaners["inbound"] is cleaner_inbound
        assert agent._cleaners["outbound"] is cleaner_outbound
