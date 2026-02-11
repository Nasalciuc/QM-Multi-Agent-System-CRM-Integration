"""
Tests for Agent 3: Quality Management Evaluation
"""
import sys
import os
import json
import re
import pytest
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# --- Fixtures ---

@pytest.fixture
def criteria_path():
    return os.path.join(os.path.dirname(__file__), '..', 'config', 'qa_criteria.yaml')


@pytest.fixture
def agent(criteria_path):
    from agents.agent_03_QualityManagement import QualityManagementAgent
    mock_client = MagicMock()
    return QualityManagementAgent(mock_client, criteria_path=criteria_path)


@pytest.fixture
def sample_evaluation(agent):
    """A valid LLM response with all 24 criteria scored."""
    criteria_keys = list(agent.EVALUATION_CRITERIA.keys())
    evaluation = {
        "criteria": {},
        "overall_assessment": "Good call overall.",
        "strengths": ["Professional greeting", "Good product knowledge", "Handled objections"],
        "improvements": ["Ask for budget", "Create urgency", "Better closing"],
        "critical_gaps": []
    }
    for i, key in enumerate(criteria_keys):
        if i % 3 == 0:
            evaluation["criteria"][key] = {"score": "YES", "evidence": "Agent did this well."}
        elif i % 3 == 1:
            evaluation["criteria"][key] = {"score": "PARTIAL", "evidence": "Partially done."}
        else:
            evaluation["criteria"][key] = {"score": "NO", "evidence": "Not observed."}
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


# --- Tests: Score Calculation ---

class TestScoreCalculation:

    def test_all_yes_scores_100(self, agent):
        """All YES should give 100%."""
        evaluation = {"criteria": {}}
        for key in agent.EVALUATION_CRITERIA:
            evaluation["criteria"][key] = {"score": "YES", "evidence": "Done."}
        result = agent.calculate_score(evaluation)
        assert result["overall_score"] == 100.0

    def test_all_no_scores_0(self, agent):
        """All NO should give 0%."""
        evaluation = {"criteria": {}}
        for key in agent.EVALUATION_CRITERIA:
            evaluation["criteria"][key] = {"score": "NO", "evidence": "Not done."}
        result = agent.calculate_score(evaluation)
        assert result["overall_score"] == 0.0

    def test_all_partial_scores_50(self, agent):
        """All PARTIAL should give 50%."""
        evaluation = {"criteria": {}}
        for key in agent.EVALUATION_CRITERIA:
            evaluation["criteria"][key] = {"score": "PARTIAL", "evidence": "Half done."}
        result = agent.calculate_score(evaluation)
        assert result["overall_score"] == 50.0

    def test_na_excluded_from_score(self, agent):
        """N/A criteria should not affect the score."""
        evaluation = {"criteria": {}}
        keys = list(agent.EVALUATION_CRITERIA.keys())
        evaluation["criteria"][keys[0]] = {"score": "YES", "evidence": "Done."}
        for key in keys[1:]:
            evaluation["criteria"][key] = {"score": "N/A", "evidence": "Not applicable."}
        result = agent.calculate_score(evaluation)
        assert result["overall_score"] == 100.0
        assert result["score_breakdown"]["na_count"] == 23

    def test_category_scores_returned(self, agent, sample_evaluation):
        """Should return scores per category."""
        result = agent.calculate_score(sample_evaluation)
        assert "category_scores" in result
        for cat in ["phone_skills", "sales_techniques", "urgency_closing", "soft_skills"]:
            assert cat in result["category_scores"]
            assert "score" in result["category_scores"][cat]

    def test_score_breakdown_counts(self, agent, sample_evaluation):
        """Breakdown should have correct YES/PARTIAL/NO/N/A counts."""
        result = agent.calculate_score(sample_evaluation)
        breakdown = result["score_breakdown"]
        total = breakdown["yes_count"] + breakdown["partial_count"] + breakdown["no_count"] + breakdown["na_count"]
        assert total == 24

    def test_empty_criteria_returns_zero(self, agent):
        """Empty or missing criteria should return 0."""
        result = agent.calculate_score({"criteria": {}})
        assert result["overall_score"] == 0
        result2 = agent.calculate_score({})
        assert result2["overall_score"] == 0


# --- Tests: Transcript Truncation ---

class TestTranscriptValidation:

    def test_long_transcript_truncated(self, agent):
        """Transcripts exceeding MAX_TRANSCRIPT_LENGTH should be truncated."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "criteria": {k: {"score": "YES", "evidence": "ok"} for k in agent.EVALUATION_CRITERIA},
            "overall_assessment": "Good.",
            "strengths": ["a", "b", "c"],
            "improvements": ["d", "e", "f"],
            "critical_gaps": []
        })
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 500
        agent.client.chat.completions.create.return_value = mock_response

        long_transcript = "word " * 50000  # way over limit
        result = agent.evaluate_call(long_transcript, "test.mp3")
        assert result.get("truncated") is True


# --- Tests: Listening Ratio ---

class TestListeningRatio:

    def test_labeled_format(self, agent):
        transcript = "Agent: Hello how are you today\nClient: I am fine thanks"
        result = agent.calculate_listening_ratio(transcript)
        assert result["agent_percentage"] > 0
        assert result["client_percentage"] > 0
        assert abs(result["agent_percentage"] + result["client_percentage"] - 100.0) < 0.1

    def test_elevenlabs_speaker_format(self, agent):
        """ElevenLabs Scribe format with Speaker 0/1 labels."""
        transcript = "Speaker 0: Hello this is John from the company\nSpeaker 1: Hi I need help with my booking"
        result = agent.calculate_listening_ratio(transcript)
        assert result["agent_percentage"] > 0
        assert result["client_percentage"] > 0
        assert result["total_words"] > 0

    def test_empty_transcript(self, agent):
        result = agent.calculate_listening_ratio("")
        assert result["agent_percentage"] == 50.0
        assert result["client_percentage"] == 50.0
        assert result["total_words"] == 0

    def test_unlabeled_lines_not_counted_as_agent(self, agent):
        """Unlabeled lines should NOT default to agent anymore."""
        transcript = "Some random text without labels\nAnother line"
        result = agent.calculate_listening_ratio(transcript)
        assert result["total_words"] == 0
