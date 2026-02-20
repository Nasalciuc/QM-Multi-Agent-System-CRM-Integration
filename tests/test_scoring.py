"""
Tests for scoring logic (calculate_score) - dedicated parameterized tests.

Separated from test_agent_03 for clarity and detailed edge-case coverage.
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch


# --- Fixture: Agent ---

@pytest.fixture
def criteria_path():
    return os.path.join(os.path.dirname(__file__), '..', 'config', 'qa_criteria.yaml')


@pytest.fixture
def mock_model_factory():
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
    with patch("agents.agent_03_evaluation.ModelFactory", autospec=True):
        from agents.agent_03_evaluation import QualityManagementAgent
        return QualityManagementAgent(
            model_factory=mock_model_factory,
            criteria_path=criteria_path,
            cache_dir=str(tmp_path / "cache"),
            enable_cache=False,
        )


# --- Helper ---

def _build_evaluation(agent, score_map: dict) -> dict:
    """Build evaluation dict from {score_value: [keys]} mapping.

    Example: {"YES": [...keys...], "NO": [...keys...]}
    """
    evaluation = {"criteria": {}}
    for score_val, keys in score_map.items():
        for key in keys:
            evaluation["criteria"][key] = {"score": score_val, "evidence": "-"}
    return evaluation


# --- Parameterized scoring tests ---

class TestScoreEdgeCases:

    @pytest.mark.parametrize("score_value,expected", [
        ("YES", 100.0),
        ("NO", 0.0),
        ("PARTIAL", 50.0),
    ])
    def test_uniform_scores(self, agent, score_value, expected):
        """When all criteria have the same score, overall == expected."""
        evaluation = {
            "criteria": {
                key: {"score": score_value, "evidence": "-"}
                for key in agent.EVALUATION_CRITERIA
            }
        }
        result = agent.calculate_score(evaluation)
        assert result["overall_score"] == expected

    def test_all_na_gives_zero(self, agent):
        """All N/A → overall_score = 0 (no scoreable criteria)."""
        evaluation = {
            "criteria": {
                key: {"score": "N/A", "evidence": "-"}
                for key in agent.EVALUATION_CRITERIA
            }
        }
        result = agent.calculate_score(evaluation)
        assert result["overall_score"] == 0

    def test_one_yes_rest_na(self, agent):
        """Single YES among all N/A → 100%."""
        keys = list(agent.EVALUATION_CRITERIA.keys())
        evaluation = {
            "criteria": {
                keys[0]: {"score": "YES", "evidence": "-"},
                **{k: {"score": "N/A", "evidence": "-"} for k in keys[1:]},
            }
        }
        result = agent.calculate_score(evaluation)
        assert result["overall_score"] == 100.0

    def test_one_no_rest_na(self, agent):
        """Single NO among all N/A → 0%."""
        keys = list(agent.EVALUATION_CRITERIA.keys())
        evaluation = {
            "criteria": {
                keys[0]: {"score": "NO", "evidence": "-"},
                **{k: {"score": "N/A", "evidence": "-"} for k in keys[1:]},
            }
        }
        result = agent.calculate_score(evaluation)
        assert result["overall_score"] == 0.0

    def test_half_yes_half_no(self, agent):
        """50% YES, 50% NO → 50% (assuming equal weights)."""
        keys = list(agent.EVALUATION_CRITERIA.keys())
        mid = len(keys) // 2
        evaluation = {
            "criteria": {
                **{k: {"score": "YES", "evidence": "-"} for k in keys[:mid]},
                **{k: {"score": "NO", "evidence": "-"} for k in keys[mid:]},
            }
        }
        result = agent.calculate_score(evaluation)
        assert 45.0 <= result["overall_score"] <= 55.0  # allow slight weight variance

    def test_mixed_with_na(self, agent):
        """Mixed scores with some N/A — N/A excluded from denominator."""
        keys = list(agent.EVALUATION_CRITERIA.keys())
        evaluation = {"criteria": {}}
        for i, key in enumerate(keys):
            if i < 5:
                evaluation["criteria"][key] = {"score": "YES", "evidence": "-"}
            elif i < 10:
                evaluation["criteria"][key] = {"score": "NO", "evidence": "-"}
            else:
                evaluation["criteria"][key] = {"score": "N/A", "evidence": "-"}
        result = agent.calculate_score(evaluation)
        # Only 10 criteria are scoreable; N/A excluded
        assert result["score_breakdown"]["na_count"] == 38
        assert 0 < result["overall_score"] < 100


class TestScoreBreakdown:

    def test_breakdown_counts_correct(self, agent):
        """Verify breakdown counts match actual input."""
        keys = list(agent.EVALUATION_CRITERIA.keys())
        evaluation = {"criteria": {}}
        # 16 YES, 16 PARTIAL, 8 NO, 8 N/A = 48
        for i, key in enumerate(keys):
            if i < 16:
                evaluation["criteria"][key] = {"score": "YES", "evidence": "-"}
            elif i < 32:
                evaluation["criteria"][key] = {"score": "PARTIAL", "evidence": "-"}
            elif i < 40:
                evaluation["criteria"][key] = {"score": "NO", "evidence": "-"}
            else:
                evaluation["criteria"][key] = {"score": "N/A", "evidence": "-"}

        result = agent.calculate_score(evaluation)
        bd = result["score_breakdown"]
        assert bd["yes_count"] == 16
        assert bd["partial_count"] == 16
        assert bd["no_count"] == 8
        assert bd["na_count"] == 8

    def test_total_points_and_weight(self, agent):
        """total_points / total_weight * 100 == overall_score."""
        keys = list(agent.EVALUATION_CRITERIA.keys())
        evaluation = {
            "criteria": {
                k: {"score": "YES", "evidence": "-"} for k in keys[:12]
            }
        }
        # Add NO for the rest
        for k in keys[12:]:
            evaluation["criteria"][k] = {"score": "NO", "evidence": "-"}

        result = agent.calculate_score(evaluation)
        if result["total_weight"] > 0:
            manual = result["total_points"] / result["total_weight"] * 100
            assert abs(result["overall_score"] - round(manual, 1)) < 0.15


class TestCategoryScores:

    def test_all_categories_present(self, agent):
        evaluation = {
            "criteria": {
                k: {"score": "YES", "evidence": "-"}
                for k in agent.EVALUATION_CRITERIA
            }
        }
        result = agent.calculate_score(evaluation)
        expected = {
            "opening", "interview", "psychological_framing", "first_call_closing",
            "second_call_opening", "strategic_presentation", "creating_certainty",
            "second_call_objection_handling", "commitment_closing", "communication",
        }
        for cat in expected:
            assert cat in result["category_scores"]
            assert result["category_scores"][cat]["score"] == 100.0

    def test_category_count_matches(self, agent):
        """Category count in score result should match criteria YAML."""
        evaluation = {
            "criteria": {
                k: {"score": "YES", "evidence": "-"}
                for k in agent.EVALUATION_CRITERIA
            }
        }
        result = agent.calculate_score(evaluation)
        assert result["category_scores"]["opening"]["count"] == 5
        assert result["category_scores"]["interview"]["count"] == 4
        assert result["category_scores"]["psychological_framing"]["count"] == 4
        assert result["category_scores"]["first_call_closing"]["count"] == 7
        assert result["category_scores"]["second_call_opening"]["count"] == 4
        assert result["category_scores"]["strategic_presentation"]["count"] == 5
        assert result["category_scores"]["creating_certainty"]["count"] == 4
        assert result["category_scores"]["second_call_objection_handling"]["count"] == 3
        assert result["category_scores"]["commitment_closing"]["count"] == 4
        assert result["category_scores"]["communication"]["count"] == 8
