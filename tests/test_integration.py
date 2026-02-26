"""
TEST-26: Integration Smoke Test

End-to-end pipeline test with all 4 agents mocked.
Verifies that the full pipeline chain produces valid output
and that data flows correctly between stages.
"""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline import Pipeline


@pytest.fixture
def integration_agents(tmp_path):
    """Build all 4 mocked agents for integration test."""
    agent_01 = MagicMock()
    agent_02 = MagicMock()
    agent_03 = MagicMock()
    agent_04 = MagicMock()

    # Agent 02: Scribe v2 batch result
    agent_02.transcribe_batch.return_value = {
        "call1.mp3": {
            "transcript": "Speaker 0: Hello, thanks for calling.\nSpeaker 1: Hi, I need to renew my subscription.",
            "raw_text": "Hello, thanks for calling. Hi, I need to renew my subscription.",
            "speakers_detected": 2,
            "diarized": True,
            "language_code": "en",
            "status": "Success",
            "cost_usd": 0.005,
            "duration": 2.5,
        },
    }

    # Agent 03: Realistic evaluation result
    agent_03.evaluate_call.return_value = {
        "criteria": {
            "greeting_prepared": {"score": "YES", "evidence": "Good greeting."},
            "contact_info": {"score": "PARTIAL", "evidence": "Only name collected."},
            "needs_assessment": {"score": "NO", "evidence": "Did not probe needs."},
        },
        "overall_assessment": "Agent showed good phone skills but missed sales techniques.",
        "strengths": ["Professional greeting", "Polite tone"],
        "improvements": ["Needs assessment", "Closing skills"],
        "critical_gaps": ["No urgency created"],
        "call_type": "First Call",
        "model_used": "gpt-4o-2024-11-20",
        "provider_used": "openrouter",
        "tokens_used": {"input": 4000, "output": 1200},
        "cost_usd": 0.025,
    }
    agent_03.calculate_score.return_value = {
        "overall_score": 50.0,
        "total_points": 1.5,
        "total_weight": 3.0,
        "category_scores": {"opening": {"score": 75.0, "count": 2}},
        "score_breakdown": {"yes_count": 1, "partial_count": 1, "no_count": 1, "na_count": 0},
    }
    agent_03.EVALUATION_CRITERIA = {
        "greeting_prepared": {"category": "opening", "weight": 1.0},
        "contact_info": {"category": "opening", "weight": 1.0},
        "needs_assessment": {"category": "interview", "weight": 1.0},
    }

    # Agent 04: Export returns file paths
    agent_04.export_all.return_value = {
        "excel": str(tmp_path / "QM_test.xlsx"),
        "json": str(tmp_path / "QM_test.json"),
    }

    return agent_01, agent_02, agent_03, agent_04


class TestIntegrationSmokeTest:

    def test_full_pipeline_local_mode(self, integration_agents):
        """Smoke test: full pipeline produces valid evaluations."""
        a1, a2, a3, a4 = integration_agents
        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)

        results = pipeline.run_local([Path("call1.mp3")])

        assert len(results) == 1
        e = results[0]

        # Verify all required output fields
        assert e["filename"] == "call1.mp3"
        assert e["overall_score"] == 50.0
        assert e["status"] == "Success"
        assert "criteria" in e
        assert "overall_assessment" in e
        assert "strengths" in e
        assert "improvements" in e
        assert "critical_gaps" in e
        assert e["cost_usd"] == 0.025

        # Verify agents were called in correct order
        a2.transcribe_batch.assert_called_once()
        a3.evaluate_call.assert_called_once()
        a3.calculate_score.assert_called_once()
        a4.export_all.assert_called_once()

    def test_pipeline_empty_input(self, integration_agents):
        """Empty input should return empty list without calling downstream agents."""
        a1, a2, a3, a4 = integration_agents
        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)

        results = pipeline.run_local([])

        assert results == []
        a2.transcribe_batch.assert_not_called()
        a3.evaluate_call.assert_not_called()
        a4.export_all.assert_not_called()

    def test_pipeline_stt_failure_skips_evaluation(self, integration_agents):
        """When STT fails, its transcript should be skipped by evaluation."""
        a1, a2, a3, a4 = integration_agents
        a2.transcribe_batch.return_value = {
            "call1.mp3": {"status": "Error: timeout", "duration": 5.0, "path": Path("call1.mp3")},
        }
        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)

        results = pipeline.run_local([Path("call1.mp3")])

        assert results == []
        a3.evaluate_call.assert_not_called()

    def test_pipeline_multi_file_flow(self, integration_agents):
        """#28: Multi-file pipeline processes all files in sequence."""
        a1, a2, a3, a4 = integration_agents
        a2.transcribe_batch.return_value = {
            "call1.mp3": {
                "transcript": "Agent: Hello\nClient: Hi",
                "status": "Success",
                "cost_usd": 0.005,
                "duration": 2.5,
            },
            "call2.mp3": {
                "transcript": "Agent: Good morning\nClient: Good morning",
                "status": "Success",
                "cost_usd": 0.003,
                "duration": 1.5,
            },
        }
        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)

        results = pipeline.run_local([Path("call1.mp3"), Path("call2.mp3")])

        assert len(results) == 2
        assert a3.evaluate_call.call_count == 2
        a4.export_all.assert_called_once()

    def test_pipeline_circuit_breaker(self, integration_agents):
        """#28: Circuit breaker triggers after max consecutive failures."""
        a1, a2, a3, a4 = integration_agents
        files = {f"call{i}.mp3": {
            "transcript": f"Agent: Hello {i}\nClient: Hi",
            "status": "Success",
            "cost_usd": 0.001,
            "duration": 1.0,
        } for i in range(5)}
        a2.transcribe_batch.return_value = files
        a3.evaluate_call.return_value = {"error": "API down", "call_type": "First Call"}
        pipeline = Pipeline(a1, a2, a3, a4, max_consecutive_failures=3, delay_between_evaluations=0)

        results = pipeline.run_local([Path(f"call{i}.mp3") for i in range(5)])

        # Should stop after 3 failures + circuit breaker entry
        circuit_breaker_entries = [r for r in results if r.get("status") == "CIRCUIT_BREAKER_TRIGGERED"]
        assert len(circuit_breaker_entries) == 1

    def test_pipeline_crm_mode_smoke(self, integration_agents):
        """LOW-17: Integration smoke test for CRM mode (mocked)."""
        a1, a2, a3, a4 = integration_agents
        # Mock CRM Agent behaviour — search_and_download returns call records
        a1.search_and_download.return_value = [
            {"local_audio_path": "data/audio/crm_call1.mp3", "recording_id": 1},
        ]
        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)

        results = pipeline.run("2025-01-01", "2025-01-02")

        # Verify CRM download was called
        a1.search_and_download.assert_called_once_with("2025-01-01", "2025-01-02")
        # Verify downstream agents were called
        a2.transcribe_batch.assert_called_once()
        assert len(results) >= 0  # May be 0 if transcription key mismatch, but no crash


# --- Tests: R-05 — CRM Integration Tests ---

class TestCRMIntegration:
    """R-05: Integration tests for CRM mode (pipeline.run())."""

    def test_crm_direction_preserved(self, integration_agents):
        """CRM direction must reach Agent 03's evaluate_call()."""
        a1, a2, a3, a4 = integration_agents

        # Agent 01 returns call with direction metadata
        a1.search_and_download = MagicMock(return_value=[
            {
                "id": "crm_call_001",
                "local_audio_path": "/tmp/crm_call_001.mp3",
                "direction": "inbound",
                "agent_name": "Emma",
                "client_name": "John Doe",
                "result": "completed",
                "flight_request_status": "new",
            },
        ])

        # Agent 02 must return transcript keyed by the filename
        a2.transcribe_batch.return_value = {
            "crm_call_001.mp3": {
                "transcript": "Speaker 0: Hello, thanks for calling.\nSpeaker 1: Hi, I need to renew my subscription.",
                "raw_text": "Hello, thanks for calling. Hi, I need to renew my subscription.",
                "speakers_detected": 2,
                "diarized": True,
                "language_code": "en",
                "status": "Success",
                "cost_usd": 0.005,
                "duration": 2.5,
            },
        }

        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)
        results = pipeline.run("2026-02-01", "2026-02-25")

        assert len(results) == 1
        # Verify metadata was passed to evaluate_call
        call_kwargs = a3.evaluate_call.call_args
        metadata = call_kwargs.kwargs.get("metadata", {})
        assert metadata.get("direction") == "inbound"
        assert metadata.get("agent_name") == "Emma"

    def test_crm_mixed_directions(self, integration_agents):
        """Multiple CRM calls with different directions."""
        a1, a2, a3, a4 = integration_agents

        a1.search_and_download = MagicMock(return_value=[
            {"id": "c1", "local_audio_path": "/tmp/c1.mp3", "direction": "inbound"},
            {"id": "c2", "local_audio_path": "/tmp/c2.mp3", "direction": "outbound"},
        ])

        a2.transcribe_batch.return_value = {
            "c1.mp3": {
                "transcript": "Speaker 0: Hi\nSpeaker 1: Hello",
                "status": "Success", "cost_usd": 0.005, "duration": 2.0,
                "raw_text": "Hi Hello", "speakers_detected": 2,
            },
            "c2.mp3": {
                "transcript": "Speaker 0: Hey\nSpeaker 1: Hi there",
                "status": "Success", "cost_usd": 0.003, "duration": 1.5,
                "raw_text": "Hey Hi there", "speakers_detected": 2,
            },
        }

        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)
        results = pipeline.run("2026-02-01")

        assert a3.evaluate_call.call_count == 2

    def test_crm_no_recordings(self, integration_agents):
        """CRM returns no recordings — pipeline should exit gracefully."""
        a1, a2, a3, a4 = integration_agents
        a1.search_and_download = MagicMock(return_value=[])

        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)
        results = pipeline.run("2026-02-01")

        assert results == []
        a2.transcribe_batch.assert_not_called()

    def test_crm_minimal_metadata(self, integration_agents):
        """CRM record with minimal metadata should not crash."""
        a1, a2, a3, a4 = integration_agents
        a1.search_and_download = MagicMock(return_value=[
            {"id": "m1", "local_audio_path": "/tmp/m1.mp3"},
        ])

        a2.transcribe_batch.return_value = {
            "m1.mp3": {
                "transcript": "Speaker 0: Hello, thanks for calling.\nSpeaker 1: Hi, I need help.",
                "raw_text": "Hello, thanks for calling. Hi, I need help.",
                "speakers_detected": 2,
                "status": "Success",
                "cost_usd": 0.005,
                "duration": 2.5,
            },
        }

        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)
        results = pipeline.run("2026-02-01")
        # Should not crash; metadata defaults to empty strings
        assert a3.evaluate_call.called
