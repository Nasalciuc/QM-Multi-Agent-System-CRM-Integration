"""
Tests for src/pipeline.py

Tests pipeline orchestration, circuit breaker, rate limiting,
graceful shutdown, and provider tracking.
"""
import sys
import os
import signal
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import time

import pytest

from pipeline import Pipeline, _GracefulShutdown


# --- Fixtures ---

@pytest.fixture(autouse=True)
def reset_shutdown():
    """Reset graceful shutdown state between tests."""
    _GracefulShutdown.reset()
    yield
    _GracefulShutdown.reset()


@pytest.fixture
def mock_agents(tmp_path):
    """Create all 4 mock agents."""
    agent_01 = MagicMock()  # AudioFileFinder
    agent_02 = MagicMock()  # ElevenLabsSTTAgent
    agent_03 = MagicMock()  # QualityManagementAgent
    agent_04 = MagicMock()  # IntegrationAgent

    # Default: agent_02 returns 2 successful transcripts (Scribe v2 format)
    agent_02.transcribe_batch.return_value = {
        "call1.mp3": {
            "transcript": "Speaker 0: Hello\nSpeaker 1: Hi there how are you",
            "raw_text": "Hello Hi there how are you",
            "speakers_detected": 2,
            "diarized": True,
            "language_code": "en",
            "status": "Success",
            "cost_usd": 0.01,
            "duration": 5.0,
        },
        "call2.mp3": {
            "transcript": "Speaker 0: Welcome\nSpeaker 1: Thanks for calling us",
            "raw_text": "Welcome Thanks for calling us",
            "speakers_detected": 2,
            "diarized": True,
            "language_code": "en",
            "status": "Success",
            "cost_usd": 0.01,
            "duration": 3.0,
        },
    }

    # Default: agent_03 returns success
    agent_03.evaluate_call.return_value = {
        "criteria": {"greeting": {"score": "YES", "evidence": "ok"}},
        "overall_assessment": "Good",
        "strengths": [],
        "improvements": [],
        "call_type": "First Call",
        "model_used": "gpt-4o",
        "provider_used": "openrouter",
        "cost_usd": 0.03,
    }
    agent_03.calculate_score.return_value = {
        "overall_score": 85.0,
        "category_scores": {},
        "score_breakdown": {"yes_count": 1, "partial_count": 0, "no_count": 0, "na_count": 0},
    }
    agent_03.EVALUATION_CRITERIA = {"greeting": {"category": "phone_skills", "weight": 1.0}}

    agent_04.export_all.return_value = {"excel": "test.xlsx", "json": "test.json"}

    return agent_01, agent_02, agent_03, agent_04


@pytest.fixture
def pipeline(mock_agents):
    a1, a2, a3, a4 = mock_agents
    return Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)


# --- Tests: Basic Pipeline Flow ---

class TestPipelineFlow:

    def test_run_local_processes_files(self, pipeline, mock_agents):
        a1, a2, a3, a4 = mock_agents
        files = [Path("call1.mp3"), Path("call2.mp3")]
        results = pipeline.run_local(files)
        assert len(results) == 2
        assert a2.transcribe_batch.called
        assert a3.evaluate_call.call_count == 2
        assert a4.export_all.called

    def test_run_local_empty_files(self, pipeline):
        results = pipeline.run_local([])
        assert results == []

    def test_evaluations_have_required_fields(self, pipeline):
        results = pipeline.run_local([Path("call1.mp3")])
        assert len(results) > 0
        e = results[0]
        assert "filename" in e
        assert "overall_score" in e
        assert "cost_usd" in e
        assert "status" in e


# --- Tests: Circuit Breaker (HIGH-3) ---

class TestCircuitBreaker:

    def test_breaks_after_consecutive_failures(self, mock_agents):
        a1, a2, a3, a4 = mock_agents
        a3.evaluate_call.return_value = {"error": "API down", "call_type": "First Call"}
        pipeline = Pipeline(a1, a2, a3, a4, max_consecutive_failures=2, delay_between_evaluations=0)

        results = pipeline.run_local([Path("call1.mp3")])
        # Should contain circuit breaker marker
        has_breaker = any(e.get("status") == "CIRCUIT_BREAKER_TRIGGERED" for e in results)
        assert has_breaker

    def test_resets_on_success(self, mock_agents):
        a1, a2, a3, a4 = mock_agents
        # First call fails, second succeeds
        a3.evaluate_call.side_effect = [
            {"error": "temporary", "call_type": "First Call"},
            {
                "criteria": {"greeting": {"score": "YES", "evidence": "ok"}},
                "call_type": "First Call",
                "model_used": "gpt-4o",
                "provider_used": "openrouter",
                "cost_usd": 0.03,
            },
        ]
        pipeline = Pipeline(a1, a2, a3, a4, max_consecutive_failures=3, delay_between_evaluations=0)
        results = pipeline.run_local([Path("call1.mp3")])
        # No circuit breaker should trigger (only 1 failure then success)
        has_breaker = any(e.get("status") == "CIRCUIT_BREAKER_TRIGGERED" for e in results)
        assert not has_breaker


# --- Tests: Rate Limiting (HIGH-2) ---

class TestRateLimiting:

    def test_delay_between_evaluations(self, mock_agents):
        a1, a2, a3, a4 = mock_agents
        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0.1)
        start = time.time()
        pipeline.run_local([Path("call1.mp3"), Path("call2.mp3")])
        elapsed = time.time() - start
        # Should have at least ~0.1s delay for the 2nd call
        assert elapsed >= 0.05  # allow some margin


# --- Tests: Provider Tracking (HIGH-12) ---

class TestProviderTracking:

    def test_tracks_providers_used(self, pipeline):
        pipeline.run_local([Path("call1.mp3")])
        assert "openrouter" in pipeline._providers_used

    def test_tracks_multiple_providers(self, mock_agents):
        a1, a2, a3, a4 = mock_agents
        a3.evaluate_call.side_effect = [
            {"criteria": {}, "call_type": "First Call", "model_used": "gpt-4o",
             "provider_used": "openrouter", "cost_usd": 0.03},
            {"criteria": {}, "call_type": "First Call", "model_used": "claude",
             "provider_used": "fallback", "cost_usd": 0.05},
        ]
        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)
        pipeline.run_local([Path("call1.mp3"), Path("call2.mp3")])
        assert len(pipeline._providers_used) == 2


# --- Tests: STT Cost Aggregation (MED-8) ---

class TestSTTCost:

    def test_stt_costs_aggregated(self, pipeline, mock_agents):
        pipeline.run_local([Path("call1.mp3")])
        assert hasattr(pipeline, '_stt_cost')
        assert pipeline._stt_cost == 0.02  # 0.01 * 2 transcripts


# --- Tests: Graceful Shutdown (MED-3) ---

class TestGracefulShutdown:

    def test_shutdown_flag(self):
        assert not _GracefulShutdown.is_triggered()
        _GracefulShutdown.trigger(signal.SIGINT, None)
        assert _GracefulShutdown.is_triggered()
        _GracefulShutdown.reset()
        assert not _GracefulShutdown.is_triggered()


# --- Tests: Data Flow Contracts (TEST-25) ---

class TestDataFlowContracts:
    """Verify data shapes flowing between pipeline stages."""

    def test_stt_batch_output_shape(self, mock_agents):
        """Agent 2 batch output must have transcript, status, and v2 fields."""
        _, a2, _, _ = mock_agents
        batch = a2.transcribe_batch.return_value
        for filename, data in batch.items():
            assert isinstance(filename, str)
            assert "transcript" in data
            assert "status" in data
            assert "raw_text" in data
            assert "speakers_detected" in data

    def test_evaluation_output_shape(self, pipeline, mock_agents):
        """Pipeline evaluation output must have required fields."""
        results = pipeline.run_local([Path("call1.mp3")])
        assert len(results) > 0
        e = results[0]
        required = {"filename", "transcript", "overall_score", "score_data",
                     "criteria", "overall_assessment", "status", "cost_usd"}
        assert required.issubset(set(e.keys())), f"Missing: {required - set(e.keys())}"

    def test_circuit_breaker_sentinel_filtered_from_export(self, mock_agents):
        """Circuit breaker rows must NOT reach agent_04.export_all()."""
        a1, a2, a3, a4 = mock_agents
        a3.evaluate_call.return_value = {"error": "API down", "call_type": "First Call"}
        pipeline = Pipeline(a1, a2, a3, a4, max_consecutive_failures=1, delay_between_evaluations=0)
        pipeline.run_local([Path("call1.mp3")])
        # export_all should not be called (all evals failed / only breaker sentinel)
        if a4.export_all.called:
            args = a4.export_all.call_args[0][0]
            for e in args:
                assert e.get("status") != "CIRCUIT_BREAKER_TRIGGERED"
