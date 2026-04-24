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
    """Reset graceful shutdown state between tests.
    HIGH-9: _GracefulShutdown is now instance-based, but we create
    a fresh instance for standalone tests.
    """
    shutdown = _GracefulShutdown()
    yield shutdown
    # No global state to clean up anymore


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
    agent_03.EVALUATION_CRITERIA = {"greeting": {"category": "opening", "weight": 1.0}}

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
        """HIGH-9: Instance-based shutdown with threading.Event."""
        shutdown = _GracefulShutdown()
        assert not shutdown.is_triggered()
        shutdown.trigger(signal.SIGINT, None)
        assert shutdown.is_triggered()
        shutdown.reset()
        assert not shutdown.is_triggered()

    def test_shutdown_reset_at_pipeline_start(self, mock_agents):
        """MED-NEW-15: Stale shutdown state should be cleared at run start."""
        a1, a2, a3, a4 = mock_agents
        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)
        # Trigger shutdown before running
        pipeline._shutdown.trigger(signal.SIGINT, None)
        assert pipeline._shutdown.is_triggered()

        pipeline.run_local([Path("call1.mp3")])
        # After run, the shutdown should have been reset at start
        # and evaluations should have proceeded
        assert a3.evaluate_call.called

    def test_interruptible_sleep(self):
        """MED-NEW-11: _interruptible_sleep should exit early on shutdown signal."""
        # Create a pipeline to test instance method
        a1, a2, a3, a4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)
        pipeline._shutdown.reset()

        import threading as th
        def trigger_after():
            time.sleep(0.1)
            pipeline._shutdown.trigger(signal.SIGINT, None)
        t = th.Thread(target=trigger_after, daemon=True)
        t.start()

        start = time.time()
        pipeline._interruptible_sleep(5.0, granularity=0.05)
        elapsed = time.time() - start
        # Should exit well before 5s
        assert elapsed < 1.0, f"Sleep was not interrupted ({elapsed:.2f}s)"

    def test_signal_handlers_restored_after_run(self, mock_agents):
        """NEW-03: Signal handlers must be restored after run() completes."""
        a1, a2, a3, a4 = mock_agents
        original_handler = signal.getsignal(signal.SIGINT)

        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)
        pipeline.run_local([Path("call1.mp3")])

        restored_handler = signal.getsignal(signal.SIGINT)
        assert restored_handler is original_handler, (
            "SIGINT handler was not restored after run_local()"
        )

    def test_signal_handlers_restored_on_error(self, mock_agents):
        """NEW-03: Signal handlers must be restored even when run() raises."""
        a1, a2, a3, a4 = mock_agents
        a2.transcribe_batch.side_effect = RuntimeError("STT failure")

        original_handler = signal.getsignal(signal.SIGINT)
        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)

        with pytest.raises(RuntimeError, match="STT failure"):
            pipeline.run_local([Path("call1.mp3")])

        restored_handler = signal.getsignal(signal.SIGINT)
        assert restored_handler is original_handler, (
            "SIGINT handler was not restored after error"
        )


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

    def test_evaluation_includes_word_count(self, pipeline, mock_agents):
        """TASK-2: Evaluation output must include word_count as int."""
        results = pipeline.run_local([Path("call1.mp3")])
        assert len(results) > 0
        assert "word_count" in results[0]
        assert isinstance(results[0]["word_count"], int)

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


# --- Tests: R-01 — CRM Metadata Passthrough ---

class TestCRMMetadataPassthrough:

    def test_crm_metadata_reaches_evaluate_call(self, mock_agents):
        """R-01: CRM metadata must reach evaluate_call() in run() mode."""
        a1, a2, a3, a4 = mock_agents

        # Simulate CRM returning calls with metadata
        a1.search_and_download = MagicMock(return_value=[
            {
                "id": "call_123",
                "local_audio_path": "/tmp/call_123.mp3",
                "direction": "inbound",
                "agent_name": "Emma",
                "client_name": "Matt Kern",
                "result": "completed",
                "flight_request_status": "new",
            }
        ])

        # Agent 02 must return transcript keyed by filename
        a2.transcribe_batch.return_value = {
            "call_123.mp3": {
                "transcript": "Speaker 0: Hello\nSpeaker 1: Hi there how are you",
                "raw_text": "Hello Hi there how are you",
                "speakers_detected": 2,
                "diarized": True,
                "language_code": "en",
                "status": "Success",
                "cost_usd": 0.01,
                "duration": 5.0,
            },
        }

        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)
        pipeline.run("2026-02-01", "2026-02-25")

        # Verify evaluate_call received metadata with direction
        assert a3.evaluate_call.called
        call_kwargs = a3.evaluate_call.call_args
        metadata = call_kwargs.kwargs.get("metadata", {})
        assert metadata.get("direction") == "inbound"
        assert metadata.get("agent_name") == "Emma"
        assert metadata.get("call_id") == "call_123"

    def test_run_local_works_without_metadata(self, pipeline):
        """R-01: run_local() must continue working without CRM metadata."""
        results = pipeline.run_local([Path("call1.mp3")])
        assert len(results) > 0  # Existing behavior preserved

    def test_crm_metadata_missing_direction_defaults_gracefully(self, mock_agents):
        """R-01: CRM record without direction should not crash."""
        a1, a2, a3, a4 = mock_agents
        a1.search_and_download = MagicMock(return_value=[
            {
                "id": "call_456",
                "local_audio_path": "/tmp/call_456.mp3",
                # No "direction" key
            }
        ])
        a2.transcribe_batch.return_value = {
            "call_456.mp3": {
                "transcript": "Speaker 0: Hello\nSpeaker 1: Hi there how are you",
                "status": "Success",
                "cost_usd": 0.01,
                "duration": 5.0,
                "raw_text": "hello",
                "speakers_detected": 2,
            },
        }
        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)
        pipeline.run("2026-02-01")
        assert a3.evaluate_call.called
        # Metadata should have empty direction string (not crash)
        meta = a3.evaluate_call.call_args.kwargs.get("metadata", {})
        assert meta.get("direction") == ""


# --- Tests: R-03 + R-04 — Export Sanitization ---

class TestExportSanitization:

    def test_blocked_keys_not_in_export(self, mock_agents):
        """R-03 + R-04: _EXPORT_BLOCKED_KEYS must be stripped before export."""
        a1, a2, a3, a4 = mock_agents
        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)
        pipeline.run_local([Path("call1.mp3")])

        if a4.export_all.called:
            exported = a4.export_all.call_args[0][0]  # first positional arg
            for evaluation in exported:
                for blocked_key in ("raw_response", "raw_text",
                                    "system_prompt", "user_prompt"):
                    assert blocked_key not in evaluation, \
                        f"Blocked key '{blocked_key}' found in exported evaluation"

    def test_transcript_present_in_export(self, mock_agents):
        """Unredacted transcript must be present in exported evaluations (EU-DPA policy)."""
        a1, a2, a3, a4 = mock_agents
        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)
        pipeline.run_local([Path("call1.mp3")])

        if a4.export_all.called:
            exported = a4.export_all.call_args[0][0]
            assert any("transcript" in e and e["transcript"] for e in exported)

    def test_raw_transcript_still_in_memory(self, mock_agents):
        """Raw transcript should remain in the in-memory evaluation list."""
        a1, a2, a3, a4 = mock_agents
        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)
        results = pipeline.run_local([Path("call1.mp3")])
        # In-memory results should still have transcript
        assert any("transcript" in r for r in results)

    # --- Pipeline integrity tests ---

    def test_transcript_forwarded_unredacted(self, mock_agents):
        """Transcript flows from Agent 02 through pipeline to export unchanged (no masking)."""
        a1, a2, a3, a4 = mock_agents
        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)
        pipeline.run_local([Path("call1.mp3")])

        if a4.export_all.called:
            exported = a4.export_all.call_args[0][0]
            for e in exported:
                # Full transcript must survive into export; no PII-tag placeholders
                assert e.get("transcript"), "Transcript missing from exported data"
                for tag in ("[PHONE]", "[EMAIL]", "[SSN]", "[CC_NUMBER]",
                            "[SPELLED_PII]", "[NATO_SPELLED]"):
                    assert tag not in e["transcript"], \
                        f"Unexpected PII tag {tag} in transcript — redaction should be removed"

    def test_no_pii_fields_in_exported_data(self, mock_agents):
        """Legacy PII fields must not appear in export after PIIRedactor removal."""
        a1, a2, a3, a4 = mock_agents
        pipeline = Pipeline(a1, a2, a3, a4, delay_between_evaluations=0)
        pipeline.run_local([Path("call1.mp3")])

        if a4.export_all.called:
            exported = a4.export_all.call_args[0][0]
            for e in exported:
                for key in ("transcript_redacted", "pii_redacted",
                            "pii_total_redactions"):
                    assert key not in e, \
                        f"Legacy PII field '{key}' leaked into export"
