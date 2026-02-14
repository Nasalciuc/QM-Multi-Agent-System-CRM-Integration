"""
Tests for Caching & Cost Optimization Features

Tests:
  - In-memory LRU cache in InferenceEngine (P2)
  - Cost budget enforcement in Pipeline (P3)
  - STT cache integration in ElevenLabsSTTAgent (P1)
  - Enhanced pipeline summary (P5)
"""
import sys
import os
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from collections import OrderedDict

import pytest

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ===========================================================================
# P2: In-Memory LRU Cache in InferenceEngine
# ===========================================================================

class TestInferenceEngineLRU:
    """Test L1 in-memory LRU cache in InferenceEngine."""

    @pytest.fixture
    def engine(self, tmp_path):
        with patch("inference.inference_engine.ModelFactory") as MockFactory, \
             patch("inference.inference_engine.PromptLoader"):
            factory = MockFactory.return_value
            factory.primary.model_name = "test-model"
            from inference.inference_engine import InferenceEngine
            return InferenceEngine(
                model_factory=factory,
                cache_dir=str(tmp_path / "cache"),
                enable_cache=True,
                memory_cache_maxsize=3,
            )

    def test_cache_stats_initial(self, engine):
        stats = engine.cache_stats
        assert stats["memory_hits"] == 0
        assert stats["memory_misses"] == 0
        assert stats["disk_hits"] == 0
        assert stats["disk_misses"] == 0
        assert stats["memory_cache_size"] == 0
        assert stats["memory_cache_maxsize"] == 3

    def test_promote_to_memory(self, engine):
        engine._promote_to_memory("key1", {"data": "value1"})
        assert "key1" in engine._memory_cache
        assert engine._memory_cache["key1"] == {"data": "value1"}

    def test_promote_moves_to_end(self, engine):
        engine._promote_to_memory("key1", {"data": "v1"})
        engine._promote_to_memory("key2", {"data": "v2"})
        engine._promote_to_memory("key1", {"data": "v1"})
        keys = list(engine._memory_cache.keys())
        assert keys[-1] == "key1"

    def test_lru_eviction(self, engine):
        """With maxsize=3, adding 4th entry should evict the oldest."""
        engine._promote_to_memory("k1", {"d": 1})
        engine._promote_to_memory("k2", {"d": 2})
        engine._promote_to_memory("k3", {"d": 3})
        assert len(engine._memory_cache) == 3
        engine._promote_to_memory("k4", {"d": 4})
        assert len(engine._memory_cache) == 3
        assert "k1" not in engine._memory_cache
        assert "k4" in engine._memory_cache

    def test_lru_access_prevents_eviction(self, engine):
        """Accessing an entry moves it to end, preventing eviction."""
        engine._promote_to_memory("k1", {"d": 1})
        engine._promote_to_memory("k2", {"d": 2})
        engine._promote_to_memory("k3", {"d": 3})
        # Access k1 to move it to end
        engine._promote_to_memory("k1", {"d": 1})
        # Add k4 — should evict k2 (oldest), not k1
        engine._promote_to_memory("k4", {"d": 4})
        assert "k1" in engine._memory_cache
        assert "k2" not in engine._memory_cache

    def test_combined_hit_rate(self, engine):
        """Test combined hit rate calculation."""
        engine._memory_hits = 3
        engine._memory_misses = 7
        engine._disk_hits = 2
        stats = engine.cache_stats
        assert stats["combined_hit_rate_pct"] == 50.0  # (3+2)/10

    def test_maxsize_minimum_one(self, tmp_path):
        """memory_cache_maxsize should be at least 1."""
        with patch("inference.inference_engine.ModelFactory") as MockFactory, \
             patch("inference.inference_engine.PromptLoader"):
            factory = MockFactory.return_value
            factory.primary.model_name = "test-model"
            from inference.inference_engine import InferenceEngine
            engine = InferenceEngine(
                model_factory=factory,
                cache_dir=str(tmp_path / "cache"),
                memory_cache_maxsize=0,  # Should be clamped to 1
            )
            assert engine._memory_cache_maxsize == 1


# ===========================================================================
# P3: Cost Budget Enforcement in Pipeline
# ===========================================================================

class TestCostBudget:
    """Test hard budget enforcement in Pipeline."""

    @pytest.fixture
    def pipeline(self):
        from pipeline import Pipeline
        agent_01 = MagicMock()
        agent_02 = MagicMock()
        agent_02.stt_cache = MagicMock()
        agent_02.stt_cache.stats = {"hits": 0, "misses": 0, "saves": 0,
                                    "errors": 0, "total_lookups": 0,
                                    "hit_rate_pct": 0.0}
        agent_03 = MagicMock()
        agent_04 = MagicMock()
        return Pipeline(agent_01, agent_02, agent_03, agent_04,
                        max_budget_usd=1.00)

    def test_budget_stored(self, pipeline):
        assert pipeline._max_budget_usd == 1.00

    def test_budget_zero_is_unlimited(self):
        from pipeline import Pipeline
        p = Pipeline(MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                     max_budget_usd=0)
        assert p._max_budget_usd == 0

    def test_80pct_warning_flag_initial(self, pipeline):
        assert pipeline._budget_80_warned is False

    def test_budget_default_zero(self):
        from pipeline import Pipeline
        p = Pipeline(MagicMock(), MagicMock(), MagicMock(), MagicMock())
        assert p._max_budget_usd == 0.0


# ===========================================================================
# P1: STT Cache Integration in ElevenLabsSTTAgent
# ===========================================================================

class TestSTTCacheIntegration:
    """Test STT cache integration in agent_02_transcription."""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        # Build a minimal Scribe v2 result
        result = MagicMock()
        result.words = []
        result.text = "Hello world"
        result.language_code = "en"
        client.speech_to_text.convert.return_value = result
        return client

    @pytest.fixture
    def agent_with_cache(self, mock_client, tmp_path):
        from agents.agent_02_transcription import ElevenLabsSTTAgent
        return ElevenLabsSTTAgent(
            client=mock_client,
            persist_transcripts=False,
            stt_cache_dir=str(tmp_path / "stt_cache"),
            enable_stt_cache=True,
            stt_cache_ttl_days=1,
        )

    @pytest.fixture
    def agent_no_cache(self, mock_client, tmp_path):
        from agents.agent_02_transcription import ElevenLabsSTTAgent
        return ElevenLabsSTTAgent(
            client=mock_client,
            persist_transcripts=False,
            enable_stt_cache=False,
        )

    @pytest.fixture
    def fake_audio(self, tmp_path):
        f = tmp_path / "call1.mp3"
        f.write_bytes(b"ID3" + b"\x00" * 100)
        return f

    def test_stt_cache_property(self, agent_with_cache):
        assert agent_with_cache.stt_cache is not None
        assert agent_with_cache.stt_cache.enabled is True

    def test_stt_cache_disabled(self, agent_no_cache):
        assert agent_no_cache.stt_cache.enabled is False

    def test_batch_marks_cached_false_on_api_call(self, agent_with_cache, fake_audio):
        """First call should hit API and mark cached=False."""
        results = agent_with_cache.transcribe_batch([fake_audio])
        filename = fake_audio.name
        assert results[filename]["cached"] is False
        assert results[filename]["status"] == "Success"

    def test_batch_uses_cache_on_second_call(self, agent_with_cache, fake_audio, mock_client):
        """Second call to same file should use cache (cached=True, no API call)."""
        # First call — hits API
        agent_with_cache.transcribe_batch([fake_audio])
        mock_client.speech_to_text.convert.reset_mock()

        # Second call — should use cache
        results = agent_with_cache.transcribe_batch([fake_audio])
        filename = fake_audio.name
        assert results[filename]["cached"] is True
        assert results[filename]["cost_usd"] == 0.0
        assert results[filename]["credits_used"] == 0
        # API should NOT have been called
        mock_client.speech_to_text.convert.assert_not_called()

    def test_cache_stats_after_batch(self, agent_with_cache, fake_audio):
        agent_with_cache.transcribe_batch([fake_audio])
        stats = agent_with_cache.stt_cache.stats
        assert stats["misses"] == 1
        assert stats["saves"] == 1

        agent_with_cache.transcribe_batch([fake_audio])
        stats = agent_with_cache.stt_cache.stats
        assert stats["hits"] == 1


# ===========================================================================
# P5: Enhanced Pipeline Summary
# ===========================================================================

class TestEnhancedSummary:
    """Test enhanced print_summary output."""

    @pytest.fixture
    def pipeline(self):
        from pipeline import Pipeline
        agent_01 = MagicMock()
        agent_02 = MagicMock()
        agent_02.stt_cache = MagicMock()
        agent_02.stt_cache.stats = {"hits": 2, "misses": 3, "saves": 3,
                                    "errors": 0, "total_lookups": 5,
                                    "hit_rate_pct": 40.0}
        agent_03 = MagicMock()
        agent_03._engine = MagicMock()
        agent_03._engine.cache_stats = {
            "memory_hits": 1, "memory_misses": 4,
            "disk_hits": 2, "disk_misses": 2,
            "total_lookups": 5, "combined_hit_rate_pct": 60.0,
            "memory_cache_size": 3, "memory_cache_maxsize": 128,
        }
        agent_04 = MagicMock()
        p = Pipeline(agent_01, agent_02, agent_03, agent_04, max_budget_usd=10.0)
        p._stt_cost = 0.50
        p._providers_used = {"openrouter"}
        return p

    def test_summary_no_evaluations(self, pipeline, caplog):
        import logging
        with caplog.at_level(logging.INFO, logger="qa_system.pipeline"):
            pipeline.print_summary([])
        assert any("No evaluations" in msg for msg in caplog.messages)

    def test_summary_with_scores(self, pipeline, caplog):
        evals = [
            {"filename": "call1.mp3", "call_type": "First Call",
             "overall_score": 85.0, "cost_usd": 0.003,
             "tokens_used": {"input": 1500, "output": 800}},
            {"filename": "call2.mp3", "call_type": "Follow-up Call",
             "overall_score": 72.0, "cost_usd": 0.003,
             "tokens_used": {"input": 1400, "output": 700}},
        ]
        import logging
        with caplog.at_level(logging.INFO, logger="qa_system.pipeline"):
            pipeline.print_summary(evals)
        combined = " ".join(caplog.messages)
        assert "PIPELINE SUMMARY" in combined
        assert "85.0" in combined
        assert "72.0" in combined

    def test_summary_budget_section(self, pipeline, caplog):
        evals = [
            {"filename": "c.mp3", "call_type": "First Call",
             "overall_score": 90.0, "cost_usd": 0.005,
             "tokens_used": {"input": 1000, "output": 500}},
        ]
        import logging
        with caplog.at_level(logging.INFO, logger="qa_system.pipeline"):
            pipeline.print_summary(evals)
        combined = " ".join(caplog.messages)
        assert "$10.00" in combined  # Budget
        assert "Remaining" in combined

    def test_summary_no_budget(self, caplog):
        from pipeline import Pipeline
        agent_02 = MagicMock()
        agent_02.stt_cache = MagicMock()
        agent_02.stt_cache.stats = {"hits": 0, "misses": 0, "saves": 0,
                                    "errors": 0, "total_lookups": 0,
                                    "hit_rate_pct": 0.0}
        p = Pipeline(MagicMock(), agent_02, MagicMock(), MagicMock(),
                     max_budget_usd=0)
        evals = [
            {"filename": "c.mp3", "call_type": "First Call",
             "overall_score": 90.0, "cost_usd": 0.005,
             "tokens_used": {"input": 1000, "output": 500}},
        ]
        import logging
        with caplog.at_level(logging.INFO, logger="qa_system.pipeline"):
            p.print_summary(evals)
        combined = " ".join(caplog.messages)
        assert "Budget" not in combined
