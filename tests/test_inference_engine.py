"""
Tests for src/inference/inference_engine.py

Tests cache sanitization (CRIT-1), atomic writes (CRIT-2),
cache key with model (MED-1), JSON serializer (CRIT-4).
"""
import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inference.inference_engine import InferenceEngine, _json_serializer
from core.base_llm import LLMResponse


# --- Fixtures ---

@pytest.fixture
def mock_factory():
    factory = MagicMock()
    factory.primary.model_name = "gpt-4o-2024-11-20"
    factory.primary.provider_name = "openrouter"
    factory.token_limits = {"max_input_tokens": 30000, "max_output_tokens": 4096}
    factory.primary_pricing = {"input_per_1m": 2.50, "output_per_1m": 10.00}
    return factory


@pytest.fixture
def engine(mock_factory, tmp_path):
    return InferenceEngine(
        model_factory=mock_factory,
        cache_dir=str(tmp_path / "cache"),
        enable_cache=True,
    )


@pytest.fixture
def engine_no_cache(mock_factory):
    return InferenceEngine(
        model_factory=mock_factory,
        enable_cache=False,
    )


@pytest.fixture
def sample_llm_response():
    return LLMResponse(
        text='{"criteria": {}, "overall_assessment": "Good", "strengths": [], "improvements": []}',
        input_tokens=5000,
        output_tokens=1500,
        cost_usd=0.0275,
        model="gpt-4o-2024-11-20",
        provider="openrouter",
        elapsed_seconds=2.5,
        raw_response=MagicMock(),  # should NOT be saved to cache
    )


# --- Tests: Cache Sanitization (CRIT-1) ---

class TestCacheSanitization:

    def test_cache_excludes_raw_response(self, engine, tmp_path):
        """CRIT-1: raw_response should never be written to cache."""
        data = {
            "criteria": {"test": {"score": "YES"}},
            "overall_assessment": "Good",
            "model_used": "gpt-4o",
            "raw_response": "SHOULD_NOT_APPEAR",
            "secret_key": "ALSO_SHOULD_NOT_APPEAR",
        }
        key = "test_key_123"
        engine._save_cache(key, data)

        cached_path = tmp_path / "cache" / f"{key}.json"
        assert cached_path.exists()
        with open(cached_path) as f:
            cached = json.load(f)

        assert "raw_response" not in cached
        assert "secret_key" not in cached
        assert "criteria" in cached
        assert "overall_assessment" in cached
        assert "model_used" in cached

    def test_cache_only_whitelisted_keys(self, engine, tmp_path):
        """CRIT-1: Only CACHE_SAFE_KEYS should persist."""
        data = {
            "criteria": {"test": "value"},
            "overall_assessment": "OK",
            "strengths": ["a"],
            "improvements": ["b"],
            "critical_gaps": [],
            "call_type": "First Call",
            "model_used": "gpt-4o",
            "provider_used": "openrouter",
            "tokens_used": {"input": 100, "output": 50},
            "cost_usd": 0.01,
            "eval_time_seconds": 1.5,
            "dangerous_field": "NOPE",
            "api_key": "sk-secret",
        }
        engine._save_cache("safe_key", data)
        cached = engine._load_cache("safe_key")
        assert cached is not None
        assert "dangerous_field" not in cached
        assert "api_key" not in cached
        assert cached["criteria"] == {"test": "value"}


# --- Tests: Atomic Writes (CRIT-2) ---

class TestAtomicWrites:

    def test_cache_file_created_atomically(self, engine, tmp_path):
        """CRIT-2: Cache should use atomic write (temp file + rename)."""
        data = {"criteria": {}, "model_used": "test"}
        engine._save_cache("atomic_test", data)

        cache_dir = tmp_path / "cache"
        # No .tmp files should remain
        tmp_files = list(cache_dir.glob("*.tmp"))
        assert len(tmp_files) == 0
        # But the final file should exist
        assert (cache_dir / "atomic_test.json").exists()

    def test_cache_load_returns_none_for_corrupt(self, engine, tmp_path):
        """Corrupt cache file should return None, not crash."""
        cache_dir = tmp_path / "cache"
        corrupt_path = cache_dir / "corrupt.json"
        corrupt_path.write_text("{invalid json")
        assert engine._load_cache("corrupt") is None


# --- Tests: Cache Key (MED-1) ---

class TestCacheKey:

    def test_cache_key_includes_model(self):
        """MED-1: Different models should produce different cache keys."""
        key1 = InferenceEngine._cache_key("transcript", "First Call", 24, model="gpt-4o")
        key2 = InferenceEngine._cache_key("transcript", "First Call", 24, model="claude-3")
        assert key1 != key2

    def test_cache_key_deterministic(self):
        key1 = InferenceEngine._cache_key("same", "same", 10, model="same")
        key2 = InferenceEngine._cache_key("same", "same", 10, model="same")
        assert key1 == key2

    def test_cache_key_includes_criteria_hash(self):
        """Cache invalidates when criteria content changes."""
        key1 = InferenceEngine._cache_key("t", "First Call", 2, model="m", criteria_hash="abc123")
        key2 = InferenceEngine._cache_key("t", "First Call", 2, model="m", criteria_hash="def456")
        assert key1 != key2

    def test_cache_key_includes_prompt_hash(self):
        """Cache invalidates when prompt template changes."""
        key1 = InferenceEngine._cache_key("t", "First Call", 2, model="m", prompt_hash="v1")
        key2 = InferenceEngine._cache_key("t", "First Call", 2, model="m", prompt_hash="v2")
        assert key1 != key2


# --- Tests: JSON Serializer (CRIT-4) ---

class TestJsonSerializer:

    def test_serializes_path(self):
        result = _json_serializer(Path("/app/data/file.txt"))
        # Path.__str__ uses OS-native separators; serializer uses str()
        assert "file.txt" in result

    def test_serializes_datetime(self):
        from datetime import datetime
        dt = datetime(2026, 2, 12, 10, 30, 0)
        result = _json_serializer(dt)
        assert "2026-02-12" in result

    def test_serializes_set(self):
        result = _json_serializer({"b", "a", "c"})
        assert result == ["a", "b", "c"]

    def test_rejects_unknown_type(self):
        with pytest.raises(TypeError, match="Non-serializable"):
            _json_serializer(MagicMock())


# --- Tests: Evaluate Flow ---

class TestEvaluateFlow:

    def test_evaluate_returns_metadata(self, engine, mock_factory):
        """Evaluate should return evaluation with metadata fields."""
        mock_response = LLMResponse(
            text=json.dumps({
                "criteria": {"test_crit": {"score": "YES", "evidence": "Good"}},
                "overall_assessment": "OK",
                "strengths": [],
                "improvements": [],
            }),
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            model="gpt-4o",
            provider="openrouter",
            elapsed_seconds=1.0,
        )
        mock_factory.chat_with_fallback.return_value = mock_response

        result = engine.evaluate(
            transcript="Agent: Hello\nClient: Hi there how are you",
            call_type="First Call",
            criteria={"test_crit": {"category": "phone_skills", "weight": 1.0, "description": "Test"}},
        )

        assert result["call_type"] == "First Call"
        assert result["model_used"] == "gpt-4o"
        assert result["provider_used"] == "openrouter"
        assert "cost_usd" in result

    def test_evaluate_caches_result(self, engine, mock_factory, tmp_path):
        """Results should be cached on success."""
        mock_response = LLMResponse(
            text=json.dumps({
                "criteria": {"c1": {"score": "YES", "evidence": "ok"}},
                "overall_assessment": "OK",
                "strengths": [],
                "improvements": [],
            }),
            input_tokens=100, output_tokens=50, cost_usd=0.01,
            model="gpt-4o", provider="openrouter", elapsed_seconds=1.0,
        )
        mock_factory.chat_with_fallback.return_value = mock_response

        engine.evaluate("transcript", "First Call", {"c1": {"category": "a", "weight": 1, "description": "d"}})

        # Second call should use cache (chat_with_fallback only called once)
        engine.evaluate("transcript", "First Call", {"c1": {"category": "a", "weight": 1, "description": "d"}})
        assert mock_factory.chat_with_fallback.call_count == 1

    def test_evaluate_returns_error_on_all_retries_fail(self, engine_no_cache, mock_factory):
        """Should return error dict when all retries fail."""
        mock_factory.chat_with_fallback.side_effect = RuntimeError("All providers down")

        result = engine_no_cache.evaluate(
            "transcript", "First Call",
            {"c1": {"category": "a", "weight": 1, "description": "d"}},
            max_retries=1,
        )
        assert "error" in result
