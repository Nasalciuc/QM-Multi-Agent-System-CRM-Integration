"""
Tests for src/core/model_factory.py

Tests provider creation, fallback behavior, config validation.
"""
import sys
import os
import json
import yaml
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.model_factory import ModelFactory
from core.base_llm import LLMResponse


# --- Fixtures ---

@pytest.fixture
def models_config(tmp_path):
    """Create a minimal models.yaml config file."""
    config = {
        "primary": {
            "provider": "test-primary",
            "model": "test-model",
            "base_url": "https://test.api/v1",
            "api_key_env": "TEST_PRIMARY_KEY",
            "pricing": {"input_per_1m": 2.50, "output_per_1m": 10.00},
        },
        "fallbacks": [
            {
                "provider": "test-fallback",
                "model": "fallback-model",
                "base_url": "https://fallback.api/v1",
                "api_key_env": "TEST_FALLBACK_KEY",
                "pricing": {"input_per_1m": 3.00, "output_per_1m": 15.00},
            },
        ],
        "token_limits": {
            "max_input_tokens": 30000,
            "max_output_tokens": 4096,
            "cost_warning_threshold_usd": 0.50,
        },
    }
    config_path = tmp_path / "models.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)


@pytest.fixture
def models_no_fallback(tmp_path):
    """Config with primary only, no fallbacks."""
    config = {
        "primary": {
            "provider": "test-only",
            "model": "test-model",
            "base_url": "https://test.api/v1",
            "api_key_env": "TEST_PRIMARY_KEY",
            "pricing": {"input_per_1m": 2.50, "output_per_1m": 10.00},
        },
        "token_limits": {
            "max_input_tokens": 30000,
            "max_output_tokens": 4096,
        },
    }
    config_path = tmp_path / "models_simple.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)


# --- Tests: Config Loading ---

class TestConfigLoading:

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            ModelFactory(config_path="nonexistent.yaml")

    def test_missing_primary_raises(self, tmp_path):
        config_path = tmp_path / "bad.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"fallbacks": []}, f)
        with pytest.raises(ValueError, match="No 'primary'"):
            with patch.dict(os.environ, {}, clear=False):
                ModelFactory(config_path=str(config_path))

    def test_missing_api_key_raises(self, models_config):
        """Primary provider without env var should raise."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing env var"):
                ModelFactory(config_path=models_config)


# --- Tests: Provider Creation ---

class TestProviderCreation:

    def test_primary_created(self, models_config):
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": "test-key"}):
            factory = ModelFactory(config_path=models_config)
            assert factory.primary.provider_name == "test-primary"
            assert factory.primary.model_name == "test-model"

    def test_fallback_skipped_without_key(self, models_config):
        """Fallback without env var should be skipped, not error."""
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": "test-key"}, clear=True):
            factory = ModelFactory(config_path=models_config)
            assert len(factory.providers) == 1  # primary only

    def test_fallback_added_with_key(self, models_config):
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": "k1", "TEST_FALLBACK_KEY": "k2"}):
            factory = ModelFactory(config_path=models_config)
            assert len(factory.providers) == 2

    def test_primary_only_no_fallbacks(self, models_no_fallback):
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": "test-key"}):
            factory = ModelFactory(config_path=models_no_fallback)
            assert len(factory.providers) == 1


# --- Tests: Token Limits & Pricing ---

class TestTokenLimits:

    def test_token_limits(self, models_config):
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": "test-key"}):
            factory = ModelFactory(config_path=models_config)
            limits = factory.token_limits
            assert limits["max_input_tokens"] == 30000
            assert limits["max_output_tokens"] == 4096

    def test_primary_pricing(self, models_config):
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": "test-key"}):
            factory = ModelFactory(config_path=models_config)
            pricing = factory.primary_pricing
            assert pricing["input_per_1m"] == 2.50
            assert pricing["output_per_1m"] == 10.00


# --- Tests: Fallback Behavior ---

class TestFallbackBehavior:

    def test_primary_success_no_fallback(self, models_config):
        """When primary succeeds, fallback is not called."""
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": "k1", "TEST_FALLBACK_KEY": "k2"}):
            factory = ModelFactory(config_path=models_config)
            expected_response = LLMResponse(
                text="ok", input_tokens=10, output_tokens=5,
                cost_usd=0.001, model="test-model",
                provider="test-primary", elapsed_seconds=0.5,
            )
            factory.providers[0].chat = MagicMock(return_value=expected_response)
            factory.providers[1].chat = MagicMock()

            result = factory.chat_with_fallback("sys", "usr")
            assert result.provider == "test-primary"
            factory.providers[1].chat.assert_not_called()

    def test_fallback_used_on_primary_failure(self, models_config):
        """When primary fails, fallback should be used."""
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": "k1", "TEST_FALLBACK_KEY": "k2"}):
            factory = ModelFactory(config_path=models_config)
            factory.providers[0].chat = MagicMock(side_effect=RuntimeError("primary down"))
            fallback_response = LLMResponse(
                text="ok", input_tokens=10, output_tokens=5,
                cost_usd=0.002, model="fallback-model",
                provider="test-fallback", elapsed_seconds=0.5,
            )
            factory.providers[1].chat = MagicMock(return_value=fallback_response)

            result = factory.chat_with_fallback("sys", "usr")
            assert result.provider == "test-fallback"

    def test_all_providers_fail_raises(self, models_config):
        """When all providers fail, should raise RuntimeError."""
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": "k1", "TEST_FALLBACK_KEY": "k2"}):
            factory = ModelFactory(config_path=models_config)
            factory.providers[0].chat = MagicMock(side_effect=RuntimeError("down"))
            factory.providers[1].chat = MagicMock(side_effect=RuntimeError("also down"))

            with pytest.raises(RuntimeError, match="All LLM providers failed"):
                factory.chat_with_fallback("sys", "usr")
