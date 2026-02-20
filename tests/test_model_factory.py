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

from core.model_factory import ModelFactory
from core.base_llm import (
    LLMResponse,
    LLMQuotaExhaustedError, LLMRateLimitError,
    LLMInvalidConfigError, LLMServerError,
)


class CustomException(Exception):
    """Custom exception with response and status_code attributes for testing."""
    def __init__(self, message, response=None, status_code=None):
        super().__init__(message)
        self.response = response
        # Only set status_code if explicitly provided
        if status_code is not None:
            self.status_code = status_code


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
            with pytest.raises(ValueError, match="Missing or empty env var"):
                ModelFactory(config_path=models_config)

    def test_whitespace_only_api_key_raises(self, models_config):
        """#29: Whitespace-only API key should fail validation."""
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": "   "}, clear=True):
            with pytest.raises(ValueError, match="Missing or empty env var"):
                ModelFactory(config_path=models_config)

    def test_empty_string_api_key_raises(self, models_config):
        """#29: Empty string API key should fail validation."""
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": ""}, clear=True):
            with pytest.raises(ValueError, match="Missing or empty env var"):
                ModelFactory(config_path=models_config)

    def test_tab_newline_api_key_raises(self, models_config):
        """#29: Tab/newline-only API key should fail validation."""
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": "\t\n"}, clear=True):
            with pytest.raises(ValueError, match="Missing or empty env var"):
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


# --- Tests: Typed Exception Handling (Fix #5) ---


class TestTypedExceptionHandling:

    def test_quota_exhausted_disables_provider(self, models_config):
        """LLMQuotaExhaustedError should disable the provider and fall through."""
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": "k1", "TEST_FALLBACK_KEY": "k2"}):
            factory = ModelFactory(config_path=models_config)
            factory.providers[0].chat = MagicMock(
                side_effect=LLMQuotaExhaustedError("quota gone")
            )
            fallback_resp = LLMResponse(
                text="ok", input_tokens=10, output_tokens=5,
                cost_usd=0.001, model="fallback-model",
                provider="test-fallback", elapsed_seconds=0.5,
            )
            factory.providers[1].chat = MagicMock(return_value=fallback_resp)

            result = factory.chat_with_fallback("sys", "usr")
            assert result.provider == "test-fallback"
            assert "test-primary" in factory._disabled_providers

    def test_invalid_config_disables_provider(self, models_config):
        """LLMInvalidConfigError should disable the provider."""
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": "k1", "TEST_FALLBACK_KEY": "k2"}):
            factory = ModelFactory(config_path=models_config)
            factory.providers[0].chat = MagicMock(
                side_effect=LLMInvalidConfigError("bad key")
            )
            fallback_resp = LLMResponse(
                text="ok", input_tokens=10, output_tokens=5,
                cost_usd=0.001, model="fallback-model",
                provider="test-fallback", elapsed_seconds=0.5,
            )
            factory.providers[1].chat = MagicMock(return_value=fallback_resp)

            result = factory.chat_with_fallback("sys", "usr")
            assert result.provider == "test-fallback"
            assert "test-primary" in factory._disabled_providers

    def test_disabled_provider_skipped_on_next_call(self, models_config):
        """A disabled provider should be skipped on subsequent calls."""
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": "k1", "TEST_FALLBACK_KEY": "k2"}):
            factory = ModelFactory(config_path=models_config)
            factory._disabled_providers.add("test-primary")
            fallback_resp = LLMResponse(
                text="ok", input_tokens=10, output_tokens=5,
                cost_usd=0.001, model="fallback-model",
                provider="test-fallback", elapsed_seconds=0.5,
            )
            factory.providers[1].chat = MagicMock(return_value=fallback_resp)
            factory.providers[0].chat = MagicMock()

            result = factory.chat_with_fallback("sys", "usr")
            assert result.provider == "test-fallback"
            # Primary should NOT have been called
            factory.providers[0].chat.assert_not_called()

    def test_reset_disabled_providers(self, models_config):
        """reset_disabled_providers() should clear the disabled set."""
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": "k1", "TEST_FALLBACK_KEY": "k2"}):
            factory = ModelFactory(config_path=models_config)
            factory._disabled_providers.add("test-primary")
            factory.reset_disabled_providers()
            assert len(factory._disabled_providers) == 0

    def test_rate_limit_retries(self, models_config):
        """LLMRateLimitError should retry once before falling through."""
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": "k1", "TEST_FALLBACK_KEY": "k2"}):
            factory = ModelFactory(config_path=models_config)
            success_resp = LLMResponse(
                text="ok", input_tokens=10, output_tokens=5,
                cost_usd=0.001, model="test-model",
                provider="test-primary", elapsed_seconds=0.5,
            )
            # First call: rate limited, retry succeeds
            factory.providers[0].chat = MagicMock(
                side_effect=[LLMRateLimitError("wait", retry_after=0.01), success_resp]
            )
            result = factory.chat_with_fallback("sys", "usr")
            # Called once in loop + once on retry = 2 calls total
            assert factory.providers[0].chat.call_count == 2
            assert result.provider == "test-primary"

    def test_server_error_falls_through(self, models_config):
        """LLMServerError should fall through to fallback."""
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": "k1", "TEST_FALLBACK_KEY": "k2"}):
            factory = ModelFactory(config_path=models_config)
            factory.providers[0].chat = MagicMock(
                side_effect=LLMServerError("500 error")
            )
            fallback_resp = LLMResponse(
                text="ok", input_tokens=10, output_tokens=5,
                cost_usd=0.001, model="fallback-model",
                provider="test-fallback", elapsed_seconds=0.5,
            )
            factory.providers[1].chat = MagicMock(return_value=fallback_resp)

            result = factory.chat_with_fallback("sys", "usr")
            assert result.provider == "test-fallback"
            # Provider should NOT be disabled (server error is transient)
            assert "test-primary" not in factory._disabled_providers
    def test_402_payment_required_disables_provider(self, models_config):
        """402 Payment Required should disable the provider like quota exhaustion."""
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": "k1", "TEST_FALLBACK_KEY": "k2"}):
            factory = ModelFactory(config_path=models_config)
            factory.providers[0].chat = MagicMock(
                side_effect=LLMQuotaExhaustedError("credits exhausted")
            )
            fallback_resp = LLMResponse(
                text="ok", input_tokens=10, output_tokens=5,
                cost_usd=0.001, model="fallback-model",
                provider="test-fallback", elapsed_seconds=0.5,
            )
            factory.providers[1].chat = MagicMock(return_value=fallback_resp)

            result = factory.chat_with_fallback("sys", "usr")
            assert result.provider == "test-fallback"
            assert "test-primary" in factory._disabled_providers

            # Second call should skip primary entirely
            factory.providers[0].chat.reset_mock()
            result2 = factory.chat_with_fallback("sys", "usr")
            factory.providers[0].chat.assert_not_called()

    def test_invalid_model_id_disables_provider(self, models_config):
        """'not a valid model ID' should be classified as InvalidConfig."""
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": "k1", "TEST_FALLBACK_KEY": "k2"}):
            factory = ModelFactory(config_path=models_config)
            factory.providers[0].chat = MagicMock(
                side_effect=LLMInvalidConfigError("not a valid model ID")
            )
            fallback_resp = LLMResponse(
                text="ok", input_tokens=10, output_tokens=5,
                cost_usd=0.001, model="fallback-model",
                provider="test-fallback", elapsed_seconds=0.5,
            )
            factory.providers[1].chat = MagicMock(return_value=fallback_resp)

            result = factory.chat_with_fallback("sys", "usr")
            assert "test-primary" in factory._disabled_providers


# --- Tests: CRIT-02 — 402 Status Extraction from Response Object ---

class TestHTTP402StatusExtraction:

    def test_402_from_response_object_raises_quota_exhausted(self):
        """CRIT-02: When status_code is on exc.response (not exc directly),
        _classify_and_raise should still detect 402 and raise LLMQuotaExhaustedError."""
        from core.openai_client import OpenAIClient

        # Create mock exception where status_code is ONLY on .response
        mock_response = MagicMock()
        mock_response.status_code = 402
        mock_exc = CustomException("Error code: 402 - Payment Required", response=mock_response)

        # Ensure direct status_code is not set
        assert not hasattr(mock_exc, "status_code")

        client = MagicMock(spec=OpenAIClient)
        client._provider = "test-openrouter"
        # Call the real method
        with pytest.raises(LLMQuotaExhaustedError, match="Payment required"):
            OpenAIClient._classify_and_raise(client, mock_exc)

    def test_402_from_direct_status_code_raises_quota_exhausted(self):
        """402 directly on exc.status_code should also raise LLMQuotaExhaustedError."""
        from core.openai_client import OpenAIClient

        mock_exc = CustomException("requires more credits", status_code=402)

        client = MagicMock(spec=OpenAIClient)
        client._provider = "test-openrouter"
        with pytest.raises(LLMQuotaExhaustedError):
            OpenAIClient._classify_and_raise(client, mock_exc)

    def test_402_disables_provider_in_factory(self, models_config):
        """CRIT-02: Full integration — 402 should disable provider AND skip on next call."""
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": "k1", "TEST_FALLBACK_KEY": "k2"}):
            factory = ModelFactory(config_path=models_config)
            factory.providers[0].chat = MagicMock(
                side_effect=LLMQuotaExhaustedError("[openrouter] Payment required / credits exhausted: 402")
            )
            fallback_resp = LLMResponse(
                text="ok", input_tokens=10, output_tokens=5,
                cost_usd=0.001, model="fallback-model",
                provider="test-fallback", elapsed_seconds=0.5,
            )
            factory.providers[1].chat = MagicMock(return_value=fallback_resp)

            result = factory.chat_with_fallback("sys", "usr")
            assert result.provider == "test-fallback"
            assert "test-primary" in factory._disabled_providers

            # Second call should skip primary entirely
            factory.providers[0].chat.reset_mock()
            result2 = factory.chat_with_fallback("sys", "usr")
            factory.providers[0].chat.assert_not_called()
            assert result2.provider == "test-fallback"