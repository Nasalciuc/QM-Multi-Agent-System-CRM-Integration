"""
Model Factory — Creates LLM Clients with Automatic Fallback

Loads config/models.yaml, instantiates the primary + fallback providers,
and exposes chat_with_fallback() that tries each provider in order.

This is the single most important missing piece — if OpenRouter is down,
the system now falls through to Claude Sonnet, then direct OpenAI.
"""

import os
import logging
from typing import List, Optional

import yaml
from pathlib import Path

from core.base_llm import BaseLLM, LLMResponse
from core.openai_client import OpenAIClient

logger = logging.getLogger("qa_system.core")


class ModelFactory:
    """Factory that creates LLM clients from config and provides fallback routing.

    Usage:
        factory = ModelFactory()                         # loads config/models.yaml
        response = factory.chat_with_fallback(sys, usr)  # tries primary, then fallbacks
    """

    def __init__(self, config_path: str = "config/models.yaml"):
        self._config_path = config_path
        self._config = self._load_config(config_path)
        self._providers: List[BaseLLM] = []
        self._primary: Optional[BaseLLM] = None
        self._build_providers()

    # ── Public API ──────────────────────────────────────────────────

    @property
    def primary(self) -> BaseLLM:
        """The primary (first-choice) LLM provider."""
        if self._primary is None:
            raise RuntimeError("No LLM providers configured")
        return self._primary

    @property
    def providers(self) -> List[BaseLLM]:
        """All configured providers in priority order."""
        return list(self._providers)

    @property
    def token_limits(self) -> dict:
        """Token limits from config."""
        return self._config.get("token_limits", {
            "max_input_tokens": 30000,
            "max_output_tokens": 4096,
            "cost_warning_threshold_usd": 0.50,
        })

    @property
    def primary_pricing(self) -> dict:
        """Pricing from the primary provider (input_per_1m, output_per_1m)."""
        if self._primary is not None:
            return self._primary.pricing
        return {"input_per_1m": 0.0, "output_per_1m": 0.0}

    def chat_with_fallback(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Try primary provider first, then each fallback in order.

        Args:
            system_prompt: System-level instructions.
            user_prompt: User message content.
            temperature: LLM temperature.
            max_tokens: Max response tokens.
            json_mode: Request JSON output format.

        Returns:
            LLMResponse from the first successful provider.

        Raises:
            RuntimeError: If all providers fail.
        """
        errors = []

        for provider in self._providers:
            try:
                logger.info(f"Trying provider: {provider.provider_name} ({provider.model_name})")
                response = provider.chat(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )
                if provider != self._primary and self._primary is not None:
                    logger.warning(
                        f"Used fallback provider: {provider.provider_name} "
                        f"(primary {self._primary.provider_name} was unavailable)"
                    )
                return response

            except Exception as e:
                logger.warning(f"Provider {provider.provider_name} failed: {e}")
                errors.append((provider.provider_name, str(e)))
                continue

        error_details = "; ".join(f"{name}: {err}" for name, err in errors)
        raise RuntimeError(f"All LLM providers failed: {error_details}")

    # ── Private ─────────────────────────────────────────────────────

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Load models.yaml config."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _build_providers(self) -> None:
        """Instantiate primary and fallback providers from config."""
        # Primary
        primary_cfg = self._config.get("primary")
        if not primary_cfg:
            raise ValueError("No 'primary' model defined in models.yaml")

        primary_client = self._create_client(primary_cfg)
        self._primary = primary_client
        self._providers.append(primary_client)

        # Fallbacks
        for fallback_cfg in self._config.get("fallbacks", []):
            api_key = os.environ.get(fallback_cfg["api_key_env"], "")
            if not api_key:
                logger.info(
                    f"Skipping fallback {fallback_cfg['provider']}: "
                    f"env var {fallback_cfg['api_key_env']} not set"
                )
                continue
            try:
                client = self._create_client(fallback_cfg)
                self._providers.append(client)
            except Exception as e:
                logger.warning(f"Failed to create fallback {fallback_cfg['provider']}: {e}")

        logger.info(
            f"ModelFactory ready | {len(self._providers)} provider(s): "
            f"{[p.provider_name for p in self._providers]}"
        )

    @staticmethod
    def _create_client(cfg: dict) -> OpenAIClient:
        """Create an OpenAIClient from a provider config dict."""
        api_key = os.environ.get(cfg["api_key_env"], "")
        if not api_key:
            raise ValueError(f"Missing env var: {cfg['api_key_env']}")

        return OpenAIClient(
            base_url=cfg["base_url"],
            api_key=api_key,
            model=cfg["model"],
            provider=cfg["provider"],
            pricing=cfg.get("pricing", {}),
        )
