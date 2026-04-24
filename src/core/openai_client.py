"""
OpenAI-Compatible LLM Client

Works with both OpenRouter and direct OpenAI — they use the same SDK,
just different base_url and api_key values.

Replaces the ~30 lines of API call + parsing in agent_03.evaluate_call().
"""

import time
import logging
from typing import Dict, Optional

from openai import OpenAI

from core.base_llm import (
    BaseLLM, LLMResponse,
    LLMQuotaExhaustedError, LLMRateLimitError,
    LLMInvalidConfigError, LLMServerError,
)
from structured_logger import emit_metric

logger = logging.getLogger("qa_system.core")


class OpenAIClient(BaseLLM):
    """OpenAI-compatible LLM client (OpenRouter / direct OpenAI).

    Usage:
        client = OpenAIClient(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            model="gpt-4o-2024-11-20",
            provider="openrouter",
            pricing={"input_per_1m": 2.50, "output_per_1m": 10.00},
        )
        response = client.chat(system_prompt, user_prompt, json_mode=True)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        provider: str,
        pricing: Optional[Dict[str, float]] = None,
        timeout: float = 120.0,
    ):
        self._base_url = base_url
        self._model = model
        self._provider = provider
        self._pricing = pricing or {"input_per_1m": 0.0, "output_per_1m": 0.0}

        # HIGH-6: Explicit timeout prevents indefinite hangs
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        logger.info(f"OpenAIClient initialized | provider={provider} model={model} base_url={base_url}")

    # ── BaseLLM interface ───────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send chat completion and return standardized LLMResponse."""
        kwargs = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        start = time.time()
        try:
            response = self._client.chat.completions.create(**kwargs)  # type: ignore[call-overload]
        except Exception as e:
            self._classify_and_raise(e)
        elapsed = time.time() - start

        # Parse response
        text = response.choices[0].message.content.strip()
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost = self.calculate_cost(input_tokens, output_tokens)

        logger.debug(
            f"LLM call | provider={self._provider} model={self._model} "
            f"in={input_tokens} out={output_tokens} cost=${cost:.4f} elapsed={elapsed:.1f}s"
        )

        # HIGH-02: Structured metric for successful LLM call
        emit_metric(
            "llm_call", provider=self._provider, model=self._model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost, elapsed_s=round(elapsed, 2), success=True,
        )

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            model=self._model,
            provider=self._provider,
            elapsed_seconds=round(elapsed, 2),
            raw_response=response,
        )

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD from token counts."""
        cost = (
            input_tokens / 1_000_000 * self._pricing["input_per_1m"]
            + output_tokens / 1_000_000 * self._pricing["output_per_1m"]
        )
        return round(cost, 6)

    def is_available(self) -> bool:
        """CRIT-NEW-3: Lightweight health check — list models instead of
        burning a real completion (which costs credits).

        Falls back to a minimal completion only if models.list() is
        not supported by the provider (e.g. some OpenRouter endpoints).
        """
        try:
            # Cheap auth-only call: list available models
            models = self._client.models.list()
            return bool(models)
        except Exception:
            # Fallback: some providers don't support models.list()
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                    temperature=0.0,
                )
                return bool(response.choices)
            except Exception as e:
                logger.warning(f"Health check failed for {self._provider}: {e}")
                return False

    def _classify_and_raise(self, exc: Exception) -> None:
        """Classify an OpenAI SDK exception into a typed LLM error and re-raise.

        Fix #5: Maps HTTP status codes and error messages to the
        appropriate LLMError subclass for smart fallback handling.
        """
        msg = str(exc).lower()

        # OpenAI SDK wraps HTTP errors in openai.APIStatusError subclasses
        status_code = getattr(exc, "status_code", None)

        # CRIT-02: Fallback — extract status_code from the httpx Response object
        # that the OpenAI SDK stores on some error types (e.g. APIStatusError).
        if status_code is None:
            response = getattr(exc, "response", None)
            if response is not None:
                status_code = getattr(response, "status_code", None)

        # HIGH-02: Structured metric for LLM errors
        emit_metric(
            "llm_error", provider=self._provider,
            error_type=type(exc).__name__, status_code=status_code,
        )

        if status_code == 401 or "authentication" in msg or "invalid api key" in msg:
            raise LLMInvalidConfigError(
                f"[{self._provider}] Authentication failed: {exc}"
            ) from exc

        # 402 Payment Required — OpenRouter sends this when credits are exhausted
        if status_code == 402 or "can only afford" in msg or "requires more credits" in msg:
            raise LLMQuotaExhaustedError(
                f"[{self._provider}] Payment required / credits exhausted: {exc}"
            ) from exc

        if status_code == 429:
            # Distinguish quota exhaustion from rate limiting
            if "quota" in msg or "billing" in msg or "exceeded" in msg:
                raise LLMQuotaExhaustedError(
                    f"[{self._provider}] Quota exhausted: {exc}"
                ) from exc
            retry_after = getattr(exc, "headers", {})
            if hasattr(retry_after, "get"):
                retry_after = float(retry_after.get("retry-after", 0)) or None
            else:
                retry_after = None
            raise LLMRateLimitError(
                f"[{self._provider}] Rate limited: {exc}",
                retry_after=retry_after,
            ) from exc

        if status_code is not None and status_code >= 500:
            raise LLMServerError(
                f"[{self._provider}] Server error ({status_code}): {exc}"
            ) from exc

        if "model" in msg and ("not found" in msg or "does not exist" in msg or "not a valid" in msg):
            raise LLMInvalidConfigError(
                f"[{self._provider}] Model not found: {exc}"
            ) from exc

        # Re-raise unknown errors as-is
        raise
