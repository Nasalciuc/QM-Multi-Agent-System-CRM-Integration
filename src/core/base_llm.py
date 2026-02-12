"""
Base LLM Abstract Class & LLMResponse Dataclass

Defines the contract that all LLM providers must implement.
Every provider returns the same LLMResponse shape.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str
    provider: str
    elapsed_seconds: float
    raw_response: Optional[object] = field(default=None, repr=False)


class BaseLLM(ABC):
    """Abstract base class for LLM providers.

    All providers must implement:
      - chat()           → send messages, get LLMResponse
      - calculate_cost() → compute USD cost from token counts
      - is_available()   → lightweight health check
    """

    @abstractmethod
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send a chat completion request and return standardized response.

        Args:
            system_prompt: System-level instructions.
            user_prompt: User message / transcript to evaluate.
            temperature: Sampling temperature (0.0–2.0).
            max_tokens: Maximum tokens in the response.
            json_mode: If True, request JSON output format.

        Returns:
            LLMResponse with text, token counts, cost, and metadata.

        Raises:
            Exception: On API errors after internal handling.
        """
        ...

    @abstractmethod
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate USD cost from token counts using provider pricing.

        Args:
            input_tokens: Number of prompt tokens.
            output_tokens: Number of completion tokens.

        Returns:
            Cost in USD (float).
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Lightweight health check — can this provider accept requests?

        Returns:
            True if the provider is reachable and authenticated.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier (e.g. 'openrouter', 'openai-direct')."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier string (e.g. 'gpt-4o-2024-11-20')."""
        ...

    @property
    def pricing(self) -> dict:
        """Pricing dict with input_per_1m and output_per_1m keys."""
        return getattr(self, '_pricing', {"input_per_1m": 0.0, "output_per_1m": 0.0})
