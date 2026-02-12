"""
Token Counter

Estimates token counts for transcripts and calculates expected costs
before making API calls. Prevents cost explosion from very long calls.

Uses tiktoken for accuracy when available, falls back to word-based estimate.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger("qa_system.processing")

# Try to import tiktoken for accurate counting
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None  # type: ignore[assignment]
    _TIKTOKEN_AVAILABLE = False
    logger.info("tiktoken not installed — using word-based token estimates")

# Fallback multiplier: multilingual safety margin ≈ 1.8 tokens per word
_WORD_TO_TOKEN_RATIO = 1.8


class TokenCounter:
    """Estimate token counts and costs for text content.

    Usage:
        counter = TokenCounter(model="gpt-4o-2024-11-20")
        info = counter.analyze("long transcript text...", max_tokens=30000)
        if info["needs_truncation"]:
            # handle truncation
    """

    def __init__(
        self,
        model: str = "gpt-4o-2024-11-20",
        pricing: Optional[Dict[str, float]] = None,
    ):
        self._model = model
        self._pricing = pricing or {"input_per_1m": 2.50, "output_per_1m": 10.00}
        self._encoder = None

        if _TIKTOKEN_AVAILABLE and tiktoken is not None:
            try:
                self._encoder = tiktoken.encoding_for_model(model)
            except KeyError:
                # Model not in tiktoken's registry — use cl100k_base (GPT-4 family)
                self._encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Uses tiktoken if available, otherwise word-based estimate.
        """
        if not text:
            return 0

        if self._encoder:
            return len(self._encoder.encode(text))

        # Fallback: word count * 1.3
        return int(len(text.split()) * _WORD_TO_TOKEN_RATIO)

    def analyze(self, text: str, max_tokens: int = 30000) -> Dict:
        """Analyze text for token count, cost estimate, and truncation need.

        Args:
            text: Input text (transcript).
            max_tokens: Maximum allowed input tokens.

        Returns:
            Dict with: token_count, needs_truncation, estimated_cost_usd,
                       excess_tokens, method ('tiktoken' or 'estimate').
        """
        token_count = self.count_tokens(text)
        needs_truncation = token_count > max_tokens
        excess = max(0, token_count - max_tokens)

        # Estimate cost (input only — output depends on response)
        estimated_input_cost = token_count / 1_000_000 * self._pricing["input_per_1m"]
        # Assume ~1500 output tokens for a typical QA evaluation response
        estimated_output_cost = 1500 / 1_000_000 * self._pricing["output_per_1m"]
        estimated_total = estimated_input_cost + estimated_output_cost

        return {
            "token_count": token_count,
            "needs_truncation": needs_truncation,
            "excess_tokens": excess,
            "max_tokens": max_tokens,
            "estimated_cost_usd": round(estimated_total, 4),
            "method": "tiktoken" if self._encoder else "estimate",
        }

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate USD cost from token counts."""
        cost = (
            input_tokens / 1_000_000 * self._pricing["input_per_1m"]
            + output_tokens / 1_000_000 * self._pricing["output_per_1m"]
        )
        return round(cost, 6)
