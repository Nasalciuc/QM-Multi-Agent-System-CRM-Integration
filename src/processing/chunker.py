"""
Transcript Chunker

Handles transcripts that exceed the LLM's token limit by:
  - Truncating with smart preservation (keep beginning + end)
  - The beginning contains greeting/intro criteria
  - The end contains closing/farewell criteria

Start with truncation; chunk-and-merge scoring is a future option.
"""

import logging
from typing import Optional

from src.processing.token_counter import TokenCounter

logger = logging.getLogger("qa_system.processing")

TRUNCATION_MARKER = "\n\n[...transcript truncated due to length...]\n\n"

# Default ratios for preserving call structure (greeting at start, closing at end)
DEFAULT_KEEP_START_RATIO = 0.6
DEFAULT_KEEP_END_RATIO = 0.4


class TranscriptChunker:
    """Truncate or split long transcripts for LLM evaluation.

    Usage:
        chunker = TranscriptChunker(max_tokens=30000)
        result = chunker.truncate(transcript)
        if result["truncated"]:
            print(f"Removed {result['removed_tokens']} tokens")
    """

    def __init__(
        self,
        max_tokens: int = 30000,
        keep_start_ratio: float = DEFAULT_KEEP_START_RATIO,
        keep_end_ratio: float = DEFAULT_KEEP_END_RATIO,
        token_counter: Optional[TokenCounter] = None,
    ):
        """
        Args:
            max_tokens: Maximum token count for output.
            keep_start_ratio: Fraction of budget for the beginning (greeting criteria).
            keep_end_ratio: Fraction of budget for the end (closing criteria).
            token_counter: TokenCounter instance. Created if not provided.
        """
        self.max_tokens = max_tokens
        self.keep_start_ratio = keep_start_ratio
        self.keep_end_ratio = keep_end_ratio
        self._counter = token_counter or TokenCounter()

    def truncate(self, transcript: str) -> dict:
        """Truncate transcript if it exceeds max_tokens.

        Preserves the beginning and end of the call (which contain
        greeting and closing evaluation criteria).

        Returns:
            Dict with:
              - text: truncated (or original) transcript
              - truncated: bool
              - original_tokens: int
              - final_tokens: int
              - removed_tokens: int
        """
        original_tokens = self._counter.count_tokens(transcript)

        if original_tokens <= self.max_tokens:
            return {
                "text": transcript,
                "truncated": False,
                "original_tokens": original_tokens,
                "final_tokens": original_tokens,
                "removed_tokens": 0,
            }

        # Reserve tokens for the truncation marker
        marker_tokens = self._counter.count_tokens(TRUNCATION_MARKER)
        budget = self.max_tokens - marker_tokens

        start_budget = int(budget * self.keep_start_ratio)
        end_budget = budget - start_budget

        # Split by lines to avoid cutting mid-sentence
        lines = transcript.split("\n")

        # Build start portion
        start_lines = []
        start_tokens = 0
        for line in lines:
            line_tokens = self._counter.count_tokens(line + "\n")
            if start_tokens + line_tokens > start_budget:
                break
            start_lines.append(line)
            start_tokens += line_tokens

        # Build end portion (from the end)
        end_lines = []
        end_tokens = 0
        for line in reversed(lines):
            line_tokens = self._counter.count_tokens(line + "\n")
            if end_tokens + line_tokens > end_budget:
                break
            end_lines.insert(0, line)
            end_tokens += line_tokens

        # Combine
        truncated_text = "\n".join(start_lines) + TRUNCATION_MARKER + "\n".join(end_lines)
        final_tokens = self._counter.count_tokens(truncated_text)

        logger.warning(
            f"Transcript truncated: {original_tokens} → {final_tokens} tokens "
            f"(removed {original_tokens - final_tokens})"
        )

        return {
            "text": truncated_text,
            "truncated": True,
            "original_tokens": original_tokens,
            "final_tokens": final_tokens,
            "removed_tokens": original_tokens - final_tokens,
        }
