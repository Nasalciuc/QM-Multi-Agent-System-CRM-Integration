"""
Transcript Chunker

Handles transcripts that exceed the LLM's token limit by:
  - Truncating with smart preservation (keep beginning + middle + end)
  - REAL-07: The beginning contains greeting/intro criteria,
    the middle contains negotiation/presentation, the end contains closing.

Start with truncation; chunk-and-merge scoring is a future option.
"""

import logging
from typing import Optional

from processing.token_counter import TokenCounter

logger = logging.getLogger("qa_system.processing")

TRUNCATION_MARKER = "\n\n[...transcript truncated due to length...]\n\n"

# REAL-07: 30% beginning (greeting+interview) + 40% middle (negotiation) + 30% end (closing)
DEFAULT_KEEP_START_RATIO = 0.30
DEFAULT_KEEP_MIDDLE_RATIO = 0.40
DEFAULT_KEEP_END_RATIO = 0.30


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
        keep_middle_ratio: float = DEFAULT_KEEP_MIDDLE_RATIO,
        keep_end_ratio: float = DEFAULT_KEEP_END_RATIO,
        token_counter: Optional[TokenCounter] = None,
    ):
        """
        Args:
            max_tokens: Maximum token count for output.
            keep_start_ratio: Fraction of budget for the beginning (greeting/interview).
            keep_middle_ratio: Fraction of budget for the middle (negotiation/presentation).
            keep_end_ratio: Fraction of budget for the end (closing criteria).
            token_counter: TokenCounter instance. Created if not provided.
        """
        self.max_tokens = max_tokens
        self.keep_start_ratio = keep_start_ratio
        self.keep_middle_ratio = keep_middle_ratio
        self.keep_end_ratio = keep_end_ratio
        self._counter = token_counter or TokenCounter()

    def truncate(self, transcript: str) -> dict:
        """Truncate transcript if it exceeds max_tokens.

        REAL-07: Preserves beginning (30%), middle (40%), and end (30%)
        of the call so negotiation/presentation content is not lost.

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

        # Reserve tokens for TWO truncation markers
        marker_tokens = self._counter.count_tokens(TRUNCATION_MARKER) * 2
        budget = self.max_tokens - marker_tokens

        start_budget = int(budget * self.keep_start_ratio)
        middle_budget = int(budget * self.keep_middle_ratio)
        end_budget = budget - start_budget - middle_budget

        # Split by lines to avoid cutting mid-sentence
        lines = transcript.split("\n")
        total_lines = len(lines)

        # Build start portion
        start_lines = []
        start_tokens = 0
        for line in lines:
            line_tokens = self._counter.count_tokens(line + "\n")
            if start_tokens + line_tokens > start_budget:
                break
            start_lines.append(line)
            start_tokens += line_tokens
        start_end_idx = len(start_lines)

        # Build end portion (from the end)
        end_lines = []
        end_tokens = 0
        for line in reversed(lines):
            line_tokens = self._counter.count_tokens(line + "\n")
            if end_tokens + line_tokens > end_budget:
                break
            end_lines.insert(0, line)
            end_tokens += line_tokens
        end_start_idx = total_lines - len(end_lines)

        # Build middle portion (centered in the gap between start and end)
        if start_end_idx < end_start_idx:
            gap_lines = lines[start_end_idx:end_start_idx]
            mid_point = len(gap_lines) // 2
            middle_lines = []
            middle_tokens = 0
            # Expand outward from midpoint
            left = mid_point
            right = mid_point
            while left >= 0 or right < len(gap_lines):
                # Try right
                if right < len(gap_lines):
                    lt = self._counter.count_tokens(gap_lines[right] + "\n")
                    if middle_tokens + lt <= middle_budget:
                        middle_lines.append((right, gap_lines[right]))
                        middle_tokens += lt
                    right += 1
                # Try left
                if left >= 0 and left != mid_point:
                    lt = self._counter.count_tokens(gap_lines[left] + "\n")
                    if middle_tokens + lt <= middle_budget:
                        middle_lines.append((left, gap_lines[left]))
                        middle_tokens += lt
                    left -= 1
                else:
                    left -= 1
                if middle_tokens >= middle_budget:
                    break
            # Sort by position for correct order
            middle_lines.sort(key=lambda x: x[0])
            middle_text = [line for _, line in middle_lines]
        else:
            middle_text = []

        # Combine with markers
        parts = ["\n".join(start_lines)]
        parts.append(TRUNCATION_MARKER)
        if middle_text:
            parts.append("\n".join(middle_text))
            parts.append(TRUNCATION_MARKER)
        parts.append("\n".join(end_lines))
        truncated_text = "".join(parts)
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
