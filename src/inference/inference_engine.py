"""
Inference Engine

Orchestrates the full LLM evaluation cycle:
  build prompts → call LLM (with fallback) → parse response → validate → retry

Agent 03 becomes a thin wrapper that calls InferenceEngine.evaluate()
and then calculate_score().
"""

import hashlib
import json
import os
import tempfile
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Set

from src.core.model_factory import ModelFactory
from src.core.base_llm import LLMResponse
from src.prompts.templates import PromptLoader
from src.inference.response_parser import ResponseParser, ValidationError

logger = logging.getLogger("qa_system.inference")


def _json_serializer(obj):
    """Explicit JSON serializer — replaces dangerous `default=str`.

    CRIT-4: Only handles known types; raises TypeError for unexpected objects
    instead of silently stringifying them (which can corrupt data).
    """
    if isinstance(obj, Path):
        return str(obj)
    try:
        from datetime import datetime, date
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
    except ImportError:
        pass
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Non-serializable object: {type(obj).__name__}")


class InferenceEngine:
    """Orchestrate LLM-based evaluation with retries and caching.

    Usage:
        engine = InferenceEngine(model_factory, prompt_loader)
        result = engine.evaluate(
            transcript="...",
            call_type="First Call",
            criteria=applicable_criteria,
        )
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        prompt_loader: Optional[PromptLoader] = None,
        cache_dir: Optional[str] = "data/cache",
        enable_cache: bool = True,
    ):
        self._factory = model_factory
        self._prompts = prompt_loader or PromptLoader()
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._enable_cache = enable_cache and self._cache_dir is not None
        if self._enable_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # Whitelist of keys safe to persist in cache (CRIT-1: no raw_response, no secrets)
    _CACHE_SAFE_KEYS = frozenset({
        "criteria", "overall_assessment", "strengths", "improvements",
        "critical_gaps", "call_type", "model_used", "provider_used",
        "tokens_used", "cost_usd", "eval_time_seconds", "is_followup",
        "truncated", "pii_redacted",
    })

    def evaluate(
        self,
        transcript: str,
        call_type: str,
        criteria: Dict[str, Dict],
        max_retries: int = 2,
    ) -> Dict:
        """Run full evaluation cycle: prompts → LLM → parse → validate.

        Args:
            transcript: Cleaned transcript text.
            call_type: "First Call" or "Follow-up Call".
            criteria: Dict of applicable criteria {key: {description, ...}}.
            max_retries: Max retry attempts on validation failure.

        Returns:
            Evaluation dict with criteria scores, metadata, and cost info.
        """
        expected_keys = set(criteria.keys())
        criteria_count = len(criteria)

        # Build criteria text for prompt
        criteria_text = self._build_criteria_text(criteria)
        first_key = list(criteria.keys())[0] if criteria else "criterion_1"

        # Check cache
        cache_key = self._cache_key(
            transcript, call_type, criteria_count,
            model=self._factory.primary.model_name,
        )
        if self._enable_cache:
            cached = self._load_cache(cache_key)
            if cached:
                logger.info(f"Cache hit for {call_type} evaluation (key={cache_key[:12]})")
                return cached

        # Build prompts
        system_prompt = self._prompts.render(
            "qa_system",
            call_type=call_type,
            criteria_count=criteria_count,
        )
        user_prompt = self._prompts.render(
            "qa_user",
            call_type=call_type,
            transcript=transcript,
            criteria_count=criteria_count,
            criteria_text=criteria_text,
            first_criterion_key=first_key,
        )

        # Retry loop
        parser = ResponseParser(expected_keys=expected_keys)
        raw_text = ""

        for attempt in range(max_retries + 1):
            try:
                # Call LLM with fallback
                llm_response: LLMResponse = self._factory.chat_with_fallback(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    json_mode=True,
                )
                raw_text = llm_response.text

                # Parse and validate
                evaluation = parser.parse(raw_text)

                # Add metadata
                evaluation["call_type"] = call_type
                evaluation["model_used"] = llm_response.model
                evaluation["provider_used"] = llm_response.provider
                evaluation["tokens_used"] = {
                    "input": llm_response.input_tokens,
                    "output": llm_response.output_tokens,
                }
                evaluation["cost_usd"] = llm_response.cost_usd
                evaluation["eval_time_seconds"] = llm_response.elapsed_seconds

                # Cache the result
                if self._enable_cache:
                    self._save_cache(cache_key, evaluation)

                logger.info(
                    f"Evaluation complete | {call_type} | {criteria_count} criteria | "
                    f"provider={llm_response.provider} | ${llm_response.cost_usd:.4f} | "
                    f"{llm_response.elapsed_seconds:.1f}s"
                )
                return evaluation

            except ValidationError as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Validation failed (attempt {attempt + 1}): {e}. Retrying..."
                    )
                    continue
                logger.error(f"Validation failed after {max_retries + 1} attempts: {e}")
                return {
                    "error": f"Validation error: {e}",
                    "call_type": call_type,
                    "raw_response": raw_text,
                }

            except Exception as e:
                if attempt < max_retries:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        f"API error (attempt {attempt + 1}): {e}. Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                    continue
                logger.error(f"Evaluation failed: {e}")
                return {"error": str(e), "call_type": call_type}

        return {"error": "Max retries exceeded", "call_type": call_type}

    # ── Private helpers ─────────────────────────────────────────────

    @staticmethod
    def _build_criteria_text(criteria: Dict[str, Dict]) -> str:
        """Format criteria dict into numbered prompt text."""
        lines = []
        for i, (key, crit) in enumerate(criteria.items(), 1):
            lines.append(
                f"{i}. **{key}** (Category: {crit['category']}, Weight: {crit['weight']})\n"
                f"   {crit['description']}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _cache_key(transcript: str, call_type: str, criteria_count: int,
                   model: str = "") -> str:
        """Generate cache key from transcript + evaluation parameters + model."""
        content = f"{model}|{call_type}|{criteria_count}|{transcript}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _load_cache(self, key: str) -> Optional[Dict]:
        """Load cached evaluation result."""
        if not self._cache_dir:
            return None
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _save_cache(self, key: str, data: Dict) -> None:
        """Save evaluation result to cache using atomic write and sanitized data.

        CRIT-1: Only whitelisted keys are persisted (no raw_response, no secrets).
        CRIT-2: Writes to a temp file first, then renames (atomic on POSIX).
        CRIT-4: Uses _json_serializer instead of default=str.
        """
        if not self._cache_dir:
            return
        path = self._cache_dir / f"{key}.json"
        safe_data = {k: v for k, v in data.items() if k in self._CACHE_SAFE_KEYS}
        fd = None
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._cache_dir), suffix=".json.tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                fd = None  # os.fdopen takes ownership
                json.dump(safe_data, f, indent=2, ensure_ascii=False,
                          default=_json_serializer)
            os.replace(tmp_path, str(path))  # atomic on all platforms
        except OSError as e:
            logger.warning(f"Failed to write cache: {e}")
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        finally:
            if fd is not None:
                os.close(fd)
