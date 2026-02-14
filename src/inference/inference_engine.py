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
import threading
from collections import OrderedDict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional, Set

from core.model_factory import ModelFactory
from core.base_llm import LLMResponse
from prompts.templates import PromptLoader
from inference.response_parser import ResponseParser, ValidationError
from utils import json_serializer as _json_serializer  # HIGH-6: shared serializer

logger = logging.getLogger("qa_system.inference")

# Lazy-loaded PII redactor for sanitising error responses (CRIT-NEW-1)
_pii_redactor_instance = None


def _get_pii_redactor():
    """Lazy-load PIIRedactor to avoid circular imports."""
    global _pii_redactor_instance
    if _pii_redactor_instance is None:
        from processing.pii_redactor import PIIRedactor
        _pii_redactor_instance = PIIRedactor()
    return _pii_redactor_instance


# HIGH-6: _json_serializer removed — now imported from utils


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
        cache_ttl_seconds: int = 7 * 24 * 3600,  # MED-20: 7-day default TTL
        memory_cache_maxsize: int = 128,  # L1 in-memory LRU cache size
    ):
        self._factory = model_factory
        self._prompts = prompt_loader or PromptLoader()
        self._cache_dir: Optional[Path] = Path(cache_dir) if cache_dir else None
        self._enable_cache = enable_cache and self._cache_dir is not None
        self._cache_ttl = cache_ttl_seconds
        self._cache_lock = threading.Lock()  # CRIT-4: Thread-safe cache access

        # L1: In-memory LRU cache (OrderedDict, moves to end on access)
        self._memory_cache: OrderedDict[str, Dict] = OrderedDict()
        self._memory_cache_maxsize = max(1, memory_cache_maxsize)
        self._memory_hits = 0
        self._memory_misses = 0
        self._disk_hits = 0
        self._disk_misses = 0
        self._cache_write_failures = 0  # MED-13: Track write failures

        if self._enable_cache and self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._cleanup_cache()  # MED-20: Remove stale entries at startup

    # Whitelist of keys safe to persist in cache (CRIT-1: no raw_response, no secrets)
    _CACHE_SAFE_KEYS = frozenset({
        "criteria", "overall_assessment", "strengths", "improvements",
        "critical_gaps", "call_type", "model_used", "provider_used",
        "tokens_used", "cost_usd", "eval_time_seconds", "is_followup",
        "truncated", "pii_redacted", "prompt_hash",
    })

    # Required keys that must be present when loading from cache (#2)
    _CACHE_REQUIRED_KEYS = frozenset({"criteria", "overall_assessment"})

    def _cleanup_cache(self) -> None:
        """MED-20: Remove cache entries older than TTL at startup."""
        if not self._cache_dir:
            return
        now = time.time()
        removed = 0
        for path in self._cache_dir.glob("*.json"):
            try:
                age = now - path.stat().st_mtime
                if age > self._cache_ttl:
                    path.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                pass
        if removed:
            logger.info(f"Cache cleanup: removed {removed} stale entries")

    @property
    def cache_stats(self) -> Dict:
        """Return combined L1 (memory) + L2 (disk) cache statistics."""
        total_lookups = self._memory_hits + self._memory_misses
        return {
            "memory_hits": self._memory_hits,
            "memory_misses": self._memory_misses,
            "disk_hits": self._disk_hits,
            "disk_misses": self._disk_misses,
            "cache_write_failures": self._cache_write_failures,  # MED-13
            "total_lookups": total_lookups,
            "combined_hit_rate_pct": round(
                (self._memory_hits + self._disk_hits) / max(1, total_lookups) * 100, 1
            ),
            "memory_cache_size": len(self._memory_cache),
            "memory_cache_maxsize": self._memory_cache_maxsize,
        }

    def _promote_to_memory(self, key: str, data: Dict) -> None:
        """Promote a disk cache entry to the in-memory LRU cache.

        HIGH-5: Thread-safe — acquires _cache_lock before mutating the
        OrderedDict to prevent race conditions in concurrent access.
        Evicts the oldest entry if the cache exceeds maxsize.
        """
        with self._cache_lock:
            if key in self._memory_cache:
                self._memory_cache.move_to_end(key)
                return
            self._memory_cache[key] = data
            while len(self._memory_cache) > self._memory_cache_maxsize:
                self._memory_cache.popitem(last=False)  # Evict oldest

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

        Raises:
            ValueError: If transcript is empty, binary, or exceeds size limit.
        """
        # #3: Input validation before LLM call
        if not transcript or not transcript.strip():
            raise ValueError("Transcript is empty or whitespace-only")
        if "\x00" in transcript:
            raise ValueError("Transcript contains binary data (null bytes)")
        _MAX_TRANSCRIPT_CHARS = 500_000
        if len(transcript) > _MAX_TRANSCRIPT_CHARS:
            raise ValueError(
                f"Transcript too large: {len(transcript)} chars "
                f"(limit {_MAX_TRANSCRIPT_CHARS})"
            )
        expected_keys = set(criteria.keys())
        criteria_count = len(criteria)

        # Build criteria text for prompt
        criteria_text = self._build_criteria_text(criteria)
        first_key = list(criteria.keys())[0] if criteria else "criterion_1"

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

        # Check cache — include criteria content and prompt hashes for invalidation
        criteria_content = json.dumps(criteria, sort_keys=True, default=str)
        criteria_hash = hashlib.sha256(criteria_content.encode()).hexdigest()[:16]
        prompt_hash = hashlib.sha256(
            (system_prompt + user_prompt).encode()
        ).hexdigest()[:16]

        cache_key = self._cache_key(
            transcript, call_type, criteria_count,
            model=self._factory.primary.model_name,
            criteria_hash=criteria_hash,
            prompt_hash=prompt_hash,
        )
        if self._enable_cache:
            # L1: Check in-memory LRU cache first
            if cache_key in self._memory_cache:
                self._memory_cache.move_to_end(cache_key)
                self._memory_hits += 1
                logger.info(f"L1 memory cache hit for {call_type} evaluation (key={cache_key[:12]})")
                return self._memory_cache[cache_key]
            self._memory_misses += 1

            # L2: Check disk cache
            cached = self._load_cache(cache_key)
            if cached:
                self._disk_hits += 1
                # Promote to L1 for faster future lookups
                self._promote_to_memory(cache_key, cached)
                logger.info(f"L2 disk cache hit for {call_type} evaluation (key={cache_key[:12]})")
                return cached
            self._disk_misses += 1

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
                # #26: Prompt template version tracking
                evaluation["prompt_hash"] = prompt_hash

                # Cache the result — only if it has no error key (HIGH-NEW-7)
                if self._enable_cache and "error" not in evaluation:
                    self._save_cache(cache_key, evaluation)
                    # Also promote to L1 memory cache
                    self._promote_to_memory(cache_key, evaluation)

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
                # CRIT-NEW-1: PII-redact raw LLM text before returning —
                # this dict may flow all the way to export/JSON files.
                redacted_raw = _get_pii_redactor().redact(raw_text)["text"] if raw_text else ""
                return {
                    "error": f"Validation error: {e}",
                    "call_type": call_type,
                    "raw_response": redacted_raw,
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
                   model: str = "", criteria_hash: str = "",
                   prompt_hash: str = "", temperature: float = 0.1) -> str:
        """Generate cache key from transcript + evaluation parameters + model + criteria content.

        MED-NEW-12: Includes temperature so different temperature runs
        don't serve stale results from a different temperature setting.
        """
        content = (
            f"{model}|{call_type}|{criteria_count}|{criteria_hash}|"
            f"{prompt_hash}|{temperature}|{transcript}"
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def _load_cache(self, key: str) -> Optional[Dict]:
        """Load cached evaluation result (thread-safe, TTL-aware)."""
        if not self._cache_dir:
            return None
        path = self._cache_dir / f"{key}.json"
        with self._cache_lock:  # CRIT-4: Prevent race condition
            if path.exists():
                try:
                    # MED-20: Check TTL — expire stale cache entries
                    age = time.time() - path.stat().st_mtime
                    if age > self._cache_ttl:
                        logger.debug(f"Cache expired (age={age:.0f}s > ttl={self._cache_ttl}s): {key[:12]}")
                        path.unlink(missing_ok=True)
                        return None
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # #2: Validate cache entry has required keys
                    if not isinstance(data, dict) or not self._CACHE_REQUIRED_KEYS.issubset(data.keys()):
                        logger.warning(f"Cache entry invalid (missing required keys): {key[:12]}")
                        path.unlink(missing_ok=True)
                        return None
                    return data
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
        with self._cache_lock:  # CRIT-4: Thread-safe write
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
                self._cache_write_failures += 1  # MED-13: Track failures
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            finally:
                if fd is not None:
                    os.close(fd)
