"""
Agent 3: Quality Management Evaluation

Purpose: Evaluate call transcripts using LLM
Now a thin wrapper that delegates to:
  - src/core/          → LLM abstraction + fallback
  - src/prompts/       → prompt templates
  - src/processing/    → transcript cleaning, PII, token counting
  - src/inference/     → response parsing + retry orchestration

What remains here: detect_call_type(), calculate_score(),
calculate_listening_ratio(), and evaluate_call() as the wiring layer.
"""

from typing import Dict, List, Tuple, Optional
import re
import logging

from utils import load_criteria, safe_log_filename
from core.model_factory import ModelFactory
from prompts.templates import PromptLoader
from processing.transcript_cleaner import TranscriptCleaner
from processing.token_counter import TokenCounter
from processing.chunker import TranscriptChunker
from processing.pii_redactor import PIIRedactor
from inference.inference_engine import InferenceEngine
from error_codes import ErrorCode

logger = logging.getLogger("qa_system.agents")

# CRIT-03: Minimum transcript requirements to avoid wasting LLM tokens
MIN_TRANSCRIPT_WORDS = 50        # Skip if fewer than 50 words
MIN_TRANSCRIPT_CHARS = 200       # Skip if fewer than 200 characters


class QualityManagementAgent:
    """Quality Management Agent — thin orchestration wrapper.

    Usage:
        factory = ModelFactory()
        agent_qm = QualityManagementAgent(factory)
        evaluation = agent_qm.evaluate_call(transcript, "call1.mp3")
        score_data = agent_qm.calculate_score(evaluation)
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        criteria_path: str = "config/qa_criteria.yaml",
        prompts_dir: Optional[str] = None,
        cache_dir: str = "data/cache",
        enable_cache: bool = True,
        min_transcript_words: int = 20,
    ):
        """
        Args:
            model_factory: ModelFactory with primary + fallback providers.
            criteria_path: Path to qa_criteria.yaml.
            prompts_dir: Directory with prompt templates (auto-detected if None).
            cache_dir: Directory for LLM response cache.
            enable_cache: Enable response caching.
            min_transcript_words: Reject transcripts shorter than this (DESIGN-23).
        """
        self.EVALUATION_CRITERIA = load_criteria(criteria_path)

        # Processing pipeline — Fix #6: direction-keyed cleaner cache
        self._cleaners: Dict[str, TranscriptCleaner] = {
            "outbound": TranscriptCleaner(direction="outbound"),
            "inbound": TranscriptCleaner(direction="inbound"),
        }
        self._redactor = PIIRedactor()
        # HIGH-4: Pass actual pricing from primary provider, not token_limits
        self._counter = TokenCounter(
            model=model_factory.primary.model_name,
            pricing=model_factory.primary_pricing,
        )
        self._chunker = TranscriptChunker(
            max_tokens=model_factory.token_limits.get("max_input_tokens", 30000),
            token_counter=self._counter,
        )

        # Inference engine
        prompt_loader = PromptLoader(prompts_dir)
        self._engine = InferenceEngine(
            model_factory=model_factory,
            prompt_loader=prompt_loader,
            cache_dir=cache_dir,
            enable_cache=enable_cache,
        )

        # DESIGN-23: Configurable minimum transcript length
        self.MIN_TRANSCRIPT_WORDS = min_transcript_words

        logger.info(
            f"QualityManagementAgent initialized | "
            f"Criteria: {len(self.EVALUATION_CRITERIA)} | "
            f"Provider: {model_factory.primary.provider_name}"
        )

    def reset_providers(self) -> None:
        """NEW-06: Public method to re-enable providers disabled in a previous run.

        Delegates to the internal inference engine's factory so pipeline.py
        does not need to reach across multiple private attributes.
        """
        if hasattr(self._engine, '_factory'):
            self._engine._factory.reset_disabled_providers()

    def detect_call_type(
        self,
        filename: str,
        metadata: Optional[Dict] = None,
    ) -> Tuple[bool, str]:
        """Detect if call is first or follow-up.

        HIGH-NEW-5: Checks metadata first (CRM flight request
        status, call result field), then falls back to filename heuristic.

        Args:
            filename: Audio filename (used as fallback heuristic).
            metadata: Optional call record dict with 'direction', 'result',
                      'flight_request_status', 'agent_name', etc.

        Returns:
            (is_followup, call_type_label)
        """
        # Priority 1: explicit metadata tag (if caller passes it)
        if metadata:
            # CRM "result" field or custom tag
            result_field = str(metadata.get("result", "")).lower()
            if result_field in ("follow-up", "followup", "callback"):
                return True, "Follow-up Call"

            # CRM API: check flight_request_status for follow-up indicators
            fr_status = str(metadata.get("flight_request_status", "")).lower()
            if "follow" in fr_status or "callback" in fr_status:
                return True, "Follow-up Call"

            # CRM API: log agent/client info for traceability
            if metadata.get("agent_name"):
                logger.debug(f"CRM metadata — Agent: {metadata['agent_name']}, "
                             f"Client: {metadata.get('client_name', 'N/A')}")

        # Priority 2: filename heuristic (works for manually-named files)
        filename_lower = filename.lower()
        follow_up_indicators = ["2nd", "second", "follow", "follow-up", "followup"]
        is_followup = any(indicator in filename_lower for indicator in follow_up_indicators)
        call_type = "Follow-up Call" if is_followup else "First Call"
        return is_followup, call_type

    def evaluate_call(self, transcript: str, filename: str, max_retries: int = 2,
                      metadata: Optional[Dict] = None) -> Dict:
        """Evaluate a call transcript against all applicable criteria.

        Pipeline: validate → clean → redact PII → truncate → call LLM → parse → validate.

        Args:
            transcript: Raw transcript text.
            filename: Audio filename for call-type detection.
            max_retries: Max LLM retry attempts.
            metadata: Optional CRM record dict. MED-11: If it contains a
                      ``direction`` key ("inbound" or "outbound"), a per-call
                      TranscriptCleaner is created with the correct direction
                      so that Speaker 0/1 labels map to Agent/Client properly.

        Returns evaluation dict with criteria scores, evidence, assessment.
        """
        # CRIT-03: Validate transcript minimum length before expensive LLM call
        transcript_stripped = transcript.strip() if transcript else ""
        word_count = len(transcript_stripped.split()) if transcript_stripped else 0
        char_count = len(transcript_stripped)
        if (not transcript_stripped
                or word_count < MIN_TRANSCRIPT_WORDS
                or char_count < MIN_TRANSCRIPT_CHARS):
            logger.warning(
                f"Transcript too short for evaluation: {word_count} words, "
                f"{char_count} chars (min: {MIN_TRANSCRIPT_WORDS} words / "
                f"{MIN_TRANSCRIPT_CHARS} chars). File: {safe_log_filename(filename)}"
            )
            _, call_type = self.detect_call_type(filename, metadata)
            return {
                "error": f"Transcript too short ({word_count} words)",
                "error_code": ErrorCode.TRANSCRIPT_TOO_SHORT,
                "status": "TOO_SHORT",
                "call_type": call_type,
                "word_count": word_count,
                "cost_usd": 0.0,
                "criteria": {},
                "overall_assessment": f"Transcript too short for evaluation ({word_count} words, minimum {MIN_TRANSCRIPT_WORDS})",
                "strengths": [],
                "improvements": [],
                "critical_gaps": [],
            }

        is_followup, call_type = self.detect_call_type(filename, metadata)

        # 1. Clean transcript (normalize speaker labels)
        # MED-11: Use direction from CRM metadata when available
        # Fix #6: Use cached cleaners instead of creating per-call instances
        direction = (metadata or {}).get("direction", "").lower()
        if direction in self._cleaners:
            cleaner = self._cleaners[direction]
            logger.debug(f"Using per-call direction '{direction}' for {safe_log_filename(filename)}")
        else:
            cleaner = self._cleaners["outbound"]  # default
        cleaned = cleaner.clean(transcript)

        # 2. Redact PII
        redaction = self._redactor.redact(cleaned)
        processed_transcript = redaction["text"]
        if redaction["total_redactions"] > 0:
            logger.info(f"PII redacted in {safe_log_filename(filename)}: {redaction['pii_found']}")

        # 3. Truncate if needed
        chunk_result = self._chunker.truncate(processed_transcript)
        processed_transcript = chunk_result["text"]
        truncated = chunk_result["truncated"]

        # 4. Filter criteria by call_applicability
        applicable_criteria = {}
        skipped_criteria = []  # MED-NEW-14: Track skipped criteria
        # NEW — call_applicability filtering
        target = "second_only" if is_followup else "first_only"
        for key, crit in self.EVALUATION_CRITERIA.items():
            applicability = crit.get("call_applicability", "both")
            if applicability == "both" or applicability == target:
                applicable_criteria[key] = crit
            else:
                skipped_criteria.append(key)

        if skipped_criteria:
            logger.info(
                f"Follow-up call '{safe_log_filename(filename)}': skipped {len(skipped_criteria)} "
                f"first-call-only criteria: {', '.join(skipped_criteria)}"
            )

        # 5. Run inference engine (LLM call + parsing + retry)
        evaluation = self._engine.evaluate(
            transcript=processed_transcript,
            call_type=call_type,
            criteria=applicable_criteria,
            max_retries=max_retries,
        )

        # Add extra metadata
        evaluation["is_followup"] = is_followup
        if truncated:
            evaluation["truncated"] = True
        if redaction["total_redactions"] > 0:
            evaluation["pii_redacted"] = redaction["pii_found"]

        return evaluation

    def calculate_score(self, evaluation: Dict) -> Dict:
        """
        Calculate overall score (0-100) from YES/PARTIAL/NO scores.
        YES = 1.0 * weight, PARTIAL = 0.5 * weight, NO = 0.0, N/A = skip
        """
        criteria_results = evaluation.get("criteria", {})
        if not criteria_results:
            return {
                "overall_score": 0, "total_points": 0, "total_weight": 0,
                "category_scores": {}, "score_breakdown": {},
            }

        total_points = 0
        total_weight = 0
        category_data = {}
        yes_count = 0
        partial_count = 0
        no_count = 0
        na_count = 0

        for key, result in criteria_results.items():
            score = result.get("score", "").upper()
            criteria_def = self.EVALUATION_CRITERIA.get(key, {})
            weight = criteria_def.get("weight", 1.0)
            category = criteria_def.get("category", "unknown")

            if score == "N/A":
                na_count += 1
                continue

            if category not in category_data:
                category_data[category] = {"points": 0, "weight": 0, "count": 0}

            if score == "YES":
                points = 1.0 * weight
                yes_count += 1
            elif score == "PARTIAL":
                points = 0.5 * weight
                partial_count += 1
            else:  # NO
                points = 0.0
                no_count += 1

            total_points += points
            total_weight += weight
            category_data[category]["points"] += points
            category_data[category]["weight"] += weight
            category_data[category]["count"] += 1

        # Overall score
        overall_score = (total_points / total_weight * 100) if total_weight > 0 else 0

        # Category scores
        category_scores = {}
        for cat, data in category_data.items():
            cat_score = (data["points"] / data["weight"] * 100) if data["weight"] > 0 else 0
            category_scores[cat] = {
                "score": round(cat_score, 1),
                "count": data["count"],
            }

        return {
            "overall_score": round(overall_score, 1),
            "total_points": round(total_points, 2),
            "total_weight": round(total_weight, 2),
            "category_scores": category_scores,
            "score_breakdown": {
                "yes_count": yes_count,
                "partial_count": partial_count,
                "no_count": no_count,
                "na_count": na_count,
            },
        }

    def calculate_listening_ratio(self, transcript: str) -> Dict[str, float]:
        """Estimate agent vs client talking percentage from transcript.

        Expects cleaned transcripts with Agent:/Client: labels
        (produced by TranscriptCleaner). Lines without recognized
        speaker labels are skipped.
        """
        lines = transcript.strip().split("\n")
        agent_words = 0
        client_words = 0

        agent_prefixes = ("agent:", "representative:", "rep:", "advisor:")
        client_prefixes = ("client:", "customer:", "caller:", "guest:")

        for line in lines:
            line_lower = line.lower().strip()
            if not line_lower:
                continue
            word_count = len(line.split())

            if line_lower.startswith(agent_prefixes):
                agent_words += word_count
            elif line_lower.startswith(client_prefixes):
                client_words += word_count
            # Skip unlabeled lines

        total = agent_words + client_words
        if total == 0:
            return {"agent_percentage": 50.0, "client_percentage": 50.0, "total_words": 0}

        return {
            "agent_percentage": round(agent_words / total * 100, 1),
            "client_percentage": round(client_words / total * 100, 1),
            "total_words": total,
        }
