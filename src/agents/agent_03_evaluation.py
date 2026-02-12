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

from src.utils import load_criteria
from src.core.model_factory import ModelFactory
from src.prompts.templates import PromptLoader
from src.processing.transcript_cleaner import TranscriptCleaner
from src.processing.token_counter import TokenCounter
from src.processing.chunker import TranscriptChunker
from src.processing.pii_redactor import PIIRedactor
from src.inference.inference_engine import InferenceEngine

logger = logging.getLogger("qa_system.agents")


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
    ):
        """
        Args:
            model_factory: ModelFactory with primary + fallback providers.
            criteria_path: Path to qa_criteria.yaml.
            prompts_dir: Directory with prompt templates (auto-detected if None).
            cache_dir: Directory for LLM response cache.
            enable_cache: Enable response caching.
        """
        self.EVALUATION_CRITERIA = load_criteria(criteria_path)

        # Processing pipeline
        self._cleaner = TranscriptCleaner()
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

        logger.info(
            f"QualityManagementAgent initialized | "
            f"Criteria: {len(self.EVALUATION_CRITERIA)} | "
            f"Provider: {model_factory.primary.provider_name}"
        )

    def detect_call_type(self, filename: str) -> Tuple[bool, str]:
        """Detect if call is first or follow-up from filename."""
        filename_lower = filename.lower()
        follow_up_indicators = ["2nd", "second", "follow", "follow-up", "followup"]
        is_followup = any(indicator in filename_lower for indicator in follow_up_indicators)
        call_type = "Follow-up Call" if is_followup else "First Call"
        return is_followup, call_type

    # Minimum word count to justify an LLM evaluation (CRIT-5)
    MIN_TRANSCRIPT_WORDS = 20

    def evaluate_call(self, transcript: str, filename: str, max_retries: int = 2) -> Dict:
        """Evaluate a call transcript against all applicable criteria.

        Pipeline: validate → clean → redact PII → truncate → call LLM → parse → validate.

        Returns evaluation dict with criteria scores, evidence, assessment.
        """
        # CRIT-5: Reject transcripts too short for meaningful evaluation
        word_count = len(transcript.split()) if transcript else 0
        if word_count < self.MIN_TRANSCRIPT_WORDS:
            logger.warning(
                f"Transcript too short for {filename}: {word_count} words "
                f"(min {self.MIN_TRANSCRIPT_WORDS}). Skipping LLM call."
            )
            _, call_type = self.detect_call_type(filename)
            return {
                "error": f"Transcript too short ({word_count} words)",
                "call_type": call_type,
                "word_count": word_count,
            }

        is_followup, call_type = self.detect_call_type(filename)

        # 1. Clean transcript (normalize speaker labels)
        cleaned = self._cleaner.clean(transcript)

        # 2. Redact PII
        redaction = self._redactor.redact(cleaned)
        processed_transcript = redaction["text"]
        if redaction["total_redactions"] > 0:
            logger.info(f"PII redacted in {filename}: {redaction['pii_found']}")

        # 3. Truncate if needed
        chunk_result = self._chunker.truncate(processed_transcript)
        processed_transcript = chunk_result["text"]
        truncated = chunk_result["truncated"]

        # 4. Filter criteria: skip first_call_only for follow-ups
        applicable_criteria = {}
        for key, criteria in self.EVALUATION_CRITERIA.items():
            if is_followup and criteria["first_call_only"]:
                continue
            applicable_criteria[key] = criteria

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

        Supports cleaned labels (Agent:/Client:) and raw ElevenLabs
        (Speaker 0:/1:) formats. Cleaned transcripts (post-TranscriptCleaner)
        will always have Agent:/Client: labels.

        Note: For raw transcripts with Speaker N: format, the first speaker
        is assigned as agent (outbound assumption). For accurate results,
        run TranscriptCleaner first.
        """
        lines = transcript.strip().split("\n")
        agent_words = 0
        client_words = 0

        # Priority 1: Cleaned labels
        agent_prefixes = ("agent:", "representative:", "rep:", "advisor:")
        client_prefixes = ("client:", "customer:", "caller:", "guest:")

        # Priority 2: Raw Speaker N: labels (fallback)
        first_speaker = None
        has_speaker_labels = False
        for line in lines:
            line_stripped = line.lower().strip()
            match = re.match(r'^(speaker \d+):', line_stripped)
            if match:
                if first_speaker is None:
                    first_speaker = match.group(1)
                has_speaker_labels = True
                break

        for line in lines:
            line_lower = line.lower().strip()
            if not line_lower:
                continue
            word_count = len(line.split())

            # Check explicit labels first
            is_agent = line_lower.startswith(agent_prefixes)
            is_client = line_lower.startswith(client_prefixes)

            # Fallback: Speaker N: format
            if not is_agent and not is_client and has_speaker_labels:
                speaker_match = re.match(r'^(speaker \d+):', line_lower)
                if speaker_match:
                    if first_speaker and speaker_match.group(1) == first_speaker:
                        is_agent = True
                    else:
                        is_client = True

            if is_agent:
                agent_words += word_count
            elif is_client:
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
