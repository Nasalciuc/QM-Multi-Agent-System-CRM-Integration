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

from typing import Any, Dict, List, Tuple, Optional
import re
import logging

from utils import load_criteria, safe_log_filename
from core.model_factory import ModelFactory
from prompts.templates import PromptLoader
from processing.transcript_cleaner import TranscriptCleaner
from processing.token_counter import TokenCounter
from processing.chunker import TranscriptChunker
from inference.inference_engine import InferenceEngine
from error_codes import ErrorCode

logger = logging.getLogger("qa_system.agents")

# P2-FIX-2: Content-based follow-up detection signals.
# Checked against the first 1000 chars (lowercased) of the transcript.
_FOLLOWUP_SIGNALS = [
    "it's me again",
    "i'm calling you back",
    "calling you back",
    "we spoke earlier",
    "we spoke before",
    "we spoke yesterday",
    "we spoke last",
    "we talked earlier",
    "we talked before",
    "we talked yesterday",
    "we talked last",
    "sorry i missed",
    "returning your call",
    "following up",
    "as we discussed",
    "as i mentioned",
    "i called earlier",
    "i called before",
    "i called yesterday",
    "you called me",
    "you left a message",
    "per our conversation",
    "continuing our conversation",
    "further to our call",
    # B3-FIX-2: Travel sales follow-up signals (from production Batch 3)
    "i looked into",
    "i checked the",
    "i checked a couple",
    "i found some",
    "i got the prices",
    "i got some options",
    "here are the options",
    "here are the deals",
    "i was working on",
    "i've been working on",
    "as promised",
]

# P4-FIX-4: Words that intro-patterns may capture but are NOT agent names.
# Gerunds (-ing words > 3 chars) are also filtered automatically.
_AGENT_NAME_STOPLIST = frozenset({
    "calling", "leaving", "working", "looking", "going", "trying",
    "speaking", "reaching", "getting", "checking", "helping",
    "here", "there", "just", "also", "actually", "basically",
    "someone", "anyone", "everyone", "nobody",
    "sure", "glad", "happy", "sorry",
    "the", "this", "that", "what",
})

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
        # REAL-06: Use remove_fillers=False so LLM sees exact wording for
        # verbatim evidence quotes that match the original transcript.
        self._cleaners: Dict[str, TranscriptCleaner] = {
            "outbound": TranscriptCleaner(direction="outbound", remove_fillers=False),
            "inbound": TranscriptCleaner(direction="inbound", remove_fillers=False),
        }
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
        transcript: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Detect if call is first or follow-up.

        HIGH-NEW-5: Checks metadata first (CRM flight request
        status, call result field), then falls back to filename heuristic.

        P2-FIX-2: If metadata and filename both say "First Call", checks
        transcript content for follow-up signals (calling back, spoken
        before, etc.) in the first 1000 chars.

        Args:
            filename: Audio filename (used as fallback heuristic).
            metadata: Optional call record dict with 'direction', 'result',
                      'flight_request_status', 'agent_name', etc.
            transcript: Optional transcript text for content-based detection.

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
        if is_followup:
            return True, "Follow-up Call"

        # P2-FIX-2: Priority 3 — content-based transcript analysis
        if transcript:
            snippet = transcript[:1000].lower()
            for signal in _FOLLOWUP_SIGNALS:
                if signal in snippet:
                    logger.info(
                        f"P2-FIX-2: Follow-up detected by transcript signal "
                        f"'{signal}' in {safe_log_filename(filename)}"
                    )
                    return True, "Follow-up Call"

        return False, "First Call"

    @staticmethod
    def detect_agents_in_transcript(cleaned_transcript: str) -> List[str]:
        """REAL-02: Detect multiple agents in a transcript via self-introduction.

        Scans Agent:-labeled lines for "my name is X", "this is X calling/from",
        etc. Returns list of distinct agent names found.

        P4-FIX-4: Filters out gerunds (-ing words) and common false-positive
        words that match intro patterns but aren't real agent names.
        """
        intro_patterns = [
            re.compile(r"\bmy name is (\w+)", re.IGNORECASE),
            re.compile(r"\bthis is (\w+)\s+(?:from|calling|with)\b", re.IGNORECASE),
            re.compile(r"\bI'?m (\w+)\s+(?:from|calling|with)\b", re.IGNORECASE),
        ]
        agents: list = []
        seen: set = set()
        for line in cleaned_transcript.split("\n"):
            if not line.lower().startswith("agent:"):
                continue
            for pattern in intro_patterns:
                m = pattern.search(line)
                if m:
                    name = m.group(1).capitalize()
                    name_lower = name.lower()
                    # P4-FIX-4: Filter gerunds and stoplist words
                    if name_lower in _AGENT_NAME_STOPLIST:
                        continue
                    if name_lower.endswith("ing") and len(name_lower) > 3:
                        continue
                    if name_lower not in seen:
                        seen.add(name_lower)
                        agents.append(name)
        return agents

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
            _, call_type = self.detect_call_type(filename, metadata, transcript=transcript)
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

        is_followup, call_type = self.detect_call_type(filename, metadata, transcript=transcript)

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

        # 2. Prepare transcript for LLM evaluation
        processed_transcript = cleaned

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

        # MOD-02: Count criteria with PII tags in evidence (benchmark metric)
        _PII_TAGS = ["[PHONE]", "[EMAIL]", "[SPELLED_PII]", "[NATO_SPELLED]",
                     "[PNR]", "[BOOKING_REF]", "[CC_NUMBER]", "[SSN]"]
        pii_affected_count = 0
        for _key, _result in evaluation.get("criteria", {}).items():
            _evidence = _result.get("evidence", "")
            if any(tag in _evidence for tag in _PII_TAGS):
                pii_affected_count += 1
        evaluation["pii_affected_criteria_count"] = pii_affected_count
        if pii_affected_count > 0:
            logger.info(
                f"MOD-02: {pii_affected_count} criteria have PII tags in evidence "
                f"for {safe_log_filename(filename)}"
            )

        # Add extra metadata
        evaluation["is_followup"] = is_followup
        # MOD-10: Store speaker detection method for benchmark transparency
        evaluation["speaker_detection_method"] = getattr(cleaner, '_last_detection_method', None)
        evaluation["speaker_agent_name"] = getattr(cleaner, '_last_agent_name', None)
        if truncated:
            evaluation["truncated"] = True

        # MOD-07: Collect pipeline warnings for benchmark transparency
        pipeline_warnings = []
        if truncated:
            removed_pct = chunk_result.get("removed_percentage", 0)
            pipeline_warnings.append(f"TRUNCATED: {removed_pct}% of transcript removed")
        if pii_affected_count > 0:
            pipeline_warnings.append(
                f"PII_IN_EVIDENCE: {pii_affected_count} criteria affected"
            )
        _detection_method = getattr(cleaner, '_last_detection_method', None)
        if _detection_method == "direction":
            pipeline_warnings.append(
                "LOW_CONFIDENCE_SPEAKER: direction-based heuristic used"
            )
        if pipeline_warnings:
            evaluation["pipeline_warnings"] = pipeline_warnings
            logger.info(
                f"MOD-07: Pipeline warnings for {safe_log_filename(filename)}: "
                f"{pipeline_warnings}"
            )

        # REAL-02: Multi-agent detection
        agents_detected = self.detect_agents_in_transcript(cleaned)
        if len(agents_detected) > 1:
            evaluation["agents_detected"] = agents_detected
            evaluation.setdefault("critical_gaps", []).append(
                f"MULTI_AGENT: {len(agents_detected)} agents detected in call — "
                f"{', '.join(agents_detected)}. QA score may mix multiple agents' performance."
            )
            logger.warning(
                f"REAL-02: Multi-agent call detected in {safe_log_filename(filename)}: "
                f"{agents_detected}"
            )

        return evaluation

    def calculate_score(self, evaluation: Dict) -> Dict:
        """
        Calculate overall score (0-100) from YES/PARTIAL/NO scores.
        YES = 1.0 * weight, PARTIAL = 0.5 * weight, NO = 0.0, N/A = skip

        TASK-6: Also computes a *confidence* metric (0.0–1.0) representing
        how reliable the score is.  Factors:
          - criteria_coverage: fraction of criteria that were actually scored
            (not N/A).
          - agreement_ratio: proportion of YES+NO answers vs PARTIAL (PARTIAL
            indicates the LLM was uncertain).
          - sample_size: small penalty when fewer than 10 criteria were scored.
        """
        criteria_results = evaluation.get("criteria", {})
        if not criteria_results:
            return {
                "overall_score": 0, "total_points": 0, "total_weight": 0,
                "category_scores": {}, "score_breakdown": {},
                "confidence": 0.0, "confidence_factors": {},
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

        # TASK-6: Confidence calculation ──────────────────────────────
        scored_count = yes_count + partial_count + no_count
        total_criteria = scored_count + na_count

        # Factor 1: criteria_coverage — what fraction was actually scored
        criteria_coverage = scored_count / max(1, total_criteria)

        # Factor 2: agreement_ratio — YES+NO are decisive; PARTIAL is uncertain
        agreement_ratio = (
            (yes_count + no_count) / max(1, scored_count)
            if scored_count > 0 else 0.0
        )

        # Factor 3: sample_size — small penalty when < 10 scored criteria
        sample_factor = min(1.0, scored_count / 10.0)

        # Weighted combination
        confidence = round(
            0.50 * criteria_coverage
            + 0.30 * agreement_ratio
            + 0.20 * sample_factor,
            3,
        )

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
            "confidence": confidence,
            "confidence_factors": {
                "criteria_coverage": round(criteria_coverage, 3),
                "agreement_ratio": round(agreement_ratio, 3),
                "sample_factor": round(sample_factor, 3),
            },
        }

    def calculate_listening_ratio(self, transcript: str) -> Dict[str, Any]:
        """Estimate agent vs client talking percentage from transcript.

        Expects cleaned transcripts with Agent:/Client: labels
        (produced by TranscriptCleaner). Lines without recognized
        speaker labels are skipped.

        REAL-03: Includes sanity checks for impossible ratios that
        suggest speaker mapping issues.
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

        # REAL-03: Sanity checks
        if total == 0:
            return {"agent_percentage": 50.0, "client_percentage": 50.0,
                    "total_words": 0, "ratio_warning": "no_labeled_lines"}

        if agent_words == 0:
            logger.warning(
                "REAL-03: Listening ratio — agent has 0 words. "
                "Likely speaker mapping issue."
            )
            return {"agent_percentage": 0.0, "client_percentage": 100.0,
                    "total_words": total, "ratio_warning": "agent_zero_words"}

        if client_words == 0:
            logger.warning(
                "REAL-03: Listening ratio — client has 0 words. "
                "Likely speaker mapping issue."
            )
            return {"agent_percentage": 100.0, "client_percentage": 0.0,
                    "total_words": total, "ratio_warning": "client_zero_words"}

        agent_pct = round(agent_words / total * 100, 1)
        if agent_pct > 95 or agent_pct < 5:
            logger.warning(
                f"REAL-03: Extreme listening ratio ({agent_pct}% agent) — "
                f"possible speaker mapping issue"
            )

        return {
            "agent_percentage": agent_pct,
            "client_percentage": round(client_words / total * 100, 1),
            "total_words": total,
        }
