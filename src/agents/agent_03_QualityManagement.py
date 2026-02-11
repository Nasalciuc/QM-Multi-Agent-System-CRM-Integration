"""
Agent 3: Quality Management Evaluation

Purpose: Evaluate call transcripts using LLM (OpenAI/OpenRouter)
Model: gpt-4o-2024-11-20 (via OpenAI or OpenRouter)
Criteria: 24 total across 4 categories
Scoring: YES / PARTIAL / NO / N/A per criterion
Output: JSON with scores, evidence, assessment
"""

from typing import Dict, List, Tuple
import json
import time
import re
import logging
import sys
import os

# Add parent dir to path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_criteria

logger = logging.getLogger("qa_system")


class QualityManagementAgent:
    """Quality Management Agent"""

    OPENAI_MODEL = "gpt-4o-2024-11-20"

    PRICING = {
        "input_per_1m": 2.50,
        "output_per_1m": 10.00
    }

    # Maximum transcript length (chars) to send to LLM (~30k tokens)
    MAX_TRANSCRIPT_LENGTH = 120_000

    def __init__(self, openai_client, criteria_path: str = "config/qa_criteria.yaml"):
        """
        Initialize with OpenAI client (can point to OpenRouter).
        Loads evaluation criteria from YAML (single source of truth).

        Usage:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ['OPENROUTER_API_KEY'],
                           base_url="https://openrouter.ai/api/v1")
            agent_qm = QualityManagementAgent(client)
        """
        self.client = openai_client
        self.model = self.OPENAI_MODEL
        self.EVALUATION_CRITERIA = load_criteria(criteria_path)
        print(f"QualityManagementAgent initialized | Model: {self.model} | Criteria: {len(self.EVALUATION_CRITERIA)}")

    def detect_call_type(self, filename: str) -> Tuple[bool, str]:
        """Detect if call is first or follow-up from filename."""
        filename_lower = filename.lower()
        follow_up_indicators = ["2nd", "second", "follow", "follow-up", "followup"]
        is_followup = any(indicator in filename_lower for indicator in follow_up_indicators)
        call_type = "Follow-up Call" if is_followup else "First Call"
        return is_followup, call_type

    def evaluate_call(self, transcript: str, filename: str, max_retries: int = 2) -> Dict:
        """
        Evaluate a call transcript against all applicable criteria.
        Returns evaluation dict with criteria scores, evidence, assessment.
        """
        is_followup, call_type = self.detect_call_type(filename)

        # Truncate transcript if too long to avoid exceeding LLM context window
        truncated = False
        if len(transcript) > self.MAX_TRANSCRIPT_LENGTH:
            transcript = transcript[:self.MAX_TRANSCRIPT_LENGTH]
            truncated = True
            logger.warning(f"Transcript truncated for {filename}: {len(transcript)} chars -> {self.MAX_TRANSCRIPT_LENGTH}")

        # Filter criteria: skip first_call_only for follow-ups
        applicable_criteria = {}
        for key, criteria in self.EVALUATION_CRITERIA.items():
            if is_followup and criteria["first_call_only"]:
                continue
            applicable_criteria[key] = criteria

        criteria_count = len(applicable_criteria)

        # Build criteria list for prompt
        criteria_text = ""
        for i, (key, criteria) in enumerate(applicable_criteria.items(), 1):
            criteria_text += f"{i}. **{key}** (Category: {criteria['category']}, Weight: {criteria['weight']})\n"
            criteria_text += f"   {criteria['description']}\n\n"

        # System prompt
        system_prompt = f"""You are an expert Call Center Quality Management Analyst for Buy Business Class, a premium travel company.

You are evaluating a {call_type}. Evaluate the transcript against {criteria_count} criteria below.

For EACH criterion, provide:
- "score": exactly one of "YES", "PARTIAL", "NO", or "N/A"
- "evidence": brief quote or explanation from the transcript supporting your score

Scoring guide:
- YES = The agent clearly and fully demonstrated this behavior
- PARTIAL = The agent partially demonstrated this behavior or did it incompletely
- NO = The agent did not demonstrate this behavior at all
- N/A = This criterion is not applicable to this call

Also provide:
- "overall_assessment": 2-3 sentence professional summary
- "strengths": list of 3 key strengths
- "improvements": list of 3 areas for improvement
- "critical_gaps": list of any critical issues found (can be empty)

Respond ONLY with valid JSON."""

        # User prompt
        user_prompt = f"""Evaluate this {call_type} transcript:

---TRANSCRIPT START---
{transcript}
---TRANSCRIPT END---

Evaluate against these {criteria_count} criteria:

{criteria_text}

Return JSON with this exact structure:
{{
    "criteria": {{
        "{list(applicable_criteria.keys())[0]}": {{"score": "YES|PARTIAL|NO|N/A", "evidence": "..."}},
        ... (all {criteria_count} criteria)
    }},
    "overall_assessment": "2-3 sentence summary",
    "strengths": ["strength1", "strength2", "strength3"],
    "improvements": ["improvement1", "improvement2", "improvement3"],
    "critical_gaps": ["gap1", ...]
}}"""

        # Call API with retry
        raw_text = ""
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                    response_format={"type": "json_object"}
                )

                elapsed = time.time() - start_time

                # Parse response
                raw_text = response.choices[0].message.content.strip()
                evaluation = json.loads(raw_text)

                # Token usage and cost
                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                cost_usd = (input_tokens / 1_000_000 * self.PRICING["input_per_1m"] +
                           output_tokens / 1_000_000 * self.PRICING["output_per_1m"])

                # Validate: all criteria present with valid scores
                valid_scores = {"YES", "PARTIAL", "NO", "N/A"}
                criteria_result = evaluation.get("criteria", {})
                missing = [k for k in applicable_criteria if k not in criteria_result]
                invalid = [k for k, v in criteria_result.items()
                          if v.get("score", "").upper() not in valid_scores]

                if missing or invalid:
                    if attempt < max_retries:
                        logger.warning(f"Validation failed (attempt {attempt+1}): missing={missing}, invalid={invalid}. Retrying...")
                        continue
                    else:
                        logger.warning(f"Validation issues after {max_retries+1} attempts: missing={missing}, invalid={invalid}")

                # Normalize scores to uppercase
                for key in criteria_result:
                    criteria_result[key]["score"] = criteria_result[key]["score"].upper()

                # Add metadata
                evaluation["call_type"] = call_type
                evaluation["is_followup"] = is_followup
                evaluation["model_used"] = self.model
                evaluation["tokens_used"] = {"input": input_tokens, "output": output_tokens}
                evaluation["cost_usd"] = round(cost_usd, 6)
                evaluation["eval_time_seconds"] = round(elapsed, 2)
                if truncated:
                    evaluation["truncated"] = True

                logger.info(f"Evaluated: {filename} | {call_type} | {criteria_count} criteria | ${cost_usd:.4f} | {elapsed:.1f}s")
                return evaluation

            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    logger.warning(f"JSON parse error (attempt {attempt+1}): {e}. Retrying...")
                    continue
                logger.error(f"JSON parse failed after {max_retries+1} attempts: {e}")
                return {"error": f"JSON parse error: {e}", "call_type": call_type, "raw_response": raw_text}

            except Exception as e:
                if attempt < max_retries:
                    wait = 2 ** (attempt + 1)  # 2s, 4s exponential backoff
                    logger.warning(f"API error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                logger.error(f"Evaluation failed for {filename}: {e}")
                return {"error": str(e), "call_type": call_type}
        
        # Fallback (should never reach here)
        return {"error": "Max retries exceeded", "call_type": call_type}

    def calculate_score(self, evaluation: Dict) -> Dict:
        """
        Calculate overall score (0-100) from YES/PARTIAL/NO scores.
        YES = 1.0 * weight, PARTIAL = 0.5 * weight, NO = 0.0 * weight, N/A = skip
        """
        criteria_results = evaluation.get("criteria", {})
        if not criteria_results:
            return {"overall_score": 0, "total_points": 0, "total_weight": 0,
                    "category_scores": {}, "score_breakdown": {}}

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
                "count": data["count"]
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
                "na_count": na_count
            }
        }

    def calculate_listening_ratio(self, transcript: str) -> Dict[str, float]:
        """Estimate agent vs client talking percentage from transcript.
        Supports ElevenLabs Scribe format (Speaker 0/1) and labeled format (Agent:/Client:)."""
        lines = transcript.strip().split("\n")
        agent_words = 0
        client_words = 0

        # ElevenLabs speaker labels
        agent_prefixes = ["agent:", "representative:", "rep:", "advisor:", "speaker 0:", "speaker 1:"]
        client_prefixes = ["client:", "customer:", "caller:", "guest:", "speaker 2:", "speaker 3:"]

        # Auto-detect: if transcript uses "Speaker X:" format, assign first speaker as agent
        first_speaker = None
        for line in lines:
            line_stripped = line.lower().strip()
            match = re.match(r'^(speaker \d+):', line_stripped)
            if match:
                first_speaker = match.group(1)
                break

        for line in lines:
            line_lower = line.lower().strip()
            word_count = len(line.split())
            if not line_lower or word_count == 0:
                continue

            # Check explicit labels first
            is_agent = any(line_lower.startswith(p) for p in agent_prefixes[:4])
            is_client = any(line_lower.startswith(p) for p in client_prefixes[:4])

            # Handle Speaker N: format — first speaker = agent
            if not is_agent and not is_client:
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
            # Skip unlabeled lines instead of defaulting to agent

        total = agent_words + client_words
        if total == 0:
            return {"agent_percentage": 50.0, "client_percentage": 50.0, "total_words": 0}

        return {
            "agent_percentage": round(agent_words / total * 100, 1),
            "client_percentage": round(client_words / total * 100, 1),
            "total_words": total
        }
