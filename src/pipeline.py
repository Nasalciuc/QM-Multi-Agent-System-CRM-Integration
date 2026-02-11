"""
Pipeline Orchestrator

Purpose: Run Agent 1 -> 2 -> 3 -> 4 in sequence

Flow:
    Agent 1 (RingCentral/Local) -> List of audio file paths
    Agent 2 (ElevenLabs)        -> Dict of {filename: {transcript, duration, ...}}
    Agent 3 (QualityMgmt)       -> List of evaluation dicts
    Agent 4 (Integration)       -> Excel/CSV/JSON files
"""

from typing import List, Dict, Optional
from pathlib import Path
import time
import logging

logger = logging.getLogger("qa_system.pipeline")


class Pipeline:
    """
    Orchestrates the 4-agent QA pipeline.

    Usage (from RingCentral):
        pipeline = Pipeline(agent_rc, agent_stt, agent_qm, agent_integration)
        results = pipeline.run(date_from="2025-02-01", date_to="2025-02-11")

    Usage (from local files):
        pipeline = Pipeline(agent_finder, agent_stt, agent_qm, agent_integration)
        results = pipeline.run_local(audio_files=[Path("data/audio/call1.mp3"), ...])
    """

    def __init__(self, agent_01, agent_02, agent_03, agent_04, max_consecutive_failures: int = 3):
        """Store all 4 agents and initialize tracking."""
        self.audio_agent = agent_01      # RingCentralAgent or AudioFileFinder
        self.stt_agent = agent_02        # ElevenLabsSTTAgent
        self.qa_agent = agent_03         # QualityManagementAgent
        self.integration_agent = agent_04 # IntegrationAgent
        self.total_cost = 0.0
        self.max_consecutive_failures = max_consecutive_failures

    def run(self, date_from: str, date_to: Optional[str] = None) -> List[Dict]:
        """Full pipeline: RingCentral -> ElevenLabs -> QA -> Export"""
        print(f"\n{'='*60}")
        print(f"  QA Pipeline: RingCentral Mode")
        print(f"  Period: {date_from} to {date_to or 'now'}")
        print(f"{'='*60}\n")

        pipeline_start = time.time()

        # Step 1: Download recordings from RingCentral
        print("STEP 1: Downloading recordings from RingCentral...")
        calls = self.audio_agent.search_and_download(date_from, date_to)
        audio_files = [Path(c["local_audio_path"]) for c in calls if c.get("local_audio_path")]
        print(f"  -> {len(audio_files)} audio files ready\n")

        if not audio_files:
            print("  No recordings found. Pipeline complete.")
            return []

        # Steps 2-4: Process audio files
        evaluations = self._process_audio_files(audio_files)

        elapsed = time.time() - pipeline_start
        print(f"\nPipeline complete in {elapsed:.1f}s")

        return evaluations

    def run_local(self, audio_files: List[Path]) -> List[Dict]:
        """Pipeline from local files (skip RingCentral download)."""
        print(f"\n{'='*60}")
        print(f"  QA Pipeline: Local Mode")
        print(f"  Files: {len(audio_files)}")
        print(f"{'='*60}\n")

        pipeline_start = time.time()

        if not audio_files:
            print("  No audio files provided. Pipeline complete.")
            return []

        evaluations = self._process_audio_files(audio_files)

        elapsed = time.time() - pipeline_start
        print(f"\nPipeline complete in {elapsed:.1f}s")

        return evaluations

    def _process_audio_files(self, audio_files: List[Path]) -> List[Dict]:
        """Steps 2-4: Transcribe, evaluate, export."""

        # Step 2: Transcribe with ElevenLabs
        print("STEP 2: Transcribing with ElevenLabs Scribe v1...")
        transcripts = self.stt_agent.transcribe_batch(audio_files)
        success_count = sum(1 for v in transcripts.values() if v.get("status") == "Success")
        print(f"  -> {success_count} transcripts ready\n")

        # Step 3: Evaluate with QA Agent
        print("STEP 3: Evaluating with QualityManagementAgent...")
        evaluations = []
        eval_count = 0
        consecutive_failures = 0

        for filename, data in transcripts.items():
            if data.get("status") != "Success":
                continue

            eval_count += 1
            print(f"  Evaluating {eval_count}: {filename}...", end="\r")

            evaluation = self.qa_agent.evaluate_call(data["transcript"], filename)

            # Circuit breaker: stop after N consecutive LLM failures
            if "error" in evaluation:
                consecutive_failures += 1
                logger.warning(f"Evaluation failed for {filename}: {evaluation['error']} "
                               f"({consecutive_failures}/{self.max_consecutive_failures} consecutive)")
                if consecutive_failures >= self.max_consecutive_failures:
                    logger.error(f"Circuit breaker triggered: {consecutive_failures} consecutive failures. "
                                 f"Stopping evaluations.")
                    print(f"\n  STOPPED: {consecutive_failures} consecutive failures (API may be down)")
                    break
                continue

            consecutive_failures = 0  # Reset on success
            score_data = self.qa_agent.calculate_score(evaluation)

            cost = evaluation.get("cost_usd", 0)
            self.total_cost += cost

            evaluations.append({
                "filename": filename,
                "transcript": data["transcript"],
                "duration_min": data.get("duration", 0),
                "call_type": evaluation.get("call_type", "Unknown"),
                "overall_score": score_data["overall_score"],
                "score_data": score_data,
                "criteria": evaluation.get("criteria", {}),
                "overall_assessment": evaluation.get("overall_assessment", ""),
                "strengths": evaluation.get("strengths", []),
                "improvements": evaluation.get("improvements", []),
                "critical_gaps": evaluation.get("critical_gaps", []),
                "model_used": evaluation.get("model_used", ""),
                "tokens_used": evaluation.get("tokens_used", {}),
                "cost_usd": cost,
                "status": "Success" if "error" not in evaluation else evaluation["error"]
            })

        print(f"\n  -> {len(evaluations)} evaluations complete\n")

        if not evaluations:
            print("  No evaluations to export.")
            return evaluations

        # Step 4: Export results
        print("STEP 4: Exporting results...")
        files = self.integration_agent.export_all(evaluations, self.qa_agent.EVALUATION_CRITERIA)

        # Print summary
        self.print_summary(evaluations)

        return evaluations

    def print_summary(self, evaluations: List[Dict]) -> None:
        """Print evaluation summary."""
        if not evaluations:
            print("No evaluations to summarize.")
            return

        scores = [e["overall_score"] for e in evaluations]
        avg_score = sum(scores) / len(scores)
        total_cost = sum(e.get("cost_usd", 0) for e in evaluations)

        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}")
        print(f"  Calls processed: {len(evaluations)}")
        print(f"  Average score:   {avg_score:.1f}/100")
        print(f"  Best score:      {max(scores):.1f}/100")
        print(f"  Worst score:     {min(scores):.1f}/100")
        print(f"  Total cost:      ${total_cost:.4f}")
        print(f"{'='*60}")

        # Per-file scores
        print(f"\n  {'File':<40} {'Type':<15} {'Score':>6}")
        print(f"  {'-'*40} {'-'*15} {'-'*6}")
        for e in evaluations:
            print(f"  {e['filename']:<40} {e['call_type']:<15} {e['overall_score']:>5.1f}")
