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
import signal
import time
import logging

logger = logging.getLogger("qa_system.pipeline")


class _GracefulShutdown:
    """MED-3: Handle SIGINT/SIGTERM for graceful shutdown."""
    _triggered = False

    @classmethod
    def trigger(cls, signum, frame):
        cls._triggered = True
        logger.warning(f"Shutdown signal received ({signum}). Finishing current evaluation...")

    @classmethod
    def is_triggered(cls) -> bool:
        return cls._triggered

    @classmethod
    def reset(cls):
        cls._triggered = False


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

    def __init__(self, agent_01, agent_02, agent_03, agent_04,
                 max_consecutive_failures: int = 3,
                 delay_between_evaluations: float = 1.0,
                 cost_warning_threshold_usd: float = 0.50):
        """Store all 4 agents and initialize tracking.

        Args:
            agent_01: RingCentralAgent or AudioFileFinder
            agent_02: ElevenLabsSTTAgent
            agent_03: QualityManagementAgent
            agent_04: IntegrationAgent
            max_consecutive_failures: Circuit breaker threshold.
            delay_between_evaluations: Seconds to wait between LLM calls (HIGH-2).
            cost_warning_threshold_usd: Warn/stop if cumulative batch cost exceeds this.
        """
        self.audio_agent = agent_01
        self.stt_agent = agent_02
        self.qa_agent = agent_03
        self.integration_agent = agent_04
        self.total_cost = 0.0
        self.max_consecutive_failures = max_consecutive_failures
        self.delay_between_evaluations = delay_between_evaluations
        self.cost_warning_threshold_usd = cost_warning_threshold_usd
        self._providers_used: set = set()  # HIGH-12: track which providers were used
        self._cost_warning_issued = False

        # CRIT-1: Register signal handlers here (not at module import time)
        signal.signal(signal.SIGINT, _GracefulShutdown.trigger)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, _GracefulShutdown.trigger)

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
        # MED-8: Aggregate STT costs
        self._stt_cost = sum(v.get("cost_usd", 0) for v in transcripts.values())
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

            # MED-3: Check for graceful shutdown signal
            if _GracefulShutdown.is_triggered():
                logger.warning("Graceful shutdown: stopping evaluation loop.")
                print(f"\n  STOPPED: Shutdown signal received.")
                break

            # HIGH-2: Rate limiting between LLM calls
            if eval_count > 1 and self.delay_between_evaluations > 0:
                time.sleep(self.delay_between_evaluations)

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
                    # HIGH-3: Flag incomplete results
                    remaining = sum(1 for fn, d in transcripts.items()
                                    if d.get("status") == "Success" and fn not in
                                    {e.get("filename") for e in evaluations})
                    evaluations.append({
                        "filename": "CIRCUIT_BREAKER",
                        "status": "CIRCUIT_BREAKER_TRIGGERED",
                        "remaining_calls": remaining,
                        "error": f"{consecutive_failures} consecutive API failures",
                    })
                    break
                continue

            consecutive_failures = 0  # Reset on success
            score_data = self.qa_agent.calculate_score(evaluation)

            cost = evaluation.get("cost_usd", 0)
            self.total_cost += cost

            # Cost budget guard: warn when cumulative cost exceeds threshold
            if (self.cost_warning_threshold_usd > 0
                    and self.total_cost >= self.cost_warning_threshold_usd
                    and not self._cost_warning_issued):
                self._cost_warning_issued = True
                logger.warning(
                    f"Cost budget warning: cumulative LLM cost ${self.total_cost:.4f} "
                    f"exceeds threshold ${self.cost_warning_threshold_usd:.2f}"
                )
                print(f"\n  WARNING: Cumulative cost ${self.total_cost:.4f} "
                      f"exceeds threshold ${self.cost_warning_threshold_usd:.2f}")

            # HIGH-12: Track which providers were used
            provider = evaluation.get("provider_used", evaluation.get("model_used", "unknown"))
            self._providers_used.add(provider)

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

        # HIGH-3/10: Filter out circuit breaker sentinel rows before export
        exportable = [e for e in evaluations if e.get("status") != "CIRCUIT_BREAKER_TRIGGERED"]

        # Step 4: Export results
        print("STEP 4: Exporting results...")
        if not exportable:
            print("  No valid evaluations to export (all circuit-breaker).")
            return evaluations
        files = self.integration_agent.export_all(exportable, self.qa_agent.EVALUATION_CRITERIA)

        # Print summary
        self.print_summary(evaluations)

        return evaluations

    def print_summary(self, evaluations: List[Dict]) -> None:
        """Print evaluation summary."""
        if not evaluations:
            print("No evaluations to summarize.")
            return

        scores = [e["overall_score"] for e in evaluations if "overall_score" in e]
        if not scores:
            print("No scored evaluations to summarize.")
            return
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
        # MED-8: Include STT cost in summary
        stt_cost = getattr(self, '_stt_cost', 0)
        if stt_cost > 0:
            print(f"  STT cost:        ${stt_cost:.4f}")
            print(f"  Combined cost:   ${total_cost + stt_cost:.4f}")
        # HIGH-12: Show which providers were used
        if self._providers_used:
            print(f"  Providers used:  {', '.join(sorted(self._providers_used))}")
            if len(self._providers_used) > 1:
                logger.warning(f"Multiple providers used (possible fallback): {self._providers_used}")
        print(f"{'='*60}")

        # Per-file scores
        print(f"\n  {'File':<40} {'Type':<15} {'Score':>6}")
        print(f"  {'-'*40} {'-'*15} {'-'*6}")
        for e in evaluations:
            if "overall_score" not in e:
                continue
            print(f"  {e['filename']:<40} {e['call_type']:<15} {e['overall_score']:>5.1f}")
