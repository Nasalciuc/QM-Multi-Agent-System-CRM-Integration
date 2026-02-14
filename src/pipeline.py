"""
Pipeline Orchestrator

Purpose: Run Agent 1 -> 2 -> 3 -> 4 in sequence

Flow:
    Agent 1 (CRM/Local)        -> List of audio file paths
    Agent 2 (ElevenLabs)        -> Dict of {filename: {transcript, duration, ...}}
    Agent 3 (QualityMgmt)       -> List of evaluation dicts
    Agent 4 (Integration)       -> Excel/CSV/JSON files
"""

from typing import List, Dict, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import signal
import threading
import time
import logging

from utils import safe_log_filename

logger = logging.getLogger("qa_system.pipeline")


class _GracefulShutdown:
    """MED-3 / HIGH-9: Handle SIGINT/SIGTERM for graceful shutdown.

    HIGH-9: Converted from class-level state to instance with threading.Event
    for thread-safe signalling and cleaner state management.
    MED-NEW-15: reset() is called at pipeline start so stale state
    from a previous run in the same process does not carry over.
    """

    def __init__(self):
        self._event = threading.Event()

    def trigger(self, signum=None, frame=None):
        """Signal handler callback — sets the shutdown event."""
        self._event.set()
        logger.warning(f"Shutdown signal received ({signum}). Finishing current evaluation...")

    def is_triggered(self) -> bool:
        return self._event.is_set()

    def reset(self):
        """Reset shutdown flag — MUST be called at pipeline start."""
        self._event.clear()

    def wait(self, timeout: float) -> bool:
        """Wait up to timeout seconds. Returns True if shutdown was triggered."""
        return self._event.wait(timeout)


class Pipeline:
    """
    Orchestrates the 4-agent QA pipeline.

    Usage (from CRM API):
        pipeline = Pipeline(agent_crm, agent_stt, agent_qm, agent_integration)
        results = pipeline.run(date_from="2025-02-01", date_to="2025-02-11")

    Usage (from local files):
        pipeline = Pipeline(agent_finder, agent_stt, agent_qm, agent_integration)
        results = pipeline.run_local(audio_files=[Path("data/audio/call1.mp3"), ...])
    """

    def __init__(self, agent_01, agent_02, agent_03, agent_04,
                 max_consecutive_failures: int = 3,
                 delay_between_evaluations: float = 1.0,
                 cost_warning_threshold_usd: float = 0.50,
                 max_workers: int = 1,
                 max_budget_usd: float = 0.0):
        """Store all 4 agents and initialize tracking.

        Args:
            agent_01: CRMAgent or AudioFileFinder
            agent_02: ElevenLabsSTTAgent
            agent_03: QualityManagementAgent
            agent_04: IntegrationAgent
            max_consecutive_failures: Circuit breaker threshold.
            delay_between_evaluations: Seconds to wait between LLM calls (HIGH-2).
            cost_warning_threshold_usd: Warn/stop if cumulative batch cost exceeds this.
            max_workers: Number of parallel evaluation workers (HIGH-NEW-8).
                         Set to 1 for sequential processing.
            max_budget_usd: Hard budget limit. 0 = unlimited.
                            Pipeline stops evaluations when this is exceeded.
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
        self._max_workers = max(1, max_workers)  # HIGH-NEW-8: concurrency level
        self._max_budget_usd = max_budget_usd  # P3: Hard budget limit
        self._budget_80_warned = False  # P3: 80% budget warning flag
        self._stt_cost = 0.0  # CRIT-4: Initialize STT cost for budget correctness

        # HIGH-9: Instance-based graceful shutdown with threading.Event
        self._shutdown = _GracefulShutdown()

        # CRIT-1: Register signal handlers, save old ones so we can restore later
        self._old_sigint = signal.signal(signal.SIGINT, self._shutdown.trigger)
        self._old_sigterm = None
        if hasattr(signal, 'SIGTERM'):
            self._old_sigterm = signal.signal(signal.SIGTERM, self._shutdown.trigger)

    def _restore_signals(self) -> None:
        """Restore original signal handlers saved during __init__."""
        if self._old_sigint is not None:
            signal.signal(signal.SIGINT, self._old_sigint)
        if self._old_sigterm is not None and hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._old_sigterm)

    def _interruptible_sleep(self, seconds: float, granularity: float = 0.25) -> None:
        """MED-NEW-11: Sleep in small increments so shutdown signals
        are noticed promptly instead of blocking for the full duration.
        HIGH-9: Uses instance _shutdown.wait() for thread-safe signalling.
        """
        remaining = seconds
        while remaining > 0 and not self._shutdown.is_triggered():
            time.sleep(min(granularity, remaining))
            remaining -= granularity

    @staticmethod
    def _check_disk_space(min_free_mb: int = 500) -> None:
        """#7: Check available disk space before processing.

        Raises:
            RuntimeError: If free disk space is below min_free_mb.
        """
        usage = shutil.disk_usage(".")
        free_mb = usage.free / (1024 * 1024)
        if free_mb < min_free_mb:
            raise RuntimeError(
                f"Insufficient disk space: {free_mb:.0f} MB free "
                f"(minimum {min_free_mb} MB required)"
            )
        logger.debug(f"Disk space check: {free_mb:.0f} MB free")

    def run(self, date_from: str, date_to: Optional[str] = None) -> List[Dict]:
        """Full pipeline: Audio source -> ElevenLabs -> QA -> Export"""

        logger.info(f"QA Pipeline: CRM Mode | Period: {date_from} to {date_to or 'now'}")

        pipeline_start = time.time()
        self._shutdown.reset()  # MED-NEW-15: Clear stale shutdown state

        # Step 1: Download recordings
        logger.info("STEP 1: Downloading recordings from CRM...")
        calls = self.audio_agent.search_and_download(date_from, date_to)
        audio_files = [Path(c["local_audio_path"]) for c in calls if c.get("local_audio_path")]
        logger.info(f"STEP 1 complete: {len(audio_files)} audio files ready")

        if not audio_files:
            logger.info("No recordings found. Pipeline complete.")
            return []

        # Steps 2-4: Process audio files
        evaluations = self._process_audio_files(audio_files)

        elapsed = time.time() - pipeline_start
        logger.info(f"Pipeline complete in {elapsed:.1f}s")

        return evaluations

    def run_local(self, audio_files: List[Path]) -> List[Dict]:
        """Pipeline from local files (skip CRM download)."""
        logger.info(f"QA Pipeline: Local Mode | Files: {len(audio_files)}")

        pipeline_start = time.time()
        self._shutdown.reset()  # MED-NEW-15: Clear stale shutdown state

        if not audio_files:
            logger.info("No audio files provided. Pipeline complete.")
            return []

        evaluations = self._process_audio_files(audio_files)

        elapsed = time.time() - pipeline_start
        logger.info(f"Pipeline complete in {elapsed:.1f}s")

        return evaluations

    def _process_audio_files(self, audio_files: List[Path]) -> List[Dict]:
        """Steps 2-4: Transcribe, evaluate, export."""

        # Fix #5: Re-enable providers disabled in previous run
        if hasattr(self.qa_agent, '_engine') and hasattr(self.qa_agent._engine, '_factory'):
            self.qa_agent._engine._factory.reset_disabled_providers()

        # #7: Check disk space before processing
        self._check_disk_space()

        # Step 2: Transcribe with ElevenLabs
        logger.info("STEP 2: Transcribing with ElevenLabs Scribe v2...")
        transcripts = self.stt_agent.transcribe_batch(audio_files)
        success_count = sum(1 for v in transcripts.values() if v.get("status") == "Success")
        # MED-8: Aggregate STT costs
        self._stt_cost = sum(v.get("cost_usd", 0) for v in transcripts.values())
        self._transcripts = transcripts  # Store for summary access
        logger.info(f"STEP 2 complete: {success_count} transcripts ready")

        # Step 3: Evaluate with QA Agent
        logger.info("STEP 3: Evaluating with QualityManagementAgent...")
        evaluations = []
        eval_count = 0
        consecutive_failures = 0

        for filename, data in transcripts.items():
            if data.get("status") != "Success":
                continue

            eval_count += 1
            logger.info(f"Evaluating {eval_count}: {safe_log_filename(filename)}...")

            # MED-3: Check for graceful shutdown signal
            if self._shutdown.is_triggered():
                logger.warning("Graceful shutdown: stopping evaluation loop.")
                break

            # HIGH-2: Rate limiting between LLM calls
            # MED-NEW-11: Use interruptible sleep so shutdown signals
            # are honoured during the delay window.
            if eval_count > 1 and self.delay_between_evaluations > 0:
                self._interruptible_sleep(self.delay_between_evaluations)
                if self._shutdown.is_triggered():
                    logger.warning("Graceful shutdown during rate-limit delay.")
                    break

            evaluation = self.qa_agent.evaluate_call(data["transcript"], filename)

            # Circuit breaker: stop after N consecutive LLM failures
            if "error" in evaluation:
                consecutive_failures += 1
                logger.warning(f"Evaluation failed for {safe_log_filename(filename)}: {evaluation['error']} "
                               f"({consecutive_failures}/{self.max_consecutive_failures} consecutive)")
                if consecutive_failures >= self.max_consecutive_failures:
                    logger.error(f"Circuit breaker triggered: {consecutive_failures} consecutive failures. "
                                 f"Stopping evaluations.")
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

            # HIGH-NEW-6 + #9: Only reset circuit breaker on *genuine* success
            # — soft failures (validation, too-short) and evaluations with
            # too few scored criteria should NOT reset it
            scored_criteria = sum(
                1 for c in evaluation.get("criteria", {}).values()
                if isinstance(c, dict) and c.get("score", "").upper() in ("YES", "PARTIAL", "NO")
            )
            if evaluation.get("status") != "TOO_SHORT" and scored_criteria > 0:
                consecutive_failures = 0
            score_data = self.qa_agent.calculate_score(evaluation)

            cost = evaluation.get("cost_usd", 0)
            self.total_cost += cost

            # P3: Hard budget enforcement — stop when exceeded
            combined_cost = self.total_cost + self._stt_cost
            if self._max_budget_usd > 0:
                # 80% warning
                if (combined_cost >= self._max_budget_usd * 0.8
                        and not self._budget_80_warned):
                    self._budget_80_warned = True
                    logger.warning(
                        f"Budget 80% warning: ${combined_cost:.4f} / "
                        f"${self._max_budget_usd:.2f}"
                    )
                # Hard stop at 100%
                if combined_cost >= self._max_budget_usd:
                    logger.error(
                        f"Budget exceeded: ${combined_cost:.4f} >= "
                        f"${self._max_budget_usd:.2f}. Stopping."
                    )
                    break

            # Cost budget guard: warn when cumulative cost exceeds threshold
            if (self.cost_warning_threshold_usd > 0
                    and self.total_cost >= self.cost_warning_threshold_usd
                    and not self._cost_warning_issued):
                self._cost_warning_issued = True
                logger.warning(
                    f"Cost budget warning: cumulative LLM cost ${self.total_cost:.4f} "
                    f"exceeds threshold ${self.cost_warning_threshold_usd:.2f}"
                )

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

        logger.info(f"STEP 3 complete: {len(evaluations)} evaluations")

        if not evaluations:
            logger.info("No evaluations to export.")
            return evaluations

        # HIGH-3/10: Filter out circuit breaker sentinel rows before export
        exportable = [e for e in evaluations if e.get("status") != "CIRCUIT_BREAKER_TRIGGERED"]

        # Step 4: Export results
        logger.info("STEP 4: Exporting results...")
        if not exportable:
            logger.info("No valid evaluations to export (all circuit-breaker).")
            return evaluations
        files = self.integration_agent.export_all(exportable, self.qa_agent.EVALUATION_CRITERIA)

        # Print summary
        self.print_summary(evaluations)

        return evaluations

    def print_summary(self, evaluations: List[Dict]) -> None:
        """Print comprehensive evaluation summary with cost & cache breakdown.

        HIGH-7: Uses logger.info instead of print for structured output.
        """
        if not evaluations:
            logger.info("No evaluations to summarize.")
            return

        scores = [e["overall_score"] for e in evaluations if "overall_score" in e]
        if not scores:
            logger.info("No scored evaluations to summarize.")
            return

        avg_score = sum(scores) / len(scores)
        llm_cost = sum(e.get("cost_usd", 0) for e in evaluations)
        stt_cost = self._stt_cost
        combined_cost = llm_cost + stt_cost

        # Token utilization
        total_input_tokens = sum(
            e.get("tokens_used", {}).get("input", 0) for e in evaluations
        )
        total_output_tokens = sum(
            e.get("tokens_used", {}).get("output", 0) for e in evaluations
        )

        # Build summary lines for a single structured log message
        std_dev_line = ""
        if len(scores) > 1:
            std_dev = (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5
            std_dev_line = f" | StdDev: {std_dev:.1f}"

        logger.info(
            f"PIPELINE SUMMARY | Evaluated: {len(scores)} | "
            f"Avg: {avg_score:.1f}/100 | Best: {max(scores):.1f} | "
            f"Worst: {min(scores):.1f}{std_dev_line}"
        )

        # ── Cost Breakdown ───────────────────────────────────────────
        stt_part = f" | STT: ${stt_cost:.4f}" if stt_cost > 0 else ""
        logger.info(
            f"Cost breakdown | LLM: ${llm_cost:.4f}{stt_part} | "
            f"Total: ${combined_cost:.4f}"
        )

        # STT cache savings
        stt_cache_stats = {}
        if hasattr(self.stt_agent, 'stt_cache') and hasattr(self.stt_agent.stt_cache, 'stats'):
            try:
                stats = self.stt_agent.stt_cache.stats
                if isinstance(stats, dict):
                    stt_cache_stats = stats
            except Exception:
                pass
        if stt_cache_stats.get("hits", 0) > 0:
            cached_savings = sum(
                (v.get("duration", 0) or 0) * 0.005
                for v in getattr(self, '_transcripts', {}).values()
                if v.get("cached")
            )
            logger.info(f"STT cache saved: ${cached_savings:.4f}")

        # Budget status
        if self._max_budget_usd > 0:
            remaining = self._max_budget_usd - combined_cost
            pct_used = combined_cost / self._max_budget_usd * 100
            logger.info(
                f"Budget status | Budget: ${self._max_budget_usd:.2f} | "
                f"Used: ${combined_cost:.4f} ({pct_used:.1f}%) | "
                f"Remaining: ${remaining:.4f}"
            )

        # ── Token Utilization ────────────────────────────────────────
        logger.info(
            f"Token utilization | Input: {total_input_tokens:,} | "
            f"Output: {total_output_tokens:,} | "
            f"Total: {total_input_tokens + total_output_tokens:,}"
        )

        # ── Cache Performance ────────────────────────────────────────
        if stt_cache_stats:
            logger.info(
                f"STT cache: {stt_cache_stats['hits']} hits / "
                f"{stt_cache_stats['total_lookups']} lookups "
                f"({stt_cache_stats['hit_rate_pct']}%)"
            )

        # LLM cache
        llm_stats = {}
        if hasattr(self.qa_agent, '_engine') and hasattr(self.qa_agent._engine, 'cache_stats'):
            try:
                cs = self.qa_agent._engine.cache_stats
                if isinstance(cs, dict):
                    llm_stats = cs
            except Exception:
                pass
        if llm_stats:
            logger.info(
                f"LLM cache | L1: {llm_stats['memory_hits']} hits "
                f"({llm_stats['memory_cache_size']}/{llm_stats['memory_cache_maxsize']} slots) | "
                f"L2: {llm_stats['disk_hits']} hits "
                f"({llm_stats['combined_hit_rate_pct']}% combined)"
            )

        # ── Providers Used ───────────────────────────────────────────
        if self._providers_used:
            logger.info(f"Providers used: {', '.join(sorted(self._providers_used))}")
            if len(self._providers_used) > 1:
                logger.warning(f"Multiple providers used (possible fallback): {self._providers_used}")

        # ── Per-File Table ───────────────────────────────────────────
        for e in evaluations:
            if "overall_score" not in e:
                continue
            logger.info(
                f"  {e['filename']} | {e.get('call_type', 'Unknown')} | "
                f"Score: {e['overall_score']:.1f} | Cost: ${e.get('cost_usd', 0):.4f}"
            )
