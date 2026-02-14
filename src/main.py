"""
Main CLI Entry Point

Usage:
    python src/main.py --date-from 2025-02-01 --date-to 2025-02-11
    python src/main.py --local data/audio/call1.mp3 data/audio/call2.mp3
    python src/main.py --folder data/audio
"""

import argparse
import os
import sys
from datetime import datetime as _dt
from pathlib import Path
from typing import Optional

from utils import setup_logging, load_config, load_env, validate_env, validate_models_config
from agents.agent_01_audio import AudioFileFinder, CRMAgent
from agents.agent_02_transcription import ElevenLabsSTTAgent
from agents.agent_03_evaluation import QualityManagementAgent
from agents.agent_04_export import IntegrationAgent
from core.model_factory import ModelFactory
from pipeline import Pipeline

# Required env vars per mode
_BASE_ENV_KEYS = ['ELEVENLABS_API_KEY', 'OPENROUTER_API_KEY']
_CRM_ENV_KEYS = ['CRM_AI_TOKEN']

# Cost constants (match agent_02_transcription.py)
_STT_COST_PER_MINUTE = 0.005
# Rough LLM cost estimate per evaluation (~1500 input + ~800 output tokens)
_LLM_COST_PER_EVAL_ESTIMATE = 0.003


def _estimate_audio_duration(audio_path: Path) -> Optional[float]:
    """Estimate audio duration in minutes using pydub (best-effort)."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(audio_path))
        return round(len(audio) / 1000 / 60, 2)
    except Exception:
        return None


def _run_dry_run(args, config, logger):
    """P4: Estimate pipeline costs without making any API calls.

    Scans audio files, estimates STT and LLM costs, reports totals.
    """
    from agents.agent_01_audio import AudioFileFinder, CRMAgent

    print(f"\n{'='*60}")
    print(f"  DRY RUN — Cost Estimation (no API calls)")
    print(f"{'='*60}\n")

    audio_files = []

    if args.local:
        audio_files = [Path(f) for f in args.local if Path(f).exists()]
    elif args.folder:
        extensions = tuple(config.get("audio_extensions", [".mp3", ".wav", ".m4a"]))
        finder = AudioFileFinder(folder_path=args.folder, extensions=extensions)
        audio_files = finder.find_all()
    elif args.date_from:
        print("  NOTE: CRM dry-run cannot count files without API call.")
        print("  Provide --folder or --local for accurate estimates.\n")
        return

    if not audio_files:
        print("  No audio files found.")
        return

    # Check STT cache for potential hits
    el_config = config.get("elevenlabs", {})
    from inference.stt_cache import STTCache
    stt_cache = STTCache(
        cache_dir=el_config.get("stt_cache_dir", "data/stt_cache"),
        enable=el_config.get("enable_stt_cache", True),
        ttl_seconds=el_config.get("stt_cache_ttl_days", 30) * 24 * 3600,
    )

    total_duration = 0.0
    cached_count = 0
    uncached_count = 0

    print(f"  {'File':<40} {'Duration':>8} {'Cached':>8} {'STT Cost':>10}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*10}")

    for audio_path in audio_files:
        duration = _estimate_audio_duration(audio_path) or 0
        total_duration += duration
        dur_str = f"{duration:.1f} min" if duration else "N/A"

        # Check if cached
        cache_key = STTCache.cache_key(
            audio_path,
            model_id=el_config.get("model", "scribe_v2"),
            diarize=el_config.get("diarize", True),
            num_speakers=el_config.get("num_speakers"),
            language_code=el_config.get("language_code"),
        )
        is_cached = stt_cache.load(cache_key) is not None
        if is_cached:
            cached_count += 1
            stt_cost = 0.0
        else:
            uncached_count += 1
            stt_cost = duration * _STT_COST_PER_MINUTE

        print(f"  {audio_path.name:<40} {dur_str:>8} {'YES' if is_cached else 'NO':>8} "
              f"${stt_cost:>9.4f}")

    # Totals
    uncached_duration = sum(
        (_estimate_audio_duration(f) or 0) for f in audio_files
    ) - sum(
        (_estimate_audio_duration(f) or 0) for f in audio_files
        if stt_cache.load(STTCache.cache_key(
            f,
            model_id=el_config.get("model", "scribe_v2"),
            diarize=el_config.get("diarize", True),
            num_speakers=el_config.get("num_speakers"),
            language_code=el_config.get("language_code"),
        )) is not None
    )

    stt_total = uncached_duration * _STT_COST_PER_MINUTE
    llm_total = len(audio_files) * _LLM_COST_PER_EVAL_ESTIMATE
    grand_total = stt_total + llm_total

    print(f"\n  {'Summary':}")
    print(f"    Files:              {len(audio_files)}")
    print(f"    Total duration:     {total_duration:.1f} min")
    print(f"    Cached (skip STT):  {cached_count}")
    print(f"    Need STT API:       {uncached_count}")
    print(f"    Est. STT cost:      ${stt_total:.4f}")
    print(f"    Est. LLM cost:      ${llm_total:.4f}")
    print(f"    Est. total cost:    ${grand_total:.4f}")

    if args.budget and args.budget > 0:
        if grand_total > args.budget:
            print(f"\n    ⚠ OVER BUDGET: ${grand_total:.4f} > ${args.budget:.2f}")
        else:
            print(f"\n    Within budget: ${grand_total:.4f} / ${args.budget:.2f}")

    print()


def _validate_date(value: str, label: str) -> str:
    """MED-16: Validate date string is YYYY-MM-DD and not in the future.

    Returns the validated date string, or calls sys.exit(1) on failure.
    """
    try:
        parsed = _dt.strptime(value, "%Y-%m-%d")
    except ValueError:
        logging_mod = __import__("logging")
        logging_mod.getLogger("qa_system").error(
            f"Invalid {label} date '{value}': expected YYYY-MM-DD format"
        )
        sys.exit(1)
    if parsed.date() > _dt.now().date():
        logging_mod = __import__("logging")
        logging_mod.getLogger("qa_system").error(
            f"{label} date '{value}' is in the future"
        )
        sys.exit(1)
    return value


def main():
    parser = argparse.ArgumentParser(description="Call Center QA System")
    parser.add_argument('--date-from', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--date-to', help='End date (YYYY-MM-DD)')
    parser.add_argument('--local', nargs='+', help='Local audio file paths')
    parser.add_argument('--folder', help='Local audio folder path')
    parser.add_argument('--agent-id', type=int, help='Filter by agent ID (CRM mode)')
    parser.add_argument('--check', action='store_true', help='Health check: validate env, config, exit 0')
    parser.add_argument('--force', action='store_true', help='Force re-run even if lock file exists (#25)')
    parser.add_argument('--budget', type=float, default=0.0,
                        help='Max budget in USD (0 = unlimited). Pipeline stops when exceeded.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Estimate costs without making API calls')
    args = parser.parse_args()

    # Load environment and config
    load_env()
    logger = setup_logging()
    config = load_config()

    # MED-16: Validate date inputs early
    if args.date_from:
        _validate_date(args.date_from, "--date-from")
    if args.date_to:
        _validate_date(args.date_to, "--date-to")

    # MED-19: Health check mode — validate environment and exit
    if args.check:
        models_config = validate_models_config()
        # CRM connectivity check
        if os.environ.get('CRM_AI_TOKEN'):
            try:
                test_agent = CRMAgent(api_token=os.environ['CRM_AI_TOKEN'])
                test_agent.search_recordings(date_from="2026-01-01", date_to="2026-01-01")
                test_agent.close()
                logger.info("CRM API connectivity: OK")
            except Exception as e:
                logger.warning(f"CRM API connectivity check failed: {e}")
        logger.info("Health check passed")
        print("OK — config valid, env loaded")
        sys.exit(0)

    # HIGH-9: Validate models.yaml at startup for clear error messages
    models_config = validate_models_config()
    logger.info(f"models.yaml validated: {len(models_config.get('fallbacks', []))} fallback(s) configured")

    # Validate required env vars upfront
    required = list(_BASE_ENV_KEYS)
    if args.date_from:
        required.extend(_CRM_ENV_KEYS)
    validate_env(required)

    # #25: Idempotency — prevent concurrent/duplicate runs
    # HIGH-10: Verify PID in lock file is still running before rejecting
    lock_file = Path("data/.pipeline.lock")
    if lock_file.exists() and not args.force:
        try:
            stored_pid = int(lock_file.read_text().strip())
            try:
                os.kill(stored_pid, 0)  # Check if process exists (signal 0 = no-op)
                # Process exists — lock is valid
                logger.error(
                    f"Pipeline lock file exists (PID {stored_pid} is running). "
                    "Use --force to override."
                )
                sys.exit(1)
            except (OSError, ProcessLookupError):
                # PID not running — stale lock file, safe to remove
                logger.warning(
                    f"Stale lock file found (PID {stored_pid} not running). Removing."
                )
                lock_file.unlink(missing_ok=True)
        except (ValueError, OSError):
            # Can't read/parse lock file — treat as stale
            logger.warning("Could not parse lock file PID. Removing stale lock.")
            lock_file.unlink(missing_ok=True)
    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file.write_text(str(os.getpid()))
    except OSError as e:
        logger.warning(f"Could not create lock file: {e}")

    # ── P4: Dry-run mode — estimate costs without API calls ──────
    if args.dry_run:
        _run_dry_run(args, config, logger)
        try:
            lock_file.unlink(missing_ok=True)
        except OSError:
            pass
        sys.exit(0)

    # Initialize Agent 2: ElevenLabs STT (Scribe v2 with diarization)
    from elevenlabs import ElevenLabs
    el_config = config.get("elevenlabs", {})
    el_client = ElevenLabs(api_key=os.environ.get('ELEVENLABS_API_KEY'))
    agent_stt = ElevenLabsSTTAgent(
        el_client,
        persist_transcripts=el_config.get("persist_transcripts", True),
        transcripts_folder=el_config.get("output_folder", "data/transcripts"),
        model_id=el_config.get("model", "scribe_v2"),
        diarize=el_config.get("diarize", True),
        num_speakers=el_config.get("num_speakers"),
        diarization_threshold=el_config.get("diarization_threshold"),
        tag_audio_events=el_config.get("tag_audio_events", False),
        language_code=el_config.get("language_code"),
        keyterms=el_config.get("keyterms", []),
        delay_between_calls=el_config.get("delay_between_calls", 0.5),
        stt_cache_dir=el_config.get("stt_cache_dir", "data/stt_cache"),
        enable_stt_cache=el_config.get("enable_stt_cache", True),
        stt_cache_ttl_days=el_config.get("stt_cache_ttl_days", 30),
    )

    # Initialize Agent 3: QualityManagement (with ModelFactory fallback)
    model_factory = ModelFactory()
    agent_qm = QualityManagementAgent(model_factory)

    # Initialize Agent 4: Integration
    integration_config = config.get("integration", {})
    agent_integration = IntegrationAgent(
        output_folder=integration_config.get("output_folder", "data/evaluations"),
        webhook_url=os.environ.get('WEBHOOK_URL', ''),
        webhook_secret=os.environ.get('WEBHOOK_SECRET', ''),
    )

    # Mode: Local files
    if args.local:
        audio_files = [Path(f) for f in args.local]
        agent_finder = AudioFileFinder(folder_path=".")
        pipeline = Pipeline(agent_finder, agent_stt, agent_qm, agent_integration,
                            max_budget_usd=args.budget)
        results = pipeline.run_local(audio_files)

    # Mode: Local folder
    elif args.folder:
        extensions = tuple(config.get("audio_extensions", [".mp3", ".wav", ".m4a"]))
        agent_finder = AudioFileFinder(folder_path=args.folder, extensions=extensions)
        audio_files = agent_finder.find_all()

        if not audio_files:
            print(f"No audio files found in {args.folder}")
            sys.exit(1)

        # Print file info
        for f in audio_files:
            info = agent_finder.get_info(f)
            duration = agent_finder.get_duration(f)
            dur_str = f"{duration:.1f} min" if duration else "N/A"
            print(f"  {info['name']} | {info['size_mb']:.1f} MB | {dur_str}")

        pipeline = Pipeline(agent_finder, agent_stt, agent_qm, agent_integration,
                            max_budget_usd=args.budget)
        results = pipeline.run_local(audio_files)

    # Mode: CRM API
    elif args.date_from:
        crm_config = config.get("crm", {})
        agent_audio = CRMAgent(
            api_token=os.environ['CRM_AI_TOKEN'],
            base_url=crm_config.get("base_url", CRMAgent.BASE_URL),
            download_folder=crm_config.get("download_folder", "data/audio"),
            delay_seconds=crm_config.get("delay_seconds", 1.5),
            agent_id=args.agent_id,
        )

        # CRIT-3: Ensure CRMAgent httpx client is closed even on error
        try:
            pipeline = Pipeline(agent_audio, agent_stt, agent_qm, agent_integration,
                                max_budget_usd=args.budget)
            results = pipeline.run(args.date_from, args.date_to)
        finally:
            agent_audio.close()

    else:
        parser.print_help()
        sys.exit(1)

    print(f"\nDone! {len(results)} calls evaluated.")

    # #25: Remove lock file on successful completion
    try:
        lock_file.unlink(missing_ok=True)
    except OSError:
        pass


if __name__ == "__main__":
    main()
