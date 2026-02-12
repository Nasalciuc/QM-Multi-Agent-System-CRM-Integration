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
from pathlib import Path

from utils import setup_logging, load_config, load_env, validate_env, validate_models_config
from agents.agent_01_audio import AudioFileFinder, RingCentralAgent
from agents.agent_02_transcription import ElevenLabsSTTAgent
from agents.agent_03_evaluation import QualityManagementAgent
from agents.agent_04_export import IntegrationAgent
from core.model_factory import ModelFactory
from pipeline import Pipeline

# Required env vars per mode
_BASE_ENV_KEYS = ['ELEVENLABS_API_KEY', 'OPENROUTER_API_KEY']
_RC_ENV_KEYS = ['RC_APP_CLIENT_ID', 'RC_APP_CLIENT_SECRET', 'RC_SERVER_URL', 'RC_USER_JWT']


def main():
    parser = argparse.ArgumentParser(description="Call Center QA System")
    parser.add_argument('--date-from', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--date-to', help='End date (YYYY-MM-DD)')
    parser.add_argument('--local', nargs='+', help='Local audio file paths')
    parser.add_argument('--folder', help='Local audio folder path')
    parser.add_argument('--check', action='store_true', help='Health check: validate env, config, exit 0')
    args = parser.parse_args()

    # Load environment and config
    load_env()
    logger = setup_logging()
    config = load_config()

    # MED-19: Health check mode — validate environment and exit
    if args.check:
        models_config = validate_models_config()
        logger.info("Health check passed")
        print("OK — config valid, env loaded")
        sys.exit(0)

    # HIGH-9: Validate models.yaml at startup for clear error messages
    models_config = validate_models_config()
    logger.info(f"models.yaml validated: {len(models_config.get('fallbacks', []))} fallback(s) configured")

    # Validate required env vars upfront
    required = list(_BASE_ENV_KEYS)
    if args.date_from:
        required.extend(_RC_ENV_KEYS)
    validate_env(required)

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
    )

    # Initialize Agent 3: QualityManagement (with ModelFactory fallback)
    model_factory = ModelFactory()
    agent_qm = QualityManagementAgent(model_factory)

    # Initialize Agent 4: Integration
    integration_config = config.get("integration", {})
    agent_integration = IntegrationAgent(
        output_folder=integration_config.get("output_folder", "data/evaluations"),
        webhook_url=os.environ.get('WEBHOOK_URL', '')
    )

    # Mode: Local files
    if args.local:
        audio_files = [Path(f) for f in args.local]
        agent_finder = AudioFileFinder(folder_path=".")
        pipeline = Pipeline(agent_finder, agent_stt, agent_qm, agent_integration)
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

        pipeline = Pipeline(agent_finder, agent_stt, agent_qm, agent_integration)
        results = pipeline.run_local(audio_files)

    # Mode: RingCentral API
    elif args.date_from:
        from ringcentral import SDK
        rc_config = config.get("ringcentral", {})

        sdk = SDK(
            os.environ['RC_APP_CLIENT_ID'],
            os.environ['RC_APP_CLIENT_SECRET'],
            os.environ['RC_SERVER_URL']
        )
        platform = sdk.platform()
        platform.login(jwt=os.environ['RC_USER_JWT'])

        agent_rc = RingCentralAgent(
            platform,
            download_folder=rc_config.get("download_folder", "data/audio"),
            delay_seconds=rc_config.get("delay_seconds", 1.5)
        )

        pipeline = Pipeline(agent_rc, agent_stt, agent_qm, agent_integration)
        results = pipeline.run(args.date_from, args.date_to)

    else:
        parser.print_help()
        sys.exit(1)

    print(f"\nDone! {len(results)} calls evaluated.")


if __name__ == "__main__":
    main()
