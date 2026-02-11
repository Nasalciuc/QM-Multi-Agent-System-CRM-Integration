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

from utils import setup_logging, load_config, load_env, validate_env
from agents.agent_01_RingcentralCall import AudioFileFinder, RingCentralAgent
from agents.agent_02_Transcribition import ElevenLabsSTTAgent
from agents.agent_03_QualityManagement import QualityManagementAgent
from agents.agent_04_ResultSending import IntegrationAgent
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
    args = parser.parse_args()

    # Load environment and config
    load_env()
    logger = setup_logging()
    config = load_config()

    # Validate required env vars upfront
    required = list(_BASE_ENV_KEYS)
    if args.date_from:
        required.extend(_RC_ENV_KEYS)
    validate_env(required)

    # Initialize Agent 2: ElevenLabs STT
    from elevenlabs import ElevenLabs
    el_client = ElevenLabs(api_key=os.environ.get('ELEVENLABS_API_KEY'))
    agent_stt = ElevenLabsSTTAgent(el_client)

    # Initialize Agent 3: QualityManagement (OpenRouter)
    from openai import OpenAI
    oa_client = OpenAI(
        api_key=os.environ.get('OPENROUTER_API_KEY'),
        base_url="https://openrouter.ai/api/v1"
    )
    agent_qm = QualityManagementAgent(oa_client)

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
