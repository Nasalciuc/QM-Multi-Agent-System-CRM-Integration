"""
Main CLI Entry Point

Usage:
    python src/main.py --date-from 2025-02-01 --date-to 2025-02-11
    python src/main.py --local data/audio/call1.mp3 data/audio/call2.mp3
    python src/main.py --folder data/audio

TODO:
- Parse CLI arguments
- Load config from config/agents.yaml and .env
- Initialize all 4 agents
- Run pipeline
- Print summary
"""

import argparse
import os
from pathlib import Path


def main():
    """
    TODO:
    1. Parse args
    2. Load .env with python-dotenv
    3. Load config/agents.yaml
    4. Initialize agents:
       - RingCentralAgent or AudioFileFinder (depends on args)
       - ElevenLabsSTTAgent(elevenlabs_client)
       - QualityManagementAgent(openai_client)
       - IntegrationAgent(output_folder, webhook_url)
    5. Create Pipeline(agent_01, agent_02, agent_03, agent_04)
    6. If --local or --folder: pipeline.run_local(audio_files)
       If --date-from: pipeline.run(date_from, date_to)
    7. Print summary

    Agent initialization pattern:
        # ElevenLabs
        from elevenlabs import ElevenLabs
        el_client = ElevenLabs(api_key=os.environ['ELEVENLABS_API_KEY'])
        agent_stt = ElevenLabsSTTAgent(el_client)

        # OpenAI/OpenRouter
        from openai import OpenAI
        oa_client = OpenAI(
            api_key=os.environ['OPENROUTER_API_KEY'],
            base_url="https://openrouter.ai/api/v1"
        )
        agent_qm = QualityManagementAgent(oa_client)
    """
    parser = argparse.ArgumentParser(description="Call Center QA System")
    parser.add_argument('--date-from', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--date-to', help='End date (YYYY-MM-DD)')
    parser.add_argument('--local', nargs='+', help='Local audio file paths')
    parser.add_argument('--folder', help='Local audio folder path')
    args = parser.parse_args()

    # TODO: Implement


if __name__ == "__main__":
    main()
