"""
Pipeline Orchestrator

Purpose: Run Agent 1 -> 2 -> 3 -> 4 in sequence

Flow:
    Agent 1 (RingCentral)  -> List of audio files + metadata
    Agent 2 (ElevenLabs)   -> List of transcripts with diarization
    Agent 3 (OpenRouter)   -> List of QA evaluations (24 criteria scored)
    Agent 4 (Integration)  -> Delivery confirmation

TODO:
    - Initialize all 4 agents from config
    - Run agents in sequence, passing output -> input
    - Handle partial failures (if 1 call fails, continue with rest)
    - Track total cost (ElevenLabs + OpenRouter)
    - Save intermediate results to data/ folders
    - Log progress
"""


class Pipeline:
    """
    Orchestrates the 4-agent QA pipeline.

    TODO: Implement these methods:

    __init__(self, config: dict)
        - Create all 4 agent instances from config
        - Set up cost tracking (simple dict)

    run(self, date_from: str, date_to: str) -> list[dict]
        - Step 1: agent_01.search_and_download(date_from, date_to)
        - Step 2: For each audio -> agent_02.transcribe(audio_path)
        - Step 3: For each transcript -> agent_03.evaluate(transcript)
        - Step 4: For each evaluation -> agent_04.deliver(evaluation)
        - Return list of results
        - Print cost summary at end

    process_single_call(self, call_id: str) -> dict
        - Same flow but for one specific call
        - Useful for testing
    """

    def __init__(self, config: dict):
        # TODO: Implement
        pass

    def run(self, date_from: str, date_to: str) -> list:
        # TODO: Implement
        pass

    def process_single_call(self, call_id: str) -> dict:
        # TODO: Implement
        pass
