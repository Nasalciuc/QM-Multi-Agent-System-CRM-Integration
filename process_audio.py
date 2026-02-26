#!/usr/bin/env python3
"""
Process newly added audio files: transcribe and evaluate.
"""
import sys
import os
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load .env
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Ensure src/ is in path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.model_factory import ModelFactory
from agents.agent_02_transcription import ElevenLabsSTTAgent
from agents.agent_03_evaluation import QualityManagementAgent
from logging_setup import setup_logging

setup_logging()
logger = logging.getLogger("qa_system")

AUDIO_DIR = Path(__file__).parent / "data" / "audio"
CACHE_DIR = Path(__file__).parent / "data" / "cache"
STT_CACHE_DIR = Path(__file__).parent / "data" / "stt_cache"

def main():
    """Process audio files: transcribe and evaluate."""
    audio_files = sorted([f for f in AUDIO_DIR.glob("*.mp3") if f.is_file()])
    
    if not audio_files:
        print("[x] No audio files found in data/audio/")
        return
    
    print(f"\n[*] Found {len(audio_files)} audio file(s)")
    for f in audio_files:
        print(f"    - {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
    
    # Initialize factory
    try:
        factory = ModelFactory()
        print(f"\n[OK] Factory initialized (primary: {factory.primary.provider_name})")
    except Exception as e:
        print(f"[x] Factory error: {e}")
        return
    
    # Initialize ElevenLabs STT client
    try:
        from elevenlabs import ElevenLabs
        el_key = os.environ.get("ELEVENLABS_API_KEY")
        if not el_key:
            print("[x] ELEVENLABS_API_KEY not set")
            return
        el_client = ElevenLabs(api_key=el_key)
        print("[OK] ElevenLabs client initialized")
    except Exception as e:
        print(f"[x] ElevenLabs client error: {e}")
        return
    
    # Initialize agents
    try:
        stt_agent = ElevenLabsSTTAgent(
            client=el_client,
            persist_transcripts=True,
            transcripts_folder=str(AUDIO_DIR.parent / "transcripts"),
            diarize=True,
            num_speakers=2,
            enable_stt_cache=True,
            stt_cache_dir=str(STT_CACHE_DIR),
            preprocess_audio=True,
        )
        print("[OK] STT Agent initialized")
    except Exception as e:
        print(f"[x] STT Agent error: {e}")
        return
    
    try:
        qa_agent = QualityManagementAgent(
            model_factory=factory,
            cache_dir=str(CACHE_DIR),
            enable_cache=True,
        )
        print("[OK] QA Agent initialized")
    except Exception as e:
        print(f"[x] QA Agent error: {e}")
        return
    
    # Process audio files
    print(f"\n{'='*60}")
    print("TRANSCRIPTION & EVALUATION")
    print(f"{'='*60}\n")
    
    results = {}
    for idx, audio_file in enumerate(audio_files, 1):
        print(f"\n[{idx}/{len(audio_files)}] Processing: {audio_file.name}")
        print("-" * 60)
        
        try:
            # Transcribe
            print(f"  [*] Transcribing...")
            transcript_result = stt_agent.transcribe(audio_file)
            transcript = transcript_result["text"]
            
            if not transcript or len(transcript.split()) < 50:
                print(f"  [x] Transcript too short ({len(transcript.split())} words)")
                results[audio_file.name] = {"status": "SKIP", "reason": "transcript_too_short"}
                continue
            
            print(f"  [OK] Transcript: {len(transcript.split())} words, {len(transcript)} chars")
            
            # Evaluate
            print(f"  [*] Evaluating...")
            evaluation = qa_agent.evaluate_call(
                transcript,
                filename=audio_file.name,
            )
            
            # Score
            score_result = qa_agent.calculate_score(evaluation)
            overall_score = score_result.get("overall_score", 0)
            
            print(f"  [OK] Evaluation complete")
            print(f"       Score: {overall_score:.1f}/100")
            
            if evaluation.get("agents_detected"):
                print(f"       [!] Multi-agent detected: {', '.join(evaluation['agents_detected'])}")
            
            if evaluation.get("critical_gaps"):
                print(f"       [!] Critical gaps: {len(evaluation['critical_gaps'])} found")
            
            results[audio_file.name] = {
                "status": "OK",
                "score": overall_score,
                "agents": evaluation.get("agents_detected", []),
                "critical_gaps": evaluation.get("critical_gaps", []),
            }
            
        except Exception as e:
            print(f"  [x] Error: {type(e).__name__}: {str(e)[:100]}")
            results[audio_file.name] = {"status": "ERROR", "error": str(e)[:100]}
    
    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}\n")
    
    ok_count = sum(1 for r in results.values() if r["status"] == "OK")
    error_count = sum(1 for r in results.values() if r["status"] == "ERROR")
    skip_count = sum(1 for r in results.values() if r["status"] == "SKIP")
    
    for filename, result in results.items():
        status = result["status"]
        if status == "OK":
            score = result.get("score", 0)
            print(f"[OK] {filename}")
            print(f"     Score: {score:.1f}/100")
            if result["agents"]:
                print(f"     Agents: {', '.join(result['agents'])}")
        elif status == "ERROR":
            print(f"[x]  {filename}")
            print(f"     Error: {result.get('error', 'Unknown')}")
        else:
            print(f"[~]  {filename}")
            print(f"     Reason: {result.get('reason', 'Unknown')}")
    
    print(f"\nSummary: {ok_count} OK, {error_count} ERROR, {skip_count} SKIP")

if __name__ == "__main__":
    main()
