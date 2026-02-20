#!/usr/bin/env python3
"""Extract transcripts from 3 audio files and display them."""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load .env
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

from elevenlabs import ElevenLabs
from agents.agent_02_transcription import ElevenLabsSTTAgent

AUDIO_DIR = Path(__file__).parent / "data" / "audio"

def main():
    print("\n" + "="*70)
    print("TRANSCRIPTION EXTRACTION — 3 AUDIO FILES")
    print("="*70)
    
    audio_files = sorted([f for f in AUDIO_DIR.glob("*.mp3") if f.is_file()])
    
    if not audio_files:
        print("[x] No audio files found in data/audio/")
        return
    
    print(f"\n[*] Found {len(audio_files)} audio file(s)")
    for f in audio_files:
        print(f"    - {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
    
    # Initialize ElevenLabs client
    try:
        el_key = os.environ.get("ELEVENLABS_API_KEY")
        if not el_key:
            print("[x] ELEVENLABS_API_KEY not set")
            return
        el_client = ElevenLabs(api_key=el_key)
        print("[OK] ElevenLabs client initialized")
    except Exception as e:
        print(f"[x] ElevenLabs client error: {e}")
        return
    
    # Initialize STT agent
    try:
        stt_agent = ElevenLabsSTTAgent(
            client=el_client,
            persist_transcripts=True,
            transcripts_folder=str(AUDIO_DIR.parent / "transcripts"),
            diarize=True,
            num_speakers=2,
            enable_stt_cache=False,  # Fresh transcripts
            preprocess_audio=True,
        )
        print("[OK] STT Agent initialized\n")
    except Exception as e:
        print(f"[x] STT Agent error: {e}")
        return
    
    # Process each audio file
    for idx, audio_file in enumerate(audio_files, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(audio_files)}] {audio_file.name}")
        print(f"{'='*70}\n")
        
        try:
            print(f"[*] Transcribing...")
            result = stt_agent.transcribe(audio_file)
            transcript = result["text"]
            
            print(f"[OK] Transcription complete")
            print(f"    Words: {len(transcript.split()):,}")
            print(f"    Characters: {len(transcript):,}\n")
            
            print("TRANSCRIPT:\n")
            print(transcript)
            
            # Save to file
            transcript_file = AUDIO_DIR.parent / "transcripts" / f"{audio_file.stem}.txt"
            transcript_file.write_text(transcript)
            print(f"\n[OK] Saved: {transcript_file.relative_to(Path.cwd())}")
            
        except Exception as e:
            print(f"[x] Transcription error: {type(e).__name__}: {str(e)[:200]}")

if __name__ == "__main__":
    main()
