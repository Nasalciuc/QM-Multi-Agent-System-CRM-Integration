#!/usr/bin/env python3
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Verify keys
keys_needed = ["OPENROUTER_API_KEY", "ELEVENLABS_API_KEY"]
for key in keys_needed:
    val = os.getenv(key)
    if val:
        print(f"[OK] {key}: {val[:20]}...")
    else:
        print(f"[x] {key}: NOT SET")
