#!/usr/bin/env python3
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Verify keys
keys_needed = ["MISTRAL_API_KEY", "ELEVENLABS_API_KEY"]
keys_optional = ["OPENAI_API_KEY", "CRM_AI_TOKEN"]
for key in keys_needed:
    val = os.getenv(key)
    if val:
        print(f"[OK] {key}: {val[:20]}...")
    else:
        print(f"[x] {key}: NOT SET")

# Check optional keys
print("\n--- Optional ---")
for key in keys_optional:
    val = os.getenv(key)
    if val:
        print(f"[OK] {key}: {val[:20]}...")
    else:
        print(f"[--] {key}: not set (optional)")
