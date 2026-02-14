"""
Test CRMAgent download functionality
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents.agent_01_audio import CRMAgent

token = os.environ.get('CRM_AI_TOKEN')
if not token:
    print("ERROR: CRM_AI_TOKEN not set")
    sys.exit(1)

print("=" * 60)
print("  CRMAgent Download Test")
print("=" * 60)

agent = CRMAgent(
    api_token=token,
    agent_id=248,
    download_folder="data/audio",
)

print("\n[1] Searching for recordings...")
try:
    calls = agent.search_recordings("2026-02-10", "2026-02-13")
    print(f"  Found: {len(calls)} calls")
    
    if calls:
        print(f"\n[2] Downloading first 3 recordings...")
        for i, call in enumerate(calls[:3], 1):
            print(f"\n  [{i}] Call: {call['id']}")
            print(f"      Started: {call['startTime']}")
            print(f"      Duration: {call['duration']}s")
            print(f"      Recording URL: {call['recording_url'][:60]}...")
            
            filepath = agent.download_audio(call)
            if filepath:
                print(f"      ✓ Downloaded to: {filepath}")
            else:
                print(f"      ✗ Download failed")
    
    agent.close()
    print("\n" + "=" * 60)
    print("  Test complete")
    print("=" * 60)
    
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
