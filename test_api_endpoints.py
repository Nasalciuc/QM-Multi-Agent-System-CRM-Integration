"""
CRM API - Check if there's an alternative download endpoint
"""
import httpx
import os
import urllib3
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

TOKEN = os.environ.get("CRM_AI_TOKEN")
BASE_URL = "https://crm.buybusinessclass.com/ai"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "*/*",  # Accept any format
}

print("Testing various download endpoints...\n")

# Get call ID
call_id = "AL_MJsarY_-MGc1A"
print(f"Call ID: {call_id}\n")

endpoints = [
    f"{BASE_URL}/call/{call_id}/recording",
    f"{BASE_URL}/call/{call_id}/download",
    f"{BASE_URL}/call/{call_id}/audio",
    f"{BASE_URL}/calls/{call_id}/recording",
    f"{BASE_URL}/recording/{call_id}",
    f"{BASE_URL}/recording/{call_id}/download",
]

for endpoint in endpoints:
    try:
        print(f"[TEST] {endpoint}")
        r = httpx.get(endpoint, headers=headers, verify=False, timeout=10)
        print(f"  Status: {r.status_code}")
        print(f"  Content-Type: {r.headers.get('content-type')}")
        print(f"  Content-Length: {len(r.content)} bytes")
        
        if r.status_code == 200 and not r.content.startswith(b'<!DOCTYPE'):
            print(f"  ✓ POSSIBLE AUDIO ENDPOINT!")
            if r.content.startswith(b'ID3') or r.content.startswith(b'\xff\xfb'):
                print(f"    ✓✓ CONFIRMED MP3 AUDIO!")
        print()
    except Exception as e:
        print(f"  Error: {e}\n")
