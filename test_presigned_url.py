"""
Test if recording URLs are pre-signed (don't need auth)
"""
import httpx
import os
import urllib3
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

TOKEN = os.environ.get("CRM_AI_TOKEN")
headers_with_auth = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json",
}

# Get a recording URL
r = httpx.get(
    "https://crm.buybusinessclass.com/ai/call-recordings",
    params={"agent_id": 248, "limit": 1},
    headers=headers_with_auth,
    verify=False,
)

data = r.json()
if data.get("items") and data["items"][0].get("calls"):
    url = data["items"][0]["calls"][0]["recording_url"]
    print(f"Recording URL: {url[:100]}...")
    print()
    
    # Test 1: WITH Authorization header
    print("[TEST 1] Try WITH Authorization header...")
    try:
        r1 = httpx.get(url, headers=headers_with_auth, timeout=10, verify=False)
        print(f"  Status: {r1.status_code}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print()
    
    # Test 2: WITHOUT Authorization header, follow redirects
    print("[TEST 2] Try WITHOUT Authorization header (pre-signed URL)...")
    try:
        r2 = httpx.get(url, timeout=10, verify=False, follow_redirects=True)
        print(f"  Status: {r2.status_code}")
        print(f"  Content-Type: {r2.headers.get('content-type')}")
        print(f"  Content-Length: {len(r2.content)} bytes")
        if r2.status_code == 200 and len(r2.content) > 0:
            # Try saving
            with open("test_download.mp3", "wb") as f:
                f.write(r2.content)
            print(f"  ✓ Downloaded and saved to test_download.mp3 ({len(r2.content)} bytes)")
    except Exception as e:
        print(f"  Error: {e}")
