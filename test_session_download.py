"""
Test recording download with session/cookies
"""
import httpx
import os
import urllib3
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

TOKEN = os.environ.get("CRM_AI_TOKEN")
if not TOKEN:
    print("ERROR: CRM_AI_TOKEN not set")
    exit(1)

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json",
}

print("Getting recording URL...")
r = httpx.get(
    "https://crm.buybusinessclass.com/ai/call-recordings",
    params={"agent_id": 248, "limit": 1},
    headers=headers,
    verify=False,
)

data = r.json()
if data.get("items") and data["items"][0].get("calls"):
    url = data["items"][0]["calls"][0]["recording_url"]
    print(f"URL: {url[:100]}...")
    print()
    
    # Try with session that maintains cookies
    print("[TEST] Download with persistent session...")
    with httpx.Client(verify=False) as client:
        # First, authenticate by getting the API
        auth_resp = client.get(
            "https://crm.buybusinessclass.com/ai/call-recordings",
            params={"agent_id": 248, "limit": 1},
            headers=headers,
        )
        print(f"Auth status: {auth_resp.status_code}")
        print(f"Cookies after auth: {dict(client.cookies)}")
        print()
        
        # Now try to get recording with same session
        print("Downloading recording with session...")
        dl_resp = client.get(url, follow_redirects=True)
        print(f"Status: {dl_resp.status_code}")
        print(f"Content-Type: {dl_resp.headers.get('content-type')}")
        print(f"Content-Length: {len(dl_resp.content)} bytes")
        
        # Check if it's audio or HTML
        if dl_resp.content.startswith(b'<!DOCTYPE') or dl_resp.content.startswith(b'<html'):
            print("ERROR: Got HTML instead of audio")
            print(f"First 200 chars: {dl_resp.content[:200]}")
        elif dl_resp.content.startswith(b'ID3') or dl_resp.content.startswith(b'\xff\xfb'):
            print("✓ AUDIO FILE (MP3)")
            # Save it
            with open("test_audio_session.mp3", "wb") as f:
                f.write(dl_resp.content)
            print(f"  Saved to test_audio_session.mp3 ({len(dl_resp.content)} bytes)")
        else:
            print(f"UNKNOWN: Starts with {dl_resp.content[:10]}")
