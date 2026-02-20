"""
Simple CRM API test - just try to connect and see what we get
"""
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.environ.get("CRM_AI_TOKEN")
if not TOKEN:
    print("ERROR: CRM_AI_TOKEN not found")
    exit(1)

# NEW-10: Honour CRM_CA_BUNDLE env var instead of verify=False
_SSL_VERIFY = os.environ.get("CRM_CA_BUNDLE", True)

print(f"Token loaded: {TOKEN[:20]}...")
print()

BASE_URL = "https://crm.buybusinessclass.com/ai"
headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json",
}

print("Testing connection...")
print(f"URL: {BASE_URL}/call-recordings")
print()

try:
    print("Sending request...")
    r = httpx.get(
        f"{BASE_URL}/call-recordings",
        params={"agent_id": 248, "limit": 10},
        headers=headers,
        timeout=15,
        verify=_SSL_VERIFY,
    )
    
    print(f"Status: {r.status_code}")
    print(f"Headers: {dict(r.headers)}")
    print()
    print("Response body:")
    print(r.text[:1000])
    
except httpx.ConnectError as e:
    print(f"Connection error: {e}")
except httpx.TimeoutException as e:
    print(f"Timeout error: {e}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
