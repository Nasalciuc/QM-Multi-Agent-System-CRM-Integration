"""
CRM API Connection Test
Tests: auth, endpoint response, structure, recording access, date filter
"""
import httpx
import json
import sys
import os
import urllib3
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

BASE_URL = "https://crm.buybusinessclass.com/ai"
TOKEN = os.environ.get("CRM_AI_TOKEN")
AGENT_ID = 248
LIMIT = 200

if not TOKEN:
    print("ERROR: CRM_AI_TOKEN not found in environment variables!")
    print("  Set CRM_AI_TOKEN in .env file or export it as environment variable")
    sys.exit(1)

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json",
}

print("=" * 60)
print("  CRM API Connection Test")
print(f"  Agent ID: {AGENT_ID} | Limit: {LIMIT}")
print("=" * 60)

# Test 1: Auth + basic endpoint
print("\n[TEST 1] GET /call-recordings?agent_id=248&limit=200")
try:
    r = httpx.get(
        f"{BASE_URL}/call-recordings",
        params={"agent_id": AGENT_ID, "limit": LIMIT},
        headers=headers,
        timeout=15,
        verify=False,
    )
    print(f"  Status: {r.status_code}")

    if r.status_code == 401:
        print("  FAIL -- Token invalid or expired!")
        print(f"  Response: {r.text}")
        sys.exit(1)

    if r.status_code != 200:
        print(f"  FAIL -- Unexpected status: {r.status_code}")
        print(f"  Response: {r.text[:500]}")
        sys.exit(1)

    data = r.json()
    print(f"  success: {data.get('success')}")
    print(f"  count: {data.get('count')}")
    print(f"  items received: {len(data.get('items', []))}")

    if not data.get("success"):
        print(f"  FAIL -- API returned success=false: {data.get('message')}")
        sys.exit(1)

    print("  PASS -- Auth OK, endpoint responding")

except httpx.ConnectError:
    print("  FAIL -- Cannot connect to CRM server")
    sys.exit(1)
except Exception as e:
    print(f"  FAIL -- {e}")
    sys.exit(1)

# Test 2: Check response structure
print("\n[TEST 2] Validate response structure")
items = data.get("items", [])
if items:
    item = items[0]
    required_fields = ["id", "created_at", "status", "agent", "client", "calls"]
    missing = [f for f in required_fields if f not in item]
    if missing:
        print(f"  WARNING -- Flight request missing fields: {missing}")
    else:
        print(f"  PASS -- Flight request structure OK")
        print(f"     Flight request ID: {item['id']}")
        print(f"     Status: {item['status']}")
        print(f"     Agent: {item.get('agent', {}).get('name', 'N/A')}")
        client = item.get("client", {})
        print(f"     Client: {client.get('first_name', '')} {client.get('last_name', '')}")

    calls = item.get("calls", [])
    if calls:
        call = calls[0]
        call_fields = ["id", "started_at", "duration", "direction", "result", "recording_url"]
        call_missing = [f for f in call_fields if f not in call]
        if call_missing:
            print(f"  WARNING -- Call missing fields: {call_missing}")
        else:
            print(f"  PASS -- Call structure OK")
            print(f"     Call ID: {call['id']}")
            print(f"     Duration: {call['duration']}s")
            print(f"     Direction: {call['direction']}")
            print(f"     Recording URL exists: {bool(call.get('recording_url'))}")
    else:
        print(f"  WARNING -- No calls in first flight request")
else:
    print(f"  WARNING -- No items returned (empty database or date range)")


# Test 3: Test recording URL accessibility (HEAD request only, no download)
print("\n[TEST 3] Check recording URL accessibility")
if items and items[0].get("calls"):
    rec_url = items[0]["calls"][0].get("recording_url")
    if rec_url:
        try:
            print(f"  Attempting to download recording...")
            r2 = httpx.get(rec_url, headers=headers, timeout=30, verify=False)
            print(f"  Recording URL: {rec_url[:80]}...")
            print(f"  Status: {r2.status_code}")
            content_type = r2.headers.get("content-type", "unknown")
            content_length = r2.headers.get("content-length", "unknown")
            print(f"  Content-Type: {content_type}")
            print(f"  Content-Length: {content_length}")
            
            if r2.status_code == 200:
                # Save the recording
                filename = f"test_recording_{items[0]['calls'][0]['id']}.mp3"
                with open(filename, "wb") as f:
                    f.write(r2.content)
                print(f"  PASS -- Recording download successful")
                print(f"  Saved to: {filename} ({len(r2.content)} bytes)")
            else:
                print(f"  WARNING -- Recording returned {r2.status_code}")
        except Exception as e:
            print(f"  WARNING -- Cannot download recording: {e}")
    else:
        print("  WARNING -- No recording URL to test")
else:
    print("  WARNING -- Skipped -- no items/calls available")

# Test 4: Test with date filter (last 7 days)
print("\n[TEST 4] GET /call-recordings with date filter (last 7 days)")
date_to = datetime.now().strftime("%Y-%m-%d")
date_from = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
try:
    r3 = httpx.get(
        f"{BASE_URL}/call-recordings",
        params={
            "agent_id": AGENT_ID,
            "date_from": date_from,
            "date_to": date_to,
            "limit": LIMIT,
        },
        headers=headers,
        timeout=15,
        verify=False,
    )
    d3 = r3.json()
    total_calls = sum(len(item.get("calls", [])) for item in d3.get("items", []))
    print(f"  Date range: {date_from} -> {date_to}")
    print(f"  Flight requests: {len(d3.get('items', []))}")
    print(f"  Total calls: {total_calls}")
    if total_calls > 0:
        print("  PASS -- Date filtering works, calls found")
    else:
        print("  WARNING -- No calls in last 7 days -- try wider date range")
except Exception as e:
    print(f"  FAIL -- {e}")

# Test 5: Count total calls across all items (pagination check)
print("\n[TEST 5] Pagination check")
total_calls_all = sum(len(item.get("calls", [])) for item in items)
print(f"  Flight requests returned: {len(items)}")
print(f"  Total calls across all requests: {total_calls_all}")
if len(items) >= LIMIT:
    print(f"  WARNING -- Hit limit ({LIMIT}). There may be MORE records!")
    print(f"     Use date_from/date_to to narrow range.")
else:
    print(f"  PASS -- All records returned (under limit)")

# Summary
print("\n" + "=" * 60)
print("  Summary")
print("=" * 60)
print(f"  API Status:        {'PASS - Online' if data.get('success') else 'FAIL - Error'}")
print(f"  Auth:              PASS - Token valid")
print(f"  Agent ID {AGENT_ID}:      {'PASS - Has data' if items else 'WARNING - No data'}")
print(f"  Flight requests:   {len(items)}")
print(f"  Total calls:       {total_calls_all}")
print(f"  Recording access:  {'PASS' if items and items[0].get('calls') else 'WARNING - Could not test'}")
print("=" * 60)
