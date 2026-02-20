#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from processing.pii_redactor import PIIRedactor

# Test REAL-04
print("=" * 60)
print("REAL-04: NATO Spelling Test")
print("=" * 60)
redactor = PIIRedactor()
nato_test = "That's D as in Denver, A Alpha, N Nancy, N Nancy, Y Yankee, G Gary."
result = redactor.redact(nato_test)
print(f"Input: {nato_test}")
print(f"Output: {result['text']}")
print(f"Spelled PII count: {result['pii_found'].get('spelled_pii', 0)}")

# Test REAL-05  
print("\n" + "=" * 60)
print("REAL-05: Multi-line Phone Test")
print("=" * 60)
multiline_test = """Agent: it's a seven oh eight.
Client: Okay.
Agent: Three, two, three.
Client: All right.
Agent: Two, eight, one, five."""

result2 = redactor.redact(multiline_test)
print(f"Input:\n{multiline_test}")
print(f"\nOutput:\n{result2['text']}")
print(f"Phone count: {result2['pii_found'].get('phone', 0)}")
