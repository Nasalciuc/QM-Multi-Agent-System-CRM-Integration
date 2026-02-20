#!/usr/bin/env python3
import re

# Current pattern from code
_NATO_SPELLED_PATTERN = re.compile(
    r"(?:[A-Za-z]\s*(?:,\s*)?(?:as in|for|like)\s+\w+[,.\s]*){4,}",
    re.IGNORECASE,
)

test_inputs = [
    "That's D as in Denver, A Alpha, N Nancy, N Nancy, Y Yankee, G Gary.",
    "D as in Denver, A as in Apple, N as in Nancy, N as in Nancy, Y as in Yankee, G as in Gary.",
    "D, A, N, N, Y, G, R, A, H, A, M",
]

print("NATO Pattern Test")
print("=" * 70)

for test in test_inputs:
    matches = _NATO_SPELLED_PATTERN.findall(test)
    print(f"\nInput: {test}")
    print(f"Matches: {matches}")
    print(f"Count: {len(matches)}")

# Test with corrected pattern that handles "A Alpha" style
_NATO_CORRECTED = re.compile(
    r"(?:"
    r"[A-Za-z]\s+[A-Za-z]{2,}"  # "A Alpha" or "D Denver"
    r"(?:\s+as\s+in)?"            # optional "as in"
    r"[,.\s]*"
    r"){4,}",
    re.IGNORECASE,
)

print("\n\n" + "=" * 70)
print("CORRECTED Pattern (handles 'A Alpha' style)")
print("=" * 70)

for test in test_inputs:
    matches = _NATO_CORRECTED.findall(test)
    print(f"\nInput: {test}")
    print(f"Matches: {matches}")
    print(f"Count: {len(matches)}")
