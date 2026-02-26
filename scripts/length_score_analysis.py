#!/usr/bin/env python3
"""Correlation analysis: transcript length vs score. Usage: python scripts/length_score_analysis.py <eval.json>"""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/length_score_analysis.py <eval.json>")
        sys.exit(1)

    data = json.loads(Path(sys.argv[1]).read_text())
    pairs = [
        (
            e.get("word_count", len(e.get("transcript_redacted", "").split())),
            e.get("overall_score", 0),
        )
        for e in data
        if e.get("overall_score", 0) > 0
    ]
    pairs = [(w, s) for w, s in pairs if w > 0]

    if len(pairs) < 3:
        print(f"Need 3+ data points, got {len(pairs)}")
        return

    n = len(pairs)
    sx = sum(w for w, _ in pairs)
    sy = sum(s for _, s in pairs)
    sxy = sum(w * s for w, s in pairs)
    sx2 = sum(w**2 for w, _ in pairs)
    sy2 = sum(s**2 for _, s in pairs)
    dx = (n * sx2 - sx**2) ** 0.5
    dy = (n * sy2 - sy**2) ** 0.5
    r = (n * sxy - sx * sy) / (dx * dy) if dx and dy else 0

    print(f"\nLength vs Score: n={n}, r={r:.3f}, r²={r**2:.3f}")
    strength = "Weak" if abs(r) < 0.3 else "Moderate" if abs(r) < 0.7 else "Strong"
    print(f"Interpretation: {strength} correlation\n")
    for w, s in sorted(pairs):
        print(f"  {w:>8,} words | {s:>6.1f}/100")


if __name__ == "__main__":
    main()
