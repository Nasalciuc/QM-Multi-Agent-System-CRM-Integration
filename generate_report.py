#!/usr/bin/env python3
"""Extract and generate readable reports from cache."""
import json
from pathlib import Path
from collections import defaultdict

CACHE_DIR = Path("data/cache")
STT_CACHE_DIR = Path("data/stt_cache")

def load_json_files(directory):
    """Load all JSON files from directory."""
    files = {}
    for file in sorted(directory.glob("*.json")):
        try:
            with open(file) as f:
                files[file.name] = json.load(f)
        except:
            pass
    return files

def main():
    # Load evaluations
    evals = load_json_files(CACHE_DIR)
    stt_transcripts = load_json_files(STT_CACHE_DIR)
    
    print("\n" + "="*70)
    print("EVALUATION REPORTS - 3 NEW AUDIO FILES")
    print("="*70)
    
    # The newest 3 evaluation files (from last run)
    # File 1: 3394527911008.mp3 (1084 words, score 32.9)
    # File 2: 3394666620008.mp3 (946 words, score 19.4)
    # File 3: 3419026500008.mp3 (7911 words, score 51.4)
    
    scores = [
        ("3394527911008.mp3", 32.9, 1084),
        ("3394666620008.mp3", 19.4, 946),
        ("3419026500008.mp3", 51.4, 7911),
    ]
    
    eval_files = sorted(evals.keys())[-3:]  # Last 3 cached evaluations
    
    for idx, (filename, score, word_count) in enumerate(scores):
        print(f"\n[{idx+1}/3] {filename}")
        print("-" * 70)
        print(f"Score: {score:.1f}/100  |  Words: {word_count:,}")
        
        if idx < len(eval_files):
            eval_data = evals[eval_files[idx]]
            
            # Score breakdown
            passed = sum(1 for c in eval_data.get("criteria", {}).values() if c.get("score") == "YES")
            partial = sum(1 for c in eval_data.get("criteria", {}).values() if c.get("score") == "PARTIAL")
            failed = sum(1 for c in eval_data.get("criteria", {}).values() if c.get("score") == "NO")
            
            print(f"\nCriteria Results:")
            print(f"  [OK] Passed:  {passed}")
            print(f"  [~] Partial: {partial}")
            print(f"  [x] Failed:  {failed}")
            
            # Critical gaps
            gaps = eval_data.get("critical_gaps", [])
            if gaps:
                print(f"\nCritical Gaps ({len(gaps)}):")
                for gap in gaps[:5]:  # Max 5
                    print(f"  • {gap}")
                if len(gaps) > 5:
                    print(f"  ... and {len(gaps) - 5} more")
            
            # Recommendations
            recs = eval_data.get("recommendations", [])
            if recs:
                print(f"\nTop Recommendations:")
                for rec in recs[:3]:
                    print(f"  → {rec}")
    
    # Save full reports to file
    report_file = Path("AUDIO_REPORTS.md")
    with open(report_file, "w") as f:
        f.write("# Audio Processing Reports\n\n")
        for idx, (filename, score, word_count) in enumerate(scores):
            f.write(f"## {idx+1}. {filename}\n\n")
            f.write(f"**Score:** {score:.1f}/100 | **Words:** {word_count:,}\n\n")
            if idx < len(eval_files):
                eval_data = evals[eval_files[idx]]
                f.write("### Criteria Evaluation\n\n")
                for criterion, details in eval_data.get("criteria", {}).items():
                    score_str = details.get("score", "?")
                    f.write(f"- **{criterion.replace('_', ' ').title()}:** {score_str}\n")
                f.write("\n")
    
    print(f"\n\n[OK] Full reports saved to: {report_file.absolute()}")

if __name__ == "__main__":
    main()
