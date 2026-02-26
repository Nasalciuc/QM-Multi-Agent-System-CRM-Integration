#!/usr/bin/env python3
"""Clear LLM evaluation cache. Usage: python scripts/clear_cache.py [--dry-run]"""
import argparse
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"


def main():
    parser = argparse.ArgumentParser(description="Clear LLM evaluation cache")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without actually deleting")
    args = parser.parse_args()

    files = sorted(CACHE_DIR.glob("*.json")) if CACHE_DIR.exists() else []
    if not files:
        print("Cache empty.")
        return

    for f in files:
        label = "[DRY]" if args.dry_run else "[DEL]"
        print(f"  {label} {f.name} ({f.stat().st_size / 1024:.1f}KB)")
        if not args.dry_run:
            f.unlink()

    action = "Would delete" if args.dry_run else "Deleted"
    print(f"\n{action} {len(files)} files.")


if __name__ == "__main__":
    main()
