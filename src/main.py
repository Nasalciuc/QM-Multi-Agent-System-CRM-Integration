"""
Main CLI Entry Point

Usage:
    python src/main.py --date-from 2025-02-01 --date-to 2025-02-11
    python src/main.py --call-id ABC123          # Process single call
    python src/main.py --days-back 7             # Last 7 days

TODO:
    - Parse CLI arguments (argparse)
    - Load config from config/agents.yaml
    - Load .env credentials
    - Create Pipeline instance
    - Run pipeline
    - Export results to CSV
    - Print summary (calls processed, avg score, total cost)
"""

import argparse
from pathlib import Path


def parse_args():
    """
    TODO: Implement argument parser

    Arguments to support:
        --date-from     Start date (YYYY-MM-DD)
        --date-to       End date (YYYY-MM-DD)
        --days-back     Process last N days (default: 7)
        --call-id       Process single call by ID
        --export-csv    Export results to CSV (default: True)
        --dry-run       Show what would be processed without doing it
    """
    # TODO: Implement
    pass


def main():
    """
    TODO: Implement main flow:
        1. args = parse_args()
        2. config = load config/agents.yaml + .env
        3. pipeline = Pipeline(config)
        4. results = pipeline.run(date_from, date_to)
        5. save results to data/evaluations/
        6. export CSV to data/exports/
        7. print summary
    """
    # TODO: Implement
    pass


if __name__ == "__main__":
    main()
