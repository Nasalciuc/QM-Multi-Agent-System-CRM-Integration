"""
Utility Functions

Shared helpers used across agents.
"""

import json
import logging
import os
import yaml
from pathlib import Path
from typing import Any, List
from dotenv import load_dotenv


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging with console handler."""
    logger = logging.getLogger("qa_system")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def load_config(config_path: str = "config/agents.yaml") -> dict:
    """Load YAML config and return as dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env() -> None:
    """Load .env using python-dotenv."""
    load_dotenv()


def validate_env(required_keys: List[str]) -> None:
    """Validate that all required environment variables are set.
    Raises SystemExit with clear message listing all missing keys."""
    missing = [k for k in required_keys if not os.environ.get(k)]
    if missing:
        print(f"\nERROR: Missing required environment variables:")
        for key in missing:
            print(f"  - {key}")
        print(f"\nSet them in .env or export them before running.")
        raise SystemExit(1)


def save_json(data: Any, filepath: str) -> None:
    """Write JSON with indent=2, ensure_ascii=False."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """Read and return JSON file contents."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_criteria(config_path: str = "config/qa_criteria.yaml") -> dict:
    """Load QA criteria from YAML and flatten to {key: {description, category, weight, first_call_only}} format."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Criteria file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    criteria = {}
    for category, items in raw.items():
        for key, props in items.items():
            criteria[key] = {
                "description": props["description"],
                "category": category,
                "weight": props.get("weight", 1.0),
                "first_call_only": props.get("first_call_only", False),
            }
    return criteria
