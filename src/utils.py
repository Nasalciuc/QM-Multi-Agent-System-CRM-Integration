"""
Utility Functions

Shared helpers used across agents:
  - setup_logging: YAML-based logging config via dictConfig
  - load_config: Load YAML config files
  - load_env: Load .env file
  - validate_env: Check required environment variables
  - load_criteria: Load QA criteria from YAML
"""

import logging
import logging.config
import os
import yaml
from pathlib import Path
from typing import List
from dotenv import load_dotenv


def setup_logging(config_path: str = "config/logging.yaml", default_level: str = "INFO") -> logging.Logger:
    """Configure logging from YAML config using dictConfig.

    Falls back to basic console logging if config file is missing.
    """
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Ensure log directory exists for file handler
        for handler in config.get("handlers", {}).values():
            if "filename" in handler:
                Path(handler["filename"]).parent.mkdir(parents=True, exist_ok=True)

        logging.config.dictConfig(config)
    else:
        # Fallback: basic console logging
        logging.basicConfig(
            level=getattr(logging, default_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    return logging.getLogger("qa_system")


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
