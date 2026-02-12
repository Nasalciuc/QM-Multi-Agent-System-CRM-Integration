"""
Utility Functions

Shared helpers used across agents:
  - setup_logging: YAML-based logging config via dictConfig
  - load_config: Load YAML config files
  - load_env: Load .env file
  - validate_env: Check required environment variables
  - load_criteria: Load QA criteria from YAML
  - JsonLogFormatter: JSON structured logging formatter
"""

import datetime
import json
import logging
import logging.config
import os
import traceback
import yaml
from pathlib import Path
from typing import List
from dotenv import load_dotenv


class JsonLogFormatter(logging.Formatter):
    """JSON structured log formatter for production use.

    Outputs one JSON object per line with standard fields:
      timestamp, level, logger, message, filename, lineno

    Referenced by config/logging.yaml as ``utils.JsonLogFormatter``.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "filename": record.filename,
            "lineno": record.lineno,
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = traceback.format_exception(*record.exc_info)
        if hasattr(record, "extra_data"):
            log_entry["extra"] = record.extra_data
        return json.dumps(log_entry, default=str)


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


def validate_models_config(config_path: str = "config/models.yaml") -> dict:
    """Load and validate models.yaml structure at startup (HIGH-9).

    Checks required keys in primary/fallback provider configs and token_limits.
    Raises ValueError with clear message if config is malformed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"models.yaml must be a YAML mapping, got {type(config).__name__}")

    # Validate primary provider
    if "primary" not in config:
        raise ValueError("models.yaml missing required key: 'primary'")

    _REQUIRED_PROVIDER_KEYS = {"provider", "model", "base_url", "api_key_env"}
    primary = config["primary"]
    missing = _REQUIRED_PROVIDER_KEYS - set(primary.keys())
    if missing:
        raise ValueError(f"models.yaml primary provider missing keys: {missing}")

    # Validate fallbacks (optional array)
    for i, fb in enumerate(config.get("fallbacks", [])):
        fb_missing = _REQUIRED_PROVIDER_KEYS - set(fb.keys())
        if fb_missing:
            raise ValueError(f"models.yaml fallback[{i}] ({fb.get('provider', '?')}) missing keys: {fb_missing}")

    # Validate token_limits
    if "token_limits" not in config:
        raise ValueError("models.yaml missing required key: 'token_limits'")

    tl = config["token_limits"]
    for key in ("max_input_tokens", "max_output_tokens"):
        if key not in tl:
            raise ValueError(f"models.yaml token_limits missing key: '{key}'")

    return config
