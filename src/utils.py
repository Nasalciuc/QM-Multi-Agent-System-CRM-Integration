"""
Utility Functions

Shared helpers used across agents.
"""

import json
import logging
import yaml
from pathlib import Path
from typing import Any
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
