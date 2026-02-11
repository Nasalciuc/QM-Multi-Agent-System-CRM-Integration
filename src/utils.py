"""
Utility Functions

Shared helpers used across agents.

TODO:
- setup_logging() -> Configure basic logging
- load_config(path) -> Load YAML config
- load_env() -> Load .env
- save_json(data, filepath) -> Write JSON
- load_json(filepath) -> Read JSON
"""

import json
import logging
from pathlib import Path
from typing import Any


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    TODO: Configure logging
    - Console handler
    - Format: "%(asctime)s [%(levelname)s] %(message)s"
    """
    # TODO: Implement
    pass


def load_config(config_path: str = "config/agents.yaml") -> dict:
    """
    TODO: Load YAML config and return as dict
    """
    # TODO: Implement
    pass


def load_env() -> None:
    """
    TODO: Load .env using python-dotenv
    """
    # TODO: Implement
    pass


def save_json(data: Any, filepath: str) -> None:
    """
    TODO: json.dump with indent=2, ensure_ascii=False
    """
    # TODO: Implement
    pass


def load_json(filepath: str) -> Any:
    """
    TODO: Read and return JSON file contents
    """
    # TODO: Implement
    pass
