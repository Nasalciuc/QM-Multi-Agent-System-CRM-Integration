"""
Utility Functions — Facade Module (#21)

Re-exports from config_loader and logging_setup for backward compatibility.
All existing imports (e.g. ``from utils import setup_logging``) continue to work.

Actual implementations live in:
  - config_loader.py: load_config, load_env, validate_env, load_criteria, validate_models_config
  - logging_setup.py: setup_logging, JsonLogFormatter
"""

# Re-export everything so existing imports keep working
from config_loader import (  # noqa: F401
    load_config,
    load_env,
    validate_env,
    load_criteria,
    validate_models_config,
)
from logging_setup import (  # noqa: F401
    setup_logging,
    JsonLogFormatter,
)

__all__ = [
    "setup_logging",
    "load_config",
    "load_env",
    "validate_env",
    "load_criteria",
    "validate_models_config",
    "JsonLogFormatter",
]
