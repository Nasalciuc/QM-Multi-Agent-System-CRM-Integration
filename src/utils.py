"""
Utility Functions — Facade Module (#21)

Re-exports from config_loader and logging_setup for backward compatibility.
All existing imports (e.g. ``from utils import setup_logging``) continue to work.

Actual implementations live in:
  - config_loader.py: load_config, load_env, validate_env, load_criteria, validate_models_config
  - logging_setup.py: setup_logging, JsonLogFormatter
"""

from datetime import datetime, date
from decimal import Decimal
from pathlib import Path

# MED-NEW-13: Optional numpy import at module level
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

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


def json_serializer(obj):
    """HIGH-6: Shared JSON serializer — single source of truth.

    Replaces dangerous `default=str` with explicit type handling.
    CRIT-4: Only handles known types; raises TypeError for unexpected objects.
    MED-21: Also handles numpy scalars, Decimal, and bytes.
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)
    # MED-21: numpy scalar → Python native
    if _HAS_NUMPY:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    # MED-21: Decimal → float
    if isinstance(obj, Decimal):
        return float(obj)
    # MED-21: bytes → utf-8 string
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    raise TypeError(f"Non-serializable object: {type(obj).__name__}")


__all__ = [
    "setup_logging",
    "load_config",
    "load_env",
    "validate_env",
    "load_criteria",
    "validate_models_config",
    "JsonLogFormatter",
    "json_serializer",
]
