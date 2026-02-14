"""
Utility Functions — Facade Module (#21)

Re-exports from config_loader and logging_setup for backward compatibility.
All existing imports (e.g. ``from utils import setup_logging``) continue to work.

Actual implementations live in:
  - config_loader.py: load_config, load_env, validate_env, load_criteria, validate_models_config
  - logging_setup.py: setup_logging, JsonLogFormatter
"""

import re as _re
import hashlib as _hashlib
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


def safe_log_filename(filename: str) -> str:
    """Sanitize a filename for safe use in log messages.

    Strips PII-bearing path components and replaces anything that
    is not alphanumeric, dash, underscore, or dot with an underscore.
    Appends a short hash so collisions are effectively impossible.

    Example:
        >>> safe_log_filename("John Doe_2nd_call.mp3")
        'John_Doe_2nd_call_a1b2c3.mp3'
    """
    if not filename:
        return "unknown"
    name = Path(filename).name  # strip directory components
    stem = Path(name).stem
    suffix = Path(name).suffix  # e.g. ".mp3"
    # Replace anything outside [A-Za-z0-9_\-] with underscore
    safe_stem = _re.sub(r"[^\w\-]", "_", stem)
    # Collapse multiple underscores
    safe_stem = _re.sub(r"_+", "_", safe_stem).strip("_")
    # Append short hash of the original filename for traceability
    short_hash = _hashlib.sha256(name.encode()).hexdigest()[:6]
    return f"{safe_stem}_{short_hash}{suffix}"


__all__ = [
    "setup_logging",
    "load_config",
    "load_env",
    "validate_env",
    "load_criteria",
    "validate_models_config",
    "JsonLogFormatter",
    "json_serializer",
    "safe_log_filename",
]
