"""
Logging Setup (#21)

Extracted from utils.py — logging configuration and JSON formatter.
"""

import datetime
import json
import logging
import logging.config
import traceback
import yaml
from pathlib import Path


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
    #24: When QA_LOG_FORMAT=json env var is set, adds json_file handler
    to all loggers automatically.
    """
    import os

    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Ensure log directory exists for file handler
        for handler in config.get("handlers", {}).values():
            if "filename" in handler:
                Path(handler["filename"]).parent.mkdir(parents=True, exist_ok=True)

        # #24: Auto-wire JSON handler when QA_LOG_FORMAT=json
        if os.environ.get("QA_LOG_FORMAT", "").strip().lower() == "json":
            if "json_file" in config.get("handlers", {}):
                for logger_cfg in config.get("loggers", {}).values():
                    handlers = logger_cfg.get("handlers", [])
                    if "json_file" not in handlers:
                        handlers.append("json_file")
                root_handlers = config.get("root", {}).get("handlers", [])
                if "json_file" not in root_handlers:
                    root_handlers.append("json_file")

        logging.config.dictConfig(config)
    else:
        # Fallback: basic console logging
        logging.basicConfig(
            level=getattr(logging, default_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    return logging.getLogger("qa_system")
