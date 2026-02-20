"""
Structured JSON Logger (HIGH-02)

Emits structured JSON line events to data/metrics.jsonl for ingestion
by monitoring tools (Datadog, CloudWatch, ELK).

Usage:
    from structured_logger import emit_metric
    emit_metric("llm_call", provider="openrouter", cost_usd=0.03)
"""

import json
import time
from pathlib import Path

_METRICS_FILE = Path("data/metrics.jsonl")


def emit_metric(event: str, **data) -> None:
    """Emit a structured metric event as a JSON line.

    Args:
        event: Event name (e.g. "llm_call", "llm_error", "pipeline_complete").
        **data: Arbitrary key-value pairs to include in the metric.

    Never raises — metrics should never crash the pipeline.
    """
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        **data,
    }
    try:
        _METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_METRICS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass  # Never crash the pipeline for metrics
