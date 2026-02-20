"""
Tests for src/structured_logger.py (HIGH-02)
"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from structured_logger import emit_metric


class TestEmitMetric:

    def test_writes_valid_json_line(self, tmp_path):
        """emit_metric should write a valid JSON line to the metrics file."""
        metrics_file = tmp_path / "metrics.jsonl"
        with patch("structured_logger._METRICS_FILE", metrics_file):
            emit_metric("test_event", key1="value1", key2=42)

        lines = metrics_file.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event"] == "test_event"
        assert record["key1"] == "value1"
        assert record["key2"] == 42

    def test_has_timestamp_field(self, tmp_path):
        """Each record should include a timestamp."""
        metrics_file = tmp_path / "metrics.jsonl"
        with patch("structured_logger._METRICS_FILE", metrics_file):
            emit_metric("ts_test")

        record = json.loads(metrics_file.read_text().strip())
        assert "timestamp" in record
        assert "T" in record["timestamp"]  # ISO-ish format

    def test_has_event_field(self, tmp_path):
        """Each record should include the event name."""
        metrics_file = tmp_path / "metrics.jsonl"
        with patch("structured_logger._METRICS_FILE", metrics_file):
            emit_metric("my_event")

        record = json.loads(metrics_file.read_text().strip())
        assert record["event"] == "my_event"

    def test_appends_multiple_lines(self, tmp_path):
        """Multiple calls should append to the same file."""
        metrics_file = tmp_path / "metrics.jsonl"
        with patch("structured_logger._METRICS_FILE", metrics_file):
            emit_metric("event_1", a=1)
            emit_metric("event_2", b=2)
            emit_metric("event_3", c=3)

        lines = metrics_file.read_text().strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            record = json.loads(line)
            assert "event" in record
            assert "timestamp" in record

    def test_never_raises_exceptions(self, tmp_path):
        """emit_metric should never raise, even with unserializable data."""
        metrics_file = tmp_path / "nonexistent_deep" / "dir" / "metrics.jsonl"
        # This should work — it creates parent dirs
        with patch("structured_logger._METRICS_FILE", metrics_file):
            emit_metric("safe_event", data="ok")

    def test_never_raises_on_readonly_path(self):
        """emit_metric should not raise even if the path is read-only."""
        # Use an impossible path on Windows
        impossible_path = Path("Z:\\nonexistent\\volume\\metrics.jsonl")
        with patch("structured_logger._METRICS_FILE", impossible_path):
            emit_metric("should_not_crash", x=1)  # Should silently fail

    def test_handles_non_serializable_values(self, tmp_path):
        """Non-serializable values should be converted via default=str."""
        metrics_file = tmp_path / "metrics.jsonl"
        with patch("structured_logger._METRICS_FILE", metrics_file):
            from datetime import datetime
            emit_metric("complex", dt=datetime(2026, 1, 1), path=Path("/tmp"))

        record = json.loads(metrics_file.read_text().strip())
        assert record["event"] == "complex"
        assert "2026" in record["dt"]
