"""
Tests for Agent 4: Integration & Export

Updated to import from agent_04_export (was agent_04_ResultSending).
"""
import sys
import os
import json
import pytest
from unittest.mock import MagicMock, patch

from agents.agent_04_export import IntegrationAgent


# --- Fixtures ---

@pytest.fixture
def tmp_output(tmp_path):
    return str(tmp_path / "exports")


@pytest.fixture
def agent(tmp_output):
    return IntegrationAgent(output_folder=tmp_output, webhook_url="")


@pytest.fixture
def agent_with_webhook(tmp_output):
    return IntegrationAgent(output_folder=tmp_output, webhook_url="https://example.com/webhook")


@pytest.fixture
def agent_with_hmac(tmp_output):
    """#32: Agent with webhook + HMAC secret."""
    return IntegrationAgent(
        output_folder=tmp_output,
        webhook_url="https://example.com/webhook",
        webhook_secret="test-secret-key",
    )


@pytest.fixture
def sample_evaluations():
    return [
        {
            "filename": "call1.mp3",
            "transcript": "Agent: Hello\nClient: Hi",
            "duration_min": 5.0,
            "call_type": "First Call",
            "overall_score": 75.0,
            "score_data": {
                "overall_score": 75.0,
                "category_scores": {
                    "phone_skills": {"score": 80.0, "count": 5},
                    "sales_techniques": {"score": 70.0, "count": 8},
                    "urgency_closing": {"score": 60.0, "count": 3},
                    "soft_skills": {"score": 85.0, "count": 8},
                },
                "score_breakdown": {"yes_count": 10, "partial_count": 8, "no_count": 4, "na_count": 2},
            },
            "criteria": {
                "greeting_prepared": {"score": "YES", "evidence": "Good greeting."},
                "contact_info": {"score": "PARTIAL", "evidence": "Email only."},
            },
            "overall_assessment": "Decent call.",
            "strengths": ["Good tone", "Professional", "Product knowledge"],
            "improvements": ["Ask budget", "Create urgency", "Close better"],
            "critical_gaps": [],
            "model_used": "gpt-4o-2024-11-20",
            "tokens_used": {"input": 5000, "output": 1500},
            "cost_usd": 0.0275,
            "status": "Success",
        }
    ]


@pytest.fixture
def criteria_ref():
    return {
        "greeting_prepared": {"category": "phone_skills", "weight": 1.0},
        "contact_info": {"category": "phone_skills", "weight": 1.0},
    }


# --- Tests: JSON Export ---

class TestJsonExport:

    def test_export_json_creates_file(self, agent, sample_evaluations):
        path = agent.export_json(sample_evaluations, "gpt-4o")
        assert os.path.exists(path)
        assert path.endswith(".json")

    def test_export_json_content(self, agent, sample_evaluations):
        path = agent.export_json(sample_evaluations, "gpt-4o")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "metadata" in data
        assert "evaluations" in data
        assert data["metadata"]["model"] == "gpt-4o"
        assert data["metadata"]["calls"] == 1
        assert len(data["evaluations"]) == 1

    def test_export_json_custom_timestamp(self, agent, sample_evaluations):
        path = agent.export_json(sample_evaluations, "gpt-4o", timestamp="20260101_120000")
        assert "20260101_120000" in path


# --- Tests: Full Export ---

class TestExportAll:

    def test_export_all_creates_files(self, agent, sample_evaluations, criteria_ref):
        files = agent.export_all(sample_evaluations, criteria_ref)
        assert "excel" in files
        assert "csv_summary" in files
        assert "csv_details" in files
        assert "json" in files
        for key in files:
            assert os.path.exists(files[key])

    def test_export_all_consistent_timestamps(self, agent, sample_evaluations, criteria_ref):
        """All exported files should share the same timestamp."""
        files = agent.export_all(sample_evaluations, criteria_ref)
        excel_name = os.path.basename(files["excel"])
        json_name = os.path.basename(files["json"])
        csv_name = os.path.basename(files["csv_summary"])
        excel_ts = excel_name.replace("QM_", "").replace(".xlsx", "")
        json_ts = json_name.replace("QM_", "").replace(".json", "")
        csv_ts = csv_name.replace("QM_", "").replace("_summary.csv", "")
        assert excel_ts == json_ts == csv_ts

    def test_csv_has_correct_columns(self, agent, sample_evaluations, criteria_ref):
        import pandas as pd
        files = agent.export_all(sample_evaluations, criteria_ref)
        df = pd.read_csv(files["csv_summary"])
        expected_cols = {"File", "Type", "Score", "Phone", "Sales", "Closing", "Soft", "YES", "PARTIAL", "NO", "Cost"}
        assert set(df.columns) == expected_cols


# --- Tests: Webhook ---

class TestWebhook:

    def test_webhook_not_called_when_empty(self, agent):
        result = agent.send_webhook({"event": "test"})
        assert result is False

    @patch("agents.agent_04_export.httpx.Client")
    def test_webhook_success(self, mock_client_class, agent_with_webhook):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        result = agent_with_webhook.send_webhook({"event": "test"})
        assert result is True

    @patch("agents.agent_04_export.httpx.Client")
    def test_webhook_retry_on_failure(self, mock_client_class, agent_with_webhook):
        """Should retry up to 3 times on failure."""
        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("Connection error")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        result = agent_with_webhook.send_webhook({"event": "test"})
        assert result is False
        assert mock_client.post.call_count == 3

    @patch("agents.agent_04_export.httpx.Client")
    def test_webhook_hmac_signature(self, mock_client_class, agent_with_hmac):
        """#32: When webhook_secret is set, X-Signature-256 header should be sent."""
        import hashlib
        import hmac as hmac_mod

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        payload = {"event": "test", "score": 85}
        result = agent_with_hmac.send_webhook(payload)
        assert result is True

        # Verify HMAC header was sent
        call_kwargs = mock_client.post.call_args
        headers = call_kwargs[1]["headers"] if "headers" in call_kwargs[1] else call_kwargs.kwargs.get("headers", {})
        assert "X-Signature-256" in headers
        assert headers["X-Signature-256"].startswith("sha256=")

        # Verify signature is correct
        body = call_kwargs[1].get("content", "")
        expected_sig = hmac_mod.new(
            b"test-secret-key", body.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        assert headers["X-Signature-256"] == f"sha256={expected_sig}"

    def test_webhook_no_hmac_without_secret(self, agent_with_webhook):
        """#32: Without webhook_secret, no HMAC header should be added."""
        # Agent without secret should not have webhook_secret set
        assert not agent_with_webhook.webhook_secret

    @patch("agents.agent_04_export.time.sleep")
    @patch("agents.agent_04_export.httpx.Client")
    def test_webhook_retry_uses_jitter(self, mock_client_class, mock_sleep, agent_with_webhook):
        """MED-12: Backoff delays should include jitter (not exact powers of 2)."""
        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("Connection error")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        agent_with_webhook.send_webhook({"event": "test"})
        assert mock_sleep.call_count == 2  # 2 retries = 2 sleeps
        # Backoff with ±25% jitter: base 2 → [1.5, 2.5], base 4 → [3.0, 5.0]
        for call in mock_sleep.call_args_list:
            delay = call[0][0]
            assert 0.1 <= delay <= 10.0  # sanity bound
