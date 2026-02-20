"""
Tests for Agent 1: CRM API Integration

Tests CRMAgent (CRM API call recording retrieval).
All HTTP calls are mocked — no real API requests.
"""
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
from contextlib import contextmanager

import pytest

from agents.agent_01_audio import CRMAgent


# --- Fixtures ---


@pytest.fixture
def sample_crm_response():
    """Standard CRM API response with 1 flight request and 1 call."""
    return {
        "success": True,
        "count": 1,
        "items": [
            {
                "id": 145835,
                "created_at": "2026-02-10T18:47:03.000000Z",
                "status": "closed:travel_date_passed",
                "agent": {"id": 12, "name": "Dennis Martin", "email": None},
                "client": {
                    "first_name": "Stephen",
                    "last_name": "White",
                    "email": None,
                    "phone": None,
                },
                "calls": [
                    {
                        "id": "TUoA4zD7rg_8DUA",
                        "started_at": "2026-02-05T14:13:24.5700000Z",
                        "duration": 348,
                        "direction": "outbound",
                        "result": "call_connected",
                        "recording_url": "https://crm.buybusinessclass.com/storage/call_log/test.mp3",
                    }
                ],
            }
        ],
    }


@pytest.fixture
def multi_flight_response():
    """CRM response with 2 flight requests and 3 total calls."""
    return {
        "success": True,
        "count": 2,
        "items": [
            {
                "id": 1001,
                "created_at": "2026-02-10T10:00:00Z",
                "status": "open",
                "agent": {"id": 10, "name": "Alice Smith", "email": None},
                "client": {"first_name": "Bob", "last_name": "Jones", "email": None, "phone": None},
                "calls": [
                    {
                        "id": "call_A",
                        "started_at": "2026-02-05T10:00:00Z",
                        "duration": 120,
                        "direction": "inbound",
                        "result": "call_connected",
                        "recording_url": "https://crm.buybusinessclass.com/storage/a.mp3",
                    },
                    {
                        "id": "call_B",
                        "started_at": "2026-02-06T11:00:00Z",
                        "duration": 240,
                        "direction": "outbound",
                        "result": "call_connected",
                        "recording_url": "https://crm.buybusinessclass.com/storage/b.mp3",
                    },
                ],
            },
            {
                "id": 1002,
                "created_at": "2026-02-11T12:00:00Z",
                "status": "closed",
                "agent": {"id": 11, "name": "Charlie Brown", "email": None},
                "client": {"first_name": "Diana", "last_name": "Ross", "email": None, "phone": None},
                "calls": [
                    {
                        "id": "call_C",
                        "started_at": "2026-02-07T09:00:00Z",
                        "duration": 60,
                        "direction": "inbound",
                        "result": "call_connected",
                        "recording_url": "https://crm.buybusinessclass.com/storage/c.mp3",
                    },
                ],
            },
        ],
    }


@pytest.fixture
def crm_agent(tmp_path):
    """CRMAgent with mocked httpx client."""
    with patch("agents.agent_01_audio.httpx.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        agent = CRMAgent(
            api_token="test-token-123",
            download_folder=str(tmp_path / "audio"),
        )
        # Replace the internal client with the mock
        agent._client = mock_client
        yield agent


@pytest.fixture
def crm_agent_with_id(tmp_path):
    """CRMAgent with agent_id filter."""
    with patch("agents.agent_01_audio.httpx.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        agent = CRMAgent(
            api_token="test-token-123",
            download_folder=str(tmp_path / "audio"),
            agent_id=120,
        )
        agent._client = mock_client
        yield agent


# --- Tests: Constructor Validation ---


class TestCRMAgentInit:

    def test_init_empty_token_raises(self, tmp_path):
        """Empty or whitespace-only token should raise ValueError."""
        with patch("agents.agent_01_audio.httpx.Client"):
            with pytest.raises(ValueError, match="CRM API token is required"):
                CRMAgent(api_token="", download_folder=str(tmp_path / "audio"))
            with pytest.raises(ValueError, match="CRM API token is required"):
                CRMAgent(api_token="   ", download_folder=str(tmp_path / "audio"))

    def test_init_valid_token_succeeds(self, tmp_path):
        """A non-empty token should initialize without error."""
        with patch("agents.agent_01_audio.httpx.Client"):
            agent = CRMAgent(api_token="valid-token", download_folder=str(tmp_path / "audio"))
            assert agent.api_token == "valid-token"


# --- Tests: Search Recordings ---


class TestCRMAgentSearch:

    def test_search_recordings_success(self, crm_agent, multi_flight_response):
        """Should flatten nested response into 3 call records."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = multi_flight_response
        mock_response.content = json.dumps(multi_flight_response).encode()
        crm_agent._client.request.return_value = mock_response

        records = crm_agent.search_recordings("2026-02-01", "2026-02-10")

        assert len(records) == 3
        # Check required keys on each record
        required_keys = {
            "id", "flight_request_id", "startTime", "duration",
            "direction", "result", "recording_url", "agent_name",
            "client_name", "agent_id", "flight_request_status",
        }
        for rec in records:
            assert required_keys.issubset(rec.keys()), f"Missing keys: {required_keys - rec.keys()}"

        # Verify call IDs
        ids = [r["id"] for r in records]
        assert ids == ["call_A", "call_B", "call_C"]

        # Verify request was made with correct params
        call_args = crm_agent._client.request.call_args
        assert call_args[0][0] == "GET"
        params = call_args[1]["params"]
        assert params["date_from"] == "2026-02-01"
        assert params["date_to"] == "2026-02-10"
        assert params["limit"] == 200

    def test_search_recordings_with_agent_id(self, crm_agent_with_id, sample_crm_response):
        """Should include agent_id in request params."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_crm_response
        mock_response.content = json.dumps(sample_crm_response).encode()
        crm_agent_with_id._client.request.return_value = mock_response

        crm_agent_with_id.search_recordings("2026-02-01")

        call_args = crm_agent_with_id._client.request.call_args
        params = call_args[1]["params"]
        assert params["agent_id"] == 120

    def test_search_recordings_auth_failure(self, crm_agent):
        """Should raise with clear message on 401."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.content = b'{"success": false, "message": "Unauthorized"}'
        mock_response.json.return_value = {"success": False, "message": "Unauthorized"}
        crm_agent._client.request.return_value = mock_response

        with pytest.raises(RuntimeError, match="authentication failed.*401"):
            crm_agent.search_recordings("2026-02-01")

    def test_search_recordings_deduplicates_by_call_id(self, crm_agent):
        """Same call ID in two flight requests should appear only once."""
        response_data = {
            "success": True,
            "count": 2,
            "items": [
                {
                    "id": 1001,
                    "created_at": "2026-02-10T10:00:00Z",
                    "status": "open",
                    "agent": {"id": 10, "name": "Agent A", "email": None},
                    "client": {"first_name": "C", "last_name": "D", "email": None, "phone": None},
                    "calls": [
                        {
                            "id": "DUPLICATE_ID",
                            "started_at": "2026-02-05T10:00:00Z",
                            "duration": 120,
                            "direction": "inbound",
                            "result": "call_connected",
                            "recording_url": "https://crm.buybusinessclass.com/storage/dup1.mp3",
                        },
                    ],
                },
                {
                    "id": 1002,
                    "created_at": "2026-02-11T12:00:00Z",
                    "status": "closed",
                    "agent": {"id": 11, "name": "Agent B", "email": None},
                    "client": {"first_name": "E", "last_name": "F", "email": None, "phone": None},
                    "calls": [
                        {
                            "id": "DUPLICATE_ID",
                            "started_at": "2026-02-06T11:00:00Z",
                            "duration": 60,
                            "direction": "outbound",
                            "result": "call_connected",
                            "recording_url": "https://crm.buybusinessclass.com/storage/dup2.mp3",
                        },
                    ],
                },
            ],
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        mock_response.content = json.dumps(response_data).encode()
        crm_agent._client.request.return_value = mock_response

        records = crm_agent.search_recordings("2026-02-01")
        assert len(records) == 1
        assert records[0]["id"] == "DUPLICATE_ID"

    def test_search_recordings_empty_response(self, crm_agent):
        """Empty result set should return empty list."""
        response_data = {"success": True, "count": 0, "items": []}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        mock_response.content = json.dumps(response_data).encode()
        crm_agent._client.request.return_value = mock_response

        records = crm_agent.search_recordings("2026-02-01")
        assert records == []

    def test_search_recordings_count_mismatch_logs_warning(self, crm_agent, caplog):
        """Should log warning when count doesn't match items length."""
        response_data = {
            "success": True,
            "count": 5,
            "items": [
                {
                    "id": 1001,
                    "created_at": "2026-02-10T10:00:00Z",
                    "status": "open",
                    "agent": {"id": 10, "name": "A", "email": None},
                    "client": {"first_name": "B", "last_name": "C", "email": None, "phone": None},
                    "calls": [
                        {
                            "id": "c1",
                            "started_at": "2026-02-05T10:00:00Z",
                            "duration": 60,
                            "direction": "inbound",
                            "result": "call_connected",
                            "recording_url": "https://crm.buybusinessclass.com/storage/c1.mp3",
                        },
                    ],
                },
            ],
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        mock_response.content = json.dumps(response_data).encode()
        crm_agent._client.request.return_value = mock_response

        import logging
        with caplog.at_level(logging.WARNING, logger="qa_system.agents"):
            crm_agent.search_recordings("2026-02-01")

        assert any("count mismatch" in msg.lower() for msg in caplog.messages)

    def test_search_recordings_limit_warning(self, crm_agent, caplog):
        """Should warn when API returns count >= API_MAX_LIMIT (truncation risk)."""
        response_data = {
            "success": True,
            "count": 200,
            "items": [
                {
                    "id": i,
                    "created_at": "2026-02-10T10:00:00Z",
                    "status": "open",
                    "agent": {"id": 10, "name": "A", "email": None},
                    "client": {"first_name": "B", "last_name": "C", "email": None, "phone": None},
                    "calls": [
                        {
                            "id": f"c_{i}",
                            "started_at": "2026-02-05T10:00:00Z",
                            "duration": 60,
                            "direction": "inbound",
                            "result": "call_connected",
                            "recording_url": f"https://crm.buybusinessclass.com/storage/c{i}.mp3",
                        },
                    ],
                }
                for i in range(200)
            ],
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        mock_response.content = json.dumps(response_data).encode()
        crm_agent._client.request.return_value = mock_response

        import logging
        with caplog.at_level(logging.WARNING, logger="qa_system.agents"):
            crm_agent.search_recordings("2026-02-01")

        assert any("truncated" in msg.lower() for msg in caplog.messages)

    def test_search_recordings_metadata_fields(self, crm_agent, sample_crm_response):
        """Should correctly extract agent_name, client_name, flight_request fields."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_crm_response
        mock_response.content = json.dumps(sample_crm_response).encode()
        crm_agent._client.request.return_value = mock_response

        records = crm_agent.search_recordings("2026-02-01")
        rec = records[0]

        assert rec["agent_name"] == "Dennis Martin"
        assert rec["client_name"] == "Stephen White"
        assert rec["flight_request_id"] == 145835
        assert rec["flight_request_status"] == "closed:travel_date_passed"
        assert rec["agent_id"] == 12


# --- Tests: Download Audio ---


class TestCRMAgentDownload:

    def _mock_stream_context(self, status_code=200, content=b"fake audio data",
                             content_type="audio/mpeg"):
        """Helper: create a mock httpx.stream() context manager."""
        mock_stream = MagicMock()
        mock_stream.status_code = status_code
        mock_stream.headers = {"content-type": content_type}
        mock_stream.iter_bytes = MagicMock(return_value=[content] if content else [])
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        return mock_stream

    @patch("agents.agent_01_audio.httpx.stream")
    def test_download_audio_success(self, mock_httpx_stream, crm_agent):
        """Should download file via streaming and return filepath."""
        call_record = {
            "id": "abc123",
            "startTime": "2026-02-05T14:13:24Z",
            "recording_url": "https://crm.buybusinessclass.com/storage/call_log/test.mp3",
        }
        mock_httpx_stream.return_value = self._mock_stream_context(
            content=b"fake audio data"
        )

        result = crm_agent.download_audio(call_record)

        assert result is not None
        assert result.endswith(".mp3")
        assert Path(result).exists()
        assert Path(result).read_bytes() == b"fake audio data"

    def test_download_audio_file_exists_skips(self, crm_agent):
        """Should skip download if file already exists."""
        call_record = {
            "id": "existing_call",
            "startTime": "2026-02-05T14:13:24Z",
            "recording_url": "https://crm.buybusinessclass.com/storage/call_log/test.mp3",
        }
        # Create the expected file
        expected_file = crm_agent.download_folder / "existing_call_20260205.mp3"
        expected_file.parent.mkdir(parents=True, exist_ok=True)
        expected_file.write_bytes(b"already here")

        result = crm_agent.download_audio(call_record)

        assert result == str(expected_file)
        # No HTTP request should have been made
        crm_agent._client.get.assert_not_called()

    @patch("agents.agent_01_audio.httpx.stream")
    def test_download_audio_size_limit_exceeded(self, mock_httpx_stream, crm_agent, caplog):
        """Should reject files exceeding MAX_DOWNLOAD_BYTES during streaming."""
        call_record = {
            "id": "big_file",
            "startTime": "2026-02-05T14:13:24Z",
            "recording_url": "https://crm.buybusinessclass.com/storage/call_log/big.mp3",
        }
        # Create oversized content as chunks
        chunk = b"x" * (64 * 1024)
        num_chunks = (CRMAgent.MAX_DOWNLOAD_BYTES // len(chunk)) + 2
        mock_stream = MagicMock()
        mock_stream.status_code = 200
        mock_stream.headers = {"content-type": "audio/mpeg"}
        mock_stream.iter_bytes = MagicMock(return_value=[chunk] * num_chunks)
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_httpx_stream.return_value = mock_stream

        import logging
        with caplog.at_level(logging.ERROR, logger="qa_system.agents"):
            result = crm_agent.download_audio(call_record)

        assert result is None
        assert any("size limit" in msg.lower() for msg in caplog.messages)

    def test_download_audio_invalid_url_domain(self, crm_agent, caplog):
        """Should reject URLs pointing to non-CRM domains (SSRF protection)."""
        call_record = {
            "id": "evil_call",
            "startTime": "2026-02-05T14:13:24Z",
            "recording_url": "https://evil.com/steal-data.mp3",
        }
        import logging
        with caplog.at_level(logging.ERROR, logger="qa_system.agents"):
            result = crm_agent.download_audio(call_record)

        assert result is None
        assert any("domain rejected" in msg.lower() for msg in caplog.messages)
        crm_agent._client.get.assert_not_called()

    def test_download_audio_no_url(self, crm_agent):
        """Should return None when recording_url is empty or missing."""
        call_record = {
            "id": "no_url_call",
            "startTime": "2026-02-05T14:13:24Z",
            "recording_url": "",
        }
        result = crm_agent.download_audio(call_record)
        assert result is None


# --- Tests: Search and Download Integration ---


class TestCRMAgentSearchAndDownload:

    @patch("agents.agent_01_audio.httpx.stream")
    def test_search_and_download_integration(self, mock_httpx_stream, crm_agent, sample_crm_response):
        """Should search, download, and set local_audio_path on each record."""
        # Mock search
        mock_search_response = MagicMock()
        mock_search_response.status_code = 200
        mock_search_response.json.return_value = sample_crm_response
        mock_search_response.content = json.dumps(sample_crm_response).encode()
        crm_agent._client.request.return_value = mock_search_response

        # Mock streaming download
        mock_stream = MagicMock()
        mock_stream.status_code = 200
        mock_stream.headers = {"content-type": "audio/mpeg"}
        mock_stream.iter_bytes = MagicMock(return_value=[b"audio bytes"])
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_httpx_stream.return_value = mock_stream

        results = crm_agent.search_and_download("2026-02-01", "2026-02-10")

        assert len(results) == 1
        assert results[0]["local_audio_path"] is not None
        assert Path(results[0]["local_audio_path"]).exists()


# --- Tests: Close ---


class TestCRMAgentClose:

    def test_close_closes_client(self, crm_agent):
        """close() should close the httpx client."""
        crm_agent.close()
        crm_agent._client.close.assert_called_once()


# --- Tests: Context Manager (CRIT-3) ---


class TestCRMAgentContextManager:

    def test_context_manager_calls_close(self, tmp_path):
        """Using CRMAgent as context manager should call close() on exit."""
        with patch("agents.agent_01_audio.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            with CRMAgent(
                api_token="test-token", download_folder=str(tmp_path / "audio")
            ) as agent:
                agent._client = mock_client
            mock_client.close.assert_called_once()

    def test_context_manager_closes_on_error(self, tmp_path):
        """Context manager should close client even when body raises an error."""
        with patch("agents.agent_01_audio.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            with pytest.raises(ValueError):
                with CRMAgent(
                    api_token="test-token", download_folder=str(tmp_path / "audio")
                ) as agent:
                    agent._client = mock_client
                    raise ValueError("boom")
            mock_client.close.assert_called_once()


# --- Tests: Streaming Download Validation (CRIT-1) ---


class TestCRMAgentStreamingDownload:

    def _mock_stream_context(self, status_code=200, content=b"fake audio",
                             content_type="audio/mpeg"):
        mock_stream = MagicMock()
        mock_stream.status_code = status_code
        mock_stream.headers = {"content-type": content_type}
        mock_stream.iter_bytes = MagicMock(return_value=[content] if content else [])
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        return mock_stream

    @patch("agents.agent_01_audio.httpx.stream")
    def test_download_rejects_html_content_type(self, mock_httpx_stream, crm_agent, caplog):
        """Should reject recording that returns text/html (login page)."""
        call_record = {
            "id": "html_call",
            "startTime": "2026-02-05T14:13:24Z",
            "recording_url": "https://crm.buybusinessclass.com/storage/call_log/test.mp3",
        }
        mock_httpx_stream.return_value = self._mock_stream_context(
            content_type="text/html; charset=utf-8"
        )

        import logging
        with caplog.at_level(logging.ERROR, logger="qa_system.agents"):
            result = crm_agent.download_audio(call_record)

        assert result is None
        assert any("html" in msg.lower() for msg in caplog.messages)

    @patch("agents.agent_01_audio.httpx.stream")
    def test_download_empty_stream(self, mock_httpx_stream, crm_agent, caplog):
        """Should reject download when stream returns no bytes."""
        call_record = {
            "id": "empty_call",
            "startTime": "2026-02-05T14:13:24Z",
            "recording_url": "https://crm.buybusinessclass.com/storage/call_log/test.mp3",
        }
        mock_httpx_stream.return_value = self._mock_stream_context(content=b"")

        import logging
        with caplog.at_level(logging.ERROR, logger="qa_system.agents"):
            result = crm_agent.download_audio(call_record)

        assert result is None

    @patch("agents.agent_01_audio.httpx.stream")
    def test_download_401_returns_none(self, mock_httpx_stream, crm_agent):
        """Should return None on 401 auth failure."""
        call_record = {
            "id": "auth_fail",
            "startTime": "2026-02-05T14:13:24Z",
            "recording_url": "https://crm.buybusinessclass.com/storage/call_log/test.mp3",
        }
        mock_httpx_stream.return_value = self._mock_stream_context(status_code=401)

        result = crm_agent.download_audio(call_record)
        assert result is None


# --- Tests: Paginated Search (CRIT-2) ---


class TestCRMAgentPagination:

    def _make_response(self, count, num_items):
        """Create a mock CRM API response with the given count and number of items."""
        items = [
            {
                "id": i,
                "created_at": "2026-02-10T10:00:00Z",
                "status": "open",
                "agent": {"id": 10, "name": "A", "email": None},
                "client": {"first_name": "B", "last_name": "C", "email": None, "phone": None},
                "calls": [
                    {
                        "id": f"c_{i}",
                        "started_at": "2026-02-05T10:00:00Z",
                        "duration": 60,
                        "direction": "inbound",
                        "result": "call_connected",
                        "recording_url": f"https://crm.buybusinessclass.com/storage/c{i}.mp3",
                    },
                ],
            }
            for i in range(num_items)
        ]
        return {"success": True, "count": count, "items": items}

    def test_paginated_no_split_needed(self, crm_agent):
        """When results < API_MAX_LIMIT, no splitting occurs."""
        data = self._make_response(5, 5)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = data
        mock_resp.content = json.dumps(data).encode()
        crm_agent._client.request.return_value = mock_resp

        records = crm_agent.search_recordings_paginated("2026-02-01", "2026-02-10")

        assert len(records) == 5
        assert not crm_agent._last_query_truncated

    def test_paginated_splits_date_range(self, crm_agent):
        """When results >= API_MAX_LIMIT, should split and recurse."""
        full_data = self._make_response(200, 200)  # Triggers split
        half_data = self._make_response(50, 50)  # Under limit

        call_count = 0

        def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            # First call returns full (triggers split), subsequent calls return half
            if call_count == 1:
                mock_resp.json.return_value = full_data
                mock_resp.content = json.dumps(full_data).encode()
            else:
                mock_resp.json.return_value = half_data
                mock_resp.content = json.dumps(half_data).encode()
            return mock_resp

        crm_agent._client.request.side_effect = mock_request

        records = crm_agent.search_recordings_paginated("2026-02-01", "2026-02-10")

        # Should have made at least 3 calls (1 initial + 2 halves)
        assert call_count >= 3
        # Should have results from both halves  
        assert len(records) > 0

    def test_paginated_max_depth_sets_truncated_flag(self, crm_agent):
        """Should set _last_query_truncated when max depth is reached."""
        data = self._make_response(200, 200)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = data
        mock_resp.content = json.dumps(data).encode()
        crm_agent._client.request.return_value = mock_resp

        # Use _MAX_DEPTH=0 to immediately trigger truncation
        records = crm_agent.search_recordings_paginated(
            "2026-02-01", "2026-02-10", _depth=0, _MAX_DEPTH=0
        )

        assert crm_agent._last_query_truncated is True
        assert len(records) == 200

    def test_paginated_deduplicates_across_halves(self, crm_agent):
        """Records appearing in both halves should be deduplicated."""
        # Both halves return a record with the same call ID
        shared_item = {
            "id": 999,
            "created_at": "2026-02-05T10:00:00Z",
            "status": "open",
            "agent": {"id": 10, "name": "A", "email": None},
            "client": {"first_name": "B", "last_name": "C", "email": None, "phone": None},
            "calls": [
                {
                    "id": "shared_call",
                    "started_at": "2026-02-05T10:00:00Z",
                    "duration": 60,
                    "direction": "inbound",
                    "result": "call_connected",
                    "recording_url": "https://crm.buybusinessclass.com/storage/shared.mp3",
                },
            ],
        }

        full_data = self._make_response(200, 200)
        small_data_with_shared = {"success": True, "count": 1, "items": [shared_item]}

        call_count = 0

        def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            if call_count == 1:
                mock_resp.json.return_value = full_data
                mock_resp.content = json.dumps(full_data).encode()
            else:
                mock_resp.json.return_value = small_data_with_shared
                mock_resp.content = json.dumps(small_data_with_shared).encode()
            return mock_resp

        crm_agent._client.request.side_effect = mock_request

        records = crm_agent.search_recordings_paginated("2026-02-01", "2026-02-10")

        # "shared_call" should appear only once
        shared_count = sum(1 for r in records if r["id"] == "shared_call")
        assert shared_count == 1


# --- Tests: SSL Configuration (Fix #3) ---


class TestCRMAgentSSL:

    def test_ssl_disabled_by_default(self, tmp_path):
        """Without CRM_CA_BUNDLE, SSL verification defaults to True (HIGH-01)."""
        with patch("agents.agent_01_audio.httpx.Client") as mock_client_class, \
             patch.dict(os.environ, {}, clear=False):
            # Ensure CRM_CA_BUNDLE is not set
            os.environ.pop("CRM_CA_BUNDLE", None)
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            agent = CRMAgent(
                api_token="test-token",
                download_folder=str(tmp_path / "audio"),
            )
            assert agent._ssl_verify is True

    def test_ssl_uses_ca_bundle(self, tmp_path):
        """With CRM_CA_BUNDLE set, should use that path for SSL verification."""
        with patch("agents.agent_01_audio.httpx.Client") as mock_client_class, \
             patch.dict(os.environ, {"CRM_CA_BUNDLE": "/path/to/ca.crt"}):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            agent = CRMAgent(
                api_token="test-token",
                download_folder=str(tmp_path / "audio"),
            )
            assert agent._ssl_verify == "/path/to/ca.crt"

    def test_ssl_empty_ca_bundle_disables(self, tmp_path):
        """Empty CRM_CA_BUNDLE should default to True (HIGH-01)."""
        with patch("agents.agent_01_audio.httpx.Client") as mock_client_class, \
             patch.dict(os.environ, {"CRM_CA_BUNDLE": "  "}):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            agent = CRMAgent(
                api_token="test-token",
                download_folder=str(tmp_path / "audio"),
            )
            assert agent._ssl_verify is True
