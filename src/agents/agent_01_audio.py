"""
Agent 1: Audio Retrieval

Purpose: Download call recordings from CRM API or find local audio files.
Two classes:
  - AudioFileFinder: Find and analyze local audio files
  - CRMAgent: Search and download recordings via Buy Business Class CRM API
"""

from pathlib import Path
from typing import List, Dict, Optional, Union
from urllib.parse import urlparse
from datetime import datetime, timedelta
import os
import tempfile
import time
import logging

import httpx

# Note: urllib3 warning suppression removed (HIGH-01).
# Use CRM_CA_BUNDLE env var for self-signed certificates.

logger = logging.getLogger("qa_system.agents")


class AudioFileFinder:
    """Agent: Finds and analyzes audio files in local folder"""

    def __init__(self, folder_path: str, extensions: tuple = (".mp3", ".wav", ".m4a")):
        self.folder_path = Path(folder_path)
        self.extensions = extensions

    def find_all(self) -> List[Path]:
        """Find all audio files in folder, sorted by name."""
        if not self.folder_path.exists():
            logger.warning(f"Folder not found: {self.folder_path}")
            return []

        audio_files: List[Path] = []
        for ext in self.extensions:
            audio_files.extend(self.folder_path.glob(f"*{ext}"))

        audio_files.sort(key=lambda f: f.name)
        logger.info(f"Found {len(audio_files)} audio files in {self.folder_path}")
        return audio_files

    def get_info(self, file_path: Path) -> Dict:
        """Get file info: name, size in MB."""
        stat = file_path.stat()
        size_mb = stat.st_size / (1024 * 1024)
        return {
            "name": file_path.name,
            "size_mb": round(size_mb, 2),
            "path": str(file_path)
        }

    def get_duration(self, file_path: Path) -> Optional[float]:
        """Get audio duration in minutes using pydub."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(file_path))
            duration_minutes = len(audio) / 1000 / 60
            return round(duration_minutes, 2)
        except ImportError:
            logger.warning(f"pydub not installed — cannot read duration for {file_path.name}")
            return None
        except Exception as e:
            # CRIT-6: Log actual error instead of swallowing silently
            logger.warning(f"Cannot read duration for {file_path.name}: {type(e).__name__}: {e}")
            return None


class CRMAgent:
    """Agent: Download call recordings from Buy Business Class CRM API.

    Uses the company's internal CRM API at crm.buybusinessclass.com/ai.

    Usage:
        agent = CRMAgent(api_token=os.environ['CRM_AI_TOKEN'])
        calls = agent.search_and_download("2026-02-01", "2026-02-10")
    """

    MAX_FILE_SIZE_MB = 500
    MAX_DOWNLOAD_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024



    BASE_URL = "https://crm.buybusinessclass.com/ai"
    ALLOWED_DOWNLOAD_DOMAIN = "crm.buybusinessclass.com"
    API_MAX_LIMIT = 200

    def __init__(
        self,
        api_token: str,
        base_url: str = BASE_URL,
        download_folder: str = "data/audio",
        delay_seconds: float = 1.5,
        agent_id: Optional[int] = None,
    ):
        if not api_token or not api_token.strip():
            raise ValueError(
                "CRM API token is required and cannot be empty. "
                "Set the CRM_AI_TOKEN environment variable."
            )
        self.api_token = api_token
        self.base_url = base_url.rstrip("/")
        self.download_folder = Path(download_folder)
        self.delay_seconds = delay_seconds
        self.agent_id = agent_id
        self.download_folder.mkdir(parents=True, exist_ok=True)

        # SSL verification: use CRM_CA_BUNDLE env var if set, else system trust store.
        # HIGH-01: Default to True (enabled) instead of False.
        ca_bundle = os.environ.get("CRM_CA_BUNDLE", "").strip()
        self._ssl_verify: Union[str, bool]
        if ca_bundle:
            self._ssl_verify = ca_bundle
            logger.info("CRMAgent SSL: using custom CA bundle from CRM_CA_BUNDLE")
        else:
            self._ssl_verify = True
            logger.info(
                "CRMAgent SSL: using system trust store. "
                "Set CRM_CA_BUNDLE env var for custom CA bundle."
            )

        self._client = httpx.Client(
            timeout=30,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Accept": "application/json",
            },
            verify=self._ssl_verify,
        )
        self._last_query_truncated = False  # CRIT-2: Flag set when pagination detects truncation
        logger.info(
            f"CRMAgent initialized | Base URL: {self.base_url} | "
            f"Download folder: {self.download_folder}"
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_recordings(
        self, date_from: str, date_to: Optional[str] = None
    ) -> List[Dict]:
        """Search CRM API for call recordings.

        Fetches flight requests with nested calls and flattens them into
        a list of call-record dicts compatible with the rest of the pipeline.

        Returns:
            Flat list of call dicts, deduplicated by call ID.
        """
        params: Dict = {
            "date_from": date_from,
            "limit": 200,
        }
        if date_to:
            params["date_to"] = date_to
        if self.agent_id is not None:
            params["agent_id"] = self.agent_id

        logger.info(f"Searching CRM recordings from {date_from} to {date_to or 'now'}...")

        response = self._request_with_retry(
            "GET", f"{self.base_url}/call-recordings", params=params
        )

        data = response.json()
        if not data.get("success"):
            msg = data.get("message", "Unknown CRM API error")
            raise RuntimeError(f"CRM API returned success=false: {msg}")

        items = data.get("items", [])
        reported_count = data.get("count", len(items))
        if reported_count != len(items):
            logger.warning(
                f"CRM count mismatch: reported {reported_count}, "
                f"received {len(items)} flight requests"
            )

        # Flatten nested structure: flight_requests → calls
        all_records: List[Dict] = []
        for item in items:
            agent_info = item.get("agent", {})
            client_info = item.get("client", {})
            client_name = (
                f"{client_info.get('first_name', '')} "
                f"{client_info.get('last_name', '')}"
            ).strip()

            for call in item.get("calls", []):
                all_records.append({
                    "id": call["id"],
                    "flight_request_id": item["id"],
                    "startTime": call["started_at"],
                    "duration": call["duration"],
                    "direction": call.get("direction", ""),
                    "result": call.get("result", ""),
                    "recording_url": call["recording_url"],
                    "agent_id": agent_info.get("id"),
                    "agent_name": agent_info.get("name", ""),
                    "client_name": client_name,
                    "flight_request_status": item.get("status", ""),
                })

        # Deduplicate by call ID
        seen_ids: set = set()
        deduped: List[Dict] = []
        for rec in all_records:
            cid = rec["id"]
            if cid in seen_ids:
                logger.debug(f"Skipping duplicate call ID {cid}")
                continue
            seen_ids.add(cid)
            deduped.append(rec)

        if len(deduped) < len(all_records):
            logger.info(
                f"Deduplicated recordings: {len(all_records)} → {len(deduped)}"
            )

        logger.info(
            f"Found {len(deduped)} recordings across {len(items)} flight requests"
        )

        if reported_count >= self.API_MAX_LIMIT:
            logger.warning(
                f"CRM returned {self.API_MAX_LIMIT}+ flight requests — results may be truncated. "
                "Consider narrowing the date range."
            )

        return deduped

    # ------------------------------------------------------------------
    # Paginated Search (CRIT-2: date-windowing to prevent silent data loss)
    # ------------------------------------------------------------------

    def search_recordings_paginated(
        self,
        date_from: str,
        date_to: Optional[str] = None,
        *,
        _depth: int = 0,
        _MAX_DEPTH: int = 5,
    ) -> List[Dict]:
        """Search CRM recordings with automatic date-windowing pagination.

        CRIT-2: If a single query returns >= API_MAX_LIMIT results, the date
        range is split in half and each half is queried recursively to avoid
        silent data loss.

        Args:
            date_from: Start date (YYYY-MM-DD).
            date_to: End date (YYYY-MM-DD). Defaults to today.
            _depth: Internal recursion depth counter.
            _MAX_DEPTH: Maximum recursion depth (prevents infinite splitting).

        Returns:
            Deduplicated flat list of call-record dicts.
        """
        if date_to is None:
            date_to = datetime.now().strftime("%Y-%m-%d")

        records = self.search_recordings(date_from, date_to)

        # Check if results may be truncated
        if len(records) >= self.API_MAX_LIMIT:
            if _depth >= _MAX_DEPTH:
                self._last_query_truncated = True
                logger.error(
                    f"Pagination max depth ({_MAX_DEPTH}) reached for "
                    f"{date_from}–{date_to}. Results may be incomplete."
                )
                return records

            # Split date range in half
            dt_from = datetime.strptime(date_from, "%Y-%m-%d")
            dt_to = datetime.strptime(date_to, "%Y-%m-%d")
            delta = (dt_to - dt_from).days

            if delta < 1:
                # Cannot split further — single day still exceeds limit
                self._last_query_truncated = True
                logger.warning(
                    f"Single day {date_from} returned {len(records)} results (>= limit). "
                    "Cannot split further."
                )
                return records

            mid = dt_from + timedelta(days=delta // 2)
            mid_str = mid.strftime("%Y-%m-%d")

            logger.info(
                f"Pagination: splitting {date_from}–{date_to} at {mid_str} "
                f"(depth={_depth + 1})"
            )

            left = self.search_recordings_paginated(
                date_from, mid_str, _depth=_depth + 1, _MAX_DEPTH=_MAX_DEPTH
            )
            right = self.search_recordings_paginated(
                (mid + timedelta(days=1)).strftime("%Y-%m-%d"),
                date_to,
                _depth=_depth + 1,
                _MAX_DEPTH=_MAX_DEPTH,
            )

            # Deduplicate across halves by call ID
            seen_ids: set = set()
            combined: List[Dict] = []
            for rec in left + right:
                cid = rec["id"]
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    combined.append(rec)

            logger.info(
                f"Pagination merged: {len(left)} + {len(right)} → {len(combined)} "
                f"unique records for {date_from}–{date_to}"
            )
            return combined

        return records

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_audio(self, call_record: dict) -> Optional[str]:
        """Download a single recording. Returns filepath or None on error."""
        try:
            recording_url = call_record.get("recording_url", "")
            call_id = call_record.get("id", "unknown")

            if not recording_url:
                logger.warning(f"No recording URL for call {call_id}")
                return None

            # SSRF protection: validate URL domain
            parsed = urlparse(recording_url)
            if parsed.hostname != self.ALLOWED_DOWNLOAD_DOMAIN:
                logger.error(
                    f"Recording URL domain rejected for call {call_id}: "
                    f"{parsed.hostname} (expected {self.ALLOWED_DOWNLOAD_DOMAIN})"
                )
                return None

            # Build filename
            timestamp = str(call_record.get("startTime", ""))[:10].replace("-", "")
            filename = f"{call_id}_{timestamp}.mp3"
            filepath = self.download_folder / filename

            if filepath.exists():
                logger.info(f"Already exists: {filename}")
                return str(filepath)

            logger.debug(f"Downloading {filename} from {recording_url[:80]}...")

            # CRIT-1: Streaming download — prevents OOM on large files.
            # NEW-02: Use self._client.stream() to inherit SSL verify,
            # auth headers, and timeout configuration from the shared client.
            tmp_fd = None
            tmp_path = None
            try:
                with self._client.stream(
                    "GET",
                    recording_url,
                    follow_redirects=True,
                ) as stream:
                    if stream.status_code == 401:
                        logger.error(f"Recording download auth failed (401) for {call_id}")
                        return None

                    if stream.status_code != 200:
                        logger.error(f"Recording download returned {stream.status_code} for {call_id}")
                        return None

                    # Content-type validation — reject HTML login pages
                    content_type = stream.headers.get("content-type", "")
                    if "text/html" in content_type:
                        logger.error(
                            f"Recording {call_id} returned HTML instead of audio "
                            f"(content-type: {content_type}). URL may require auth."
                        )
                        return None

                    # Stream to temp file, then atomic rename
                    tmp_fd, tmp_path = tempfile.mkstemp(
                        dir=str(self.download_folder), suffix=".tmp"
                    )
                    total_bytes = 0
                    with os.fdopen(tmp_fd, "wb") as tmp_f:
                        tmp_fd = None  # os.fdopen takes ownership
                        for chunk in stream.iter_bytes(chunk_size=64 * 1024):
                            total_bytes += len(chunk)
                            if total_bytes > self.MAX_DOWNLOAD_BYTES:
                                logger.error(
                                    f"Recording {call_id} exceeds size limit during "
                                    f"download: >{self.MAX_DOWNLOAD_BYTES} bytes"
                                )
                                return None
                            tmp_f.write(chunk)

                    if total_bytes == 0:
                        logger.error(f"Recording download returned empty content for {call_id}")
                        return None

                    # Atomic rename (safe on Windows via os.replace)
                    os.replace(tmp_path, str(filepath))
                    tmp_path = None  # Prevent cleanup

            finally:
                if tmp_fd is not None:
                    os.close(tmp_fd)
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            time.sleep(self.delay_seconds)
            logger.info(f"Downloaded: {filename} ({total_bytes} bytes)")
            return str(filepath)

        except Exception as e:
            logger.error(f"Download error for call {call_record.get('id')}: {type(e).__name__}: {e}")
            return None

    # ------------------------------------------------------------------
    # Combined search + download
    # ------------------------------------------------------------------

    def search_and_download(
        self, date_from: str, date_to: Optional[str] = None
    ) -> List[Dict]:
        """Search recordings and download all. Returns enriched call list.

        This is the method that Pipeline.run() calls.
        """
        calls = self.search_recordings(date_from, date_to)
        total = len(calls)

        for i, call in enumerate(calls, 1):
            logger.info(f"Downloading {i} of {total}...")
            local_path = self.download_audio(call)
            call["local_audio_path"] = local_path

        downloaded = sum(1 for c in calls if c.get("local_audio_path"))
        logger.info(f"Downloaded {downloaded} of {total} recordings")
        return calls

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict] = None,
        max_attempts: int = 3,
    ) -> httpx.Response:
        """HTTP request with retry + backoff for 429/5xx errors."""
        for attempt in range(max_attempts):
            try:
                response = self._client.request(method, url, params=params)

                if response.status_code == 401:
                    body = response.json() if response.content else {}
                    msg = body.get("message", "Unauthorized")
                    raise RuntimeError(
                        f"CRM API authentication failed (401): {msg}. "
                        f"Check CRM_AI_TOKEN — token may be expired or invalid."
                    )

                if response.status_code == 429:
                    retry_after = int(
                        response.headers.get("Retry-After", 2 ** (attempt + 1))
                    )
                    logger.warning(
                        f"CRM API rate limited (429). "
                        f"Retrying in {retry_after}s (attempt {attempt + 1}/{max_attempts})"
                    )
                    time.sleep(retry_after)
                    continue

                if response.status_code >= 500:
                    backoff = 2 ** (attempt + 1)
                    logger.warning(
                        f"CRM API server error ({response.status_code}). "
                        f"Retrying in {backoff}s (attempt {attempt + 1}/{max_attempts})"
                    )
                    time.sleep(backoff)
                    continue

                response.raise_for_status()
                return response

            except httpx.HTTPStatusError:
                raise
            except RuntimeError:
                raise
            except Exception as e:
                if attempt < max_attempts - 1:
                    backoff = 2 ** (attempt + 1)
                    logger.warning(
                        f"CRM API request error: {e}. "
                        f"Retrying in {backoff}s (attempt {attempt + 1}/{max_attempts})"
                    )
                    time.sleep(backoff)
                else:
                    raise

        raise RuntimeError(
            f"CRM API request failed after {max_attempts} attempts"
        )

    def close(self) -> None:
        """Close the underlying httpx client."""
        self._client.close()
        logger.debug("CRMAgent httpx client closed")

    # CRIT-3: Context manager support for resource cleanup
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
