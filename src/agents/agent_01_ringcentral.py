"""
Agent 1: RingCentral Audio Retrieval

Purpose: Download call recordings via RingCentral Official SDK (JWT auth)

Dependencies:
    - ringcentral (Official SDK)
    - python-dotenv

Authentication: JWT flow (no user interaction needed)
    - Uses RC_USER_JWT from .env
    - SDK handles token refresh automatically

Rate Limiting:
    - 1.5s delay between download requests
    - Exponential backoff on 429 responses

TODO:
    - authenticate() -> bool
    - search_recordings(date_from, date_to) -> list[dict]
    - download_audio(call_record) -> str  (returns filepath)
    - _download_batch(records) -> list[str]
    - _build_call_log_params(date_from, date_to) -> dict
    - _handle_pagination(response) -> list[dict]  (follow nextPage links)
    - _extract_metadata(record) -> dict
"""

import os
import time
from pathlib import Path
from typing import Optional

# from ringcentral import SDK   # TODO: Uncomment when implementing


class RingCentralAgent:
    """
    Downloads call recordings from RingCentral.

    Usage:
        agent = RingCentralAgent(config)
        agent.authenticate()
        calls = agent.search_recordings("2025-02-01", "2025-02-11")
        for call in calls:
            filepath = agent.download_audio(call)

    Config keys needed (from config/agents.yaml):
        ringcentral.download_folder  -> "data/audio"
        ringcentral.delay_seconds    -> 1.5
        ringcentral.days_back        -> 7

    Env vars needed (from .env):
        RC_APP_CLIENT_ID
        RC_APP_CLIENT_SECRET
        RC_SERVER_URL
        RC_USER_JWT
    """

    def __init__(self, config: dict):
        """
        TODO:
            - Read credentials from os.environ
            - Store config (download_folder, delay_seconds)
            - Initialize SDK: SDK(client_id, client_secret, server_url)
            - Don't authenticate yet (call authenticate() explicitly)
        """
        # TODO: Implement
        pass

    def authenticate(self) -> bool:
        """
        Authenticate with RingCentral using JWT.

        TODO:
            - platform = sdk.platform()
            - platform.login(jwt=os.environ['RC_USER_JWT'])
            - Return True on success
            - Log error and return False on failure

        RingCentral SDK JWT auth:
            sdk = SDK(client_id, client_secret, server_url)
            platform = sdk.platform()
            platform.login(jwt='your_jwt_token')
        """
        # TODO: Implement
        pass

    def search_recordings(
        self,
        date_from: str,
        date_to: Optional[str] = None,
    ) -> list[dict]:
        """
        Search RingCentral call log for calls with recordings.

        API: GET /restapi/v1.0/account/~/call-log

        TODO:
            - Build query params:
                dateFrom, dateTo, type="Voice",
                withRecording=True, perPage=100
            - Send GET request via platform.get()
            - Handle pagination (check for navigation.nextPage)
            - Filter: only calls with recording.contentUri
            - Extract metadata for each call:
                {call_id, session_id, start_time, duration,
                 direction, from_number, to_number,
                 recording_id, content_uri}
            - Return list of call dicts

        Pagination pattern:
            response = platform.get('/restapi/v1.0/account/~/call-log', params)
            records = response.json().records
            while response.json().navigation.nextPage:
                response = platform.get(nextPage.uri)
                records += response.json().records
        """
        # TODO: Implement
        pass

    def download_audio(self, call_record: dict) -> str:
        """
        Download a single call recording.

        TODO:
            - Get content_uri from call_record
            - Make authenticated GET to content_uri:
                response = platform.get(content_uri)
            - Save to: data/audio/{call_id}_{timestamp}.mp3
            - Respect rate limit: time.sleep(self.delay_seconds)
            - Return filepath string
            - On error: log warning, return None

        Rate limiting:
            time.sleep(1.5)  # Between each download
        """
        # TODO: Implement
        pass

    def search_and_download(
        self,
        date_from: str,
        date_to: Optional[str] = None,
    ) -> list[dict]:
        """
        Convenience: search + download all in one call.

        TODO:
            - calls = self.search_recordings(date_from, date_to)
            - For each call:
                filepath = self.download_audio(call)
                call['local_audio_path'] = filepath
            - Return enriched list with local paths
            - Log progress: "Downloaded X of Y recordings"
        """
        # TODO: Implement
        pass
