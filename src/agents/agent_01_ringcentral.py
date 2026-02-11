"""
Agent 1: RingCentral Audio Retrieval

Purpose: Download call recordings using Official RingCentral SDK
Style: Matches my AudioFileFinder + RingCentral SDK pattern

TODO:
- Implement RingCentralAgent (JWT auth, search, download)
- Implement AudioFileFinder (local file discovery)
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os
import time


class AudioFileFinder:
    """Agent: Finds and analyzes audio files in local folder"""

    def __init__(self, folder_path: str, extensions: tuple = (".mp3", ".wav", ".m4a")):
        self.folder_path = Path(folder_path)
        self.extensions = extensions

    def find_all(self) -> List[Path]:
        """
        TODO:
        - Check folder exists
        - Glob for all audio extensions
        - Return sorted list of Paths
        """
        # TODO: Implement
        pass

    def get_info(self, file_path: Path) -> Dict:
        """
        TODO:
        - Get file stat (size)
        - Return {"name": ..., "size_mb": ...}
        """
        # TODO: Implement
        pass

    def get_duration(self, file_path: Path) -> Optional[float]:
        """
        TODO:
        - Use pydub AudioSegment.from_file()
        - Return duration in minutes
        - Return None if file unreadable
        """
        # TODO: Implement
        pass


class RingCentralAgent:
    """Agent: Download audio from RingCentral via Official SDK"""

    def __init__(self, platform):
        """
        TODO:
        - Store authenticated platform instance
        - Set download folder from config
        - Set rate limit delay (1.5s)

        Usage:
            from ringcentral import SDK
            sdk = SDK(client_id, client_secret, server)
            platform = sdk.platform()
            platform.login(jwt=jwt_token)
            agent_rc = RingCentralAgent(platform)
        """
        self.platform = platform
        # TODO: Implement

    def search_recordings(self, date_from: str, date_to: str = None) -> List[Dict]:
        """
        TODO:
        - Call /restapi/v1.0/account/~/call-log
        - Params: dateFrom, dateTo, type=Voice, withRecording=True, perPage=100
        - Handle pagination (follow navigation.nextPage)
        - Filter: only calls with recording.contentUri
        - Return list of call dicts with metadata
        """
        # TODO: Implement
        pass

    def download_audio(self, call_record: dict) -> Optional[str]:
        """
        TODO:
        - Get contentUri from call_record
        - Download via platform.get(contentUri)
        - Save to data/audio/{call_id}_{timestamp}.mp3
        - Rate limit: time.sleep(1.5) between downloads
        - Return filepath or None on error
        """
        # TODO: Implement
        pass

    def search_and_download(self, date_from: str, date_to: str = None) -> List[Dict]:
        """
        TODO:
        - calls = self.search_recordings(date_from, date_to)
        - For each call: download and add local_audio_path
        - Print progress: "Downloaded X of Y recordings"
        - Return enriched call list
        """
        # TODO: Implement
        pass


# TODO: Initialize like this:
# === Option A: From RingCentral API ===
# from ringcentral import SDK
# sdk = SDK(os.environ['RC_APP_CLIENT_ID'], os.environ['RC_APP_CLIENT_SECRET'], os.environ['RC_SERVER_URL'])
# platform = sdk.platform()
# platform.login(jwt=os.environ['RC_USER_JWT'])
# agent_rc = RingCentralAgent(platform)
#
# === Option B: From local folder ===
# agent_finder = AudioFileFinder(folder_path="data/audio", extensions=(".mp3", ".wav"))
# audio_files = agent_finder.find_all()
