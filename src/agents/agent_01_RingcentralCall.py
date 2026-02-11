"""
Agent 1: RingCentral Audio Retrieval

Purpose: Download call recordings using Official RingCentral SDK
Two classes:
  - AudioFileFinder: Find and analyze local audio files
  - RingCentralAgent: Search and download recordings via RingCentral API
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os
import time
import logging

logger = logging.getLogger("qa_system")


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

        audio_files = []
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
        except Exception as e:
            logger.warning(f"Cannot read duration for {file_path.name}: {e}")
            return None


class RingCentralAgent:
    """Agent: Download audio from RingCentral via Official SDK"""

    def __init__(self, platform, download_folder: str = "data/audio", delay_seconds: float = 1.5):
        """
        Initialize with authenticated RingCentral platform instance.

        Usage:
            from ringcentral import SDK
            sdk = SDK(client_id, client_secret, server)
            platform = sdk.platform()
            platform.login(jwt=jwt_token)
            agent_rc = RingCentralAgent(platform)
        """
        self.platform = platform
        self.download_folder = Path(download_folder)
        self.delay_seconds = delay_seconds
        self.download_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"RingCentralAgent initialized | Download folder: {self.download_folder}")

    def search_recordings(self, date_from: str, date_to: Optional[str] = None) -> List[Dict]:
        """
        Search call-log for recordings.
        Returns list of call dicts with recording metadata.
        """
        all_records = []
        params = {
            "dateFrom": f"{date_from}T00:00:00.000Z",
            "type": "Voice",
            "withRecording": True,
            "perPage": 100
        }
        if date_to:
            params["dateTo"] = f"{date_to}T23:59:59.999Z"

        logger.info(f"Searching recordings from {date_from} to {date_to or 'now'}...")

        url = "/restapi/v1.0/account/~/call-log"
        while url:
            response = self.platform.get(url, params)
            data = response.json_dict()

            for record in data.get("records", []):
                recording = record.get("recording")
                if recording and recording.get("contentUri"):
                    all_records.append({
                        "id": record.get("id"),
                        "sessionId": record.get("sessionId"),
                        "startTime": record.get("startTime"),
                        "duration": record.get("duration"),
                        "direction": record.get("direction"),
                        "from": record.get("from", {}).get("phoneNumber", ""),
                        "to": record.get("to", {}).get("phoneNumber", ""),
                        "contentUri": recording["contentUri"],
                        "contentType": recording.get("contentType", "audio/mpeg")
                    })

            # Handle pagination
            next_page = data.get("navigation", {}).get("nextPage", {}).get("uri")
            if next_page:
                url = next_page
                params = None  # params are in the URL already
            else:
                url = None

        logger.info(f"Found {len(all_records)} recordings with audio")
        return all_records

    def download_audio(self, call_record: dict) -> Optional[str]:
        """Download a single recording. Returns filepath or None on error."""
        try:
            content_uri = call_record["contentUri"]
            call_id = call_record.get("id", "unknown")
            timestamp = call_record.get("startTime", "")[:10].replace("-", "")
            ext = ".mp3" if "mpeg" in call_record.get("contentType", "") else ".wav"
            filename = f"{call_id}_{timestamp}{ext}"
            filepath = self.download_folder / filename

            if filepath.exists():
                logger.info(f"Already exists: {filename}")
                return str(filepath)

            response = self.platform.get(content_uri)
            with open(filepath, "wb") as f:
                f.write(response.response().content)

            time.sleep(self.delay_seconds)
            logger.info(f"Downloaded: {filename}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Download error for call {call_record.get('id')}: {e}")
            return None

    def search_and_download(self, date_from: str, date_to: Optional[str] = None) -> List[Dict]:
        """Search recordings and download all. Returns enriched call list."""
        calls = self.search_recordings(date_from, date_to)
        total = len(calls)

        for i, call in enumerate(calls, 1):
            print(f"  Downloading {i} of {total}...", end="\r")
            local_path = self.download_audio(call)
            call["local_audio_path"] = local_path

        downloaded = sum(1 for c in calls if c.get("local_audio_path"))
        logger.info(f"Downloaded {downloaded} of {total} recordings")
        return calls
