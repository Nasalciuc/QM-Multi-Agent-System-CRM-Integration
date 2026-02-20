"""
STT Transcript Cache

Content-addressable cache for ElevenLabs STT transcripts.
Uses SHA-256 of (audio bytes + model settings) as key.
Saves ~$17/run by avoiding re-transcription of already-processed audio.

TTL: 30 days (configurable).
Storage: data/stt_cache/{sha256}.json
"""

import hashlib
import json
import os
import tempfile
import time
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("qa_system.stt_cache")

# Default TTL: 30 days
DEFAULT_STT_CACHE_TTL_SECONDS = 30 * 24 * 3600


class STTCache:
    """Content-addressable cache for STT transcriptions.

    Cache key = SHA-256(audio_bytes + model_id + diarize + num_speakers + language_code).
    Each entry is a JSON file at {cache_dir}/{key}.json.

    Usage:
        cache = STTCache("data/stt_cache")
        key = cache.cache_key(audio_path, model_id="scribe_v2", diarize=True)
        cached = cache.load(key)
        if cached:
            return cached  # skip API call
        result = call_api(...)
        cache.save(key, result)
    """

    # Required keys for a valid cached transcript
    _REQUIRED_KEYS = frozenset({"text", "raw_text"})

    def __init__(
        self,
        cache_dir: str = "data/stt_cache",
        enable: bool = True,
        ttl_seconds: int = DEFAULT_STT_CACHE_TTL_SECONDS,
        max_entries: int = 0,  # MED-14: 0 = unlimited
    ):
        self._cache_dir: Optional[Path] = Path(cache_dir) if cache_dir else None
        self._enabled = enable and self._cache_dir is not None
        self._ttl = ttl_seconds
        self._max_entries = max_entries  # MED-14: LRU eviction threshold

        # Stats tracking
        self._hits = 0
        self._misses = 0
        self._saves = 0
        self._errors = 0

        if self._enabled and self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._cleanup_expired()
            entry_count = self.size()
            logger.info(
                f"STTCache initialized | dir={self._cache_dir} | ttl={self._ttl}s | "
                f"entries={entry_count}"
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def stats(self) -> Dict[str, int]:
        """Return cache hit/miss/save statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "saves": self._saves,
            "errors": self._errors,
            "total_lookups": self._hits + self._misses,
            "hit_rate_pct": round(
                self._hits / max(1, self._hits + self._misses) * 100, 1
            ),
        }

    @staticmethod
    def cache_key(
        audio_path: Path,
        model_id: str = "scribe_v2",
        diarize: bool = True,
        num_speakers: Optional[int] = None,
        language_code: Optional[str] = None,
    ) -> str:
        """Generate content-addressable cache key.

        SHA-256 of (audio file bytes + model settings) ensures:
        - Same audio → same key (content-addressable)
        - Different settings → different key (no stale results)

        Args:
            audio_path: Path to the audio file.
            model_id: STT model identifier.
            diarize: Whether diarization was enabled.
            num_speakers: Expected speaker count (None = auto).
            language_code: ISO language code (None = auto).

        Returns:
            Hex SHA-256 digest string.
        """
        hasher = hashlib.sha256()
        # Hash audio content
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        # Hash model settings
        settings = f"|{model_id}|{diarize}|{num_speakers}|{language_code}"
        hasher.update(settings.encode("utf-8"))
        return hasher.hexdigest()

    def load(self, key: str) -> Optional[Dict]:
        """Load cached transcript by key.

        Returns None on miss, expired, invalid, or disabled cache.
        """
        if not self._enabled or not self._cache_dir:
            self._misses += 1
            return None

        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            self._misses += 1
            return None

        try:
            # TTL check
            age = time.time() - path.stat().st_mtime
            if age > self._ttl:
                logger.debug(f"STT cache expired (age={age:.0f}s): {key[:12]}")
                path.unlink(missing_ok=True)
                self._misses += 1
                return None

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate required keys
            if not isinstance(data, dict) or not self._REQUIRED_KEYS.issubset(data.keys()):
                logger.warning(f"STT cache entry invalid (missing keys): {key[:12]}")
                path.unlink(missing_ok=True)
                self._misses += 1
                return None

            self._hits += 1
            logger.info(f"STT cache hit: {key[:12]}")
            return data

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"STT cache load error: {e}")
            self._errors += 1
            self._misses += 1
            return None

    def save(self, key: str, data: Dict) -> None:
        """Save transcript result to cache (atomic write).

        Uses temp file + os.replace for atomic writes.
        """
        if not self._enabled or not self._cache_dir:
            return

        path = self._cache_dir / f"{key}.json"
        # COST-03: Attach cache metadata for future migration / audit
        enriched = dict(data)
        enriched.setdefault("_cache_meta", {
            "cached_at": time.time(),
            "cache_key": key,
        })
        fd = None
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._cache_dir), suffix=".json.tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                fd = None  # os.fdopen takes ownership
                json.dump(enriched, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, str(path))
            self._saves += 1
            logger.debug(f"STT cache saved: {key[:12]}")
            # MED-14: Evict oldest entries when max_entries is exceeded
            if self._max_entries > 0:
                self._evict_lru()
        except OSError as e:
            logger.warning(f"STT cache save error: {e}")
            self._errors += 1
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        finally:
            if fd is not None:
                os.close(fd)

    def _evict_lru(self) -> None:
        """MED-14: Remove oldest cache entries (by mtime) when over max_entries."""
        if not self._cache_dir or self._max_entries <= 0:
            return
        entries = sorted(
            self._cache_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        evicted = 0
        while len(entries) > self._max_entries:
            oldest = entries.pop(0)
            try:
                oldest.unlink(missing_ok=True)
                evicted += 1
            except OSError:
                pass
        if evicted:
            logger.info(f"STT cache LRU eviction: removed {evicted} oldest entries")

    def _cleanup_expired(self) -> None:
        """Remove cache entries older than TTL at startup."""
        if not self._cache_dir:
            return
        now = time.time()
        removed = 0
        for path in self._cache_dir.glob("*.json"):
            try:
                age = now - path.stat().st_mtime
                if age > self._ttl:
                    path.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                pass
        if removed:
            logger.info(f"STT cache cleanup: removed {removed} expired entries")

    def clear(self) -> int:
        """Remove all cache entries. Returns number of entries removed."""
        if not self._cache_dir:
            return 0
        removed = 0
        for path in self._cache_dir.glob("*.json"):
            try:
                path.unlink(missing_ok=True)
                removed += 1
            except OSError:
                pass
        self._hits = 0
        self._misses = 0
        self._saves = 0
        self._errors = 0
        logger.info(f"STT cache cleared: {removed} entries removed")
        return removed

    def size(self) -> int:
        """Return number of cached entries."""
        if not self._cache_dir or not self._cache_dir.exists():
            return 0
        return sum(1 for _ in self._cache_dir.glob("*.json"))

    def cleanup_orphaned(
        self,
        current_audio_keys: Optional[set] = None,
    ) -> int:
        """COST-03: Remove cache entries whose keys are not in *current_audio_keys*.

        This is useful after changing STT settings (model, num_speakers, etc.)
        which alters cache keys, leaving the old entries orphaned.

        If *current_audio_keys* is None the method is a no-op (returns 0).

        Args:
            current_audio_keys: Set of cache key hex strings that are still valid.

        Returns:
            Number of orphaned entries removed.
        """
        if not self._enabled or not self._cache_dir or current_audio_keys is None:
            return 0

        removed = 0
        for path in list(self._cache_dir.glob("*.json")):
            key = path.stem
            if key not in current_audio_keys:
                try:
                    path.unlink(missing_ok=True)
                    removed += 1
                except OSError:
                    pass
        if removed:
            logger.info(f"COST-03: Removed {removed} orphaned cache entries")
        return removed
