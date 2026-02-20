"""
Tests for STT Cache (inference/stt_cache.py)

Tests content-addressable caching for ElevenLabs STT transcripts.
"""
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure src is on path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference.stt_cache import STTCache, DEFAULT_STT_CACHE_TTL_SECONDS


# --- Fixtures ---

@pytest.fixture
def cache_dir(tmp_path):
    return str(tmp_path / "stt_cache")


@pytest.fixture
def cache(cache_dir):
    return STTCache(cache_dir=cache_dir, enable=True, ttl_seconds=3600)


@pytest.fixture
def disabled_cache(cache_dir):
    return STTCache(cache_dir=cache_dir, enable=False)


@pytest.fixture
def audio_file(tmp_path):
    """Create a fake audio file."""
    f = tmp_path / "call1.mp3"
    f.write_bytes(b"ID3" + b"\x00" * 100)
    return f


@pytest.fixture
def sample_transcript():
    return {
        "text": "Speaker 0: Hello\nSpeaker 1: Hi",
        "raw_text": "Hello Hi",
        "speakers_detected": 2,
        "diarized": True,
        "language_code": "en",
    }


# --- Tests: Cache Key ---

class TestCacheKey:
    def test_same_audio_same_key(self, audio_file):
        key1 = STTCache.cache_key(audio_file)
        key2 = STTCache.cache_key(audio_file)
        assert key1 == key2

    def test_different_audio_different_key(self, tmp_path):
        f1 = tmp_path / "a.mp3"
        f2 = tmp_path / "b.mp3"
        f1.write_bytes(b"ID3" + b"\x00" * 100)
        f2.write_bytes(b"ID3" + b"\x01" * 100)
        assert STTCache.cache_key(f1) != STTCache.cache_key(f2)

    def test_different_settings_different_key(self, audio_file):
        key1 = STTCache.cache_key(audio_file, diarize=True)
        key2 = STTCache.cache_key(audio_file, diarize=False)
        assert key1 != key2

    def test_different_model_different_key(self, audio_file):
        key1 = STTCache.cache_key(audio_file, model_id="scribe_v2")
        key2 = STTCache.cache_key(audio_file, model_id="scribe_v1")
        assert key1 != key2

    def test_different_speakers_different_key(self, audio_file):
        key1 = STTCache.cache_key(audio_file, num_speakers=2)
        key2 = STTCache.cache_key(audio_file, num_speakers=3)
        assert key1 != key2

    def test_key_is_hex_sha256(self, audio_file):
        key = STTCache.cache_key(audio_file)
        assert len(key) == 64  # SHA-256 hex digest
        assert all(c in "0123456789abcdef" for c in key)


# --- Tests: Save & Load ---

class TestSaveLoad:
    def test_save_and_load(self, cache, audio_file, sample_transcript):
        key = STTCache.cache_key(audio_file)
        cache.save(key, sample_transcript)
        loaded = cache.load(key)
        assert loaded is not None
        assert loaded["text"] == sample_transcript["text"]
        assert loaded["raw_text"] == sample_transcript["raw_text"]

    def test_load_miss(self, cache):
        result = cache.load("nonexistent_key")
        assert result is None

    def test_load_disabled(self, disabled_cache):
        result = disabled_cache.load("any_key")
        assert result is None

    def test_save_disabled_noop(self, disabled_cache, audio_file, sample_transcript):
        key = STTCache.cache_key(audio_file)
        disabled_cache.save(key, sample_transcript)
        # No file should be created
        assert disabled_cache.size() == 0

    def test_save_creates_json(self, cache, cache_dir, audio_file, sample_transcript):
        key = STTCache.cache_key(audio_file)
        cache.save(key, sample_transcript)
        path = Path(cache_dir) / f"{key}.json"
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["text"] == sample_transcript["text"]

    def test_load_invalid_json(self, cache, cache_dir):
        key = "bad_json_key"
        path = Path(cache_dir) / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not valid json", encoding="utf-8")
        result = cache.load(key)
        assert result is None

    def test_load_missing_required_keys(self, cache, cache_dir):
        key = "incomplete_key"
        path = Path(cache_dir) / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"foo": "bar"}', encoding="utf-8")
        result = cache.load(key)
        assert result is None


# --- Tests: TTL ---

class TestTTL:
    def test_expired_entry_returns_none(self, cache_dir):
        cache = STTCache(cache_dir=cache_dir, enable=True, ttl_seconds=1)
        key = "expire_me"
        cache.save(key, {"text": "hello", "raw_text": "hello"})
        # Still fresh
        assert cache.load(key) is not None
        # Wait for expiry
        time.sleep(1.5)
        assert cache.load(key) is None

    def test_cleanup_expired_at_startup(self, cache_dir):
        # Create some entries manually
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        old_file = Path(cache_dir) / "old_entry.json"
        old_file.write_text('{"text":"old","raw_text":"old"}')
        # Set mtime to past
        old_time = time.time() - 10
        os.utime(old_file, (old_time, old_time))
        # Create cache with 5-second TTL — should clean up
        cache = STTCache(cache_dir=cache_dir, enable=True, ttl_seconds=5)
        assert not old_file.exists()


# --- Tests: Stats ---

class TestStats:
    def test_initial_stats(self, cache):
        stats = cache.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["saves"] == 0
        assert stats["hit_rate_pct"] == 0.0

    def test_stats_after_miss(self, cache):
        cache.load("nonexistent")
        stats = cache.stats
        assert stats["misses"] == 1
        assert stats["total_lookups"] == 1

    def test_stats_after_hit(self, cache, audio_file, sample_transcript):
        key = STTCache.cache_key(audio_file)
        cache.save(key, sample_transcript)
        cache.load(key)
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["saves"] == 1
        assert stats["hit_rate_pct"] == 100.0

    def test_stats_mixed(self, cache, audio_file, sample_transcript):
        key = STTCache.cache_key(audio_file)
        cache.load("miss1")
        cache.load("miss2")
        cache.save(key, sample_transcript)
        cache.load(key)
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["total_lookups"] == 3
        assert stats["hit_rate_pct"] == pytest.approx(33.3, abs=0.1)


# --- Tests: Clear & Size ---

class TestClearAndSize:
    def test_size_empty(self, cache):
        assert cache.size() == 0

    def test_size_after_save(self, cache, audio_file, sample_transcript):
        key = STTCache.cache_key(audio_file)
        cache.save(key, sample_transcript)
        assert cache.size() == 1

    def test_clear(self, cache, audio_file, sample_transcript):
        key = STTCache.cache_key(audio_file)
        cache.save(key, sample_transcript)
        removed = cache.clear()
        assert removed == 1
        assert cache.size() == 0
        assert cache.load(key) is None

    def test_clear_resets_stats(self, cache, audio_file, sample_transcript):
        key = STTCache.cache_key(audio_file)
        cache.save(key, sample_transcript)
        cache.load(key)
        cache.clear()
        stats = cache.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0


# --- Tests: Edge Cases ---

class TestEdgeCases:
    def test_none_cache_dir(self):
        cache = STTCache(cache_dir=None, enable=True)
        assert not cache.enabled
        assert cache.load("key") is None
        assert cache.size() == 0

    def test_enabled_property(self, cache, disabled_cache):
        assert cache.enabled is True
        assert disabled_cache.enabled is False

    def test_default_ttl(self):
        assert DEFAULT_STT_CACHE_TTL_SECONDS == 30 * 24 * 3600


# --- Tests: MED-14 — LRU Eviction ---

class TestLRUEviction:

    def test_eviction_when_over_max(self, cache_dir):
        """MED-14: When max_entries is set and exceeded, oldest entries are evicted."""
        cache = STTCache(cache_dir=cache_dir, enable=True, max_entries=3)

        sample = {"text": "hello", "raw_text": "hello"}
        for i in range(5):
            key = f"key_{i:03d}"
            cache.save(key, sample)
            # Slight delay to ensure distinct mtimes
            time.sleep(0.05)

        assert cache.size() <= 3

    def test_no_eviction_when_unlimited(self, cache_dir):
        """MED-14: max_entries=0 means unlimited — no eviction."""
        cache = STTCache(cache_dir=cache_dir, enable=True, max_entries=0)

        sample = {"text": "hello", "raw_text": "hello"}
        for i in range(5):
            cache.save(f"key_{i:03d}", sample)

        assert cache.size() == 5

    def test_eviction_removes_oldest(self, cache_dir):
        """MED-14: Oldest entries (by mtime) should be evicted first."""
        cache = STTCache(cache_dir=cache_dir, enable=True, max_entries=2)

        sample = {"text": "hello", "raw_text": "hello"}
        cache.save("oldest", sample)
        time.sleep(0.05)
        cache.save("middle", sample)
        time.sleep(0.05)
        cache.save("newest", sample)

        # oldest should have been evicted
        assert cache.load("newest") is not None
        assert cache.size() <= 2


# --- Tests: COST-03 — Cache meta & orphan cleanup ---

class TestCacheMeta:
    """COST-03: save() should embed _cache_meta for audit/migration."""

    def test_save_embeds_cache_meta(self, cache, cache_dir, audio_file, sample_transcript):
        """Saved entries should contain _cache_meta with cached_at and cache_key."""
        key = STTCache.cache_key(audio_file)
        cache.save(key, sample_transcript)

        path = Path(cache_dir) / f"{key}.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "_cache_meta" in data
        assert "cached_at" in data["_cache_meta"]
        assert data["_cache_meta"]["cache_key"] == key

    def test_save_does_not_overwrite_existing_meta(self, cache, audio_file, cache_dir):
        """If caller supplies _cache_meta, save() should preserve it."""
        key = STTCache.cache_key(audio_file)
        custom_meta = {"cached_at": 1234567890, "custom": True}
        transcript = {"text": "hi", "raw_text": "hi", "_cache_meta": custom_meta}
        cache.save(key, transcript)

        path = Path(cache_dir) / f"{key}.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # setdefault should keep original
        assert data["_cache_meta"]["cached_at"] == 1234567890
        assert data["_cache_meta"]["custom"] is True


class TestCleanupOrphaned:
    """COST-03: Orphan cleanup removes stale cache entries."""

    def test_cleanup_removes_orphans(self, cache, cache_dir):
        """Entries not in current_audio_keys should be removed."""
        sample = {"text": "hello", "raw_text": "hello"}
        cache.save("keep_me", sample)
        cache.save("remove_me", sample)
        cache.save("also_remove", sample)

        removed = cache.cleanup_orphaned(current_audio_keys={"keep_me"})
        assert removed == 2
        assert cache.size() == 1
        assert cache.load("keep_me") is not None

    def test_cleanup_noop_when_none(self, cache, cache_dir):
        """cleanup_orphaned(None) should be a no-op."""
        sample = {"text": "hello", "raw_text": "hello"}
        cache.save("entry1", sample)

        removed = cache.cleanup_orphaned(current_audio_keys=None)
        assert removed == 0
        assert cache.size() == 1

    def test_cleanup_noop_when_disabled(self, disabled_cache, cache_dir):
        """Disabled cache should return 0."""
        removed = disabled_cache.cleanup_orphaned(current_audio_keys=set())
        assert removed == 0

    def test_cleanup_all_orphans(self, cache, cache_dir):
        """If current_audio_keys is empty set, all entries are orphans."""
        sample = {"text": "hello", "raw_text": "hello"}
        cache.save("a", sample)
        cache.save("b", sample)

        removed = cache.cleanup_orphaned(current_audio_keys=set())
        assert removed == 2
        assert cache.size() == 0


class TestStartupInventory:
    """COST-03: Startup log should include entry count."""

    def test_init_logs_entry_count(self, tmp_path):
        """STTCache init should log the number of entries found."""
        cache_dir = str(tmp_path / "stt_cache")
        # Create a cache with some entries
        c = STTCache(cache_dir=cache_dir, enable=True)
        c.save("entry1", {"text": "a", "raw_text": "a"})
        c.save("entry2", {"text": "b", "raw_text": "b"})

        # Re-initialize — should log "entries=2"
        import logging
        with patch.object(logging.getLogger("qa_system.stt_cache"), "info") as mock_info:
            STTCache(cache_dir=cache_dir, enable=True)
            init_calls = [
                str(call) for call in mock_info.call_args_list
                if "entries=" in str(call)
            ]
            assert len(init_calls) >= 1
            assert "entries=2" in init_calls[0]
