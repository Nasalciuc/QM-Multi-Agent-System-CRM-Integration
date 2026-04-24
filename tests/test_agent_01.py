"""
Tests for Agent 1: Audio file discovery

Tests AudioFileFinder (local file scanning).
CRMAgent is tested separately in test_agent_01_crm.py.
"""

import pytest

from agents.agent_01_audio import AudioFileFinder


# --- Fixtures ---

@pytest.fixture
def audio_dir(tmp_path):
    """Create a temp directory with fake audio files."""
    for name in ["call1.mp3", "call2.wav", "call3.m4a", "notes.txt"]:
        (tmp_path / name).write_text("fake")
    return tmp_path


@pytest.fixture
def finder(audio_dir):
    return AudioFileFinder(folder_path=str(audio_dir))


# --- Tests: File Discovery ---

class TestAudioFileFinder:

    def test_find_all_audio_files(self, finder):
        """Should find all 3 audio files, ignoring .txt."""
        files = finder.find_all()
        assert len(files) == 3

    def test_find_returns_sorted(self, finder):
        """Files should be sorted by name."""
        files = finder.find_all()
        names = [f.name for f in files]
        assert names == sorted(names)

    def test_find_empty_folder(self, tmp_path):
        """Empty folder returns empty list."""
        finder = AudioFileFinder(folder_path=str(tmp_path))
        assert finder.find_all() == []

    def test_find_nonexistent_folder(self, tmp_path):
        """Non-existent folder returns empty list (no crash)."""
        finder = AudioFileFinder(folder_path=str(tmp_path / "nope"))
        assert finder.find_all() == []

    def test_get_info(self, finder, audio_dir):
        """get_info returns name, size_mb, and path."""
        info = finder.get_info(audio_dir / "call1.mp3")
        assert info["name"] == "call1.mp3"
        assert "size_mb" in info
        assert "path" in info

    def test_custom_extensions(self, tmp_path):
        """Should respect custom extension filter."""
        (tmp_path / "a.ogg").write_text("fake")
        (tmp_path / "b.mp3").write_text("fake")
        finder = AudioFileFinder(folder_path=str(tmp_path), extensions=(".ogg",))
        files = finder.find_all()
        assert len(files) == 1
        assert files[0].name == "a.ogg"

    def test_get_duration_graceful_failure(self, finder, audio_dir):
        """get_duration returns None when pydub fails on fake file."""
        duration = finder.get_duration(audio_dir / "call1.mp3")
        assert duration is None
