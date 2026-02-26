"""Tests for process_audio.py script."""
from pathlib import Path


class TestProcessAudioScript:

    def test_uses_evaluate_call(self):
        """process_audio.py should use evaluate_call for QA evaluation."""
        content = (Path(__file__).parent.parent / "process_audio.py").read_text()
        assert "evaluate_call(" in content
        assert "calculate_score(" in content

    def test_uses_stt_agent(self):
        """process_audio.py should use ElevenLabsSTTAgent."""
        content = (Path(__file__).parent.parent / "process_audio.py").read_text()
        assert "ElevenLabsSTTAgent" in content
        assert "QualityManagementAgent" in content

    def test_no_hardcoded_direction(self):
        """process_audio.py should not hardcode outbound direction."""
        content = (Path(__file__).parent.parent / "process_audio.py").read_text()
        assert '"direction": "outbound"' not in content
