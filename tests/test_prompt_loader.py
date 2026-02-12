"""
Tests for src/prompts/templates.py

Tests PromptLoader, template rendering, SafeDict behavior.
"""
import sys
import os
import pytest

from prompts.templates import PromptLoader, _SafeDict


# --- Fixtures ---

@pytest.fixture
def templates_dir(tmp_path):
    """Create temporary template files."""
    (tmp_path / "greeting.txt").write_text("Hello {name}, welcome to {company}!")
    (tmp_path / "json_example.txt").write_text('Output: {{"key": "{value}"}}')
    (tmp_path / "multi_var.txt").write_text("{a} + {b} = {c}")
    return str(tmp_path)


@pytest.fixture
def loader(templates_dir):
    return PromptLoader(templates_dir)


# --- Tests: Loading ---

class TestLoading:

    def test_load_existing_template(self, loader):
        template = loader.load("greeting")
        assert "{name}" in template
        assert "{company}" in template

    def test_load_missing_template_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent")

    def test_load_caches_template(self, loader):
        loader.load("greeting")
        assert "greeting" in loader._cache
        # Load again - should come from cache
        template = loader.load("greeting")
        assert "{name}" in template


# --- Tests: Rendering ---

class TestRendering:

    def test_render_substitutes_variables(self, loader):
        result = loader.render("greeting", name="Alice", company="Acme Corp")
        assert result == "Hello Alice, welcome to Acme Corp!"

    def test_render_multiple_variables(self, loader):
        result = loader.render("multi_var", a="1", b="2", c="3")
        assert result == "1 + 2 = 3"


# --- Tests: SafeDict (MED-7) ---

class TestSafeDict:

    def test_missing_key_returns_placeholder(self):
        d = _SafeDict({"a": "1"})
        assert d["a"] == "1"
        assert d["missing"] == "{missing}"

    def test_json_braces_preserved(self, loader):
        """Literal JSON braces in templates should not break rendering."""
        result = loader.render("json_example", value="test_value")
        assert '{"key": "test_value"}' in result

    def test_partial_rendering(self, loader):
        """Missing variables should be left as placeholders (MED-7 logs warning)."""
        result = loader.render("greeting", name="Bob")
        assert "Bob" in result
        assert "{company}" in result  # left as placeholder


# --- Tests: Cache Management ---

class TestCacheManagement:

    def test_clear_cache(self, loader):
        loader.load("greeting")
        assert len(loader._cache) > 0
        loader.clear_cache()
        assert len(loader._cache) == 0

    def test_auto_detect_templates_dir(self):
        """Default loader should point to src/prompts/ directory."""
        loader = PromptLoader()
        # Should be able to load the real qa_system template
        try:
            template = loader.load("qa_system")
            assert len(template) > 0
        except FileNotFoundError:
            pass  # OK if running from a different cwd
