"""
Prompt Template Loader

Loads .txt prompt templates from the prompts/ directory,
caches them in memory, and renders with variable substitution.
Uses simple str.format_map() — no Jinja2 dependency needed.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("qa_system.prompts")


class PromptLoader:
    """Load and render prompt templates from text files.

    Usage:
        loader = PromptLoader()
        system = loader.render("qa_system", call_type="First Call", criteria_count=24)
        user   = loader.render("qa_user", transcript="...", criteria_text="...")
    """

    def __init__(self, templates_dir: Optional[str] = None):
        """Initialize with the directory containing .txt templates.

        Args:
            templates_dir: Path to templates directory.
                           Defaults to the same directory as this file.
        """
        if templates_dir:
            self._dir = Path(templates_dir)
        else:
            self._dir = Path(__file__).parent
        self._cache: Dict[str, str] = {}
        logger.debug(f"PromptLoader initialized | templates_dir={self._dir}")

    def load(self, template_name: str) -> str:
        """Load a raw template by name (without .txt extension).

        Returns the template string with {variable} placeholders intact.
        Caches on first read.
        """
        if template_name in self._cache:
            return self._cache[template_name]

        path = self._dir / f"{template_name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")

        text = path.read_text(encoding="utf-8")
        self._cache[template_name] = text
        logger.debug(f"Loaded template: {template_name} ({len(text)} chars)")
        return text

    def render(self, template_name: str, **variables) -> str:
        """Load template and substitute variables.

        Uses str.format_map() with a SafeDict so missing keys
        are left as-is (e.g. literal JSON braces aren't broken).

        DESIGN-22: Escapes curly braces in the 'transcript' variable
        to prevent format_map() from interpreting transcript content
        (e.g. JSON samples in call recordings) as format placeholders.

        Args:
            template_name: Template name (without .txt extension).
            **variables: Key-value pairs to substitute.

        Returns:
            Rendered prompt string.
        """
        template = self.load(template_name)
        # DESIGN-22: Escape braces in transcript before substitution
        if "transcript" in variables and isinstance(variables["transcript"], str):
            variables["transcript"] = (
                variables["transcript"].replace("{", "{{").replace("}", "}}")
            )
        return template.format_map(_SafeDict(variables))

    def clear_cache(self) -> None:
        """Clear the template cache (useful for development hot-reload)."""
        self._cache.clear()


class _SafeDict(dict):
    """Dict subclass that returns the key placeholder for missing keys.

    This prevents str.format_map() from crashing on literal braces
    in JSON examples within prompt templates.

    MED-7: Logs a warning when a referenced variable is missing from the
    provided context, which helps catch prompt template typos.
    #12: When QA_STRICT_PROMPTS=1, raises KeyError instead of silently
    substituting, to catch template mistakes in development.
    """

    def __missing__(self, key: str) -> str:
        import logging
        import os
        logger = logging.getLogger("qa_system.prompts")
        # #12: Strict mode — fail loudly on missing template variables
        if os.environ.get("QA_STRICT_PROMPTS", "").strip() == "1":
            raise KeyError(
                f"Template variable '{{{key}}}' not provided and "
                f"QA_STRICT_PROMPTS=1 is set"
            )
        logger.warning(
            f"Template variable '{{{key}}}' not provided — left as placeholder"
        )
        return "{" + key + "}"
