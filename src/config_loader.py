"""
Config Loader (#21)

Extracted from utils.py — YAML config loading and validation functions.
"""

import logging
import os
import yaml
from pathlib import Path
from typing import List

logger = logging.getLogger("qa_system.config")


def load_config(config_path: str = "config/agents.yaml") -> dict:
    """Load YAML config and return as dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env() -> None:
    """Load .env using python-dotenv."""
    from dotenv import load_dotenv
    load_dotenv()


def validate_env(required_keys: List[str]) -> None:
    """Validate that all required environment variables are set and non-empty.
    Raises SystemExit with clear message listing all missing keys."""
    missing = [k for k in required_keys if not os.environ.get(k, "").strip()]
    if missing:
        # HIGH-8: Use logging instead of print for config validation errors
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        raise SystemExit(1)


def load_criteria(config_path: str = "config/qa_criteria.yaml") -> dict:
    """Load QA criteria from YAML and flatten to
    {key: {description, category, weight, call_applicability}} format.

    call_applicability values: "first_only", "second_only", "both"
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Criteria file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    criteria = {}
    for category, items in raw.items():
        for key, props in items.items():
            criteria[key] = {
                "description": props["description"],
                "category": category,
                "weight": props.get("weight", 1.0),
                "call_applicability": props.get("call_applicability", "both"),
            }
    return criteria


def validate_agents_config(config: dict) -> None:
    """TASK-3: Validate agents.yaml structure at startup.

    Checks that all required top-level sections exist and required
    sub-keys are present.  Raises ValueError with clear message if
    the config is malformed or has duplicate/missing sections.
    """
    if not isinstance(config, dict):
        raise ValueError(
            f"agents.yaml must be a YAML mapping, got {type(config).__name__}"
        )

    _REQUIRED_SECTIONS = {"crm", "elevenlabs", "integration", "pipeline"}
    missing_sections = _REQUIRED_SECTIONS - set(config.keys())
    if missing_sections:
        raise ValueError(
            f"agents.yaml missing required sections: {sorted(missing_sections)}"
        )

    # ElevenLabs required keys
    el = config["elevenlabs"]
    if not isinstance(el, dict):
        raise ValueError("agents.yaml 'elevenlabs' must be a mapping")
    _EL_REQUIRED = {"model", "cost_per_minute", "output_folder", "diarize"}
    el_missing = _EL_REQUIRED - set(el.keys())
    if el_missing:
        raise ValueError(f"agents.yaml elevenlabs missing keys: {sorted(el_missing)}")

    # CRM required keys
    crm = config["crm"]
    if not isinstance(crm, dict):
        raise ValueError("agents.yaml 'crm' must be a mapping")
    _CRM_REQUIRED = {"base_url", "download_folder"}
    crm_missing = _CRM_REQUIRED - set(crm.keys())
    if crm_missing:
        raise ValueError(f"agents.yaml crm missing keys: {sorted(crm_missing)}")

    # Pipeline required keys
    pipe = config["pipeline"]
    if not isinstance(pipe, dict):
        raise ValueError("agents.yaml 'pipeline' must be a mapping")

    # Integration required keys
    integ = config["integration"]
    if not isinstance(integ, dict):
        raise ValueError("agents.yaml 'integration' must be a mapping")
    if "output_folder" not in integ:
        raise ValueError("agents.yaml integration missing key: 'output_folder'")

    logger.info("agents.yaml validation passed")


def validate_models_config(config_path: str = "config/models.yaml") -> dict:
    """Load and validate models.yaml structure at startup.

    Checks required keys in primary/fallback provider configs and token_limits.
    Raises ValueError with clear message if config is malformed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"models.yaml must be a YAML mapping, got {type(config).__name__}")

    if "primary" not in config:
        raise ValueError("models.yaml missing required key: 'primary'")

    _REQUIRED_PROVIDER_KEYS = {"provider", "model", "base_url", "api_key_env"}
    primary = config["primary"]
    missing = _REQUIRED_PROVIDER_KEYS - set(primary.keys())
    if missing:
        raise ValueError(f"models.yaml primary provider missing keys: {missing}")

    for i, fb in enumerate(config.get("fallbacks", [])):
        fb_missing = _REQUIRED_PROVIDER_KEYS - set(fb.keys())
        if fb_missing:
            raise ValueError(f"models.yaml fallback[{i}] ({fb.get('provider', '?')}) missing keys: {fb_missing}")

    if "token_limits" not in config:
        raise ValueError("models.yaml missing required key: 'token_limits'")

    tl = config["token_limits"]
    for key in ("max_input_tokens", "max_output_tokens"):
        if key not in tl:
            raise ValueError(f"models.yaml token_limits missing key: '{key}'")

    return config
