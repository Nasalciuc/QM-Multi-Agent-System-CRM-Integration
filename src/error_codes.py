"""
Structured Error Codes (#22)

Centralised error codes for the QA system. Using string constants
(not an Enum) so they can be directly embedded in JSON output.
"""


class ErrorCode:
    """Namespace for structured error code constants."""

    # --- LLM / Inference ---
    LLM_ALL_PROVIDERS_FAILED = "LLM_ALL_PROVIDERS_FAILED"
    LLM_VALIDATION_ERROR = "LLM_VALIDATION_ERROR"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_RATE_LIMITED = "LLM_RATE_LIMITED"

    # --- STT / Transcription ---
    STT_TIMEOUT = "STT_TIMEOUT"
    STT_QUOTA_EXCEEDED = "STT_QUOTA_EXCEEDED"
    STT_INVALID_AUDIO = "STT_INVALID_AUDIO"
    STT_API_ERROR = "STT_API_ERROR"

    # --- Pipeline ---
    PIPELINE_CIRCUIT_BREAKER = "PIPELINE_CIRCUIT_BREAKER"
    PIPELINE_DISK_SPACE = "PIPELINE_DISK_SPACE"
    PIPELINE_SHUTDOWN = "PIPELINE_SHUTDOWN"

    # --- Config / Environment ---
    CONFIG_MISSING = "CONFIG_MISSING"
    CONFIG_INVALID = "CONFIG_INVALID"
    ENV_MISSING = "ENV_MISSING"
    AUTH_FAILED = "AUTH_FAILED"

    # --- Data ---
    TRANSCRIPT_TOO_SHORT = "TRANSCRIPT_TOO_SHORT"
    TRANSCRIPT_EMPTY = "TRANSCRIPT_EMPTY"
    TRANSCRIPT_TOO_LARGE = "TRANSCRIPT_TOO_LARGE"
    CACHE_INVALID = "CACHE_INVALID"
