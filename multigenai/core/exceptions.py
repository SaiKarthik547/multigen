"""
Typed exception hierarchy for MultiGenAI OS.

Using typed exceptions instead of bare RuntimeError enables:
  - Fine-grained except clauses in engine code
  - Structured error logging with metadata
  - Accurate exit codes in CLI
  - Safe retry/fallback logic in orchestration
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class MGOSError(Exception):
    """Base class for all MultiGenAI OS exceptions."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.details: dict = details or {}

    def __str__(self) -> str:
        base = super().__str__()
        if self.details:
            return f"{base} | details={self.details}"
        return base


# ---------------------------------------------------------------------------
# Core / Infrastructure
# ---------------------------------------------------------------------------

class ConfigurationError(MGOSError):
    """Raised when configuration cannot be parsed or is invalid."""


class CapabilityError(MGOSError):
    """Raised when a required system capability is unavailable (e.g., CUDA missing)."""


class InsufficientVRAMError(CapabilityError):
    """Raised when the GPU does not have enough VRAM to load a model."""

    def __init__(self, required_gb: float, available_gb: float) -> None:
        super().__init__(
            f"Insufficient VRAM: need {required_gb:.1f} GB, have {available_gb:.1f} GB available.",
            details={"required_gb": required_gb, "available_gb": available_gb},
        )
        self.required_gb = required_gb
        self.available_gb = available_gb


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

class ModelLoadError(MGOSError):
    """Raised when a model fails to load from disk or HuggingFace Hub."""

    def __init__(self, model_id: str, reason: str) -> None:
        super().__init__(
            f"Failed to load model '{model_id}': {reason}",
            details={"model_id": model_id, "reason": reason},
        )
        self.model_id = model_id


class ModelNotFoundError(ModelLoadError):
    """Raised when a requested model ID is not in the registry."""


# ---------------------------------------------------------------------------
# Prompt / LLM
# ---------------------------------------------------------------------------

class InvalidPromptError(MGOSError):
    """Raised when a prompt fails schema validation."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(
            f"Invalid prompt — field '{field}': {reason}",
            details={"field": field, "reason": reason},
        )
        self.field = field


# ---------------------------------------------------------------------------
# Engine execution
# ---------------------------------------------------------------------------

class EngineExecutionError(MGOSError):
    """Raised when an engine (image, video, audio, …) fails during generation."""

    def __init__(self, engine: str, reason: str) -> None:
        super().__init__(
            f"Engine '{engine}' execution failed: {reason}",
            details={"engine": engine, "reason": reason},
        )
        self.engine = engine


class TemporalCoherenceError(EngineExecutionError):
    """Raised when video frame embedding similarity drops below threshold."""


class IdentityDriftError(EngineExecutionError):
    """Raised when face embedding cosine similarity drops below threshold."""

    def __init__(self, frame_idx: int, similarity: float, threshold: float) -> None:
        super().__init__(
            "video_engine",
            f"Identity drift at frame {frame_idx}: similarity={similarity:.3f} < threshold={threshold:.3f}",
        )
        self.frame_idx = frame_idx
        self.similarity = similarity
        self.threshold = threshold


# ---------------------------------------------------------------------------
# Memory / Storage
# ---------------------------------------------------------------------------

class MemoryError(MGOSError):  # noqa: A001 — intentional shadow for domain clarity
    """Raised when memory/storage operations fail (identity store, world state, …)."""


class EmbeddingStoreError(MGOSError):
    """Raised when vector embedding operations fail."""


class IdentityEncoderError(MGOSError):
    """
    Raised when face embedding extraction fails.

    Common causes:
      - insightface or onnxruntime not installed
      - No face detected in the reference image
      - ArcFace embedding returned empty
    """


# ---------------------------------------------------------------------------
# LLM Provider — scoped exception hierarchy
# ---------------------------------------------------------------------------

class ProviderUnavailableError(MGOSError):
    """
    Base: LLM backend is unreachable or returned an unrecoverable error.

    Catch this when you want to handle any provider failure generically.
    Use the subclasses for specific failure modes.
    """


class ProviderTimeoutError(ProviderUnavailableError):
    """
    Raised when the LLM provider does not respond within the configured timeout.

    Maps to: requests.Timeout, socket.timeout
    """

    def __init__(self, endpoint: str, timeout_seconds: float) -> None:
        super().__init__(
            f"LLM provider timed out after {timeout_seconds}s: {endpoint}",
            details={"endpoint": endpoint, "timeout_seconds": timeout_seconds},
        )
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds


class ProviderAuthError(ProviderUnavailableError):
    """
    Raised when the LLM provider returns HTTP 401 or 403.

    Usually means: missing or invalid API key in the MGOS_LLM_API_KEY env var.
    """

    def __init__(self, endpoint: str) -> None:
        super().__init__(
            f"LLM provider authentication failed: {endpoint}. "
            "Check your MGOS_LLM_API_KEY environment variable.",
            details={"endpoint": endpoint},
        )
        self.endpoint = endpoint


class ProviderResponseError(ProviderUnavailableError):
    """
    Raised when the LLM provider returns a non-auth HTTP error (4xx/5xx).

    Distinct from ProviderAuthError so retry logic can differentiate.
    """

    def __init__(self, endpoint: str, status_code: int, body: str = "") -> None:
        super().__init__(
            f"LLM provider HTTP {status_code} from {endpoint}",
            details={"endpoint": endpoint, "status_code": status_code, "body": body[:200]},
        )
        self.status_code = status_code


class ProviderResponseFormatError(ProviderUnavailableError):
    """
    Raised when the LLM response cannot be parsed as valid JSON or fails
    schema validation, even after the retry/fix cycle.

    Maps to: json.JSONDecodeError, pydantic.ValidationError (after retries)
    """
