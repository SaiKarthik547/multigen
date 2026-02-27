"""
ModelRegistry — Lazy model loading and lifecycle management.

Design principles:
  - Models are NEVER loaded at import time (Kaggle OOM safety)
  - First `get()` call triggers loading
  - VRAM check before loading (raises InsufficientVRAMError if needed)
  - Explicit `unload()` releases GPU memory
  - Singleton pattern: one registry per process
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from multigenai.core.exceptions import InsufficientVRAMError, ModelLoadError, ModelNotFoundError
from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)

# SD1.5 fallback when SDXL cannot fit
_SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
_SD15_MODEL_ID = "runwayml/stable-diffusion-v1-5"
_SDXL_VRAM_THRESHOLD_GB = 8.0


@dataclass
class ModelEntry:
    """Metadata + reference for a registered model."""
    model_id: str
    loader: Callable[[], Any]          # Callable that loads and returns the model
    min_vram_gb: float = 0.0           # Minimum VRAM required (0 = CPU-only safe)
    instance: Optional[Any] = field(default=None, repr=False)
    loaded: bool = False
    # --- Usage tracking ---
    last_used_ts: float = field(default=0.0)          # time.time() of last get()
    load_count: int = field(default=0)                # total times loaded
    total_runtime_seconds: float = field(default=0.0) # cumulative inference time
    estimated_vram_mb: int = field(default=0)         # observed peak VRAM (updated externally)


class ModelRegistry:
    """
    Thread-safe, lazy-loading model registry.

    Usage:
        registry = ModelRegistry.instance()
        registry.register("sdxl", loader=lambda: load_sdxl(), min_vram_gb=6.0)
        model = registry.get("sdxl")   # loads on first call
        registry.unload("sdxl")        # releases memory
    """

    _singleton: Optional["ModelRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._models: Dict[str, ModelEntry] = {}
        self._model_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------

    @classmethod
    def instance(cls) -> "ModelRegistry":
        """Return the global singleton ModelRegistry."""
        if cls._singleton is None:
            with cls._lock:
                if cls._singleton is None:
                    cls._singleton = cls()
        return cls._singleton

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def register(
        self,
        model_id: str,
        loader: Callable[[], Any],
        min_vram_gb: float = 0.0,
    ) -> None:
        """
        Register a model loader without loading it yet.

        Args:
            model_id:    Unique identifier (e.g. "sdxl-base", "whisper-large").
            loader:      Zero-argument callable that returns the loaded model.
            min_vram_gb: Minimum VRAM required in GB. 0 = no VRAM requirement.
        """
        with self._model_lock:
            self._models[model_id] = ModelEntry(
                model_id=model_id,
                loader=loader,
                min_vram_gb=min_vram_gb,
            )
            LOG.debug(f"Registered model '{model_id}' (min_vram={min_vram_gb} GB, lazy).")

    def get(self, model_id: str, device_manager=None, environment=None) -> Any:
        """
        Return the loaded model instance, loading it on first access.

        Performs a VRAM pre-check against environment.vram_mb (not just
        device_manager.get_vram_info()) so behaviour is consistent whether
        or not torch is fully initialised.

        Soft downgrade: if model_id is the SDXL model and environment
        reports VRAM < 8 GB, automatically loads SD1.5 instead and logs
        a warning.

        Args:
            model_id:       Registered model identifier.
            device_manager: Optional DeviceManager for live VRAM query.
            environment:    Optional EnvironmentProfile for VRAM-aware guard.

        Raises:
            ModelNotFoundError:   model_id not registered (and fallback not registered).
            InsufficientVRAMError: not enough VRAM even after downgrade.
            ModelLoadError:       loader raised an exception.
        """
        with self._model_lock:
            # --- Soft downgrade: SDXL → SD1.5 when VRAM is too low ---
            if (
                model_id == _SDXL_MODEL_ID
                and environment is not None
                and 0 < environment.vram_mb < int(_SDXL_VRAM_THRESHOLD_GB * 1024)
            ):
                LOG.warning(
                    f"⚠ Downgrading to SD1.5 due to VRAM constraints "
                    f"({environment.vram_mb} MB < {int(_SDXL_VRAM_THRESHOLD_GB * 1024)} MB required). "
                    f"Using '{_SD15_MODEL_ID}' instead."
                )
                model_id = _SD15_MODEL_ID

            if model_id not in self._models:
                raise ModelNotFoundError(model_id, "not registered in ModelRegistry")

            entry = self._models[model_id]
            if entry.loaded:
                entry.last_used_ts = time.time()
                return entry.instance

            # --- VRAM pre-check (environment-level) ---
            if environment is not None and entry.min_vram_gb > 0:
                required_mb = int(entry.min_vram_gb * 1024)
                if 0 < environment.vram_mb < required_mb * 0.8:
                    raise InsufficientVRAMError(
                        required_gb=entry.min_vram_gb,
                        available_gb=environment.vram_mb / 1024,
                    )

            # --- Fallback to live DeviceManager VRAM query ---
            if device_manager and entry.min_vram_gb > 0:
                vram = device_manager.get_vram_info()
                if vram and vram.free_gb < entry.min_vram_gb:
                    raise InsufficientVRAMError(
                        required_gb=entry.min_vram_gb,
                        available_gb=vram.free_gb,
                    )

            LOG.info(f"Loading model '{model_id}'...")
            try:
                entry.instance = entry.loader()
                entry.loaded = True
                entry.load_count += 1
                LOG.info(f"Model '{model_id}' loaded successfully.")
            except InsufficientVRAMError:
                raise
            except Exception as exc:
                raise ModelLoadError(model_id, str(exc)) from exc

            entry.last_used_ts = time.time()
            return entry.instance

    def unload(self, model_id: str) -> None:
        """
        Unload a model and free its memory.

        Args:
            model_id: Registered model identifier.
        """
        with self._model_lock:
            if model_id not in self._models:
                raise ModelNotFoundError(model_id, "not registered")
            entry = self._models[model_id]
            if not entry.loaded:
                return
            entry.instance = None
            entry.loaded = False
            LOG.info(f"Model '{model_id}' unloaded.")

    def is_loaded(self, model_id: str) -> bool:
        """Return True if the model is currently loaded."""
        return self._models.get(model_id, ModelEntry("", lambda: None)).loaded

    def list_registered(self) -> Dict[str, bool]:
        """Return {model_id: is_loaded} for all registered models."""
        with self._model_lock:
            return {mid: e.loaded for mid, e in self._models.items()}

    def unload_all(self) -> None:
        """Unload every loaded model (called on shutdown or Kaggle post-generation)."""
        with self._model_lock:
            for mid, entry in self._models.items():
                if entry.loaded:
                    entry.instance = None
                    entry.loaded = False
                    LOG.info(f"Model '{mid}' unloaded during shutdown.")

    def update_runtime(self, model_id: str, duration_seconds: float, peak_vram_mb: int = 0) -> None:
        """
        Record post-generation stats (called by engines after each inference).

        Args:
            model_id:         Registered model identifier.
            duration_seconds: Wall-clock generation time.
            peak_vram_mb:     Peak VRAM observed during this generation.
        """
        with self._model_lock:
            if model_id in self._models:
                entry = self._models[model_id]
                entry.total_runtime_seconds += duration_seconds
                if peak_vram_mb > entry.estimated_vram_mb:
                    entry.estimated_vram_mb = peak_vram_mb

    def registry_summary(self) -> Dict[str, dict]:
        """
        Return a snapshot of all registered models and their usage stats.

        Returns:
            {model_id: {loaded, load_count, total_runtime_s, estimated_vram_mb, last_used_ts}}
        """
        with self._model_lock:
            return {
                mid: {
                    "loaded": e.loaded,
                    "load_count": e.load_count,
                    "total_runtime_s": round(e.total_runtime_seconds, 2),
                    "estimated_vram_mb": e.estimated_vram_mb,
                    "last_used_ts": e.last_used_ts,
                }
                for mid, e in self._models.items()
            }

