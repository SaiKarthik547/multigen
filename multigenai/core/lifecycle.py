"""
LifecycleManager — Global startup and shutdown orchestration.

Responsibilities:
  1. Load config (settings.py)
  2. Configure logger
  3. Run and log CapabilityReport
  4. Register atexit shutdown hooks (CUDA cache clear, model unload)

Call `LifecycleManager.startup()` once at CLI / API entry point.
"""

from __future__ import annotations

import atexit
import pathlib
from typing import Optional

from multigenai.core.config.settings import Settings, get_settings
from multigenai.core.logging.logger import configure_logging, get_logger

LOG = get_logger(__name__)


class LifecycleManager:
    """
    Manages the complete MGOS application lifecycle.

    Usage:
        lm = LifecycleManager()
        lm.startup()           # call at process start
        # ... work happens ...
        lm.shutdown()          # called automatically via atexit
    """

    def __init__(self, config_path: Optional[pathlib.Path] = None) -> None:
        self._config_path = config_path
        self._settings: Optional[Settings] = None
        self._started = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def startup(self) -> Settings:
        """
        Run the full startup sequence. Idempotent — safe to call multiple times.

        Returns:
            The loaded Settings instance.
        """
        if self._started:
            return self._settings  # type: ignore[return-value]

        # 1. Load config
        self._settings = get_settings(self._config_path)

        # 2. Configure logger
        configure_logging(
            level=self._settings.log_level,
            mode=self._settings.log_mode,
            log_file=self._settings.log_file,
        )
        LOG.info("MultiGenAI OS starting up…")

        # 3. Ensure output directory exists
        out_dir = pathlib.Path(self._settings.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        LOG.debug(f"Output directory: {out_dir.resolve()}")

        # 4. Run capability report (info level — not printed to console unless DEBUG)
        try:
            from multigenai.core.capability_report import CapabilityReport
            cap = CapabilityReport()
            cap_data = cap.to_dict()
            cuda_ok = cap_data.get("cuda", {}).get("available", False)
            LOG.info(f"Capability check — CUDA: {'✔' if cuda_ok else '✘ (CPU mode)'}")
        except Exception as exc:
            LOG.warning(f"Capability report failed (non-fatal): {exc}")

        # 5. Register shutdown hooks
        atexit.register(self.shutdown)

        self._started = True
        LOG.info("Startup complete.")
        return self._settings

    def shutdown(self) -> None:
        """Run cleanup on process exit (called via atexit or manually)."""
        if not self._started:
            return
        LOG.info("MultiGenAI OS shutting down…")

        # Unload all registered models
        try:
            from multigenai.core.model_registry import ModelRegistry
            ModelRegistry.instance().unload_all()
        except Exception as exc:
            LOG.warning(f"Model unload error during shutdown: {exc}")

        # Clear CUDA cache
        try:
            from multigenai.core.device_manager import DeviceManager
            DeviceManager().clear_cache()
        except Exception:
            pass

        self._started = False
        LOG.info("Shutdown complete.")

    @property
    def settings(self) -> Settings:
        """Return current settings (raises if startup not called yet)."""
        if self._settings is None:
            raise RuntimeError("LifecycleManager.startup() must be called first.")
        return self._settings
