"""
LifecycleManager — Global startup and shutdown orchestration.

Responsibilities:
  1. Load .env file (HF_TOKEN, optional LLM keys)
  2. Load config (settings.py)
  3. Configure logger
  4. Authenticate Hugging Face Hub (HF_TOKEN)
  5. Run and log CapabilityReport
  6. Register atexit shutdown hooks (CUDA cache clear, model unload)

Call `LifecycleManager.startup()` once at CLI / API entry point.
"""

from __future__ import annotations

import atexit
import os
import pathlib
from typing import Optional

from multigenai.core.config.settings import Settings, get_settings
from multigenai.core.logging.logger import configure_logging, get_logger

LOG = get_logger(__name__)

# Root of the project (two levels up from this file: core/ → multigenai/ → root)
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DOTENV_PATH = _PROJECT_ROOT / ".env"


def _load_dotenv() -> None:
    """Load .env from the project root into os.environ (silently skips if absent)."""
    try:
        from dotenv import load_dotenv
        loaded = load_dotenv(dotenv_path=_DOTENV_PATH, override=False)
        if loaded:
            LOG.info(f".env loaded from {_DOTENV_PATH}")
    except ImportError:
        # python-dotenv not installed — try manual fallback
        if _DOTENV_PATH.exists():
            with _DOTENV_PATH.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
            LOG.info(f".env loaded manually from {_DOTENV_PATH} (python-dotenv not installed)")


def _login_huggingface() -> None:
    """Authenticate with Hugging Face Hub using HF_TOKEN from environment."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token or token == "your_hf_token_here":
        LOG.warning(
            "HF_TOKEN not set. Add it to your .env file to enable authenticated "
            "HF Hub access (higher rate limits and gated model downloads)."
        )
        return
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        LOG.info("✅ Hugging Face Hub: authenticated successfully.")
    except Exception as exc:
        LOG.warning(f"HF Hub login failed (non-fatal): {exc}")


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
    _process_started = False
    _global_settings: Optional[Settings] = None

    def __init__(self, config_path: Optional[pathlib.Path] = None) -> None:
        self._config_path = config_path

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def startup(self) -> Settings:
        """
        Run the full startup sequence. Global per-process idempotency.

        Returns:
            The loaded Settings instance.
        """
        if LifecycleManager._process_started:
            return LifecycleManager._global_settings  # type: ignore[return-value]

        # 1. Load .env (must be before get_settings so env vars are available)
        _load_dotenv()

        # 2. Authenticate Hugging Face Hub IMMEDIATELY after loading env
        # This prevents "unauthenticated request" warnings if subsequent 
        # imports (like CapabilityReport or ModelRegistry) trigger HF Hub lookups.
        _login_huggingface()

        # 3. Load config
        LifecycleManager._global_settings = get_settings(self._config_path)

        # 4. Configure logger
        configure_logging(
            level=LifecycleManager._global_settings.log_level,
            mode=LifecycleManager._global_settings.log_mode,
            log_file=LifecycleManager._global_settings.log_file,
        )
        LOG.info("MultiGenAI OS starting up…")

        # 5. Ensure output directory exists
        out_dir = pathlib.Path(LifecycleManager._global_settings.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        LOG.debug(f"Output directory: {out_dir.resolve()}")

        # 6. Run capability report
        try:
            from multigenai.core.capability_report import CapabilityReport
            cap = CapabilityReport()
            cap_data = cap.to_dict()
            cuda_ok = cap_data.get("cuda", {}).get("available", False)
            LOG.info(f"Capability check — CUDA: {'✔' if cuda_ok else '✘ (CPU mode)'}")
        except Exception as exc:
            LOG.warning(f"Capability report failed (non-fatal): {exc}")

        # 7. Register shutdown hooks
        atexit.register(self.shutdown)

        LifecycleManager._process_started = True
        LOG.info("Startup complete.")
        return LifecycleManager._global_settings

    def shutdown(self) -> None:
        """Run cleanup on process exit (called via atexit or manually)."""
        if not LifecycleManager._process_started:
            return
            
        def safe_log(msg: str):
            try:
                LOG.info(msg)
            except Exception:
                pass

        safe_log("MultiGenAI OS shutting down…")

        # Unload all registered models
        try:
            from multigenai.core.model_registry import ModelRegistry
            ModelRegistry.instance().unload_all()
        except Exception as exc:
            try:
                LOG.warning(f"Model unload error during shutdown: {exc}")
            except Exception:
                pass

        # Clear CUDA cache
        try:
            from multigenai.core.device_manager import DeviceManager
            DeviceManager().clear_cache()
        except Exception:
            pass

        LifecycleManager._process_started = False
        safe_log("Shutdown complete.")

    @property
    def settings(self) -> Settings:
        """Return current settings (raises if startup not called yet)."""
        if LifecycleManager._global_settings is None:
            raise RuntimeError("LifecycleManager.startup() must be called first.")
        return LifecycleManager._global_settings
