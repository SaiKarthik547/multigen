"""
IPAdapterManager — Phase 15 VRAM Guard Shim

IP-Adapter has been retired from the active pipeline to stay within
the Kaggle T4 15 GB VRAM budget. The real implementation is preserved
in legacy/models/ip_adapter/ip_adapter_manager.py for future use when
hardware allows.

This shim is a safe no-op. It will raise a clear error if anyone tries
to enable IP-Adapter via config (enable_ip_adapter: true) so the
problem is immediately obvious rather than silently failing.
"""

from __future__ import annotations
from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)

_RETIRED_MSG = (
    "IPAdapter is retired (Phase 15 VRAM guard). "
    "Set enable_ip_adapter: false in config. "
    "Real implementation: legacy/models/ip_adapter/ip_adapter_manager.py"
)


class IPAdapterManager:
    """
    No-op shim for the retired IP-Adapter integration.

    The load() and apply() methods are safe no-ops so that engines
    instantiating this class with enable_ip_adapter=False continue
    to work without any changes. If actually called (i.e., the flag
    is True), raises RuntimeError immediately to surface the misconfiguration.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self.adapter_loaded = False
        LOG.debug("IPAdapterManager: retired shim instantiated (VRAM guard).")

    def load(self, pipe, model_type: str = "sdxl") -> None:
        raise RuntimeError(_RETIRED_MSG)

    def apply(self, pipe, reference_image) -> dict:
        raise RuntimeError(_RETIRED_MSG)
