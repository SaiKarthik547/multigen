"""
ControlNetManager — Phase 15 VRAM Guard Shim

ControlNet + DepthAnything has been retired from the active pipeline to
stay within the Kaggle T4 15 GB VRAM budget. The real implementations are
preserved in:
  legacy/models/controlnet/controlnet_manager_sdxl.py   (full SDXL ControlNet + DepthAnything)
  legacy/models/controlnet/controlnet_manager_stub.py   (Phase 4 stub)

This shim is a safe no-op on construction and raises a clear RuntimeError
if any method is actually invoked, making misconfiguration immediately obvious.
"""

from __future__ import annotations
from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)

_RETIRED_MSG = (
    "ControlNet + DepthAnything is retired (Phase 15 VRAM guard). "
    "Real implementation: legacy/models/controlnet/controlnet_manager_sdxl.py"
)


class ControlNetManager:
    """
    No-op shim for the retired ControlNet + DepthAnything integration.

    Instantiation is safe and free. Any call to load() or get_depth_map()
    raises RuntimeError immediately so misconfigured code surfaces fast.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self.controlnet = None          # attribute expected by image_engine.py
        self.depth_estimator = None
        self.feature_extractor = None
        LOG.debug("ControlNetManager: retired shim instantiated (VRAM guard).")

    def load(self) -> None:
        raise RuntimeError(_RETIRED_MSG)

    def get_depth_map(self, image):
        raise RuntimeError(_RETIRED_MSG)
