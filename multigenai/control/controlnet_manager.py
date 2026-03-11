"""
ControlNetManager — Phase 15 Retired Stub.

Original Phase 4 stub moved to legacy/models/controlnet/controlnet_manager_stub.py.
This file kept here to preserve the control.__init__ import chain without crashing.
"""

from __future__ import annotations


class ControlNetManager:
    """Retired Phase 4 stub. See legacy/models/controlnet/."""

    SUPPORTED_TYPES = ("depth", "seg", "pose", "canny")

    def apply(self, image, control_type: str = "depth"):
        raise NotImplementedError(
            "ControlNet is retired (Phase 15 VRAM guard). "
            "See legacy/models/controlnet/controlnet_manager_stub.py"
        )
