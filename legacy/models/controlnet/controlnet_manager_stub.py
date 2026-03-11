"""
ControlNetManager — Phase 4 stub.

Phase 4 will implement:
  - Depth map extraction + ControlNet depth conditioning
  - Segmentation conditioning
  - Lighting embedding injection
"""

from __future__ import annotations


class ControlNetManager:
    """
    Manages ControlNet model loading and conditioned image generation.

    Phase 4 implementation will support:
      - depth: MiDaS depth map → ControlNet depth
      - seg: Segmentation mask → ControlNet seg
      - pose: OpenPose keypoints → ControlNet pose
    """

    SUPPORTED_TYPES = ("depth", "seg", "pose", "canny")

    def apply(self, image, control_type: str = "depth"):
        """[Phase 4] Apply ControlNet conditioning to a generation pipeline."""
        raise NotImplementedError(f"ControlNet '{control_type}' activates in Phase 4.")
