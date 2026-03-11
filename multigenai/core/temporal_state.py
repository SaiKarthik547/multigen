"""
TemporalState — Phase 15 Global Latent Graph.

Maintains full temporal trajectory across scenes:
  - global_latent:    carries structure from scene to scene
  - latent_velocity:  directional propagation vector (used in LatentPropagator)
  - previous_latent:  output of the last generated window
  - previous_frame:   PIL image for trajectory encoding
  - identity_latent:  character anchor geometry
  - scene_index:      monotonically increasing counter
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch
from PIL import Image as PILImage


@dataclass
class TemporalState:
    """
    Stores temporal information across scenes to maintain visual and motion continuity.

    Phase 15 additions:
        global_latent   — carries compressed scene structure between every scene.
        latent_velocity — directional propagation vector from LatentPropagator.
    """
    previous_frame: Optional[PILImage.Image] = None
    previous_latent: Optional[torch.Tensor] = None
    identity_latent: Optional[torch.Tensor] = None
    global_latent: Optional[torch.Tensor] = None
    latent_velocity: Optional[torch.Tensor] = None
    scene_index: int = 0

    def reset(self) -> None:
        """Full hard reset for a new generation run."""
        self.previous_frame = None
        self.previous_latent = None
        self.identity_latent = None
        self.global_latent = None
        self.latent_velocity = None
        self.scene_index = 0
