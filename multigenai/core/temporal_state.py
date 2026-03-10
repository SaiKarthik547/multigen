from dataclasses import dataclass, field
from typing import Optional, List
import torch
import numpy as np
from PIL import Image as PILImage

@dataclass
class TemporalState:
    """
    Stores temporal information across scenes to maintain visual and motion continuity.
    """
    previous_frame: Optional[PILImage.Image] = None
    previous_latent: Optional[torch.Tensor] = None
    previous_frames: List[PILImage.Image] = field(default_factory=list)
    previous_latents: Optional[torch.Tensor] = None
    previous_seed: Optional[int] = None
    motion_field: Optional[np.ndarray] = None
    scene_index: int = 0
    identity_latent: Optional[torch.Tensor] = None
