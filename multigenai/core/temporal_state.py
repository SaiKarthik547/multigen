from dataclasses import dataclass
from typing import Optional
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
    motion_field: Optional[np.ndarray] = None
    scene_index: int = 0
