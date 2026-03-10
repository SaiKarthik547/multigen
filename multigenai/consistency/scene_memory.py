from dataclasses import dataclass
from typing import Optional
from PIL import Image as PILImage
from multigenai.core.temporal_state import TemporalState

@dataclass
class SceneState:
    character_reference_path: Optional[str] = None
    reference_frame_path: Optional[str] = None
    latent_reference: Optional[str] = None  # Generic path storage for latency caches
    seed: Optional[int] = None
    environment_prompt: Optional[str] = None
    lighting_prompt: Optional[str] = None
    style_prompt: Optional[str] = None
    temporal_state: Optional[TemporalState] = None

class SceneMemory:
    """
    Holds persistent visual and thematic state across multi-segment generation runs.
    Ensures that Character Identity (IP-Adapter) and Scene Structure (ControlNet depth)
    can be passed cleanly between ImageEngine and VideoEngine passes.
    """
    def __init__(self):
        self._state = SceneState()

    def update(self, **kwargs):
        """Update scene memory with new visual state or prompts."""
        for k, v in kwargs.items():
            if hasattr(self._state, k):
                setattr(self._state, k, v)

    def get(self) -> SceneState:
        """Retrieve the current SceneState."""
        return self._state

    def reset(self):
        """Clear all scene memory for a new generation run."""
        self._state = SceneState()
