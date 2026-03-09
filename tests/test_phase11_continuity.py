import pytest
import torch
import numpy as np
from PIL import Image
from multigenai.core.temporal_state import TemporalState
from multigenai.engines.transition_engine.engine import TransitionEngine

def test_temporal_state_initialization():
    state = TemporalState(scene_index=1)
    assert state.scene_index == 1
    assert state.previous_frame is None
    # In Phase 12, previous_latent is no longer tracked externally for AnimateDiff
    assert state.previous_latent is None

def test_transition_engine_blending():
    frames_a = [Image.new("RGB", (32, 32), color=(i + 100, 0, 0)) for i in range(10)]
    frames_b = [Image.new("RGB", (32, 32), color=(0, i + 100, 0)) for i in range(10)]
    
    blended = TransitionEngine.blend(frames_a, frames_b, window=4)
    
    # original total: 20. window=4 means 4 frames are blended.
    # Result should be: frames_a[:-4] (6) + blended (4) + frames_b[window:] (6) = 16 frames
    assert len(blended) == 16
    
    # Check middle frame (blended)
    mid_img = np.array(blended[7]) # blended[1] which is the 8th frame (index 7)
    assert mid_img[0, 0, 0] > 0 # Some Red
    assert mid_img[0, 0, 1] > 0 # Some Green
