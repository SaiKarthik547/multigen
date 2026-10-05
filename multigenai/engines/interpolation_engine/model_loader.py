"""
model_loader.py — RIFE IFNet weight loader for InterpolationEngine.

Loads the local RIFE model definition and weights (flownet.pkl) provided
in the multigenai/models/rife directory. This ensures the architecture
matches the specific trained weights used for video interpolation.

Lifecycle:
  load_rife_model(device) → (model | None, device)

On any failure, the function returns (None, device) and logs a warning;
InterpolationEngine degrades gracefully by returning the original frames unchanged.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

LOG = logging.getLogger(__name__)

# Local RIFE weights — flownet.pkl
# IFNet_2R.py: model definition (matches the local checkpoint architecture)
RIFE_REPO_ID   = "ZhiyuXu/RIFE"
RIFE_WEIGHT_FILE = "flownet.pkl"
RIFE_MODEL_FILE  = "IFNet_2R.py"


def load_rife_model(device: str) -> Tuple[Optional[object], str]:
    """
    Load the local RIFE IFNet_2R model.

    Returns
    -------
    (model, device) where model exposes::

        model.interpolate(img0: Tensor, img1: Tensor) -> Tensor

    On failure returns (None, device) — caller must handle None.
    """
    try:
        from multigenai.models.rife.rife_model import RIFEModel
        LOG.info(f"RIFE: Loading local IFNet_2R weights...")
        
        # Instantiate wrapper which loads rife47.pth
        model = RIFEModel(device=device)
        LOG.info(f"RIFE: Local model ready on device={device}")
        
        return model, device

    except Exception as exc:
        LOG.warning(f"RIFE: Could not load local model ({exc}). Interpolation will be skipped.")
        return None, device

