"""
model_loader.py — RIFE IFNet weight loader for InterpolationEngine.

Downloads the official RIFE IFNet_HDv3 weights and model definition from
HuggingFace (AlexWortega/RIFE).  The model definition is fetched alongside the
weights so we use the exact architecture the checkpoint expects — eliminating
channel-count mismatches from custom re-implementations.

Lifecycle:
  load_rife_model(device) → (model | None, device)

On any failure (missing network, bad HF token, torch absent, etc.) the function
returns (None, device) and logs a warning; InterpolationEngine degrades gracefully
by returning the original frames unchanged.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

LOG = logging.getLogger(__name__)

# Official RIFE PyTorch checkpoint — AlexWortega/RIFE on HuggingFace.
# flownet.pkl  : IFNet_HDv3 checkpoint weights
# IFNet_HDv3.py: model definition (fetched alongside weights to guarantee
#                architecture / weight compatibility)
RIFE_REPO_ID   = "AlexWortega/RIFE"
RIFE_WEIGHT_FILE = "flownet.pkl"
RIFE_MODEL_FILE  = "IFNet_HDv3.py"


def load_rife_model(device: str) -> Tuple[Optional[object], str]:
    """
    Download (once, then cached) and load the official RIFE IFNet_HDv3 model.

    Returns
    -------
    (model, device) where model exposes::

        model.inference(img0: Tensor, img1: Tensor, timestep: float) -> Tensor

    On failure returns (None, device) — caller must handle None.
    """
    try:
        import torch
        from huggingface_hub import hf_hub_download

        LOG.info(f"RIFE: Fetching weights + model definition from {RIFE_REPO_ID} ...")

        weight_path = hf_hub_download(repo_id=RIFE_REPO_ID, filename=RIFE_WEIGHT_FILE)
        model_src   = hf_hub_download(repo_id=RIFE_REPO_ID, filename=RIFE_MODEL_FILE)

        model = _load_ifnet_from_source(model_src, weight_path, device)
        LOG.info(f"RIFE: IFNet_HDv3 ready on device={device}")
        return model, device

    except ImportError as exc:
        LOG.warning(f"RIFE: Missing dependency ({exc}). Interpolation will be skipped.")
        return None, device
    except Exception as exc:
        LOG.warning(f"RIFE: Could not load model ({exc}). Interpolation will be skipped.")
        return None, device


def _load_ifnet_from_source(model_src: str, weight_path: str, device: str) -> object:
    """
    Dynamically import IFNet_HDv3 from the downloaded source file,
    load the checkpoint, and return an eval-mode model on the target device.

    Using dynamic import means we always run the exact architecture the
    checkpoint was trained with — no hand-rolled re-implementation needed.
    """
    import torch

    # ----------------------------------------------------------------
    # Dynamically import IFNet_HDv3.py into an isolated module namespace
    # so it doesn't pollute sys.modules and doesn't conflict if called
    # more than once (e.g. reloading after first failure).
    # ----------------------------------------------------------------
    module_name = "rife_ifnet_hdv3_dynamic"

    # Remove stale cached module so weights reload cleanly on retry
    sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(module_name, model_src)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create module spec from {model_src}")

    rife_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = rife_module

    try:
        spec.loader.exec_module(rife_module)   # type: ignore[attr-defined]
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise RuntimeError(f"Failed to exec IFNet_HDv3.py: {exc}") from exc

    if not hasattr(rife_module, "IFNet"):
        raise RuntimeError("IFNet_HDv3.py does not expose an 'IFNet' class.")

    # ----------------------------------------------------------------
    # Resolve device: DirectML → CPU (torch-dml has limited op support)
    # ----------------------------------------------------------------
    dev = torch.device(
        device if (torch.cuda.is_available() or device == "cpu") else "cpu"
    )

    model = rife_module.IFNet().to(dev)

    state = torch.load(weight_path, map_location=dev)

    # Some checkpoints wrap weights under a "model" key
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    model.load_state_dict(state, strict=False)
    model.eval()
    return model
