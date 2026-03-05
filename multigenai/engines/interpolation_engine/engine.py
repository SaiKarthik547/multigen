"""
InterpolationEngine — Phase 8 Temporal Enhancement.

Receives a list of PIL Images from VideoEngine (SVD-XT keyframes)
and returns an expanded list with RIFE-interpolated intermediate frames.

Strictly isolated from VideoEngine — no model sharing, no VRAM overlap.

Lifecycle contract (mirrors all other engines):
  _load_model() → interpolate() → _unload_model()

The engine degrades gracefully if RIFE weights are unavailable:
  - Logs a warning
  - Returns original frames unchanged
  - No crash, no exception propagated
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from multigenai.core.logging.logger import get_logger

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from multigenai.core.execution_context import ExecutionContext

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InterpolationResult:
    """Metadata about a completed interpolation pass."""
    frame_count_in: int
    frame_count_out: int
    factor: int
    success: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class InterpolationEngine:
    """
    RIFE-based frame interpolation engine.

    Receives SVD-XT PIL frames and inserts (factor-1) intermediate frames
    between each adjacent pair, producing a smoother, longer video sequence.

    Frame count formula:
        output = n + (n - 1) * (factor - 1)

    Example:
        16 frames, factor=2  →  16 + 15*1 = 31 frames
        16 frames, factor=3  →  16 + 15*2 = 46 frames
    """

    def __init__(self, ctx: "ExecutionContext") -> None:
        self._ctx = ctx
        self.device = ctx.device
        self._model = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """
        Lazy-load RIFE IFNet weights.
        Sets self._model to None if loading fails (graceful degradation).
        """
        if self._model is not None:
            return  # Model already cached
            
        from multigenai.engines.interpolation_engine.model_loader import load_rife_model
        self._model, self.device = load_rife_model(self.device)

    def _unload_model(self) -> None:
        """
        Release RIFE model weights and clear GPU/CPU memory.
        Follows same lifecycle as ImageEngine and VideoEngine.
        """
        if self._model is not None:
            del self._model
            self._model = None

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as exc:
            LOG.warning(f"RIFE: Error flushing CUDA memory: {exc}")

    def _interpolate_pair(
        self, f0: "PILImage", f1: "PILImage", factor: int
    ) -> List["PILImage"]:
        """
        Generate (factor - 1) intermediate frames between f0 and f1 using
        the local RIFE IFNet_2R API.

        The model's `.interpolate(img0, img1)` generates exactly the midpoint t=0.5.
        For factor=2, this implies one intermediate frame.
        For factor=3 or factor=4, recursive interpolation is used cleanly.

        Returns:
            List of (factor - 1) PIL Images (intermediate frames only).
        """
        import math
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image as PILImage
        from multigenai.models.rife.utils import image_to_tensor, tensor_to_image

        intermediates: List["PILImage"] = []

        orig_w, orig_h = f0.size
        # The user's provided `pad` recommendation: padding to multiple of 32
        pad_w = (math.ceil(orig_w / 32) * 32) - orig_w
        pad_h = (math.ceil(orig_h / 32) * 32) - orig_h

        t0 = image_to_tensor(f0).to(self.device)
        t1 = image_to_tensor(f1).to(self.device)

        # Apply reflection padding if needed
        if pad_w > 0 or pad_h > 0:
            # F.pad padding order: (left, right, top, bottom)
            padding = (0, pad_w, 0, pad_h)
            t0 = F.pad(t0, padding, mode='reflect')
            t1 = F.pad(t1, padding, mode='reflect')

        def _get_mid(tensor0, tensor1):
            mid = self._model.interpolate(tensor0, tensor1)
            # Crop back to original size
            return mid[:, :, :orig_h, :orig_w]

        if factor == 2:
            # Single mid frame
            mid_t = _get_mid(t0, t1)
            intermediates.append(tensor_to_image(mid_t))
        elif factor == 3:
            # mid1 halfway (t=0.5), mid2 halfway mid1->t1 (t=0.75)
            mid1_t = _get_mid(t0, t1)
            mid2_t = _get_mid(mid1_t, t1)
            intermediates.extend([
                tensor_to_image(mid1_t), 
                tensor_to_image(mid2_t)
            ])
        elif factor == 4:
            # Genuine recursive splitting (t=0.25, 0.5, 0.75)
            mid50_t = _get_mid(t0, t1)
            mid25_t = _get_mid(t0, mid50_t)
            mid75_t = _get_mid(mid50_t, t1)
            intermediates.extend([
                tensor_to_image(mid25_t),
                tensor_to_image(mid50_t),
                tensor_to_image(mid75_t)
            ])
        else:
            # Fallback for factor > 4
            for _ in range(1, factor):
                mid_t = _get_mid(t0, t1)
                intermediates.append(tensor_to_image(mid_t))

        return intermediates

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def interpolate(
        self, frames: List["PILImage"], factor: int
    ) -> List["PILImage"]:
        """
        Expand a list of SVD keyframes by inserting (factor-1) RIFE-generated
        intermediate frames between each adjacent pair.

        Args:
            frames:  List of PIL Images from VideoEngine
            factor:  Multiplication factor (1 = passthrough, 2 = double, etc.)

        Returns:
            Expanded list of PIL Images.
            On failure: original frames unchanged (graceful degradation).
        """
        if factor == 1 or len(frames) < 2:
            LOG.debug(f"InterpolationEngine: factor=1 or frames<2, passthrough.")
            return frames

        try:
            self._load_model()

            if self._model is None:
                LOG.warning("InterpolationEngine: RIFE model unavailable — returning original frames.")
                return frames

            LOG.info(
                f"InterpolationEngine: Interpolating {len(frames)} frames "
                f"with factor={factor}. "
                f"Expected output: {len(frames) + (len(frames) - 1) * (factor - 1)} frames."
            )

            expanded: List["PILImage"] = []
            for i in range(len(frames) - 1):
                expanded.append(frames[i])
                intermediates = self._interpolate_pair(frames[i], frames[i + 1], factor)
                expanded.extend(intermediates)

            expanded.append(frames[-1])  # always include the last keyframe

            LOG.info(
                f"InterpolationEngine: Complete. "
                f"{len(frames)} → {len(expanded)} frames."
            )
            return expanded

        except Exception as exc:
            LOG.warning(
                f"InterpolationEngine: Interpolation failed ({exc}). "
                f"Returning original frames."
            )
            return frames

        finally:
            if self._ctx.behaviour.auto_unload_after_gen:
                self._unload_model()
