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
        Generate (factor - 1) intermediate frames between f0 and f1.

        Returns:
            List of (factor - 1) PIL Images (the intermediate frames only,
            NOT including f0 or f1).
        """
        import torch
        import torch.nn.functional as F
        import numpy as np
        from PIL import Image as PILImage

        intermediates = []

        # Convert PIL → float32 tensor [1, C, H, W] in [0, 1]
        def to_tensor(img: "PILImage") -> "torch.Tensor":
            arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            return t.to(self.device if self.device != "directml" else "cpu")

        def to_pil(t: "torch.Tensor") -> "PILImage":
            arr = (t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255)
            return PILImage.fromarray(arr.astype(np.uint8))

        t0 = to_tensor(f0)
        t1 = to_tensor(f1)

        for i in range(1, factor):
            timestep = i / factor
            x = torch.cat([t0, t1, torch.zeros_like(t0)], dim=1)
            with torch.no_grad():
                mid = self._model(x)
            intermediates.append(to_pil(mid))

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
            self._unload_model()
