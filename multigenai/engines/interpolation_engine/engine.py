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
from typing import TYPE_CHECKING, List, Optional, Union

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
        self, frames: List[Union["PILImage", str]], factor: int = 2, base_fps: int = 8
    ) -> List[str]:
        """
        Expand keyframes by inserting intermediate frames.
        Works in disk-streaming mode: accepts paths, returns paths.
        
        Phase 12: Dynamic interpolation to eliminate blur.
        """
        import pathlib
        from PIL import Image
        
        # Phase 12 Fix: Interpolation applied unnecessarily if base motion is sufficient
        # Only interpolate if input FPS is below the cinematic threshold (12)
        resolved_factor = factor
        if factor > 1:
             if base_fps < 12:
                 resolved_factor = max(factor, 2)
                 LOG.info(f"InterpolationEngine: Low FPS ({base_fps}) detected. Enforcing factor={resolved_factor}.")
             else:
                 resolved_factor = 1
                 LOG.info(f"InterpolationEngine: Base FPS ({base_fps}) sufficient. Bypassing (factor=1).")
        
        if resolved_factor == 1 or len(frames) < 2:
            return [str(f) if isinstance(f, (str, pathlib.Path)) else f for f in frames]
            
        factor = resolved_factor

        try:
            self._load_model()
            if self._model is None:
                return [str(f) if isinstance(f, (str, pathlib.Path)) else f for f in frames]

            # Output cache dir for interpolated run
            # Use the parent of the first frame if it's a path
            if isinstance(frames[0], (str, pathlib.Path)):
                base_dir = pathlib.Path(frames[0]).parent.parent
            else:
                base_dir = self._ctx.output_dir / "temp"
                
            out_cache = base_dir / ".interpolated_frames"
            out_cache.mkdir(parents=True, exist_ok=True)
            
            expanded_paths: List[str] = []
            frame_idx = 0

            LOG.info(f"InterpolationEngine: Processing {len(frames)} keyframes (factor={factor}).")

            for i in range(len(frames) - 1):
                # Load Pair (Handle both PIL and Path)
                source0, source1 = frames[i], frames[i+1]
                
                # Context manager for f0
                if isinstance(source0, (str, pathlib.Path)):
                    img0_ctx = Image.open(source0)
                else:
                    img0_ctx = source0 # Already an Image
                
                # Context manager for f1
                if isinstance(source1, (str, pathlib.Path)):
                    img1_ctx = Image.open(source1)
                else:
                    img1_ctx = source1
                    
                try:
                    f0 = img0_ctx.convert("RGB")
                    f1 = img1_ctx.convert("RGB")
                    
                    # 1. Save F0
                    out_f0 = out_cache / f"frame_{frame_idx:04d}.png"
                    f0.save(out_f0)
                    expanded_paths.append(str(out_f0))
                    frame_idx += 1
                    
                    # 2. Interpolate
                    intermediates = self._interpolate_pair(f0, f1, factor)
                    for interf in intermediates:
                        out_f = out_cache / f"frame_{frame_idx:04d}.png"
                        interf.save(out_f)
                        expanded_paths.append(str(out_f))
                        frame_idx += 1
                finally:
                    # Only close if we opened it
                    if isinstance(source0, (str, pathlib.Path)):
                        img0_ctx.close()
                    if isinstance(source1, (str, pathlib.Path)):
                        img1_ctx.close()
                
            # Save Last Frame (Handle both)
            source_last = frames[-1]
            if isinstance(source_last, (str, pathlib.Path)):
                with Image.open(source_last) as img_last:
                    out_last = out_cache / f"frame_{frame_idx:04d}.png"
                    img_last.convert("RGB").save(out_last)
            else:
                out_last = out_cache / f"frame_{frame_idx:04d}.png"
                source_last.convert("RGB").save(out_last)
            
            expanded_paths.append(str(out_last))

            LOG.info(f"InterpolationEngine: Complete. {len(frames)} -> {len(expanded_paths)} frames.")
            return expanded_paths

        except Exception as exc:
            LOG.warning(f"InterpolationEngine: Failed ({exc}) - returning original paths.")
            return [str(f) if isinstance(f, (str, pathlib.Path)) else f for f in frames]

        finally:
            if self._ctx.behaviour.auto_unload_after_gen:
                self._unload_model()
