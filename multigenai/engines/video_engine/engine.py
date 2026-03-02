"""
VideoEngine — Frame-sequence video generation with temporal consistency.

Phase 1: Migrated from MultiGenAi.py with ExecutionContext injection.
Phase 5: Will add motion modules, optical flow, latent propagation, and
         per-frame identity drift detection.
"""

from __future__ import annotations

import os
import pathlib
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from multigenai.core.exceptions import EngineExecutionError
from multigenai.core.logging.logger import get_logger

if TYPE_CHECKING:
    from multigenai.core.execution_context import ExecutionContext
    from multigenai.llm.schema_validator import VideoGenerationRequest

LOG = get_logger(__name__)


@dataclass
class VideoResult:
    """Output from the VideoEngine."""
    path: str
    frame_count: int
    fps: int
    seed: int
    success: bool = True
    error: Optional[str] = None


class VideoEngine:
    """
    Generates video by producing keyframes with ImageEngine and stitching them.

    Usage:
        engine = VideoEngine(ctx)
        result = engine.run(VideoGenerationRequest(prompt="a stormy sea at night"))
    """

    def __init__(self, ctx: "ExecutionContext") -> None:
        self._ctx = ctx
        self._out_dir = pathlib.Path(ctx.settings.output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def run(self, request: "VideoGenerationRequest") -> VideoResult:
        """Generate a video from the validated request."""
        from multigenai.engines.image_engine.engine import ImageEngine, _slug
        from multigenai.llm.schema_validator import ImageGenerationRequest

        seed = request.seed if request.seed is not None else random.randint(0, 1_000_000)
        out_path = self._out_dir / f"{_slug(request.prompt)}.mp4"

        # Check MoviePy availability
        try:
            from moviepy.editor import ImageClip, concatenate_videoclips
        except ImportError:
            msg = "MoviePy not installed — install via: pip install 'multigenai[video]'"
            LOG.error(msg)
            return VideoResult(path="", frame_count=0, fps=request.fps, seed=seed, success=False, error=msg)

        image_engine = ImageEngine(self._ctx)
        frame_paths: List[str] = []

        # Resolve identity profile once before the frame loop to avoid
        # repeated disk IO (IdentityStore reads from disk on every call).
        profile = None
        if getattr(request, "identity_name", None):
            from multigenai.memory.identity_store import IdentityStore
            store = IdentityStore(self._ctx.settings.output_dir)
            profile = store.get_profile(request.identity_name)

        try:
            LOG.info(f"Generating {request.num_frames} frames (seed={seed})...")
            for i in range(request.num_frames):
                frame_prompt = f"{request.prompt}, frame {i + 1} of {request.num_frames}"

                # Resolve per-frame seed: use character persistent_seed if available
                frame_seed = seed
                if (
                    profile is not None
                    and request.seed is None
                    and profile.persistent_seed is not None
                ):
                    frame_seed = profile.persistent_seed

                img_req = ImageGenerationRequest(
                    prompt=frame_prompt,
                    negative_prompt=request.negative_prompt,
                    character_id=request.character_id,
                    scene_id=request.scene_id,
                    style_id=request.style_id,
                    width=request.width,
                    height=request.height,
                    seed=frame_seed,
                    # --- Identity propagation: same conditioning on every frame ---
                    identity_name=getattr(request, "identity_name", None),
                    identity_strength=getattr(request, "identity_strength", 0.8),
                )
                img_result = image_engine.run(img_req)
                if not img_result.success:
                    raise EngineExecutionError("video_engine", f"Frame {i + 1} generation failed.")
                frame_paths.append(img_result.path)
                LOG.info(f"  Frame {i + 1}/{request.num_frames} done.")


            LOG.info(f"Stitching {len(frame_paths)} frames at {request.fps} fps...")
            clips = [ImageClip(fp).set_duration(request.frame_duration) for fp in frame_paths]
            video = concatenate_videoclips(clips, method="compose")

            video.write_videofile(
                str(out_path),
                fps=request.fps,
                codec="libx264",
                audio=False,                 # CRITICAL for Kaggle
                preset="ultrafast",          # Faster encode, less RAM
                threads=2,
                verbose=False,
                logger=None,
            )

            # --- IMPORTANT: Explicit cleanup ---
            video.close()
            for clip in clips:
                clip.close()

            del video
            del clips

            import torch
            import gc
            torch.cuda.empty_cache()
            gc.collect()

            LOG.info(f"Video saved: {out_path}")
            return VideoResult(
                path=str(out_path),
                frame_count=len(frame_paths),
                fps=request.fps,
                seed=seed
            )

        except EngineExecutionError:
            raise
        except Exception as exc:
            LOG.error(f"Video engine error: {exc}")
            return VideoResult(path="", frame_count=0, fps=request.fps, seed=seed, success=False, error=str(exc))
        finally:
            for fp in frame_paths:
                try:
                    os.remove(fp)
                except OSError:
                    pass

    # Phase 5 hooks
    def run_with_motion_module(self, request: "VideoGenerationRequest") -> VideoResult:
        """[Phase 5] AnimateDiff motion module injection."""
        raise NotImplementedError("Motion module activates in Phase 5.")

    def run_with_optical_flow(self, request: "VideoGenerationRequest") -> VideoResult:
        """[Phase 5] Optical-flow guided temporal smoothing."""
        raise NotImplementedError("Optical flow activates in Phase 5.")
