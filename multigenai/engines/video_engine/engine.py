"""
VideoEngine — Frame-sequence video generation with temporal consistency.

Phase 1: Migrated from MultiGenAi.py with ExecutionContext injection.
Phase 5: Latent-chained sequential img2img for inter-frame stability.
         Frame 0 is a full text2img denoise; frames 1+ are img2img passes
         conditioned on the previous frame (PIL-based, not true latent chaining).
         Deterministic via a single shared torch.Generator created once per run.
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

    Phase 5 temporal pipeline:
        Frame 0  → full SDXL text2img denoise (anchor frame)
        Frame i  → SDXL img2img pass, init=prev_frame, strength=temporal_strength
                   Optional: identity similarity guard retries (max 2 per frame)
        Generator → created once, never re-seeded — noise injected by img2img strength only.

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

        # --- Phase 5: Deterministic generator — created ONCE, never re-seeded ---
        generator = None
        try:
            import torch
            device = self._ctx.device
            generator = torch.Generator(device=device).manual_seed(seed)
            LOG.debug(f"Phase 5: Generator seeded with {seed} on device={device}")
        except ImportError:
            LOG.warning("torch not available — generator disabled; reproducibility reduced.")

        # --- Phase 5: Lock prompt — no frame index drift ---
        effective_prompt = request.prompt
        motion_hint = getattr(request, "motion_hint", "")
        if motion_hint:
            effective_prompt = f"{effective_prompt}, {motion_hint}"

        # Resolve per-run seed: use character persistent_seed if available
        frame_seed = seed
        if (
            profile is not None
            and request.seed is None
            and profile.persistent_seed is not None
        ):
            frame_seed = profile.persistent_seed

        # Build a single ImageGenerationRequest shared for all frames.
        # Prompt is locked; strength is controlled by run_from_previous_frame.
        img_req = ImageGenerationRequest(
            prompt=effective_prompt,
            negative_prompt=request.negative_prompt,
            character_id=request.character_id,
            scene_id=request.scene_id,
            style_id=request.style_id,
            width=request.width,
            height=request.height,
            seed=frame_seed,
            num_inference_steps=getattr(request, "num_inference_steps", 30),
            identity_name=getattr(request, "identity_name", None),
            identity_strength=getattr(request, "identity_strength", 0.8),
        )

        temporal_strength = getattr(request, "temporal_strength", 0.25)
        identity_threshold = getattr(request, "identity_threshold", 0.55)
        prev_image = None

        try:
            LOG.info(
                f"Phase 5: Generating {request.num_frames} frames "
                f"(seed={seed}, temporal_strength={temporal_strength})..."
            )

            for i in range(request.num_frames):

                if i == 0 or prev_image is None:
                    # --- Frame 0: full text2img denoise (anchor frame) ---
                    img_result = image_engine.run(img_req)
                    if not img_result.success:
                        raise EngineExecutionError("video_engine", "Frame 1 (anchor) generation failed.")

                    # Load the saved anchor image as PIL for the next frame
                    try:
                        from PIL import Image as PILImage
                        prev_image = PILImage.open(img_result.path).copy()
                    except Exception as pil_exc:
                        LOG.warning(f"Could not load anchor frame as PIL ({pil_exc}); Phase 5 chaining disabled.")
                        prev_image = None

                    frame_paths.append(img_result.path)
                    LOG.info(f"  Frame 1/{request.num_frames} done (anchor, full denoise).")

                else:
                    # --- Frame i>0: img2img conditioned on previous frame ---
                    # Optional noise injection via LatentPropagator
                    init_image = prev_image
                    if generator is not None:
                        try:
                            from multigenai.temporal.latent_propagator import LatentPropagator
                            init_image = LatentPropagator().inject_noise(
                                prev_image, temporal_strength, generator
                            )
                        except Exception as noise_exc:
                            LOG.debug(f"inject_noise skipped: {noise_exc}")

                    # Identity similarity guard (max 2 retries per frame)
                    current_strength = temporal_strength
                    attempts = 0
                    img_result = None
                    new_image = None

                    while attempts < 2:
                        img_result, new_image = image_engine.run_from_previous_frame(
                            img_req, init_image, current_strength, generator
                        )

                        if not img_result.success:
                            break  # OOM or other failure — accept prev_image

                        # Identity guard: only runs if profile has a face embedding
                        # and insightface is installed. Fails gracefully otherwise.
                        if (
                            profile is not None
                            and getattr(profile, "face_embedding", None)
                            and img_result.path
                        ):
                            try:
                                from multigenai.identity.face_encoder import FaceEncoder
                                from multigenai.control.consistency_enforcer import ConsistencyEnforcer

                                frame_emb = FaceEncoder().extract(img_result.path)
                                similarity = ConsistencyEnforcer().check_embedding_drift(
                                    frame_emb, profile.face_embedding
                                )
                                LOG.debug(
                                    f"  Frame {i+1}: identity similarity={similarity:.3f} "
                                    f"(threshold={identity_threshold})"
                                )

                                if similarity >= identity_threshold:
                                    break  # acceptable — use this frame

                                # Below threshold: reduce strength and retry
                                current_strength *= 0.8
                                LOG.debug(
                                    f"  Frame {i+1}: drift detected — retry "
                                    f"{attempts+1}/2 with strength={current_strength:.3f}"
                                )
                                attempts += 1

                            except Exception as guard_exc:
                                # insightface not installed, no face, or encoder failed
                                LOG.debug(f"  Identity guard skipped: {guard_exc}")
                                break
                        else:
                            break  # no embedding to check → accept frame

                    if attempts >= 2:
                        LOG.warning(
                            f"  Frame {i+1}: identity drift accepted after max retries "
                            f"(final strength={current_strength:.3f})."
                        )

                    if img_result is None or not img_result.success:
                        # OOM fallback: reuse previous frame path rather than hard-fail
                        LOG.warning(
                            f"  Frame {i+1}: img2img failed — repeating previous frame."
                        )
                        if frame_paths:
                            frame_paths.append(frame_paths[-1])
                        else:
                            raise EngineExecutionError("video_engine", f"Frame {i+1} failed with no fallback.")
                    else:
                        frame_paths.append(img_result.path)
                        if new_image is not None:
                            prev_image = new_image

                    LOG.info(f"  Frame {i+1}/{request.num_frames} done.")

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

            # --- Explicit MoviePy cleanup ---
            video.close()
            for clip in clips:
                clip.close()
            del video
            del clips

            # --- Phase 5: GPU memory cleanup ---
            del prev_image          # release PIL chain reference
            # NOTE: do NOT delete generator — it is garbage collected naturally
            try:
                import gc
                import torch
                torch.cuda.empty_cache()
                gc.collect()
            except ImportError:
                pass

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

    # Phase 5 hooks (promoted from stubs)
    def run_with_motion_module(self, request: "VideoGenerationRequest") -> VideoResult:
        """[Phase 6] AnimateDiff motion module injection."""
        raise NotImplementedError("Motion module activates in Phase 6.")

    def run_with_optical_flow(self, request: "VideoGenerationRequest") -> VideoResult:
        """[Phase 6] Optical-flow guided temporal smoothing."""
        raise NotImplementedError("Optical flow activates in Phase 6.")
