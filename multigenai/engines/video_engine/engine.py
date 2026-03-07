"""
VideoEngine — True Temporal Video Generation using Stable Video Diffusion (SVD-XT).

Phase 6: Replaces the Phase 5 (Iterative SDXL img2img) hack with a pristine SVD-XT pipeline.
- SVD-XT is loaded lazily, bypassing the global ModelRegistry.
- Uses `enable_sequential_cpu_offload()` for memory efficiency on Kaggle.
- Direct `ffmpeg` byte-piping replaces slow `moviepy` dependencies.
- Strict unload lifecycle (`gc.collect()`, `empty_cache()`, `ipc_collect()`) guarantees VRAM
  is entirely cleared after generation, avoiding collisions with ImageEngine (SDXL).
"""

from __future__ import annotations

import gc
import os
import pathlib
import subprocess
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from multigenai.core.exceptions import EngineExecutionError
from multigenai.core.logging.logger import get_logger

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from multigenai.core.execution_context import ExecutionContext
    from multigenai.llm.schema_validator import VideoGenerationRequest

LOG = get_logger(__name__)

# Phase 6 Model Selection
SVD_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"


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
    True Temporal Video Generation Engine.
    
    Generates [B, T, C, H, W] latent frames from a single conditioning image in one
    forward pass using SVD-XT, encodes directly using ffmpeg, and drops all weights
    immediately from VRAM to prevent pipeline collisions.
    """

    def __init__(self, ctx: "ExecutionContext") -> None:
        self._ctx = ctx
        self._out_dir = pathlib.Path(ctx.settings.output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self.device = ctx.device
        self.pipe = None

    def _load_model(self) -> None:
        """
        Lazily loads SVD-XT directly.
        Diffusion models bypass ModelRegistry by design (Phase 7 isolation model).
        Uses sequential CPU offloading to prevent extreme memory spikes.
        """
        import torch
        from multigenai.models.temporal_svd_pipeline import TemporalStableVideoDiffusionPipeline

        # Phase 6: Core SVD-XT model loading (fp16 for speed and VRAM economy)
        # SVD-XT (25 frames) is used rather than SVD (14 frames) for better temporal depth.
        try:
            self.pipe = TemporalStableVideoDiffusionPipeline.from_pretrained(
                SVD_MODEL_ID,
                torch_dtype=torch.float16,
                variant="fp16"
            ).to(self.device)
        except Exception as e:
            LOG.error(f"Failed to load SVD pipeline: {e}")
            raise

        # Memory optimizations:
        #   sequential_cpu_offload: UNet/VAE/image_encoder move to GPU only when used
        if self.device == "cuda":
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)

    def _unload_model(self) -> None:
        """
        Forcefully unloads the SVD-XT pipeline from system and GPU memory.
        This must be called immediately after encoding to keep the environment
        sterile for incoming ImageEngine (SDXL) processes.
        """
        LOG.debug("Unloading SVD-XT and clearing VRAM...")
        if self.pipe:
            del self.pipe
            self.pipe = None

        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                LOG.debug("CUDA memory completely flushed (including IPC handles).")
        except Exception as exc:
            LOG.warning(f"Error flushing CUDA memory: {exc}")

    def _generate_video(
        self,
        request: "VideoGenerationRequest",
        conditioning_image: "PILImage",
        seed: int,
        num_frames: int,
        previous_latent: Optional[torch.Tensor] = None,
    ) -> tuple[List["PILImage"], torch.Tensor]:
        """
        Executes the SVD-XT forward pass, returning a list of PIL Images and the final latents.

        Args:
            num_frames: Effective frame count to generate.
            previous_latent: Optional latent tensor from a previous scene to maintain continuity.
        """
        import torch

        # CPU generators: single RNG stream for the unified pass
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        # Map temporal_strength (0.0–1.0) to native SVD motion_bucket_id (0–255)
        motion_bucket = max(0, min(255, int(request.temporal_strength * 255)))
        
        LOG.info(
            f"Phase 11: Generating {num_frames} frames via SVD-XT (Single-Pass Optimization). "
            f"Seed={seed}, resolution={request.width}x{request.height}, "
            f"steps={request.num_inference_steps}, motion_bucket_id={motion_bucket}."
        )

        # Safety Guard: Falls back if latent dimensions mismatch (e.g. frame count change)
        if previous_latent is not None:
            # SVD-XT diffusion latents shape: [batch, frames, channels, height/8, width/8]
            if previous_latent.ndim != 5:
                LOG.warning(f"Invalid latent rank: expected 5D, got {previous_latent.ndim}D. Resetting temporal state.")
                previous_latent = None
            elif abs(previous_latent.shape[1] - num_frames) > 1:
                LOG.warning(f"Latent frame mismatch: expected {num_frames} frames, got {previous_latent.shape[1]}. Resetting temporal state.")
                previous_latent = None
            else:
                # Spatial dimension guard: ensure height/width match (H/8, W/8)
                expected_h, expected_w = request.height // 8, request.width // 8
                if previous_latent.shape[-2:] != (expected_h, expected_w):
                    LOG.warning(
                        f"Latent spatial mismatch: expected {(expected_h, expected_w)}, "
                        f"got {previous_latent.shape[-2:]}. Resetting temporal state."
                    )
                    previous_latent = None

        if previous_latent is not None:
            # CRITICAL: ensure latent is on the correct device before injection
            previous_latent = previous_latent.to(self.device)
            LOG.info(f"Temporal latent tensor input shape: {previous_latent.shape} (device={previous_latent.device})")

        # Unified single-pass: Generate both frames and propagate-friendly latents
        # We use return_latents=True to get the 5D diffusion tensor directly from the pass
        result, final_latent = self.pipe(
            image=conditioning_image,
            num_frames=num_frames,
            num_inference_steps=request.num_inference_steps,
            height=request.height,
            width=request.width,
            generator=generator,
            motion_bucket_id=motion_bucket,
            decode_chunk_size=2,
            output_type="pil",
            latents=previous_latent,
            return_latents=True,
            return_dict=True
        )
        
        frames = result.frames[0]
        
        if final_latent is not None:
            LOG.info(f"Temporal latent tensor output captured. Shape: {final_latent.shape}")

        LOG.info(f"Generated frames: {len(frames)}")
        
        return frames, final_latent

        

    def _encode_video(self, frames: List["PILImage"], path: str, fps: int) -> None:
        """
        Stream raw RGB24 frames into ffmpeg via stdin and encode to H.264 mp4.

        Correct subprocess lifecycle:
          1. Write all frames to stdin (streaming, no bulk buffer)
          2. Close stdin (signals EOF — ffmpeg begins muxing)
          3. Wait for process exit (process state is now stable)
          4. Read stderr (safe after wait — no race with pipe close)
          5. Check return code

        Ordering matters: wait() FIRST, stderr.read() AFTER.
        Reading stderr before wait() can race with ultrafast ffmpeg exit on Kaggle,
        causing a flush-of-closed-file ValueError on stdin pipe internals.
        Never call communicate() after manual stdin writes.
        """
        import numpy as np

        if not frames:
            raise ValueError("No frames provided for encoding.")

        width, height = frames[0].size

        LOG.info(f"Encoding {len(frames)} frames ({width}x{height}) to H.264 mp4 via ffmpeg...")

        cmd = [
            "ffmpeg",
            "-y",                   # overwrite output
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",              # read from stdin
            "-an",                  # no audio track
            "-vcodec", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            path,
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            raise RuntimeError("ffmpeg is not installed or not in system PATH.")

        try:
            # Stream frames one-by-one: no peak allocation of frames × W × H × 3 bytes
            for frame in frames:
                # Broken pipe guard
                if process.poll() is not None:
                   raise RuntimeError("ffmpeg process terminated early during stream write.")
                process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())

            process.stdin.close()         # EOF signal — ffmpeg begins muxing

            return_code = process.wait()  # wait FIRST — process is fully stable after this
            stderr_bytes = process.stderr.read()  # read AFTER wait — no pipe race

            if return_code != 0:
                raise RuntimeError(
                    f"FFmpeg encoding failed (code {return_code}):\n"
                    f"{stderr_bytes.decode(errors='replace')}"
                )

        except RuntimeError:
            raise
        except Exception as exc:
            process.kill()
            process.wait()
            raise RuntimeError(f"Video encoding failed: {exc}") from exc


    # ------------------------------------------------------------------
    # Public interface — Phase 8 split API
    # GenerationManager calls generate_frames() → interpolate → encode()
    # External callers may use the backwards-compatible generate() wrapper.
    # ------------------------------------------------------------------

    def generate_frames(
        self,
        request: "VideoGenerationRequest",
        conditioning_image_path: str,
        previous_latent: Optional[torch.Tensor] = None,
    ) -> tuple[List["PILImage"], torch.Tensor, pathlib.Path, int]:
        """
        Run SVD-XT generation and return (frames, final_latent, out_path, seed).

        Does NOT encode — caller (GenerationManager) encodes after
        optional interpolation.  Returns a 4-tuple:
            (List[PIL.Image], torch.Tensor, pathlib.Path, int)
        """
        from multigenai.engines.image_engine.engine import _slug

        img_slug = _slug(request.prompt)
        out_path = self._out_dir / f"{img_slug}.mp4"

        import torch
        seed = request.seed if request.seed is not None else int(
            torch.randint(0, 1_000_000, (1,)).item()
        )

        # Effective frame count — never mutates request object
        effective_frames = request.num_frames

        from PIL import Image as PILImage
        LOG.debug(f"Loading conditioning keyframe: {conditioning_image_path}")
        init_image = PILImage.open(conditioning_image_path).convert("RGB")

        if init_image.size != (request.width, request.height):
            LOG.debug(f"Resizing conditioning image from {init_image.size} to {request.width}x{request.height}")
            init_image = init_image.resize((request.width, request.height), PILImage.Resampling.LANCZOS)

        # Adaptive pixel-area frame cap (Kaggle/T4 VRAM hardening)
        # Threshold: >600k pixels (1024×576=589k is fine; 1280×720=921k is not)
        # Cap at 16 (not 8) — interpolation depends on keyframe count for
        # output frame formula: n + (n-1)*(factor-1). Capping at 8 would
        # silently halve interpolated output vs user expectation.
        max_pixels = 600_000
        max_svd_frames = 16
        if request.width * request.height > max_pixels and effective_frames > max_svd_frames:
            LOG.warning(
                f"High resolution {request.width}x{request.height} detected "
                f"(pixels={request.width * request.height} > {max_pixels}). "
                f"Capping frames {effective_frames} → {max_svd_frames} for VRAM stability."
            )
            effective_frames = max_svd_frames

        self._load_model()

        try:
            frames, final_latent = self._generate_video(
                request, 
                init_image, 
                seed, 
                effective_frames,
                previous_latent=previous_latent
            )
            del init_image
            return frames, final_latent, out_path, seed
        finally:
            if self._ctx.behaviour.auto_unload_after_gen:
                self._unload_model()

    @staticmethod
    def encode(
        frames: list,
        out_path,
        fps: int,
        seed: int,
        requested_frames: int,
    ) -> "VideoResult":
        """
        Encode a list of PIL frames to mp4 via ffmpeg and return VideoResult.

        Defined as a staticmethod — callers (GenerationManager, generate() wrapper)
        do not need to instantiate a full VideoEngine to encode an existing frame list.

        Args:
            frames:           Final (possibly interpolated) frame list
            out_path:         pathlib.Path or str for output mp4
            fps:              Frames per second
            seed:             Seed used during generation (for VideoResult metadata)
            requested_frames: Fallback frame count if frames list is empty
        """
        _log = get_logger(__name__)
        frame_count = len(frames) if frames else requested_frames  # capture BEFORE del

        try:
            # Import _encode_video logic inline via a temporary engine-free subprocess call
            import numpy as np
            import subprocess

            if not frames:
                raise ValueError("No frames to encode.")

            width, height = frames[0].size
            _log.info(f"Encoding {frame_count} frames ({width}x{height}) to {out_path}")

            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "-",
                "-an",
                "-vcodec", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                str(out_path),
            ]

            try:
                process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            except FileNotFoundError:
                raise RuntimeError("ffmpeg is not installed or not in system PATH.")

            try:
                for frame in frames:
                    process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
                process.stdin.close()
                return_code = process.wait()
                stderr_bytes = process.stderr.read()
                if return_code != 0:
                    raise RuntimeError(
                        f"FFmpeg encoding failed (code {return_code}):\n"
                        f"{stderr_bytes.decode(errors='replace')}"
                    )
            except RuntimeError:
                raise
            except Exception as exc:
                process.kill()
                process.wait()
                raise RuntimeError(f"Video encoding failed: {exc}") from exc

            del frames  # release memory AFTER we've finished using it

            return VideoResult(
                path=str(out_path),
                frame_count=frame_count,
                fps=fps,
                seed=seed,
                success=True,
            )
        except Exception as exc:
            _log.error(f"Video encoding error: {exc}", exc_info=True)
            return VideoResult(
                path="", frame_count=0, fps=fps, seed=seed,
                success=False, error=str(exc)
            )


    def generate(
        self,
        request: "VideoGenerationRequest",
        conditioning_image_path: str,
        previous_latent: Optional[torch.Tensor] = None,
    ) -> "VideoResult":
        """
        Backwards-compatible wrapper: generate_frames() → encode().

        Direct external callers (tests, CLI) can continue using this.
        GenerationManager uses the split API for interpolation support.
        """
        try:
            frames, final_latent, out_path, seed = self.generate_frames(
                request, 
                conditioning_image_path,
                previous_latent=previous_latent
            )
        except Exception as exc:
            LOG.error(f"SVD Video engine error: {exc}", exc_info=True)
            return VideoResult(
                path="", frame_count=0, fps=request.fps, seed=0,
                success=False, error=str(exc)
            )

        return self.encode(
            frames=frames,
            out_path=out_path,
            fps=request.fps,
            seed=seed,
            requested_frames=request.num_frames,
        )

    # ------------------------------------------------------------------
    # Phase 9 Hooks (stubs)
    # ------------------------------------------------------------------
    def inject_motion_lora(self) -> None:
        """[Phase 9 Stub] Dynamic injection of motion-specific LoRAs into SVD-XT."""
        pass

    def inject_controlnet(self) -> None:
        """[Phase 9 Stub] Structural video conditioning hook."""
        pass

    def inject_depth_guidance(self) -> None:
        """[Phase 9 Stub] SVD-XT dynamic depth mask manipulation."""
        pass
