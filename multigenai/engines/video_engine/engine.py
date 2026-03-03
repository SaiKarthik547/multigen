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
        from diffusers import StableVideoDiffusionPipeline

        if self.pipe is not None:
            return

        LOG.info(f"Loading {SVD_MODEL_ID} (fp16, sequential CPU offload)...")
        
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            SVD_MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16"
        )

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

    def _generate_video(self, request: "VideoGenerationRequest", conditioning_image: "PILImage", seed: int) -> List["PILImage"]:
        """
        Executes the SVD-XT forward pass, returning a list of PIL Images.
        """
        import torch

        # CPU generator: portable across CUDA/CPU/DirectML, same sequence guaranteed
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        # Map Phase 5 temporal_strength (0.0 to 1.0) to native SVD motion_bucket_id (0 to 255)
        motion_bucket = int(request.temporal_strength * 255)
        
        LOG.info(
            f"Phase 6: Generating {request.num_frames} frames via SVD-XT. "
            f"Seed={seed}, resolution={request.width}x{request.height}, "
            f"steps={request.num_inference_steps}, motion_bucket_id={motion_bucket}."
        )

        frames = self.pipe(
            image=conditioning_image,
            num_frames=request.num_frames,
            num_inference_steps=request.num_inference_steps,
            height=request.height,
            width=request.width,
            generator=generator,
            motion_bucket_id=motion_bucket,
            decode_chunk_size=2,  # Peak VRAM optimization (reduces 3D conv batching)
        ).frames[0]

        return frames

    def _encode_video(self, frames: List["PILImage"], path: str, fps: int) -> None:
        """
        Pipes raw RGB24 frames securely into an ffmpeg subprocess.
        Bypasses `moviepy` entirely for zero-overhead, production-grade speeds.
        """
        import numpy as np
        
        if not frames:
            raise ValueError("No frames provided for encoding.")

        width, height = frames[0].size
        
        LOG.info(f"Encoding {len(frames)} frames to H.264 mp4 via ffmpeg pipe...")
        
        command = [
            "ffmpeg",
            "-y",                  # overwrite output
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",             # read from stdin
            "-an",                 # no audio
            "-vcodec", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            path
        ]

        try:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Build the entire stdin buffer first then communicate() drains all
            # pipes safely — avoids deadlock from simultaneous write + stderr fill.
            stdin_buffer = b""
            for frame in frames:
                stdin_buffer += np.array(frame).astype(np.uint8).tobytes()

            _stdout, stderr_bytes = process.communicate(input=stdin_buffer)

            if process.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg encoding failed (code {process.returncode}):\n"
                    f"{stderr_bytes.decode(errors='replace')}"
                )

        except FileNotFoundError:
            raise RuntimeError("ffmpeg is not installed or not in system PATH.")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Video encoding failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Public interface (invoked ONLY by GenerationManager in Phase 6)
    # ------------------------------------------------------------------
    def generate(self, request: "VideoGenerationRequest", conditioning_image_path: str) -> VideoResult:
        """
        Main execution flow properly decoupled from other engines.
        """
        from multigenai.engines.image_engine.engine import _slug
        
        img_slug = _slug(request.prompt)
        # Handle Windows invalid path characters if prompt mapping
        out_path = self._out_dir / f"{img_slug}.mp4"
        
        import torch
        seed = request.seed if request.seed is not None else int(
            torch.randint(0, 1_000_000, (1,)).item()
        )
        
        try:
            # 1. Load Conditioning Image 
            # (Generated by ImageEngine before VideoEngine was ever instantiated)
            from PIL import Image as PILImage
            LOG.debug(f"Loading conditioning keyframe: {conditioning_image_path}")
            init_image = PILImage.open(conditioning_image_path).convert("RGB")
            
            if init_image.size != (request.width, request.height):
                 LOG.debug(f"Resizing conditioning image from {init_image.size} to {request.width}x{request.height}")
                 init_image = init_image.resize((request.width, request.height), PILImage.Resampling.LANCZOS)

            # 2. Adaptive Constraints (Kaggle/T4 Hardening)
            # 1024x576 (or similar high res) at 16 frames OOMs on T4 (15GB). 
            # Force-cap to 8 frames at high res for stability.
            if request.width * request.height > 600000 and request.num_frames > 8:
                LOG.warning(f"High resolution {request.width}x{request.height} detected. Capping frames to 8 for VRAM stability.")
                request.num_frames = 8

            # 3. Strict Model Loading
            self._load_model()
            
            # 3. SVD-XT Latent Generation
            frames = self._generate_video(request, init_image, seed)

            # Release conditioning image — no longer needed after generation
            del init_image

            # 4. ffmpeg byte-pipe encoding
            self._encode_video(frames, str(out_path), request.fps)

            # Release frame list before model unload to minimise peak memory
            del frames

        except Exception as exc:
            LOG.error(f"SVD Video engine error: {exc}", exc_info=True)
            return VideoResult(path="", frame_count=0, fps=request.fps, seed=seed, success=False, error=str(exc))
        finally:
            # 5. Strict Model Unload (MUST run even if Generation crashes)
            self._unload_model()

        return VideoResult(
            path=str(out_path), 
            frame_count=request.num_frames, 
            fps=request.fps, 
            seed=seed, 
            success=True
        )

    # ------------------------------------------------------------------
    # Phase 6 / Phase 8 Hooks
    # ------------------------------------------------------------------
    def inject_motion_lora(self) -> None:
        """[Phase 8 Stub] Dynamic injection of motion-specific LoRAs into SVD-XT."""
        pass

    def inject_controlnet(self) -> None:
        """[Phase 8 Stub] Structural video conditioning hook."""
        pass

    def inject_depth_guidance(self) -> None:
        """[Phase 8 Stub] SVD-XT dynamic depth mask manipulation."""
        pass
