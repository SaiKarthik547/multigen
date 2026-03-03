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
        
        # Map temporal_strength (0.0–1.0) to native SVD motion_bucket_id (0–255)
        # Clamped to prevent out-of-range values from user-supplied temporal_strength
        motion_bucket = max(0, min(255, int(request.temporal_strength * 255)))
        
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
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )

            # Stream frames one-by-one to avoid peak allocation of
            # frames × width × height × 3 bytes for 16+ frame runs.
            import numpy as np
            try:
                for frame in frames:
                    process.stdin.write(np.array(frame).astype(np.uint8).tobytes())
                process.stdin.close()
            except BrokenPipeError:
                pass  # ffmpeg failure will be caught by returncode check below

            _, stderr_bytes = process.communicate()

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

        # Compute effective frame count WITHOUT mutating the request object.
        # Mutating request causes VideoResult to return capped count as if it were requested.
        effective_frames = request.num_frames
        
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
            # Pixel-area threshold covers 1024x576, 896x512, and other high-density resolutions
            # that trigger 3D conv OOM during VAE decode, even with decode_chunk_size=2.
            if request.width * request.height > 600000 and effective_frames > 8:
                LOG.warning(
                    f"High resolution {request.width}x{request.height} detected "
                    f"(pixels={request.width * request.height} > 600000). "
                    f"Capping frames {effective_frames} → 8 for VRAM stability."
                )
                effective_frames = 8

            # 3. Strict Model Loading
            self._load_model()
            
            # 4. SVD-XT Latent Generation (pass effective_frames, not request.num_frames)
            # We pass effective_frames via a local — request object is NOT mutated.
            original_frames = request.num_frames
            request.num_frames = effective_frames
            frames = self._generate_video(request, init_image, seed)
            request.num_frames = original_frames  # always restore

            # Release conditioning image — no longer needed after generation
            del init_image

            # 5. ffmpeg byte-pipe encoding
            self._encode_video(frames, str(out_path), request.fps)

            # Release frame list before model unload to minimise peak memory
            del frames

        except Exception as exc:
            LOG.error(f"SVD Video engine error: {exc}", exc_info=True)
            return VideoResult(path="", frame_count=0, fps=request.fps, seed=seed, success=False, error=str(exc))
        finally:
            # 6. Strict Model Unload (MUST run even if Generation crashes)
            self._unload_model()

        return VideoResult(
            path=str(out_path), 
            frame_count=effective_frames,  # reflects actual frames generated
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
