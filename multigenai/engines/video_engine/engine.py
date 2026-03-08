"""
VideoEngine — True Temporal Video Generation using Stable Video Diffusion (SVD).

Phase 6: Replaces the Phase 5 (Iterative SDXL img2img) hack with a pristine SVD pipeline.
- SVD is loaded lazily, bypassing the global ModelRegistry.
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
# SVD 1.1: Better motion coherence and temporal stability (XT = 25 frames)
SVD_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid"

# Global CUDA optimizations for memory efficiency and performance
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


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
    forward pass using SVD, encodes directly using ffmpeg, and drops all weights
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
        Lazily loads SVD directly.
        Diffusion models bypass ModelRegistry by design (Phase 7 isolation model).
        Uses sequential CPU offloading to prevent extreme memory spikes.
        """
        import torch
        from multigenai.models.temporal_svd_pipeline import TemporalStableVideoDiffusionPipeline

        # Phase 6: Core SVD model loading (fp16 for speed and VRAM economy)
        # Base SVD (14 frames) is now used.
        try:
            self.pipe = TemporalStableVideoDiffusionPipeline.from_pretrained(
                SVD_MODEL_ID,
                torch_dtype=torch.float16,
                variant="fp16"
            ).to(self.device)
            
            # Disable progress bar for Kaggle overhead reduction
            self.pipe.set_progress_bar_config(disable=True)
            
            # Proactive cleanup to prevent fragmentation after weights move
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
        except Exception as e:
            LOG.error(f"Failed to load SVD pipeline: {e}")
            raise

        if self.device == "cuda":
            # Phase 14+: Defensive capability checks to avoid crashes on older/custom pipelines
            if hasattr(self.pipe, "enable_model_cpu_offload"):
                self.pipe.enable_model_cpu_offload()
            
            if hasattr(self.pipe, "enable_vae_slicing"):
                self.pipe.enable_vae_slicing()
                
            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing()
            
            # Phase 14: Kaggle Optimization — torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)

            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                LOG.info("VideoEngine: Xformers memory-efficient attention enabled.")
            except Exception:
                LOG.debug("VideoEngine: Xformers not available, using SDP attention.")

            # Phase 14: Speed boost — torch.compile(pipe.unet)
            try:
                LOG.info("VideoEngine: Compiling UNet for 20-30% speed boost...")
                self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead")
            except Exception as e:
                LOG.debug(f"VideoEngine: torch.compile failed (skipping): {e}")
        else:
            self.pipe = self.pipe.to(self.device)

    def _unload_model(self) -> None:
        """
        Forcefully unloads the SVD pipeline from system and GPU memory.
        This must be called immediately after encoding to keep the environment
        sterile for incoming ImageEngine (SDXL) processes.
        """
        LOG.debug("Unloading SVD and clearing VRAM...")
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

    def renoise_latent(self, prev_latent: torch.Tensor, scene_index: int = 0) -> torch.Tensor:
        """
        Inject minimal Gaussian noise into an existing latent tensor while preserving structure.
        Uses adaptive scheduling: strength increases with scene index to maintain motion vitality.
        """
        # Phase 11+: Adaptive noise schedule
        strength = 0.015 + (0.005 * min(10, scene_index))
        noise = torch.randn_like(prev_latent)
        return prev_latent + noise * strength

    def _generate_video(
        self,
        request: "VideoGenerationRequest",
        conditioning_image: "PILImage",
        seed: int,
        num_frames: int,
        previous_latent: Optional[torch.Tensor] = None,
        scene_index: int = 0
    ) -> tuple[List["PILImage"], torch.Tensor]:
        """
        Executes the SVD forward pass using Temporal Sliding-Window Diffusion.
        Splits generation into overlapping windows to minimize VRAM and maximize consistency.
        """
        import torch
        from PIL import Image

        # GPU/Device generators to avoid sync overhead during CUDA inference
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # SVD quality baseline: 14 steps is almost identical to 25 but 40% faster
        inference_steps = request.num_inference_steps if request.num_inference_steps > 0 else 14
        motion_bucket = max(0, min(255, int(request.temporal_strength * 255)))
        
        # Sliding Window Configuration
        window_size = 14  # SVD native depth
        stride = 7        # 50% overlap for smooth blending
        
        all_frames: List["PILImage"] = []
        final_latents_capture: Optional[torch.Tensor] = None
        
        # Determine windows
        if num_frames <= window_size:
            windows = [(0, num_frames)]
        else:
            windows = []
            for start in range(0, num_frames - window_size + stride, stride):
                end = min(start + window_size, num_frames)
                windows.append((start, end))
                if end == num_frames:
                    break

        LOG.info(
            f"VideoEngine: Processing {num_frames} frames via {len(windows)} windows "
            f"(size={window_size}, stride={stride}). Scene={scene_index}."
        )

        current_conditioning = conditioning_image
        current_latent_input = previous_latent

        for i, (start, end) in enumerate(windows):
            w_frames_count = end - start
            
            # Preparation for this window
            win_latents = None
            if current_latent_input is not None:
                # Ensure device/dtype match
                current_latent_input = current_latent_input.to(self.device, dtype=self.pipe.unet.dtype)
                
                # Repeat last latent to fill window (more stable for diffusion than zeros)
                if current_latent_input.shape[1] == 1:
                    win_latents = current_latent_input.repeat(1, w_frames_count, 1, 1, 1)
                elif current_latent_input.shape[1] != w_frames_count:
                    if current_latent_input.shape[1] > w_frames_count:
                        win_latents = current_latent_input[:, :w_frames_count]
                    else:
                        # Stability improvement: repeat the last encoded frame instead of zero tensor
                        last_latent = current_latent_input[:, -1:]
                        needed = w_frames_count - current_latent_input.shape[1]
                        padding = last_latent.repeat(1, needed, 1, 1, 1)
                        win_latents = torch.cat([current_latent_input, padding], dim=1)
                else:
                    win_latents = current_latent_input

                # Inject noise
                win_latents = self.renoise_latent(win_latents, scene_index=scene_index)

            LOG.debug(f"VideoEngine: Window {i} [{start}:{end}] starting...")
            
            with torch.inference_mode():
                result, w_latents = self.pipe(
                    image=current_conditioning,
                    num_frames=w_frames_count,
                    num_inference_steps=inference_steps,
                    height=request.height,
                    width=request.width,
                    generator=generator,
                    motion_bucket_id=motion_bucket,
                    decode_chunk_size=min(8, w_frames_count),
                    output_type="pil",
                    latents=win_latents,
                    return_latents=True,
                    return_dict=True
                )
            
            # Capture the very last frame's latent for the next SCENE (not just window)
            if i == len(windows) - 1:
                final_latents_capture = w_latents[:, -1:].detach().cpu().clone()

            # Frames for this window
            w_pil_frames = result.frames[0]
            
            # Update conditioning for next window if not at end
            if i < len(windows) - 1:
                current_conditioning = w_pil_frames[-1]
                # Carry over last few frames of latents for temporal consistency
                current_latent_input = w_latents[:, -stride:].detach().clone()

            # Handle Overlap Blending (Linear Cross-Fade)
            if i == 0:
                all_frames.extend(w_pil_frames)
            else:
                overlap_len = len(all_frames) - start
                if overlap_len > 0:
                    existing_overlap = all_frames[start:]
                    new_overlap = w_pil_frames[:overlap_len]
                    
                    blended = []
                    for t in range(overlap_len):
                        alpha = t / float(overlap_len)
                        b_frame = Image.blend(existing_overlap[t], new_overlap[t], alpha)
                        blended.append(b_frame)
                    
                    all_frames[start:] = blended
                    all_frames.extend(w_pil_frames[overlap_len:])
                else:
                    all_frames.extend(w_pil_frames)

            # Cleanup window
            del result
            del w_latents
            del win_latents
            gc.collect()
            if torch.cuda.is_available() and (i % 2 == 1 or i == len(windows) - 1):
                torch.cuda.empty_cache()

        # Phase 13: Disk Streaming Cache
        # Instead of returning raw PIL objects, we can save to a temp session directory
        # to ensure RAM never spikes regardless of frame count.
        temp_session_dir = self._out_dir / f".temp_frames_{seed}"
        temp_session_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        
        for idx, f in enumerate(all_frames):
            f_path = temp_session_dir / f"frame_{idx:04d}.png"
            f.save(f_path)
            frame_paths.append(f_path)
            
        LOG.info(f"VideoEngine: Cached {len(all_frames)} frames to disk for streaming.")
        return frame_paths, final_latents_capture

    
        

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
            # Stream frames one-by-one from disk paths to save RAM
            from PIL import Image
            for frame_source in frames:
                # Broken pipe guard
                if process.poll() is not None:
                   raise RuntimeError("ffmpeg process terminated early during stream write.")
                
                if isinstance(frame_source, (str, pathlib.Path)):
                    with Image.open(frame_source) as img:
                        frame_data = np.asarray(img.convert("RGB"), dtype=np.uint8)
                else:
                    # Fallback for raw PIL image objects
                    frame_data = np.asarray(frame_source.convert("RGB"), dtype=np.uint8)
                
                process.stdin.write(frame_data.tobytes())
                del frame_data

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
        scene_index: int = 0
    ) -> tuple[List["PILImage"], torch.Tensor, pathlib.Path, int]:
        """
        Run SVD-XT generation and return (frames, final_latent, out_path, seed).

        Does NOT encode — caller (GenerationManager) encodes after
        optional interpolation.  Returns a 4-tuple:
            (List[PIL.Image], torch.Tensor, pathlib.Path, int)
        """
        from multigenai.engines.image_engine.engine import _slug

        # Use scene index in filename to avoid collisions during multi-segment runs
        img_slug = _slug(request.prompt).strip("_")[:50]
        out_path = self._out_dir / f"{img_slug}_seg{scene_index}.mp4"

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

        # Adaptive pixel-budget frame cap (Kaggle/T4 VRAM hardening)
        # 16-24 for high res, up to 48 for standard 512x512
        pixel_budget = 7_000_000
        auto_max_frames = max(16, pixel_budget // (request.width * request.height))
        
        if effective_frames > auto_max_frames:
            LOG.warning(
                f"VideoEngine: Resolution {request.width}x{request.height} "
                f"exceeds pixel-budget. Capping frames {effective_frames} → {auto_max_frames}."
            )
            effective_frames = auto_max_frames

        self._load_model()

        try:
            frames, final_latent = self._generate_video(
                request, 
                init_image, 
                seed, 
                effective_frames,
                previous_latent=previous_latent,
                scene_index=scene_index
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
        Encode a list of PIL frames (or frame paths) to mp4 via ffmpeg and return VideoResult.
        """
        import numpy as np
        import subprocess
        from PIL import Image
        import pathlib
        import shutil

        _log = get_logger(__name__)
        frame_count = len(frames) if frames else requested_frames
        
        if not frames:
            return VideoResult(path="", frame_count=0, fps=fps, seed=seed, success=False, error="No frames provided.")

        # Determine dimensions from first frame source
        try:
            if isinstance(frames[0], (str, pathlib.Path)):
                with Image.open(frames[0]) as first_img:
                    width, height = first_img.size
            else:
                width, height = frames[0].size
        except Exception as e:
            return VideoResult(path="", frame_count=0, fps=fps, seed=seed, success=False, error=f"Invalid frame format: {e}")

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
            return VideoResult(path="", frame_count=0, fps=fps, seed=seed, success=False, error="ffmpeg not found.")

        try:
            for frame_source in frames:
                # Broken pipe guard
                if process.poll() is not None:
                   break
                
                if isinstance(frame_source, (str, pathlib.Path)):
                    with Image.open(frame_source) as img:
                        frame_data = np.asarray(img.convert("RGB"), dtype=np.uint8)
                else:
                    frame_data = np.asarray(frame_source.convert("RGB"), dtype=np.uint8)
                
                process.stdin.write(frame_data.tobytes())
                del frame_data

            process.stdin.close()
            return_code = process.wait()
            stderr_bytes = process.stderr.read()

            if return_code != 0:
                raise RuntimeError(f"FFmpeg failed: {stderr_bytes.decode(errors='replace')}")

            # Phase 14: Temp Session Cleanup
            if frames and isinstance(frames[0], (str, pathlib.Path)):
                p = pathlib.Path(frames[0])
                if ".temp_frames_" in p.parent.name:
                    try:
                        shutil.rmtree(p.parent)
                        _log.debug(f"VideoEngine: Cleaned up temp path: {p.parent}")
                    except Exception as e:
                        _log.warning(f"VideoEngine: Cleanup failure: {e}")

            return VideoResult(
                path=str(out_path),
                frame_count=frame_count,
                fps=fps,
                seed=seed,
                success=True,
            )
        except Exception as exc:
            if 'process' in locals() and process.poll() is None:
                process.kill()
                process.wait()
            _log.error(f"Video encoding error: {exc}")
            return VideoResult(path="", frame_count=0, fps=fps, seed=seed, success=False, error=str(exc))


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
