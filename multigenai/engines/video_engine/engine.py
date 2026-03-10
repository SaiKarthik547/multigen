"""
VideoEngine — True Temporal Video Generation using AnimateDiff.

Phase 12: Replaces SVD with AnimateDiff for significantly better motion and control.
- Based on Stable Diffusion v1.5 + AnimateDiff Motion Adapter.
- No longer uses external optical flow (RAFT).
- Optimized for Kaggle T4 with VAE slicing and model CPU offload.
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

from multigenai.core.model_registry import ModelRegistry
 
# Phase 12 Dynamic Model Selection
registry = ModelRegistry.instance()
SD_MODEL_ID = registry.get_config_value("video_model", "runwayml/stable-diffusion-v1-5")
MOTION_ADAPTER_ID = registry.get_config_value("motion_adapter", "guoyww/animatediff-motion-adapter-v1-5-2")

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
    True Temporal Video Generation Engine using AnimateDiff.
    """

    def __init__(self, ctx: "ExecutionContext") -> None:
        self._ctx = ctx
        self._out_dir = pathlib.Path(ctx.settings.output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self.device = ctx.device
        self.pipe = None  # Ensure attribute exists even if load fails
        from multigenai.consistency.ip_adapter_manager import IPAdapterManager
        self.ip_adapter_manager = IPAdapterManager(self.device)

    def _load_model(self, use_ip_adapter: bool = False) -> None:
        """
        Lazily loads AnimateDiff pipeline.
        """
        import torch
        from diffusers import AnimateDiffPipeline, MotionAdapter, DPMSolverMultistepScheduler

        try:
            adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER_ID, torch_dtype=torch.float16)
            self.pipe = AnimateDiffPipeline.from_pretrained(
                SD_MODEL_ID,
                motion_adapter=adapter,
                torch_dtype=torch.float16,
            ).to(self.device)
            
            # Phase 12 Upgrade: DPMSolver for better speed/quality convergence
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                beta_schedule="linear",
                use_karras_sigmas=True,
                algorithm_type="sde-dpmsolver++"
            )

            # Disable progress bar for Kaggle overhead reduction
            self.pipe.set_progress_bar_config(disable=True)
            
            # Proactive cleanup to prevent fragmentation
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            if self._ctx.settings.video.enable_ip_adapter and use_ip_adapter:
                LOG.info("VideoEngine: Loading IP-Adapter for visual conditioning (SD1.5)...")
                self.ip_adapter_manager.load(self.pipe, model_type="sd15")
                # Phase 12 Fix: Increase influence for character stability
                if hasattr(self.pipe, "set_ip_adapter_scale"):
                    self.pipe.set_ip_adapter_scale(0.8)
                    LOG.info("VideoEngine: IP-Adapter scale set to 0.8")
                
        except Exception as e:
            LOG.error(f"Failed to load AnimateDiff pipeline: {e}")
            raise

        if self.device == "cuda":
            if hasattr(self.pipe, "enable_model_cpu_offload"):
                self.pipe.enable_model_cpu_offload()
            
            if hasattr(self.pipe, "enable_vae_slicing"):
                self.pipe.enable_vae_slicing()

            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing("max")
                LOG.info("VideoEngine: Max attention slicing enabled.")
                
            if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    LOG.info("VideoEngine: Xformers enabled.")
                except Exception:
                    LOG.debug("VideoEngine: Xformers not available.")

            # Phase 12 Optimized: Compile UNet for speed improvement (Kaggle T4 safe)
            if self._ctx.settings.video.enable_compile and hasattr(torch, "compile"):
                try:
                    self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=False)
                    LOG.info("VideoEngine: UNet compilation enabled (safe mode).")
                except Exception as ec:
                    LOG.debug(f"VideoEngine: UNet compilation failed: {ec}")
            
            torch.backends.cuda.enable_flash_sdp(True)
        else:
            self.pipe = self.pipe.to(self.device)

    def _unload_model(self) -> None:
        """
        Forcefully unloads the pipeline.
        """
        LOG.debug("Unloading AnimateDiff and clearing VRAM...")
        if hasattr(self, "pipe") and self.pipe is not None:
            del self.pipe
            self.pipe = None

        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as exc:
            LOG.warning(f"Error flushing CUDA memory: {exc}")

    def _generate_video(
        self,
        request: "VideoGenerationRequest",
        conditioning_image: "PILImage",
        seed: int,
        num_frames: int,
        temporal_state: "TemporalState",
        scene_index: int = 0
    ) -> tuple[List["PILImage"], torch.Tensor]:
        """
        Executes AnimateDiff generation with Temporal Sliding Windowing.
        
        Phase 12 Strategy:
        - Window 1: Frames 0-15 (16 frames)
        - Window 2: Frames 8-23 (16 frames)
        - Blend window: 8 overlapping frames (linear alpha blend)
        """
        from PIL import Image
        from multigenai.creative.scene_designer import SceneDesigner
        from multigenai.creative.prompt_compiler import PromptCompiler
        
        inference_steps = request.num_inference_steps if request.num_inference_steps > 0 else 25
        
        # Phase 13: Integrated SceneDesigner for trajectory and motion tokens
        blueprint = SceneDesigner().design_video(request, scene_index=scene_index)
        motion_prompt, negative_prompt = PromptCompiler().compile(blueprint, "animatediff")
        
        # Base kwargs
        pipe_kwargs = {
            "prompt": motion_prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": inference_steps,
            "guidance_scale": 6.5,
            "width": 768,
            "height": 512,
        }
        
        if self._ctx.settings.video.enable_ip_adapter:
            pipe_kwargs["ip_adapter_image"] = conditioning_image

        LOG.info(f"VideoEngine: Generating 2x16 frame windows with 4-frame overlap (Phase 13)...")
        
        device_type = "cuda" if self.device == "cuda" else "cpu"
        with torch.inference_mode(), torch.autocast(device_type):
            from multigenai.temporal.latent_propagator import LatentPropagator
            propagator = LatentPropagator()

            # Phase 14: Identity-Latent Conditioning Initialization
            shape = (
                1, 
                self.pipe.unet.config.in_channels, 
                16, 
                512 // self.pipe.vae_scale_factor, 
                768 // self.pipe.vae_scale_factor
            )
            base_generator = torch.Generator(device=self.device).manual_seed(seed)
            
            if hasattr(temporal_state, "identity_latent") and temporal_state.identity_latent is not None:
                id_lat = temporal_state.identity_latent.to(self.device, dtype=self.pipe.dtype)
                id_lat = id_lat.unsqueeze(2).expand(*shape)
                
                if temporal_state.previous_latent is None:
                    latents = id_lat.clone()
                else:
                    if scene_index > 0 and scene_index % 3 == 0:
                        latents = id_lat.clone()
                    else:
                        prev_lat = temporal_state.previous_latent.to(self.device, dtype=self.pipe.dtype)
                        latents = (prev_lat * 0.7) + (id_lat * 0.3)
                
                # Structural variance initialization
                latents = propagator.propagate(latents, drift=0.015, generator=base_generator)
            else:
                latents = torch.randn(shape, generator=base_generator, device=self.device, dtype=self.pipe.dtype)
            
            # --- WINDOW 1 (0-15) ---
            pipe_kwargs["num_frames"] = 16
            pipe_kwargs["latents"] = latents.clone()
            
            if "image" in self.pipe.__call__.__code__.co_varnames:
                pipe_kwargs["image"] = conditioning_image
                pipe_kwargs["strength"] = 0.75
                
            res1_raw = self.pipe(**pipe_kwargs).frames
            res1 = res1_raw[0] if isinstance(res1_raw[0], list) else res1_raw
            
            # --- WINDOW 2 (12-27) - 4 frame overlap ---
            # Phase 14 Fix: Latent noise drift for continuity
            latents = propagator.propagate(latents, drift=0.015, generator=base_generator)
            pipe_kwargs["latents"] = latents.clone()
            
            if "image" in self.pipe.__call__.__code__.co_varnames:
                pipe_kwargs["image"] = res1[-1]  # Anchor to Window 1 end
                pipe_kwargs["strength"] = 0.6    # Preserve character, allow motion
            
            res2_raw = self.pipe(**pipe_kwargs).frames
            res2 = res2_raw[0] if isinstance(res2_raw[0], list) else res2_raw
            
        # Linear Temporal Blending (4-frame overlap)
        final_frames = []
        # 1. First 12 frames from Window 1
        final_frames.extend(res1[:12])
        
        # 2. Middle 4 frames: Blended
        for i in range(4):
            alpha = (i + 1) / 5.0 # Linear ramp
            blended = Image.blend(res1[12+i], res2[i], alpha)
            final_frames.append(blended)
            
        # 3. Last 12 frames from Window 2
        final_frames.extend(res2[4:])
        
        LOG.info(f"VideoEngine: Windowing complete. Result length: {len(final_frames)}")
        return final_frames, latents

    def generate_frames(
        self,
        request: "VideoGenerationRequest",
        conditioning_image_path: str,
        temporal_state: "TemporalState",
        scene_index: int = 0
    ) -> tuple[List["PILImage"], Optional[torch.Tensor], pathlib.Path, int]:
        """
        Run AnimateDiff generation and return (frames, None, out_path, seed).
        """
        from multigenai.engines.image_engine.engine import _slug

        img_slug = _slug(request.prompt).strip("_")[:50]
        out_path = self._out_dir / f"{img_slug}_seg{scene_index}.mp4"

        import torch
        seed = request.seed if request.seed is not None else int(
            torch.randint(0, 1_000_000, (1,)).item()
        )

        # AnimateDiff usually takes 16-24 frames. 
        # User requested 3 scenes x 24 frames.
        effective_frames = 24 

        from PIL import Image as PILImage, ImageOps
        LOG.debug(f"Loading conditioning image (if needed): {conditioning_image_path}")
        init_image = PILImage.open(conditioning_image_path).convert("RGB")
        # ISSUE 7 Fix: Use ImageOps.fit to preserve aspect ratio (Phase 13: 768x512)
        request.width = 768
        request.height = 512
        init_image = ImageOps.fit(init_image, (request.width, request.height), method=PILImage.Resampling.LANCZOS)

        self._load_model(use_ip_adapter=(init_image is not None))

        try:
            frames, final_latents = self._generate_video(
                request, 
                init_image, 
                seed, 
                effective_frames,
                temporal_state=temporal_state,
                scene_index=scene_index
            )
            
            # Disk Streaming Cache
            temp_session_dir = self._out_dir / f".temp_frames_{seed}"
            temp_session_dir.mkdir(parents=True, exist_ok=True)
            frame_paths = []
            for idx, f in enumerate(frames):
                f_path = temp_session_dir / f"frame_{idx:04d}.png"
                f.save(f_path)
                frame_paths.append(f_path)

            return frame_paths, final_latents, out_path, seed
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
        Encode frames to mp4 via ffmpeg. Copied from earlier implementation for stability.
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

        try:
            if isinstance(frames[0], (str, pathlib.Path)):
                with Image.open(frames[0]) as first_img:
                    width, height = first_img.size
            else:
                width, height = frames[0].size
        except Exception as e:
            return VideoResult(path="", frame_count=0, fps=fps, seed=seed, success=False, error=f"Invalid format: {e}")

        _log.info(f"Encoding {frame_count} frames to {out_path}")

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
            process.wait()
            
            # Phase 12 Hardening: Capture stderr for diagnostics
            stderr_output = process.stderr.read().decode()
            if process.returncode != 0:
                return VideoResult(path="", frame_count=0, fps=fps, seed=seed, success=False, error=f"FFmpeg failed: {stderr_output}")

            # Cleanup temp session
            if frames and isinstance(frames[0], (str, pathlib.Path)):
                p = pathlib.Path(frames[0])
                if ".temp_frames_" in p.parent.name:
                    try: shutil.rmtree(p.parent)
                    except: pass

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
            return VideoResult(path="", frame_count=0, fps=fps, seed=seed, success=False, error=str(exc))

    def generate(
        self,
        request: "VideoGenerationRequest",
        conditioning_image_path: str,
        previous_latent: Optional[torch.Tensor] = None,
    ) -> "VideoResult":
        """Backwards compatible wrapper."""
        try:
            frames, _, out_path, seed = self.generate_frames(
                request, 
                conditioning_image_path,
                temporal_state=None,  # Not used in deprecated direct pass
                scene_index=0
            )
            return self.encode(
                frames=frames,
                out_path=out_path,
                fps=request.fps,
                seed=seed,
                requested_frames=24
            )
        except Exception as exc:
            LOG.error(f"Video engine error: {exc}", exc_info=True)
            return VideoResult(path="", frame_count=0, fps=request.fps, seed=0, success=False, error=str(exc))
