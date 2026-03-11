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
    from multigenai.core.temporal_state import TemporalState
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
        # Phase 15: IP-Adapter retired — no manager object needed

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
                # Phase 15: IP-Adapter is retired — raise immediately to surface misconfiguration
                raise RuntimeError(
                    "IP-Adapter is retired (Phase 15 VRAM guard). "
                    "Set enable_ip_adapter: false in config.yaml. "
                    "See legacy/models/ip_adapter/ip_adapter_manager.py"
                )
                
        except Exception as e:
            LOG.error(f"Failed to load AnimateDiff pipeline: {e}")
            raise

        if self.device == "cuda":
            if hasattr(self.pipe, "enable_model_cpu_offload"):
                self.pipe.enable_model_cpu_offload()
            
            if hasattr(self.pipe.vae, "enable_slicing"):
                self.pipe.vae.enable_slicing()

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

    def decode_latents(self, latents: "torch.Tensor") -> List["PILImage"]:
        """Decode latents to PIL Images correctly per Phase A."""
        import torch
        from PIL import Image

        B, C, T, H, W = latents.shape

        # correct scaling for AnimateDiff pipeline
        latents = latents / 0.18215

        # reshape for VAE decode: [B*T, C, H, W]
        latents = latents.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        with torch.no_grad():
            images = self.pipe.vae.decode(latents).sample

        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).astype("uint8")

        frames = [Image.fromarray(img) for img in images]
        return frames

    def _generate_video(
        self,
        request: "VideoGenerationRequest",
        conditioning_image: "PILImage",
        seed: int,
        num_frames: int,
        temporal_state: "TemporalState",
        scene_index: int = 0,
        keyframe_latent: Optional[torch.Tensor] = None,
    ) -> tuple[List["PILImage"], torch.Tensor]:
        """
        Executes AnimateDiff generation with Phase 15 Temporal Sliding Windowing.

        Phase 15 Strategy:
        - Window 1 → frames 0-23   (WINDOW_SIZE=24)
        - Window 2 → frames 16-39  (OVERLAP=8)
        - Latent seeded from keyframe anchor (not random) when available
        - Directional velocity propagation between windows
        """
        from PIL import Image
        from multigenai.creative.scene_designer import SceneDesigner
        from multigenai.creative.prompt_compiler import PromptCompiler

        WINDOW_SIZE = 24  # Phase 15 upgrade from 16
        OVERLAP = 8       # Phase 15 upgrade from 4

        inference_steps = request.num_inference_steps if request.num_inference_steps > 0 else 25

        # SceneDesigner: builds enriched prompt with camera + motion tokens
        blueprint = SceneDesigner().design_video(request, scene_index=scene_index)
        motion_prompt, negative_prompt = PromptCompiler().compile(blueprint, "animatediff")
        
        # Hard truncate prompt characters to ensure absolute safety (Phase 10)
        motion_prompt = motion_prompt[:220]

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

        # Anchor trajectory encoder to first frame when available
        if scene_index == 0 and conditioning_image is not None and getattr(temporal_state, "previous_frame", None) is None:
            temporal_state.previous_frame = conditioning_image

        LOG.info(f"VideoEngine: Generating 2x{WINDOW_SIZE} frame windows with {OVERLAP}-frame overlap (Phase 15)...")

        device_type = "cuda" if self.device == "cuda" else "cpu"
        with torch.inference_mode(), torch.autocast(device_type):
            from multigenai.temporal.latent_propagator import LatentPropagator
            import torch.nn.functional as F

            propagator = LatentPropagator()

            # Phase 15 latent shape: 24 frames
            shape = (
                1,
                self.pipe.unet.config.in_channels,
                WINDOW_SIZE,
                512 // self.pipe.vae_scale_factor,
                768 // self.pipe.vae_scale_factor,
            )
            H, W = shape[3], shape[4]
            base_generator = torch.Generator(device=self.device).manual_seed(seed + scene_index)

            # ----------------------------------------------------------------
            # Phase 15: Latent initialization priority chain
            # 1. Keyframe anchor latent (highest priority — from SDXL keyframe)
            # 2. Identity + previous scene latent blend
            # 3. Pure random (no prior context)
            # ----------------------------------------------------------------
            if keyframe_latent is not None:
                kf = keyframe_latent.to(self.device, dtype=self.pipe.dtype)
                if kf.shape[1] != self.pipe.unet.config.in_channels:
                    raise ValueError(
                        f"Keyframe latent channel mismatch: "
                        f"expected {self.pipe.unet.config.in_channels}, got {kf.shape[1]}"
                    )
                if kf.shape[2:] != (H, W):
                    kf = F.interpolate(kf, size=(H, W), mode="bilinear", align_corners=False)
                # Expand keyframe to full temporal window
                latents = kf.unsqueeze(2).repeat(1, 1, WINDOW_SIZE, 1, 1)
                LOG.debug("VideoEngine: Latent seeded from keyframe anchor.")

            elif hasattr(temporal_state, "identity_latent") and temporal_state.identity_latent is not None:
                id_lat = temporal_state.identity_latent.to(self.device, dtype=self.pipe.dtype)
                if id_lat.shape[2:] != (H, W):
                    id_lat = F.interpolate(id_lat, size=(H, W), mode="bilinear", align_corners=False)
                # Expand keyframe to full temporal window
                id_lat_expanded = id_lat.unsqueeze(2).repeat(1, 1, WINDOW_SIZE, 1, 1)

                if temporal_state.previous_latent is None:
                    latents = id_lat_expanded.clone()
                elif scene_index > 0 and scene_index % 3 == 0:
                    latents = id_lat_expanded.clone()
                else:
                    prev_lat = temporal_state.previous_latent.to(self.device, dtype=self.pipe.dtype)

                    # Trajectory: pull structure from previous frame
                    traj_lat = None
                    if getattr(temporal_state, "previous_frame", None) is not None:
                        try:
                            from multigenai.temporal.trajectory_encoder import TrajectoryEncoder
                            traj_lat = TrajectoryEncoder().encode(self.pipe, temporal_state.previous_frame)
                            if traj_lat is not None:
                                traj_lat = traj_lat.to(self.device, dtype=self.pipe.dtype)
                                if traj_lat.shape[2:] != (H, W):
                                    traj_lat = F.interpolate(traj_lat, size=(H, W), mode="bilinear", align_corners=False)
                                traj_lat = traj_lat.unsqueeze(2).repeat(1, 1, WINDOW_SIZE, 1, 1)
                        except Exception as e:
                            LOG.warning(f"Failed to encode trajectory latent: {e}")

                    if traj_lat is not None:
                        latents = (prev_lat * 0.6) + (id_lat_expanded * 0.25) + (traj_lat * 0.15)
                    else:
                        latents = (prev_lat * 0.7) + (id_lat_expanded * 0.3)

                # Phase I: Frame 0 Identity Injection
                latents[:, :, 0] = id_lat.squeeze(2)

                # Apply directional propagation
                latents, velocity = propagator.propagate(
                    latents,
                    prev_latent=temporal_state.previous_latent.to(self.device, dtype=self.pipe.dtype)
                    if temporal_state.previous_latent is not None else None,
                    velocity=temporal_state.latent_velocity.to(self.device, dtype=self.pipe.dtype)
                    if temporal_state.latent_velocity is not None else None,
                )
                temporal_state.latent_velocity = velocity.cpu() if velocity is not None else None
            else:
                latents = torch.randn(shape, generator=base_generator, device=self.device, dtype=self.pipe.dtype)

            latents = torch.clamp(latents, -4.0, 4.0)

            # --- WINDOW 1 (frames 0 → WINDOW_SIZE-1) ---
            pipe_kwargs["num_frames"] = WINDOW_SIZE
            pipe_kwargs["latents"] = latents.clone()

            if "image" in self.pipe.__call__.__code__.co_varnames:
                pipe_kwargs["image"] = conditioning_image
                pipe_kwargs["strength"] = 0.75

            pipe_kwargs["output_type"] = "latent"

            res1_raw = self.pipe(**pipe_kwargs).frames
            res1_lat = res1_raw[0] if isinstance(res1_raw[0], list) else res1_raw
            res1 = self.decode_latents(res1_lat) if isinstance(res1_lat, torch.Tensor) else res1_lat

            # --- WINDOW 2 (OVERLAP frames carryover + fresh noise) ---
            # Carry the LAST OVERLAP frames of window 1 as warm start, append fresh noise
            # BUG FIX: was latents[:, :, OVERLAP:] (drops first frames) → must be [:, :, -OVERLAP:]
            fresh_frames = WINDOW_SIZE - OVERLAP  # 16 frames of fresh noise
            new_noise = torch.randn(
                (1, self.pipe.unet.config.in_channels, fresh_frames, H, W),
                generator=base_generator, device=self.device, dtype=self.pipe.dtype,
            )
            latents_w2 = torch.cat([latents[:, :, -OVERLAP:], new_noise], dim=2)

            # Apply directional propagation for window 2
            latents_w2, _ = propagator.propagate(
                latents_w2,
                prev_latent=latents,
                velocity=temporal_state.latent_velocity.to(self.device, dtype=self.pipe.dtype)
                if temporal_state.latent_velocity is not None else None,
            )
            latents_w2 = torch.clamp(latents_w2, -4.0, 4.0)

            pipe_kwargs["latents"] = latents_w2
            if "image" in self.pipe.__call__.__code__.co_varnames:
                pipe_kwargs["image"] = res1[-1]   # anchor to last frame of W1
                pipe_kwargs["strength"] = 0.6

            pipe_kwargs["output_type"] = "latent"

            res2_raw = self.pipe(**pipe_kwargs).frames
            res2_lat = res2_raw[0] if isinstance(res2_raw[0], list) else res2_raw
            res2 = self.decode_latents(res2_lat) if isinstance(res2_lat, torch.Tensor) else res2_lat

        # --- Linear temporal blending over OVERLAP frames ---
        final_frames = []
        final_frames.extend(res1[:WINDOW_SIZE - OVERLAP])          # 16 clean frames from W1

        for i in range(OVERLAP):
            alpha = (i + 1) / (OVERLAP + 1)                        # linear ramp
            blended = Image.blend(res1[WINDOW_SIZE - OVERLAP + i], res2[i], alpha)
            final_frames.append(blended)

        final_frames.extend(res2[OVERLAP:])                        # 16 clean frames from W2

        del res1
        del res2
        gc.collect()

        # Phase 16: Generation Telemetry
        velocity_norm = torch.norm(velocity).item() if 'velocity' in locals() and velocity is not None else 0.0
        latent_std = latents.std().item() if latents is not None else 0.0
        vram_usage = torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0.0

        LOG.info(
            f"VideoEngine Telemetry | "
            f"Frames: {len(final_frames)} | "
            f"Latent STD: {latent_std:.3f} | "
            f"Velocity Norm: {velocity_norm:.3f} | "
            f"VRAM: {vram_usage:.0f} MB"
        )

        # Persist global_latent on CPU (before latents go out of scope or get .cpu() called externally)
        temporal_state.global_latent = latents.detach().cpu()

        return final_frames, latents.detach().cpu()

    def generate_frames(
        self,
        request: "VideoGenerationRequest",
        conditioning_image_path: str,
        temporal_state: "TemporalState",
        scene_index: int = 0,
        keyframe_latent: Optional[torch.Tensor] = None,
    ) -> tuple[List["PILImage"], Optional[torch.Tensor], pathlib.Path, int]:
        """
        Run AnimateDiff generation and return (frame_paths, final_latents, out_path, seed).

        Args:
            keyframe_latent: Optional pre-encoded latent from SDXL keyframe anchor.
                             When provided seeds the AnimateDiff latent space for
                             strong composition + character consistency.
        """
        from multigenai.engines.image_engine.engine import _slug

        img_slug = _slug(request.prompt).strip("_")[:50]
        out_path = self._out_dir / f"{img_slug}_seg{scene_index}.mp4"

        import torch
        seed = request.seed if request.seed is not None else int(
            torch.randint(0, 1_000_000, (1,)).item()
        )

        effective_frames = 24  # Phase 15: matches WINDOW_SIZE

        from PIL import Image as PILImage, ImageOps
        init_image = None
        if conditioning_image_path is not None:
            LOG.debug(f"Loading conditioning image: {conditioning_image_path}")
            init_image = PILImage.open(conditioning_image_path).convert("RGB")
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
                scene_index=scene_index,
                keyframe_latent=keyframe_latent,
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
                # Find the actual .temp_frames_ or .interpolated_frames parent directory
                temp_dir = next((pathlib.Path(*p.parts[:i+1]) for i, part in enumerate(p.parts) if ".temp_frames_" in part or ".interpolated_frames" in part), None)
                if temp_dir is not None:
                    try: shutil.rmtree(temp_dir)
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
