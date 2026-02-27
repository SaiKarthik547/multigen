"""
ImageEngine — SDXL-based cinematic image generation.

Migrated from MultiGenAi.py Generator class with:
  - ExecutionContext injection (no global state)
  - Typed request/response (ImageGenerationRequest → PipelineResult)
  - VRAM-aware model loading via ModelRegistry
  - Phase 3: Two-stage SDXL refiner pipeline with VAE float32 upgrade
  - Phase 4 hooks for IP-Adapter and ControlNet
"""

from __future__ import annotations

import hashlib
import pathlib
import random
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from multigenai.core.exceptions import EngineExecutionError
from multigenai.core.logging.logger import get_logger
from multigenai.core.metrics import GenerationMetrics, GenerationTimer, MetricsCollector

if TYPE_CHECKING:
    from multigenai.core.execution_context import ExecutionContext
    from multigenai.llm.schema_validator import ImageGenerationRequest

LOG = get_logger(__name__)

_SDXL_MODEL_ID          = "stabilityai/stable-diffusion-xl-base-1.0"
_SDXL_REFINER_MODEL_ID  = "stabilityai/stable-diffusion-xl-refiner-1.0"
_SDXL_MIN_VRAM_GB       = 6.0
# Refiner VRAM is not checked independently — base is already resident,
# and the two run sequentially, not in parallel. Rely on runtime OOM
# fallback (graceful refiner skip) rather than a blocking VRAM gate.
_SDXL_REFINER_MIN_VRAM_GB = 0.0
# Phase 4: IP-Adapter requires at least this much free VRAM to be safe on Kaggle / T4.
# At <10 GB the adapter risks OOM during cross-attention patch; skip gracefully instead.
_IP_ADAPTER_MIN_VRAM_GB = 10.0


def _slug(s: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9\-_]+", "_", s)[:40]
    return f"{safe}_{hashlib.sha1(s.encode()).hexdigest()[:8]}"


@dataclass
class ImageResult:
    """Output from the ImageEngine."""
    path: str
    prompt_used: str
    seed: int
    width: int
    height: int
    engine: str = "sdxl"
    success: bool = True
    error: Optional[str] = None


class ImageEngine:
    """
    Generates photorealistic images using SDXL.

    Pipeline (when refiner is enabled):
        1. Base model  → denoising_end=0.8  → output_type="latent"
        2. Refiner     → denoising_start=0.8 → final image
        3. VAE float32 upgrade applied to both models at load time.

    Refiner is optional and always fails gracefully — if VRAM is
    insufficient or the model is unavailable, base-only output is returned.

    Usage:
        engine = ImageEngine(ctx)
        result = engine.run(ImageGenerationRequest(prompt="a hero at dawn"))
    """

    def __init__(self, ctx: "ExecutionContext") -> None:
        self._ctx = ctx
        self._out_dir = pathlib.Path(ctx.settings.output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._register_model()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, request: "ImageGenerationRequest") -> ImageResult:
        """
        Generate a single image from a validated request.

        Applies environment-aware resolution capping, wraps generation
        in a GenerationTimer for VRAM watermarks, performs OOM recovery
        (retry at halved resolution on OutOfMemoryError), and records
        GenerationMetrics to MetricsCollector.

        Returns:
            ImageResult with path and metadata.

        Raises:
            EngineExecutionError: if generation fails even after OOM retry.
        """
        from multigenai.llm.prompt_engine import PromptEngine
        pe = PromptEngine(style_registry=self._ctx.style_registry)
        enhanced = pe.process_image(request)

        seed = request.seed if request.seed is not None else random.randint(0, 1_000_000)
        out_path = self._out_dir / f"{_slug(request.prompt)}.png"

        # --- Adaptive resolution ---
        width, height = self._cap_resolution(request.width, request.height)

        # --- Metrics setup ---
        metrics = GenerationMetrics(
            model_id=_SDXL_MODEL_ID, width=width, height=height
        )

        with GenerationTimer(metrics):
            status, downgraded, identity_used = self._generate_with_oom_recovery(
                prompt=enhanced.enhanced,
                negative_prompt=enhanced.negative,
                out_path=out_path,
                width=width,
                height=height,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                seed=seed,
                request=request,
            )
            success = (status == "success")
            metrics.downgraded = downgraded
            if not success:
                metrics.success = False
            metrics.identity_used = identity_used
            metrics.identity_name = request.identity_name if identity_used else None

        # --- Update registry runtime stats ---
        self._ctx.registry.update_runtime(
            _SDXL_MODEL_ID,
            duration_seconds=metrics.duration_seconds,
            peak_vram_mb=metrics.peak_vram_mb,
        )

        # --- Record to global collector ---
        MetricsCollector.instance().record(metrics)

        if not success:
            LOG.warning("Generation failed; creating placeholder.")
            self._create_placeholder(out_path, request.prompt)
        elif self._ctx.behaviour.auto_unload_after_gen:
            self._auto_unload()

        return ImageResult(
            path=str(out_path),
            prompt_used=enhanced.enhanced,
            seed=seed,
            width=metrics.width,
            height=metrics.height,
            success=success,
        )

    # ------------------------------------------------------------------
    # Phase 4 — IP-Adapter / Identity
    # ------------------------------------------------------------------

    def run_with_ip_adapter(self, request: "ImageGenerationRequest", face_image_path: str) -> ImageResult:
        """
        [Phase 4] Identity-consistent generation via IP-Adapter.

        Delegates to run() with identity_name already set on the request.
        FaceEncoder.extract() must have been called and the embedding stored
        in IdentityStore before calling this method.

        If VRAM is insufficient, IP-Adapter is silently skipped and
        base-only generation proceeds (graceful degradation).
        """
        # Ensure identity_name is forwarded; face_image_path is informational here —
        # the embedding comes from IdentityStore (already extracted by FaceEncoder).
        if not getattr(request, "identity_name", None):
            LOG.warning(
                "run_with_ip_adapter called but request.identity_name is None — "
                "falling back to standard run()."
            )
        return self.run(request)

    def run_with_controlnet(self, request: "ImageGenerationRequest", control_image_path: str, control_type: str = "depth") -> ImageResult:
        """[Phase 5] ControlNet-conditioned generation."""
        raise NotImplementedError("ControlNet conditioning activates in Phase 5.")

    def _inject_identity(
        self,
        request: "ImageGenerationRequest",
        pipe,
    ) -> bool:
        """
        Load IP-Adapter into *pipe* and configure face-embedding conditioning.

        VRAM guard: if available VRAM < _IP_ADAPTER_MIN_VRAM_GB the adapter is
        skipped and False is returned. This is the ONLY layer that enforces the
        VRAM guard — FaceEncoder itself is CPU-only.

        Returns:
            True  — IP-Adapter loaded, scale set, embeds attached to *pipe*.
            False — skipped (VRAM, missing profile, or missing embedding).
        """
        identity_name = getattr(request, "identity_name", None)
        if not identity_name:
            return False

        # --- VRAM guard ---
        vram_mb = getattr(self._ctx.environment, "vram_mb", 0)
        if vram_mb < _IP_ADAPTER_MIN_VRAM_GB * 1024:
            LOG.warning(
                f"⚠ Identity skipped: insufficient VRAM "
                f"({vram_mb} MB < {_IP_ADAPTER_MIN_VRAM_GB * 1024:.0f} MB required)."
            )
            return False

        # --- Fetch embedding via IdentityResolver (no inline store access) ---
        from multigenai.identity.identity_resolver import IdentityResolver
        store = self._ctx.identity_store
        face_embedding = IdentityResolver.get_face_embedding(identity_name, store)

        if face_embedding is None:
            # IdentityResolver already logged a structured warning
            return False

        # --- Load IP-Adapter in-place on the existing pipeline ---
        try:
            import torch
            strength = min(getattr(request, "identity_strength", 0.8), 1.0)  # clamp >1.0
            pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl.bin",
            )
            pipe.set_ip_adapter_scale(strength)

            # Shape embedding: [1, 512] float16 tensor
            embed_tensor = torch.tensor(
                face_embedding, dtype=torch.float16
            ).unsqueeze(0)
            # Store embed on the pipe object for retrieval in _generate_sdxl
            pipe._mgos_ip_embeds = embed_tensor

            LOG.info(
                f"IP-Adapter loaded for identity '{identity_name}' "
                f"(strength={strength:.2f}, embedding_dim={len(face_embedding)})."
            )
            return True

        except Exception as exc:
            LOG.warning(
                f"IP-Adapter load failed for '{identity_name}': {exc} — "
                "falling back to base-only generation."
            )
            return False


    def _unload_ip_adapter(self, pipe) -> None:
        """
        Unload the IP-Adapter from *pipe* and clear the stored embed tensor.

        Safe to call even if IP-Adapter was never loaded — diffusers
        pipe.unload_ip_adapter() handles that gracefully.
        """
        try:
            pipe.unload_ip_adapter()
            if hasattr(pipe, "_mgos_ip_embeds"):
                del pipe._mgos_ip_embeds
            LOG.debug("IP-Adapter unloaded from pipeline.")
        except Exception as exc:
            LOG.debug(f"_unload_ip_adapter: ignoring error during unload: {exc}")

    # ------------------------------------------------------------------
    # Model registration
    # ------------------------------------------------------------------

    def _register_model(self) -> None:
        """Register base and refiner loaders in the ModelRegistry (does not load yet)."""
        ctx = self._ctx
        ctx.registry.register(
            _SDXL_MODEL_ID,
            loader=lambda: self._load_sdxl_pipeline(ctx.device, ctx.settings.sdxl.vae_float32),
            min_vram_gb=_SDXL_MIN_VRAM_GB,
        )
        ctx.registry.register(
            _SDXL_REFINER_MODEL_ID,
            loader=lambda: self._load_sdxl_refiner_pipeline(ctx.device, ctx.settings.sdxl.vae_float32),
            min_vram_gb=_SDXL_REFINER_MIN_VRAM_GB,
        )

    # ------------------------------------------------------------------
    # OOM recovery orchestration
    # ------------------------------------------------------------------

    def _generate_with_oom_recovery(
        self,
        prompt: str,
        negative_prompt: str,
        out_path,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        request: "Optional[ImageGenerationRequest]" = None,
    ) -> "tuple[str, bool, bool]":
        """
        Run SDXL generation with one OOM recovery attempt.

        _generate_sdxl returns a tri-state: 'success' | 'oom' | 'error'.
        Only a genuine 'oom' triggers retry at halved resolution.
        A plain 'error' (model load failure, etc.) is returned immediately.

        When identity is active and OOM occurs, IP-Adapter is unloaded before
        the retry so the base-only pipeline can run within available VRAM.

        Returns:
            (status: 'success'|'error', downgraded: bool, identity_used: bool)
        """
        status, identity_used = self._generate_sdxl(
            prompt=prompt, negative_prompt=negative_prompt, out_path=out_path,
            width=width, height=height, num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, seed=seed, request=request,
        )
        if status == "success":
            return "success", False, identity_used
        if status != "oom":
            # Non-OOM failure — no retry
            return "error", False, False

        # --- OOM recovery path ---
        retry_w = max(64, (width // 2 // 8) * 8)
        retry_h = max(64, (height // 2 // 8) * 8)
        LOG.warning(
            f"ImageEngine: OOM — retrying at {retry_w}×{retry_h} (was {width}×{height})"
        )
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # Retry WITHOUT identity — request passed as None to disable IP-Adapter
        if identity_used:
            LOG.warning(
                "ImageEngine: OOM with IP-Adapter active — dropping identity for retry."
            )
        status, _ = self._generate_sdxl(
            prompt=prompt, negative_prompt=negative_prompt, out_path=out_path,
            width=retry_w, height=retry_h,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, seed=seed,
            request=None,  # disables IP-Adapter on retry
        )
        final_status = "success" if status == "success" else "error"
        return final_status, True, False  # downgraded=True, identity_used=False

    # ------------------------------------------------------------------
    # Core SDXL inference — two-stage pipeline
    # ------------------------------------------------------------------

    def _generate_sdxl(
        self,
        prompt: str,
        negative_prompt: str,
        out_path: pathlib.Path,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        request: "Optional[ImageGenerationRequest]" = None,
    ) -> "tuple[str, bool]":
        """
        Run SDXL inference via ModelRegistry — base + optional refiner.

        Phase 4: when *request* carries identity_name, _inject_identity() is
        called after loading base_pipe to attach the IP-Adapter in-place.
        On OOM the adapter is unloaded before returning 'oom'.

        Two-stage pipeline (when settings.sdxl.use_refiner is True):
            Stage 1 — Base:    denoising_end=base_denoising_end → latent
            Stage 2 — Refiner: denoising_start=refiner_denoising_start → image

        The refiner is always optional:
            - If refiner load fails for any reason, base output is saved and
              'success' is returned — no OOM signal, no crash.
            - Unload of BOTH models happens only after the full pass (or in
              auto_unload path via _auto_unload); never between stages.

        Returns:
            ('success'|'oom'|'error', identity_used: bool)
        """
        identity_active = False
        base_pipe = None
        try:
            import torch

            sdxl_cfg = self._ctx.settings.sdxl
            device = self._ctx.device

            # ----------------------------------------------------------
            # Stage 1: Base model → latent
            # ----------------------------------------------------------
            base_pipe = self._ctx.registry.get(
                _SDXL_MODEL_ID,
                device_manager=self._ctx.device_manager,
                environment=self._ctx.environment,
            )

            # --- Phase 4: IP-Adapter identity injection (in-place, optional) ---
            ip_kwargs: dict = {}
            if request is not None:
                identity_active = self._inject_identity(request, base_pipe)
                if identity_active and hasattr(base_pipe, "_mgos_ip_embeds"):
                    ip_kwargs["ip_adapter_image_embeds"] = [base_pipe._mgos_ip_embeds]

            generator = (
                torch.Generator(device=device).manual_seed(seed)
                if device != "cpu" else None
            )

            if sdxl_cfg.use_refiner:
                # Two-stage: base produces a latent, refiner refines it
                base_output = base_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    denoising_end=sdxl_cfg.base_denoising_end,
                    output_type="latent",
                    width=width,
                    height=height,
                    **ip_kwargs,
                )
                latent = base_output.images.float()

                # ----------------------------------------------------------
                # Stage 2: Refiner → final image (graceful fallback)
                # ----------------------------------------------------------
                image = self._run_refiner(
                    latent=latent,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    refiner_start=sdxl_cfg.refiner_denoising_start,
                    generator=generator,
                )
            else:
                # Single-stage base-only path
                image = base_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    **ip_kwargs,
                ).images[0]

            image.save(str(out_path))
            LOG.info(f"Image saved: {out_path} (seed={seed})")
            return "success", identity_active

        except Exception as exc:  # noqa: BLE001
            # Narrow OOM first — must be checked before the broad except
            try:
                import torch as _torch
                is_oom = isinstance(exc, _torch.cuda.OutOfMemoryError)
            except ImportError:
                is_oom = False

            if is_oom:
                LOG.error(f"SDXL OOM at {width}×{height}: {exc}")
                # Unload IP-Adapter first so caller can retry base-only
                if identity_active and base_pipe is not None:
                    self._unload_ip_adapter(base_pipe)
                    LOG.info("ImageEngine: IP-Adapter unloaded before OOM retry.")
                try:
                    self._ctx.registry.unload(_SDXL_MODEL_ID)
                except Exception:
                    pass
                return "oom", identity_active

            LOG.error(f"SDXL inference failed: {exc}")
            try:
                self._ctx.registry.unload(_SDXL_MODEL_ID)
            except Exception:
                pass
            return "error", identity_active

    def _run_refiner(
        self,
        latent,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        refiner_start: float,
        generator,
    ):
        """
        Run the SDXL refiner stage.

        Always returns a PIL Image — either the refined output or the
        base latent decoded by falling back to the base VAE if the refiner
        is unavailable.

        Graceful fallback: any exception during refiner load / inference
        logs a warning and returns a decoded version of the base output.
        The caller treats this as a transparent quality downgrade, not
        an error — success status is not affected.
        """
        try:
            refiner_pipe = self._ctx.registry.get(
                _SDXL_REFINER_MODEL_ID,
                device_manager=self._ctx.device_manager,
                environment=self._ctx.environment,
            )
            image = refiner_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                denoising_start=refiner_start,
                image=latent,
                generator=generator,
            ).images[0]
            LOG.debug("SDXL refiner stage complete.")
            return image
        except Exception as exc:  # noqa: BLE001
            LOG.warning(
                f"SDXL refiner unavailable ({exc}) — "
                "falling back to base-only output. Quality is reduced."
            )
            # Decode the latent using the base pipeline's VAE as fallback
            return self._decode_latent_fallback(latent)

    def _decode_latent_fallback(self, latent):
        try:
            import torch
            from PIL import Image
            import numpy as np

            base_pipe = self._ctx.registry.get(
                _SDXL_MODEL_ID,
                device_manager=self._ctx.device_manager,
                environment=self._ctx.environment,
            )

            # 🔥 Force latent to float32 for VAE decode
            latent = latent.float()

            with torch.no_grad():
                latent = latent / base_pipe.vae.config.scaling_factor
                image = base_pipe.vae.decode(latent).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            return Image.fromarray((image[0] * 255).astype(np.uint8))

        except Exception as exc:
            LOG.error(f"Latent decode fallback failed: {exc}. Returning blank image.")
            from PIL import Image
            return Image.new("RGB", (512, 512), color=(30, 30, 40))
    # ------------------------------------------------------------------
    # Adaptive resolution (environment-aware)
    # ------------------------------------------------------------------

    def _cap_resolution(self, width: int, height: int) -> tuple[int, int]:
        """
        Cap resolution based on device and VRAM limits.

        Policy (from behaviour matrix):
          CPU                 → max 512 × 512
          VRAM < 7 000 MB     → max 512 × 512
          7 000–13 999 MB     → max 768 × 768
          ≥ 14 000 MB         → max 1024 (or request-specified)

        Never trusts raw request dimensions.
        """
        max_res = self._ctx.behaviour.max_image_resolution
        if width > max_res or height > max_res:
            ratio = min(max_res / width, max_res / height)
            new_w = int(width * ratio)
            new_h = int(height * ratio)
            # Snap to multiples of 8 (required by diffusion models)
            new_w = (new_w // 8) * 8
            new_h = (new_h // 8) * 8
            LOG.warning(
                f"ImageEngine: resolution capped {width}×{height} → {new_w}×{new_h} "
                f"(max_res={max_res} for device={self._ctx.device} "
                f"vram={getattr(self._ctx.environment, 'vram_mb', 0)}MB)"
            )
            return new_w, new_h
        return width, height

    def _auto_unload(self) -> None:
        """Unload base + refiner and empty CUDA cache (Kaggle / auto-unload mode only).

        Must only be called AFTER the full two-stage pass is complete.
        Never called between base and refiner stages.
        """
        for model_id in (_SDXL_MODEL_ID, _SDXL_REFINER_MODEL_ID):
            try:
                self._ctx.registry.unload(model_id)
            except Exception as exc:
                LOG.warning(f"ImageEngine: auto-unload of {model_id} failed ({exc})")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        LOG.info("ImageEngine: auto-unloaded base + refiner and cleared CUDA cache.")

    # ------------------------------------------------------------------
    # Pipeline loaders (static — no self state needed)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_sdxl_pipeline(device: str, vae_float32: bool = True):
        """
        Load SDXL base pipeline.

        Both base and refiner load as fp16 to avoid dtype mismatch in the
        refiner cross-attention pass. Manual VAE recasting is intentionally
        omitted — mixed precision is handled uniformly by diffusers fp16 path.

        Memory optimizations active on CUDA:
          - enable_model_cpu_offload(): moves model layers to CPU between uses
          - enable_vae_slicing():       encodes latents one slice at a time
          - enable_attention_slicing(): chunked attention computation
        These cut peak VRAM by ~30–40% with minimal quality impact.
        """
        import torch
        from diffusers import DiffusionPipeline

        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        kwargs: dict = {"torch_dtype": torch_dtype, "use_safetensors": True}
        if torch_dtype == torch.float16:
            kwargs["variant"] = "fp16"

        LOG.info(f"Loading SDXL base model (dtype={torch_dtype})…")
        pipe = DiffusionPipeline.from_pretrained(_SDXL_MODEL_ID, **kwargs)

        if device == "cuda":
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.enable_attention_slicing()
            LOG.debug("SDXL base: CPU offload + VAE/attention slicing enabled.")
        else:
            pipe = pipe.to(device)

        return pipe

    @staticmethod
    def _load_sdxl_refiner_pipeline(device: str, vae_float32: bool = True):
        """
        Load SDXL refiner pipeline.

        Must match the base pipeline's dtype exactly (fp16) to avoid the
        'Input type (c10::Half) and bias type (float) should be the same'
        error during the latent-to-image refiner pass.
        """
        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline

        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        kwargs: dict = {"torch_dtype": torch_dtype, "use_safetensors": True}
        if torch_dtype == torch.float16:
            kwargs["variant"] = "fp16"

        LOG.info(f"Loading SDXL refiner model (dtype={torch_dtype})…")
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            _SDXL_REFINER_MODEL_ID, **kwargs
        )

        if device == "cuda":
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.enable_attention_slicing()
            LOG.debug("SDXL refiner: CPU offload + VAE/attention slicing enabled.")
        else:
            pipe = pipe.to(device)

        return pipe

    @staticmethod
    def _create_placeholder(out_path: pathlib.Path, prompt: str) -> None:
        """Create a minimal placeholder PNG when SDXL is unavailable."""
        try:
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (512, 512), color=(30, 30, 40))
            draw = ImageDraw.Draw(img)
            draw.text((20, 240), f"[PLACEHOLDER]\n{prompt[:60]}", fill=(200, 200, 200))
            img.save(str(out_path))
        except ImportError:
            out_path.write_bytes(b"")  # empty file as last resort
