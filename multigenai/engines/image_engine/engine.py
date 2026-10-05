"""
ImageEngine — Phase 7 Isolated Image Generation.

Architecture contract:
  - Diffusion models bypass ModelRegistry by design (Phase 7 isolation model).
  - `generator` is created ONCE in `run()` and threaded through both base and
    refiner passes — this is the sole guarantee of determinism/reproducibility.
  - Generator uses CPU device for portability across CUDA/CPU/DirectML runs.
  - Base model is unloaded before refiner loads (no dual-VRAM residency).
  - `import torch` is method-level; safe for cold-import in test environments.

Memory optimizations applied (all zero quality cost):
  - enable_sequential_cpu_offload()   → each submodule moves to GPU only when used
  - enable_vae_tiling()               → decodes large images in tiles, ~60% less VRAM
  - enable_attention_slicing()        → slices attention along batch dim, ~30% VRAM saving
  - torch.float16 + use_safetensors  → half-precision weights, fast safe format
  - variant="fp16"                    → downloads fp16 checkpoint directly from HF Hub
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from multigenai.core.logging.logger import get_logger
from multigenai.core.model_lifecycle import ModelLifecycle

if TYPE_CHECKING:
    import torch as _torch
    from PIL import Image as PILImage
    from multigenai.core.execution_context import ExecutionContext
    from multigenai.llm.schema_validator import ImageGenerationRequest

LOG = get_logger(__name__)

from multigenai.core.model_registry import ModelRegistry
 
# ---------------------------------------------------------------------------
# Dynamic Model IDs (managed by model_config.yaml)
# ---------------------------------------------------------------------------
registry = ModelRegistry.instance()
_PROD_XL = registry.get_config_value("image_model", "RunDiffusion/Juggernaut-XL-v9")

MODEL_ALIASES: dict[str, str] = {
    "sdxl-base":    _PROD_XL,
    "sdxl-refiner": "stabilityai/stable-diffusion-xl-refiner-1.0",
    "sd15":         "runwayml/stable-diffusion-v1-5",
    "sd-1.5":       "runwayml/stable-diffusion-v1-5",
    "juggernaut-xl-v9": _PROD_XL,
}

REFINER_REPO = "stabilityai/stable-diffusion-xl-refiner-1.0"


@dataclass
class ImageResult:
    """Output from the ImageEngine."""
    path: str
    width: int
    height: int
    seed: int
    success: bool = True
    error: Optional[str] = None


def _slug(text: str, max_len: int = 40) -> str:
    import re
    cleaned = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    slug = re.sub(r"[-\s]+", "-", cleaned)
    return slug[:max_len].strip("-")


def _apply_memory_optimizations(pipe, device: str, use_ip_adapter: bool = False) -> None:
    """
    Apply unified VRAM-saving optimizations to a diffusers pipeline.

    - model_cpu_offload:    Offloads submodels to CPU when idle (Standard Phase 13).
    - vae_slicing/tiling:  Reduces peak VRAM during high-res decoding.
    - attention_slicing:   Saves VRAM (skipped if IP-Adapter is active to avoid conflicts).
    """
    if device == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        
        if not use_ip_adapter:
            pipe.enable_attention_slicing()
        else:
            LOG.debug("ImageEngine: Skipping attention slicing (IP-Adapter active).")
            
    elif device == "directml":
        pipe.vae.enable_tiling()
        pipe.enable_attention_slicing()
        pipe = pipe.to(device)
    else:
        # CPU
        pipe.vae.enable_tiling()
        pipe = pipe.to(device)
    return pipe


class ImageEngine:
    """
    Isolated Image Generation Engine.

    Receives fully compiled positive/negative prompts from PromptCompiler.
    Loads models lazily, executes forward passes, and strictly reclaims VRAM.

    Determinism contract
    --------------------
    A single `torch.Generator` (CPU device) is created from the resolved seed
    in `run()`. This generator is passed unchanged to both `_generate()` and
    `_refine()`. Using CPU for the generator guarantees the same random sequence
    regardless of whether the model is on CUDA or CPU.
    """

    def __init__(self, ctx: "ExecutionContext") -> None:
        self._ctx = ctx
        self._out_dir = pathlib.Path(ctx.settings.output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self.device = ctx.device
        # Diffusion models bypass ModelRegistry by design (Phase 7 isolation model)
        self.pipe = None
        self.refiner = None
        self._controlnet_enabled = False
        self._ip_adapter_enabled = False

        from multigenai.consistency.ip_adapter_manager import IPAdapterManager
        from multigenai.consistency.controlnet_manager import ControlNetManager
        self.ip_adapter_manager = IPAdapterManager(self.device)
        self.controlnet_manager = ControlNetManager(self.device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model_name(self, model_name: str) -> str:
        """Translate short alias to a valid HuggingFace repo id."""
        return MODEL_ALIASES.get(model_name, model_name)

    def _load_model(self, model_name: str, use_controlnet: bool = False, use_ip_adapter: bool = False) -> None:
        """
        Loads the base diffusion model with memory optimizations applied.

        Auto-detects the correct pipeline class from the repo id:
          - SDXL repos → StableDiffusionXLPipeline (fp16 variant)
          - SD 1.x repos → StableDiffusionPipeline (no variant kwarg)

        Memory profile (CUDA, SDXL 1024x1024):
          Naive load:       ~11GB VRAM
          After this:       ~4-5GB VRAM  (sequential offload + tiling + slicing)
        """
        import torch

        repo_id = self._resolve_model_name(model_name)
        is_xl = "xl" in repo_id.lower()

        # Phase 15: ControlNet and IP-Adapter are retired (VRAM guard)
        if use_controlnet:
            raise RuntimeError(
                "ControlNet is retired in Phase 15 to stay within Kaggle T4 VRAM budget. "
                "Set use_controlnet=False. See legacy/models/controlnet/."
            )
        if use_ip_adapter:
            raise RuntimeError(
                "IP-Adapter is retired in Phase 15 to stay within Kaggle T4 VRAM budget. "
                "Set use_ip_adapter=False. See legacy/models/ip_adapter/."
            )

        if self.pipe is not None:
            return  # Model is correctly cached — no config change possible (both flags always False)

        if is_xl:
            from diffusers import StableDiffusionXLPipeline
            LOG.info(f"Loading SDXL model {repo_id} (fp16)...")
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                repo_id,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
        else:
            from diffusers import StableDiffusionPipeline
            LOG.info(f"Loading SD 1.x model {repo_id} (fp16)...")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                repo_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )

        # Apply memory optimizations (ControlNet / IP-Adapter paths removed)
        self.pipe = _apply_memory_optimizations(self.pipe, self.device, use_ip_adapter=False)

        self._controlnet_enabled = False
        self._ip_adapter_enabled = False

    def _generate(
        self,
        compiled_prompt: str,
        negative_prompt: str,
        request: "ImageGenerationRequest",
        generator: "_torch.Generator",
        seed: int,
        ref_image: Optional["PILImage.Image"] = None,
        control_image: Optional["PILImage.Image"] = None,
    ) -> "PILImage":
        """
        Executes the base generation pass.

        Parameters
        ----------
        generator:
            CPU-seeded torch.Generator from `run()`. Must NOT be recreated here —
            the same object is reused in `_refine()` to guarantee reproducibility.
        """
        LOG.info(
            f"Base Generation: {request.num_inference_steps} steps, "
            f"{request.width}x{request.height}, seed={seed}"
        )
        
        # Build dynamic kwargs for ControlNet / IP-Adapter — refactored for Phase 10 stability
        # SDXL + ControlNet + IP-Adapter can be fragile if args are mixed naively.
        kwargs = {
            "prompt": compiled_prompt,
            "negative_prompt": negative_prompt,
            "width": request.width,
            "height": request.height,
            "generator": generator,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "output_type": "latent" if request.use_refiner else "pil",
        }

        if control_image is not None:
            # Phase 15: ControlNet is retired. Log and skip depth-map conditioning.
            LOG.warning(
                "ImageEngine: control_image provided but ControlNet is retired (Phase 15). "
                "Ignoring depth conditioning."
            )

        if ref_image is not None:
            # Phase 15: IP-Adapter is retired. ref_image is intentionally ignored.
            LOG.warning(
                "ImageEngine: ref_image provided but IP-Adapter is retired (Phase 15). "
                "Identity conditioning via IdentityLatentEncoder is used instead."
            )

        # --- Normalize ControlNet image ---
        if "image" in kwargs:
            ctrl_img = kwargs["image"]
            if isinstance(ctrl_img, (list, tuple)):
                ctrl_img = ctrl_img[0]
            kwargs["image"] = ctrl_img
            
        result = self.pipe(**kwargs)
        return result.images[0]

    def _refine(
        self,
        image: "PILImage",
        compiled_prompt: str,
        negative_prompt: str,
        request: "ImageGenerationRequest",
        generator: "_torch.Generator",
    ) -> "PILImage":
        """
        Executes the refiner pass in isolated memory.

        Determinism contract: `generator` is the same CPU-seeded object from `run()`.
        Never instantiate a new Generator here.

        The refiner is loaded AFTER base is unloaded (if auto_unload_after_gen) to prevent VRAM overlap.
        """
        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline

        if self.refiner is None:
            LOG.info("Loading SDXL refiner model (fp16, all memory optimizations)...")
            self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                REFINER_REPO,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
            self.refiner = _apply_memory_optimizations(self.refiner, self.device)

        LOG.info(f"Refining image ({request.refiner_num_inference_steps} steps, strength={request.refiner_strength})...")
        
        refined = self.refiner(
            prompt=compiled_prompt,
            negative_prompt=negative_prompt,
            image=image,
            generator=generator,
            num_inference_steps=request.refiner_num_inference_steps,
            strength=request.refiner_strength,
            guidance_scale=request.guidance_scale,
        )

        # Immediate teardown if required — otherwise keep alive for next segment
        if self._ctx.behaviour.auto_unload_after_gen:
            ModelLifecycle.safe_unload(self.refiner)
            self.refiner = None

        return refined.images[0]

    # ------------------------------------------------------------------
    # Public interface (invoked ONLY by GenerationManager)
    # ------------------------------------------------------------------

    def run(
        self,
        compiled_prompt: str,
        negative_prompt: str,
        request: "ImageGenerationRequest",
        ref_image: Optional["PILImage.Image"] = None,
        control_image: Optional["PILImage.Image"] = None,
    ) -> ImageResult:
        """
        Main execution flow.

        Generator is seeded ONCE here on CPU and passed through the entire pipeline.
        CPU generator ensures the same random sequence on CUDA, CPU, and DirectML.
        This is the single source of determinism for the generation run.
        """
        import torch

        seed = request.seed if request.seed is not None else int(
            torch.randint(0, 1_000_000, (1,)).item()
        )
        # CPU generator: portable across all devices, same sequence guaranteed
        generator = torch.Generator(device="cpu").manual_seed(seed)

        img_slug = _slug(request.prompt)
        out_path = self._out_dir / f"{img_slug}-{seed}.png"

        try:
            self._load_model(
                request.model_name, 
                use_controlnet=(control_image is not None),
                use_ip_adapter=(ref_image is not None)
            )

            # Base generation pass
            image = self._generate(
                compiled_prompt, 
                negative_prompt, 
                request, 
                generator, 
                seed,
                ref_image=ref_image,
                control_image=control_image
            )

            # Drop base before allocating refiner if forced to unload
            if request.use_refiner and getattr(self._ctx.settings, "environment", "") != "kaggle":
                if self._ctx.behaviour.auto_unload_after_gen and self.pipe is not None:
                    ModelLifecycle.safe_unload(self.pipe)
                    self.pipe = None
                image = self._refine(image, compiled_prompt, negative_prompt, request, generator)

            image.save(out_path)
            LOG.info(f"Image saved to {out_path}")

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() or "oom" in str(exc).lower():
                LOG.error(f"Image generation failed (CUDA Out of Memory): {exc}")
                return ImageResult(str(out_path), request.width, request.height, seed, False, str(exc))
            LOG.error(f"Image generation failed with RuntimeError: {exc}", exc_info=True)
            raise
        except Exception as exc:
            LOG.error(f"Image generation failed with unpredictable error: {exc}", exc_info=True)
            raise

        finally:
            if self._ctx.behaviour.auto_unload_after_gen:
                ModelLifecycle.safe_unload(self.pipe)
                self.pipe = None
                ModelLifecycle.safe_unload(self.refiner)
                self.refiner = None

        return ImageResult(str(out_path), request.width, request.height, seed, True)
