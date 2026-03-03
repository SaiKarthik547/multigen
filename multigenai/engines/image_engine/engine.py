"""
ImageEngine — Phase 7 Isolated Image Generation.

Runs stable diffusion XL pipelines stripped of any global registry dependencies.
Includes explicit loading and unloading of Base and Optional Refiner passes
guaranteeing no dual-VRAM residency.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from multigenai.core.logging.logger import get_logger
from multigenai.core.model_lifecycle import ModelLifecycle

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from multigenai.core.execution_context import ExecutionContext
    from multigenai.llm.schema_validator import ImageGenerationRequest

LOG = get_logger(__name__)


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


class ImageEngine:
    """
    Isolated Image Generation Engine.

    Receives fully compiled positive/negative prompts from the PromptCompiler.
    Loads models lazily, executes forward passes, and strictly reclaims VRAM.
    """

    def __init__(self, ctx: "ExecutionContext") -> None:
        self._ctx = ctx
        self._out_dir = pathlib.Path(ctx.settings.output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self.device = ctx.device
        self.pipe = None
        self.refiner = None

    def _load_model(self, model_name: str) -> None:
        """Loads the SDXL base model with sequential CPU offloading."""
        import torch
        from diffusers import StableDiffusionXLPipeline
        
        # Default to base model if simple name provided
        repo_id = "stabilityai/stable-diffusion-xl-base-1.0" if "base" in model_name.lower() else model_name

        LOG.info(f"Loading base model {repo_id} (fp16, sequential CPU offload)...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )

        if self.device == "cuda":
            self.pipe.enable_sequential_cpu_offload()
            self.pipe.enable_attention_slicing()
        else:
            self.pipe = self.pipe.to(self.device)

    def _generate(self, compiled_prompt: str, negative_prompt: str, request: "ImageGenerationRequest") -> "PILImage":
        """Executes the base generation pass."""
        import torch
        seed = request.seed if request.seed is not None else torch.randint(0, 1_000_000, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        LOG.info(f"Base Generation: 30 steps, {request.width}x{request.height}, seed={seed}")
        result = self.pipe(
            prompt=compiled_prompt,
            negative_prompt=negative_prompt,
            width=request.width,
            height=request.height,
            generator=generator,
            num_inference_steps=30,
            output_type="latent" if request.use_refiner else "pil"
        )
        return result.images[0]

    def _refine_optional(self, image, compiled_prompt: str, request: "ImageGenerationRequest") -> "PILImage":
        """Executes the refiner pass in isolated memory."""
        if not request.use_refiner:
            return image

        from diffusers import StableDiffusionXLImg2ImgPipeline

        LOG.info("Loading SDXL refiner model...")
        self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )

        if self.device == "cuda":
            self.refiner.enable_sequential_cpu_offload()
            self.refiner.enable_attention_slicing()
        else:
            self.refiner = self.refiner.to(self.device)

        LOG.info("Refining image...")
        refined = self.refiner(
            prompt=compiled_prompt,
            image=image,
            num_inference_steps=30,
            strength=0.2 # Refiner strength
        )

        # Immediate teardown
        ModelLifecycle.safe_unload(self.refiner)
        self.refiner = None

        return refined.images[0]

    def run(self, compiled_prompt: str, negative_prompt: str, request: "ImageGenerationRequest") -> ImageResult:
        """
        Main execution flow properly decoupled from prompt generation.
        """
        seed = request.seed if request.seed is not None else 42
        img_slug = _slug(request.prompt)
        out_path = self._out_dir / f"{img_slug}-{seed}.png"

        try:
            self._load_model(request.model_name)
            
            # Base generation
            image = self._generate(compiled_prompt, negative_prompt, request)
            
            # Immediately drop base pipe before allocating refiner memory 
            # to guarantee no overlap on Kaggle GPUs
            if getattr(request, 'use_refiner', True):
                ModelLifecycle.safe_unload(self.pipe)
                self.pipe = None
                
            image = self._refine_optional(image, compiled_prompt, request)

            image.save(out_path)
            LOG.info(f"Image saved to {out_path}")

        except Exception as exc:
            LOG.error(f"Image generation failed: {exc}")
            return ImageResult(str(out_path), request.width, request.height, seed, False, str(exc))
        finally:
            ModelLifecycle.safe_unload(self.pipe)
            self.pipe = None
            ModelLifecycle.safe_unload(self.refiner)
            self.refiner = None

        return ImageResult(str(out_path), request.width, request.height, seed, True)
