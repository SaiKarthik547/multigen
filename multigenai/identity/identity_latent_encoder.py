"""
IdentityLatentEncoder — Phase 15

Extracts the foundational latent geometry of a character anchor image using
the pipeline's VAE. Ensures the extracted latent is always on the same device
as the VAE weights, even when model CPU offloading is active.

Research basis:
  - DreamVideo identity latent conditioning
  - VideoCrafter anchor latent injection
  - Tune-A-Video temporal identity
"""

from __future__ import annotations

from typing import Any
from PIL import Image
import torch


class IdentityLatentEncoder:
    """
    Encodes a PIL Image into a diffusion latent using the pipeline's VAE.
    """

    def encode(self, pipe: Any, image: Image.Image) -> torch.Tensor:
        """
        Encode character image into the pipeline latent space.

        Device is resolved from actual VAE weight parameters to avoid
        cross-device errors when model_cpu_offload() is active.

        Args:
            pipe:  The loaded diffusers pipeline (AnimateDiff or SDXL).
            image: PIL Image of the character anchor.

        Returns:
            torch.Tensor: Latent [1, C, H/8, W/8] scaled by VAE scaling_factor,
                          on the same device as the VAE weights, detached from graph.
        """
        # Resolve the real execution device. Diffusers pipelines with CPU offloading
        # will report module parameters as "cpu" while resting, but execute on "cuda".
        # pipe._execution_device is the official, safe property.
        vae_device = getattr(pipe, "_execution_device", getattr(pipe, "device", None))
        if vae_device is None:
            vae_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Preprocess PIL → normalised tensor [-1, 1] using diffusers processor
        image_tensor = pipe.image_processor.preprocess(image)

        # Cast to VAE device + fp16 to match AnimateDiff domain exactly
        image_tensor = image_tensor.to(device=vae_device, dtype=torch.float16)

        with torch.no_grad():
            latent = pipe.vae.encode(image_tensor).latent_dist.sample()

        # Scale by the VAE scaling factor (0.18215 for SD1.5, 0.13025 for SDXL)
        scaling_factor = getattr(pipe.vae.config, "scaling_factor", 0.18215)
        latent = latent * scaling_factor

        return latent.detach()
