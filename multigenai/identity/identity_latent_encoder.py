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
        import numpy as np

        # Resolve the real execution device. Diffusers pipelines with CPU offloading
        # will report module parameters as "cpu" while resting, but execute on "cuda".
        vae_device = next(pipe.vae.parameters()).device
        vae_dtype = next(pipe.vae.parameters()).dtype

        # Exact RGB normalization as specified by user
        image = image.convert("RGB")
        img_arr = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0)
        image_tensor = (image_tensor * 2 - 1).to(device=vae_device, dtype=vae_dtype)

        with torch.no_grad():
            latent = pipe.vae.encode(image_tensor).latent_dist.sample()

        # Scale by the exact VAE scaling factor requested
        latent = latent * 0.18215

        return latent.detach()
