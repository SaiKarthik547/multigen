"""
Identity Latent Encoder - Phase 14

Extracts the foundational latent geometry of a character anchor image using
the pipeline's VAE. This latent represents the core identity geometry.
"""

from typing import Any
from PIL import Image

class IdentityLatentEncoder:
    """
    Encodes a PIL Image into a diffusion latent using the pipeline's VAE.
    """

    def encode(self, pipe: Any, image: Image.Image):
        """
        Encode character image into the pipeline latent space.
        
        Args:
            pipe: The loaded diffusers pipeline (e.g. SDXL or AnimateDiff).
            image: PIL Image of the character anchor.
            
        Returns:
            torch.Tensor: Latent tensor of shape [1, 4, H/8, W/8] scaled
                          by the VAE's config.scaling_factor.
        """
        import torch

        # Use the pipeline's native image processor to guarantee correct normalization
        image_tensor = pipe.image_processor.preprocess(image)
        image_tensor = image_tensor.to(device=pipe.device, dtype=pipe.dtype)

        with torch.no_grad():
            latent = pipe.vae.encode(image_tensor).latent_dist.sample()

        # Scale by the VAE scaling factor to match expected noise magnitude
        # Usually ~0.13025 for SDXL or ~0.18215 for SD1.5
        scaling_factor = getattr(pipe.vae.config, "scaling_factor", 0.18215)
        return latent * scaling_factor
