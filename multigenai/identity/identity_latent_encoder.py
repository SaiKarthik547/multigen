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

        # Ensure we match the device of the actual VAE weights to avoid offload mismatches (Phase 14 Kaggle)
        # Initialize vae_device to a default before the try block to prevent UnboundLocalError
        # in case both try and except blocks fail to assign it (though unlikely with current logic).
        # Also, ensure mock objects or pipelines without parameters accessible via next()
        # correctly fall back to the pipe's device or 'cpu'.
        vae_device = "cpu" # Default fallback
        try:
            # Attempt to get the device from VAE parameters
            vae_device = next(pipe.vae.parameters()).device
        except (StopIteration, AttributeError):
            # Fallback for mock objects or pipelines without parameters accessible via next()
            vae_device = getattr(pipe, "device", "cpu")
            
        image_tensor = image_tensor.to(device=vae_device, dtype=pipe.dtype)

        with torch.no_grad():
            latent = pipe.vae.encode(image_tensor).latent_dist.sample()

        # Scale by the VAE scaling factor to match expected noise magnitude
        # Usually ~0.13025 for SDXL or ~0.18215 for SD1.5
        scaling_factor = getattr(pipe.vae.config, "scaling_factor", 0.18215)
        return latent * scaling_factor
