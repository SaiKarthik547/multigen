from diffusers import StableVideoDiffusionPipeline
import torch
from typing import Optional, Union, List, Dict, Any

class TemporalStableVideoDiffusionPipeline(StableVideoDiffusionPipeline):
    """
    Extended SVD pipeline that supports warm-starting from existing latents.
    """
    
    def prepare_latents(
        self,
        batch_size,
        num_frames,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None
    ):
        """
        Overrides the default latent preparation to respect the 'latents' argument.
        """
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        return super().prepare_latents(
            batch_size,
            num_frames,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator
        )

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        """
        Wraps the call to ensure latents are captured if requested.
        """
        # The standard UNet loop in SVD doesn't return latents, 
        # but we can capture them if the output_type is 'latent'.
        # For simplicity in this implementation, we rely on prepare_latents 
        # override and standard return logic.
        return super().__call__(*args, **kwargs)
