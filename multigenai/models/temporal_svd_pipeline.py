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
    def __call__(self, *args, return_latents=False, **kwargs):
        """
        Wraps the call to ensure 5D diffusion latents are captured before VAE decoding.
        
        Args:
            return_latents: If True, forces output_type="latent" and returns (output, latents).
        """
        if not return_latents:
            return super().__call__(*args, **kwargs)

        # Force output_type to "latent" to get the 5D diffusion tensor
        # [batch, frames, channels, height/8, width/8]
        temp_kwargs = kwargs.copy()
        temp_kwargs["output_type"] = "latent"
        output = super().__call__(*args, **temp_kwargs)
        
        latents = output.frames
        
        # Ensure it's 5D (batch, frames, channels, h, w)
        if latents.ndim == 4:
            latents = latents.unsqueeze(1)
            
        return output, latents
