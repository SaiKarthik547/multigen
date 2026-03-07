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
        Supports single-pass generation of both PIL frames and latents.
        
        Args:
            return_latents: If True, returns (output, latents) where output.frames
                           contain decoded/post-processed images if requested.
        """
        if not return_latents:
            return super().__call__(*args, **kwargs)

        # Force output_type to "latent" to get the 5D diffusion tensor
        # [batch, frames, channels, height/8, width/8]
        temp_kwargs = kwargs.copy()
        requested_output_type = temp_kwargs.get("output_type", "pil")
        temp_kwargs["output_type"] = "latent"
        
        output = super().__call__(*args, **temp_kwargs)
        
        latents = output.frames
        
        # Ensure it's 5D (batch, frames, channels, h, w)
        if latents.ndim == 4:
            latents = latents.unsqueeze(1)
            
        if requested_output_type == "pil":
            # Decode frames manually to save a second diffusion pass
            batch, frames, channels, h, w = latents.shape
            
            # Reshape for VAE (chunking is handled by manual loop or self.vae.decode)
            # We follow the suggestion to use self.vae.decode directly
            latents_reshaped = latents.reshape(batch * frames, channels, h, w)
            
            # scaling_factor is usually 0.18215 for SD/SVD
            scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
            
            # VAE decode
            decoded = self.vae.decode(latents_reshaped / scaling_factor).sample
            
            # Reshape back to (batch, frames, channels, H, W)
            decoded = decoded.reshape(batch, frames, *decoded.shape[1:])
            
            # Post-process to PIL images
            images = self.image_processor.postprocess(decoded, output_type="pil")
            output.frames = images

        return output, latents
