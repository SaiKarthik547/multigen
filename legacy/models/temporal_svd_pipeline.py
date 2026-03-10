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
            return latents.to(device=device, dtype=dtype).clone()

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

        if latents.ndim != 5:
            raise ValueError("Unexpected latent shape")
            
        if requested_output_type == "pil":
            # Decode frames manually to save a second diffusion pass
            batch, frames, channels, h, w = latents.shape
            
            # Ensure it's 5D (batch, frames, channels, h, w) for manual decoding
            assert latents.ndim == 5, f"Manual decode expects 5D latents, got {latents.ndim}D"
            
            # scaling_factor is usually 0.18215 for SD/SVD
            scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
            
            # High-precision VAE decoding (research-correct solution)
            # SVD VAE often produces artifacts or NaN in FP16; we force FP32.
            vae_dtype = self.vae.dtype
            self.vae.to(torch.float32)
            
            chunk_size = 4
            decoded_chunks = []
            
            for i in range(0, frames, chunk_size):
                latent_chunk = latents[:, i:i + chunk_size]
                b, f, c, h_lat, w_lat = latent_chunk.shape
                
                # Reshape for VAE: [batch * f, c, h, w]
                latent_chunk_reshaped = latent_chunk.to(torch.float32).reshape(b * f, c, h_lat, w_lat)
                
                # SVD VAE expects latents / scaling_factor
                decoded_chunk = self.vae.decode(
                    latent_chunk_reshaped / scaling_factor,
                    num_frames=f
                ).sample
                
                # Reshape back to (batch, f, C, H, W)
                decoded_chunks.append(decoded_chunk.reshape(b, f, *decoded_chunk.shape[1:]).cpu())
                del decoded_chunk
                
            # Restore model dtype
            self.vae.to(vae_dtype)
            
            # Combine all chunks: [batch, frames, channels, H, W]
            decoded = torch.cat(decoded_chunks, dim=1)
            
            # CRITICAL FIX: video_processor.postprocess() expects 4D tensor [N, C, H, W]
            # where N = batch * frames.
            b, f, c_img, h_img, w_img = decoded.shape
            decoded_reshaped = decoded.reshape(b * f, c_img, h_img, w_img)
            
            # Post-process to PIL images
            images = self.video_processor.postprocess(decoded_reshaped, output_type="pil")
            
            # Reconstruct batch structure: List[List[PIL.Image]]
            # Diffusers SVD pipelines return a list of lists of images.
            batch_frames = []
            for i in range(b):
                batch_frames.append(images[i * f : (i + 1) * f])
            
            output.frames = batch_frames

        return output, latents
