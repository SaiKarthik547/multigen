import torch
from PIL import Image

class TrajectoryEncoder:
    """
    Extracts trajectory/motion latent from the previous frame using standard VAE encoding 
    to provide structural continuity offset without heavy IP-Adapter overhead.
    """
    def encode(self, pipe, previous_frame: Image.Image) -> torch.Tensor:
        """
        Encode PIL image using the pipeline's VAE to inject continuous structure.
        Ensures exact spatial constraint map logic.
        """
        # Phase 16 Fix: Resolve true execution device resistant to CPU offloading
        vae_device = getattr(pipe, "_execution_device", getattr(pipe, "device", None))
        if vae_device is None:
            vae_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if hasattr(pipe, "image_processor") and pipe.image_processor is not None:
            processor = pipe.image_processor
        else:
            from diffusers.image_processor import VaeImageProcessor
            processor = VaeImageProcessor(vae_scale_factor=8)

        # Process image for VAE boundaries
        image_tensor = processor.preprocess(previous_frame)
        image_tensor = image_tensor.to(device=vae_device, dtype=pipe.dtype)
        
        # Standard scaling
        scaling_factor = pipe.vae.config.scaling_factor if hasattr(pipe.vae.config, "scaling_factor") else 0.18215
        dist = pipe.vae.encode(image_tensor)
        
        # Extract direct vector map
        if hasattr(dist, "latent_dist"):
            latents = dist.latent_dist.sample()
        else:
            return None
            
        latents = latents * scaling_factor
        latents = latents.to(pipe.dtype)
        return torch.clamp(latents, -4.0, 4.0)
