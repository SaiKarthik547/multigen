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
        processor = getattr(pipe, "image_processor", None)
        if processor is None:
            return None

        # Process image for VAE boundaries
        image_tensor = processor.preprocess(previous_frame)
        image_tensor = image_tensor.to(pipe.device, dtype=pipe.dtype)
        
        # Standard scaling
        scaling_factor = pipe.vae.config.scaling_factor if hasattr(pipe.vae.config, "scaling_factor") else 0.18215
        dist = pipe.vae.encode(image_tensor)
        
        # Extract direct vector map
        if hasattr(dist, "latent_dist"):
            latents = dist.latent_dist.sample()
        else:
            return None
            
        return latents * scaling_factor
