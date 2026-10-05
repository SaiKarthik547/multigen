import torch

class LatentPropagator:
    """
    Handles temporal latent propagation using velocity vectors 
    to maintain motion continuity across generated segments.
    """
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype

    def calculate_velocity(self, current_latents: torch.Tensor, previous_latents: torch.Tensor) -> torch.Tensor:
        """
        Calculate the delta (velocity) between two latent states.
        """
        if current_latents is None or previous_latents is None:
            return torch.zeros_like(current_latents) if current_latents is not None else None
            
        # Simple directional delta for Phase 15 continuity
        velocity = current_latents - previous_latents
        return velocity

    def propagate(self, latents: torch.Tensor, velocity: torch.Tensor, strength: float = 0.1) -> torch.Tensor:
        """
        Apply velocity propagation to a new noise latent to seed motion.
        """
        if velocity is None:
            return latents
            
        # Inject motion bias into the initial noise
        propagated_latents = latents + (velocity * strength)
        return propagated_latents
