"""
LatentPropagator — Phase 5: Subtle noise injection for temporal micro-motion.

Adds controlled Gaussian noise to a PIL image before each img2img pass so that
successive frames exhibit slight natural variation without hard visual cuts.

The multiplier (0.02) is intentionally tiny — it prevents the noise from
producing visible artifacts while still perturbing the diffusion trajectory
enough to keep frames from being pixel-identical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


class LatentPropagator:
    """Propagates frame state across video frames for temporal coherence."""

    def propagate(self, latents, drift: float = 0.015, generator=None):
        """
        Add subtle Gaussian noise drift to AnimateDiff latents.

        This directly manipulates the tensor without PIL conversions,
        preserving the exact VRAM footprint and dtype required for diffusion.

        Args:
            latents:   torch.Tensor (usually [1, 4, F, H, W] for video).
            drift:     Multiplier for the noise injection magnitude.
            generator: Optional torch.Generator for deterministic noise.

        Returns:
            torch.Tensor with drifted noise applied.
        """
        import torch
        
        noise = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
        return latents + noise * drift
