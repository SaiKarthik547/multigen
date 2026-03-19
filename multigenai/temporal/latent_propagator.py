"""
LatentPropagator — Phase 15 Directional Temporal Propagation.

Replaces naive random-noise injection with directional velocity propagation
derived from the latent trajectory (current − previous). This prevents the
body distortion caused by uncorrelated noise drifting structure away from
the identity anchor.

Architecture basis:
  - VideoCrafter temporal latent trajectory
  - CogVideoX directional propagation
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch


class LatentPropagator:
    """Propagates frame state across video frames for temporal coherence."""

    def propagate(
        self,
        latents: torch.Tensor,
        prev_latent: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
        drift: float = 0.015,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Directional latent propagation to maintain structural continuity.

        If a previous latent is available, uses directional velocity (current − prev)
        scaled by a small multiplier. If not (first scene), falls back to a tiny random
        drift to keep frames from being pixel-identical.

        Args:
            latents:      Current latent tensor  [1, C, F, H, W] or [1, C, H, W].
            prev_latent:  Latent from previous scene window. None on first scene.
            velocity:     Cached velocity tensor from previous call. None initially.
            drift:        Scale factor for propagation magnitude (default 0.05 dir / 0.015 noise).
            generator:    Optional torch.Generator for deterministic fallback noise.

        Returns:
            Tuple of (propagated_latents, new_velocity).
            new_velocity is None when prev_latent was None (first scene).
        """
        if prev_latent is None:
            # First scene: apply tiny random perturbation only
            noise = torch.randn(
                latents.shape,
                generator=generator,
                device=latents.device,
                dtype=latents.dtype,
            )
            return torch.clamp(latents + noise * drift, -4.0, 4.0), None

        # Align device + dtype: prev_latent may be on CPU (stored in TemporalState) while
        # latents are on CUDA (active inference). Must coerce before any arithmetic.
        if prev_latent.device != latents.device or prev_latent.dtype != latents.dtype:
            prev_latent = prev_latent.to(device=latents.device, dtype=latents.dtype)

        # Align velocity if provided
        if velocity is not None and (velocity.device != latents.device or velocity.dtype != latents.dtype):
            velocity = velocity.to(device=latents.device, dtype=latents.dtype)

        # Align shapes: prev_latent may have a different temporal length
        if prev_latent.shape != latents.shape:
            import torch.nn.functional as F
            if prev_latent.dim() == 5 and latents.dim() == 5:
                b, c, f, h, w = latents.shape
                prev_reshaped = prev_latent[:, :, :f, :, :]  # clip temporal dim
                if prev_reshaped.shape[-2:] != (h, w):
                    prev_flat = prev_reshaped.reshape(b * f, c, *prev_reshaped.shape[-2:])
                    prev_flat = F.interpolate(prev_flat, size=(h, w), mode="bilinear", align_corners=False)
                    prev_reshaped = prev_flat.reshape(b, c, f, h, w)
                prev_latent = prev_reshaped
            elif prev_latent.dim() == 4 and latents.dim() == 4:
                if prev_latent.shape[-2:] != latents.shape[-2:]:
                    prev_latent = F.interpolate(prev_latent, size=latents.shape[-2:], mode="bilinear", align_corners=False)

        # Compute or reuse directional velocity
        if velocity is None:
            velocity = latents - prev_latent

        # Directional step: nudge latents along trajectory
        propagated = latents + velocity * 0.05
        
        # Phase 16: Apply temporal smoothing to reduce pose jitter
        smoothed = self.smooth(propagated, prev_latent)
        
        return torch.clamp(smoothed, -4.0, 4.0), velocity

    def smooth(self, latent: torch.Tensor, prev: Optional[torch.Tensor]) -> torch.Tensor:
        """Phase F Temporal smoothing to reduce pose jitter."""
        if prev is None:
            return latent
        latents = 0.85 * latent + 0.15 * prev
        latents = latents.clamp(-4.0, 4.0)
        return latents
