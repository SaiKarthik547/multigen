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

    def inject_noise(self, image, strength: float, generator=None):
        """
        Add subtle Gaussian noise to a PIL image before an img2img pass.

        The noise amplitude is scaled by ``strength * 0.02`` to keep it
        imperceptible to the eye while still perturbing the diffusion
        trajectory (preventing frozen / repeated frames).

        Args:
            image:     PIL.Image — the previous frame.
            strength:  Temporal strength value (0.10–0.50). Multiplied by
                       0.02 internally; larger values → slightly more motion.
            generator: torch.Generator (optional, for deterministic noise).
                       Pass the same generator used by the img2img pipeline to
                       keep the noise seed consistent with the diffusion seed.

        Returns:
            PIL.Image with subtle noise applied.
        """
        try:
            import torch
            import numpy as np
            from PIL import Image

            image_np = np.array(image).astype("float32") / 255.0
            tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # (C, H, W)

            if generator is not None:
                noise = torch.randn_like(tensor, generator=generator)
            else:
                noise = torch.randn_like(tensor)

            tensor = tensor + noise * strength * 0.02
            tensor = torch.clamp(tensor, 0.0, 1.0)

            out_np = tensor.permute(1, 2, 0).numpy()
            out_np = (out_np * 255).astype("uint8")
            return Image.fromarray(out_np)

        except ImportError:
            # torch or numpy not available — return image unchanged
            return image

    def propagate(self, latents, flow_fields, strength: float = 0.5):
        """[Phase 6] Propagate latent codes guided by optical flow."""
        raise NotImplementedError("Optical-flow latent propagation activates in Phase 6.")
