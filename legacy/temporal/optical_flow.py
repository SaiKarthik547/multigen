"""
OpticalFlow — Phase 5 stub (ARCHIVED to legacy/).

Superseded by SVD-XT (StableVideoDiffusionPipeline) in Phase 6.
Original path: multigenai/temporal/optical_flow.py
See legacy/README.md for reuse context (Phase 9).
"""

from __future__ import annotations


class OpticalFlow:
    """Computes and applies optical flow for video temporal coherence."""

    def compute(self, frame_a, frame_b):
        """[Phase 5] Compute dense optical flow between two frames."""
        raise NotImplementedError("Optical flow computation activates in Phase 5.")

    def warp(self, frame, flow):
        """[Phase 5] Warp a frame using a computed flow field."""
        raise NotImplementedError("Frame warping activates in Phase 5.")
