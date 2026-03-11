import numpy as np
from PIL import Image as PILImage
from typing import List


class TransitionEngine:
    """
    Handles temporal smoothing and camera-style transitions across scene boundaries.

    Phase 15: Added camera_pan() for true cinematic scene transitions that
    simulate lateral camera movement rather than a hard cut or image blend.
    """

    @staticmethod
    def blend(
        frames_a: List[PILImage.Image],
        frames_b: List[PILImage.Image],
        window: int = 4,
    ) -> List[PILImage.Image]:
        """
        Linear alpha-blend between the end of frames_a and start of frames_b.
        """
        if len(frames_a) < window or len(frames_b) < window:
            return frames_a + frames_b

        overlap_a = frames_a[-window:]
        overlap_b = frames_b[:window]

        blended: List[PILImage.Image] = []
        for i in range(window):
            alpha = (i + 1) / (window + 1)
            img_a = np.array(overlap_a[i]).astype(float)
            img_b = np.array(overlap_b[i]).astype(float)
            blended_img = (1 - alpha) * img_a + alpha * img_b
            blended.append(PILImage.fromarray(blended_img.astype(np.uint8)))

        return frames_a[:-window] + blended + frames_b[window:]

    @staticmethod
    def camera_pan(
        frames: List[PILImage.Image],
        max_shift_px: int = 20,
    ) -> List[PILImage.Image]:
        """
        Phase 15: Simulate a lateral camera pan across a list of frames.

        Applies a linearly increasing horizontal pixel shift to each frame
        using np.roll to emulate camera movement. Used at scene boundaries
        to produce a continuous cinematic transition rather than a hard cut.

        Args:
            frames:        List of PIL Images (the overlap window to transform).
            max_shift_px:  Maximum horizontal pixel offset at the end of the sequence.

        Returns:
            List of PIL Images with the pan applied.
        """
        if not frames:
            return frames

        shift_values = np.linspace(0, max_shift_px, len(frames))
        result: List[PILImage.Image] = []

        for i, frame in enumerate(frames):
            arr = np.array(frame)
            shifted = np.roll(arr, int(shift_values[i]), axis=1)
            result.append(PILImage.fromarray(shifted.astype(np.uint8)))

        return result
