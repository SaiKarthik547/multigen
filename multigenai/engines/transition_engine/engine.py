import numpy as np
from PIL import Image as PILImage
from typing import List

class TransitionEngine:
    """
    Handles temporal smoothing and alpha-blending across scene boundaries.
    """
    @staticmethod
    def blend(frames_a: List[PILImage.Image], frames_b: List[PILImage.Image], window: int = 4) -> List[PILImage.Image]:
        """
        Performs linear alpha-blending between the end of frames_a and start of frames_b.
        """
        if len(frames_a) < window or len(frames_b) < window:
            return frames_a + frames_b
            
        overlap_a = frames_a[-window:]
        overlap_b = frames_b[:window]
        
        blended = []
        for i in range(window):
            # Alpha goes from near 0 (mostly A) to near 1 (mostly B)
            alpha = (i + 1) / (window + 1)
            img_a = np.array(overlap_a[i]).astype(float)
            img_b = np.array(overlap_b[i]).astype(float)
            
            blended_img = (1 - alpha) * img_a + alpha * img_b
            blended.append(PILImage.fromarray(blended_img.astype(np.uint8)))
            
        return frames_a[:-window] + blended + frames_b[window:]
