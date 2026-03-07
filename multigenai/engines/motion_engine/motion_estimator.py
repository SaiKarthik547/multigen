import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image as PILImage
from typing import Optional, List
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.transforms import functional as TF
from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)

class MotionEstimator:
    """
    RAFT-based Optical Flow estimation for temporal movement continuity.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        try:
            self.weights = Raft_Large_Weights.DEFAULT
            self.model = raft_large(weights=self.weights, progress=False).to(device)
            self.model.eval()
            LOG.info("MotionEstimator: RAFT model loaded successfully.")
        except Exception as e:
            LOG.error(f"MotionEstimator: Failed to load RAFT model: {e}")
            self.model = None

    def unload(self):
        """Reclaim RAFT model VRAM."""
        if hasattr(self, "model") and self.model is not None:
            from multigenai.core.model_lifecycle import ModelLifecycle
            ModelLifecycle.safe_unload(self.model)
            self.model = None
            LOG.info("MotionEstimator: RAFT model unloaded.")

    def estimate(self, frame_a: PILImage.Image, frame_b: PILImage.Image) -> Optional[np.ndarray]:
        """Estimate optical flow between two frames using RAFT large."""
        if self.model is None:
            return None
            
        img1 = TF.to_tensor(frame_a).unsqueeze(0).to(self.device)
        img2 = TF.to_tensor(frame_b).unsqueeze(0).to(self.device)
        
        # RAFT guard: ensure RGB input
        if img1.shape[1] != 3:
            raise ValueError(f"MotionEstimator: RAFT expects RGB input (3 channels), got {img1.shape[1]}")
        
        # RAFT expects inputs to be multiples of 8
        h, w = img1.shape[-2:]
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        
        if h != new_h or w != new_w:
            img1 = F.interpolate(img1, size=(new_h, new_w), mode='bilinear', align_corners=False)
            img2 = F.interpolate(img2, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # RAFT expects images normalized to [-1, 1]
        img1 = (img1 - 0.5) / 0.5
        img2 = (img2 - 0.5) / 0.5
        
        with torch.no_grad():
            list_of_flows = self.model(img1, img2)
            flow = list_of_flows[-1]
            
        if h != new_h or w != new_w:
             flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=False)
             flow[:, 0, :, :] *= float(w) / new_w
             flow[:, 1, :, :] *= float(h) / new_h

        return flow.squeeze(0).cpu().numpy()

    def warp_frame(self, frame: PILImage.Image, flow: np.ndarray) -> PILImage.Image:
        """
        Warp frame using RAFT optical flow with stability constraints.

        Args:
            frame: previous PIL Image
            flow: RAFT optical flow (2, H, W)

        Returns:
            warped PIL Image
        """
        import cv2
        import numpy as np

        # Convert PIL to numpy (H, W, 3)
        frame_np = np.array(frame)
        
        # Transpose flow from (2, H, W) to (H, W, 2) for OpenCV
        flow_cv = flow.transpose(1, 2, 0)
        h, w = flow_cv.shape[:2]

        # -------------------------------
        # Scene change detection
        # -------------------------------
        flow_mag_field = np.sqrt(flow_cv[..., 0]**2 + flow_cv[..., 1]**2)
        flow_mag = np.mean(flow_mag_field)
        
        LOG.debug(f"MotionEstimator: flow magnitude mean={flow_mag:.2f}")

        if flow_mag > 60:
            # large camera jump or scene cut
            LOG.info("MotionEstimator: Large scene change detected — skipping warp")
            return frame

        # -------------------------------
        # Clip extreme motion vectors
        # -------------------------------
        flow_cv = np.clip(flow_cv, -25.0, 25.0)

        # -------------------------------
        # Smooth the flow field
        # -------------------------------
        flow_cv = cv2.GaussianBlur(flow_cv, (5, 5), 0)

        # -------------------------------
        # Generate sampling grid
        # -------------------------------
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow_cv[..., 0]).astype(np.float32)
        map_y = (grid_y + flow_cv[..., 1]).astype(np.float32)

        # -------------------------------
        # Apply remap
        # -------------------------------
        try:
            warped = cv2.remap(
                frame_np,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
        except Exception as exc:
            LOG.warning(f"MotionEstimator: Warp failed — returning original frame: {exc}")
            return frame

        # -------------------------------
        # Motion stability mask
        # -------------------------------
        # Recalculate mag after clipping/smoothing
        mag = np.sqrt(flow_cv[..., 0]**2 + flow_cv[..., 1]**2)
        mask = mag < 20

        result = frame_np.copy()
        # mag is (H, W), result is (H, W, 3). Broadcast mask to all channels.
        mask = mask[..., None]
        result[np.repeat(mask, 3, axis=-1)] = warped[np.repeat(mask, 3, axis=-1)]

        return PILImage.fromarray(result)
