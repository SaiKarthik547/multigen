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

    def estimate(self, frame_a: PILImage.Image, frame_b: PILImage.Image) -> Optional[np.ndarray]:
        """Estimate optical flow between two frames using RAFT large."""
        if self.model is None:
            return None
            
        img1 = TF.to_tensor(frame_a).unsqueeze(0).to(self.device)
        img2 = TF.to_tensor(frame_b).unsqueeze(0).to(self.device)
        
        # RAFT expects inputs to be multiples of 8
        h, w = img1.shape[-2:]
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        
        if h != new_h or w != new_w:
            img1 = F.interpolate(img1, size=(new_h, new_w), mode='bilinear', align_corners=False)
            img2 = F.interpolate(img2, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # RAFT expects images normalized to [-1, 1]
        img1 = (img1 * 2) - 1
        img2 = (img2 * 2) - 1
        
        with torch.no_grad():
            list_of_flows = self.model(img1, img2)
            flow = list_of_flows[-1]
            
        if h != new_h or w != new_w:
             flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=False)
             flow[:, 0, :, :] *= float(w) / new_w
             flow[:, 1, :, :] *= float(h) / new_h

        return flow.squeeze(0).cpu().numpy()

    @staticmethod
    def warp_frame(frame: PILImage.Image, flow: np.ndarray) -> PILImage.Image:
        """Warp a frame using optical flow motion field."""
        device = "cpu"
        img = TF.to_tensor(frame).unsqueeze(0).to(device)
        flow_torch = torch.from_numpy(flow).unsqueeze(0).to(device)
        
        B, C, H, W = img.size()
        
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        
        vgrid = grid + flow_torch
        
        # Normalize grid to [-1, 1] for grid_sample
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        
        vgrid = vgrid.permute(0, 2, 3, 1)
        warped_img = F.grid_sample(img, vgrid, align_corners=True, padding_mode="reflection")
        
        return TF.to_pil_image(warped_img.squeeze(0))
