from typing import Optional
from PIL import Image as PILImage
from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)

class ControlNetManager:
    """
    Manages ControlNet loading and Depth Map generation using memory-safe
    DepthAnythingSmall (LiheYoung/depth-anything-small-hf).
    """
    def __init__(self, device: str):
        self.device = device
        self.controlnet = None
        self.depth_estimator = None
        self.feature_extractor = None

    def load(self) -> None:
        """Lazy load the ControlNet model and Depth Estimator components."""
        import torch
        from diffusers import ControlNetModel
        from transformers import DPTImageProcessor, DPTForDepthEstimation
        
        if self.controlnet is not None:
             return

        LOG.info("Loading ControlNet (depth-sdxl-1.0) and DepthAnythingSmall...")

        self.controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(self.device)
        
        self.depth_estimator = DPTForDepthEstimation.from_pretrained(
            "LiheYoung/depth-anything-small-hf"
        ).to("cpu")
        
        self.feature_extractor = DPTImageProcessor.from_pretrained(
            "LiheYoung/depth-anything-small-hf"
        )

    def get_depth_map(self, image: PILImage.Image) -> PILImage.Image:
        """
        Generate a depth map from the reference frame to structurally condition 
        the next segment. Dynamically matches input spatial resolution.
        """
        import numpy as np
        import torch
        from PIL import Image
        
        LOG.debug(f"Generating depth map from {image.size} reference frame...")

        # DepthAnythingSmall estimation runs on CPU to save VRAM
        image_tensor = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cpu")
        
        with torch.no_grad():
            depth_map = self.depth_estimator(image_tensor).predicted_depth

        w, h = image.size
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        depth_image_tensor = torch.cat([depth_map] * 3, dim=1)

        depth_image_tensor = depth_image_tensor.permute(0, 2, 3, 1).cpu().numpy()[0]
        depth_image = Image.fromarray((depth_image_tensor * 255.0).clip(0, 255).astype(np.uint8))
        return depth_image
