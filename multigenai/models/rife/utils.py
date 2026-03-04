import numpy as np
import torch
from PIL import Image

def image_to_tensor(img):
    """Convert PIL image to float32 tensor [1, 3, H, W] in [0, 1]."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
    return tensor

def tensor_to_image(t):
    """Convert float32 tensor [1, 3, H, W] to PIL image."""
    t = t.squeeze(0).permute(1,2,0).clamp(0, 1).cpu().numpy()
    t = (t * 255).astype(np.uint8)
    return Image.fromarray(t)
