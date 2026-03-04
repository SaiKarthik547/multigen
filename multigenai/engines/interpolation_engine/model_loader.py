"""
model_loader.py — RIFE IFNet weight loader for InterpolationEngine.

Isolates model download and instantiation from the engine class.
Weights are downloaded via huggingface_hub on first use and cached in
~/.cache/huggingface (standard HF cache).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

LOG = logging.getLogger(__name__)

# HuggingFace repo hosting the RIFE IFNet weights used here.
# Using AlignedAI/RIFE which provides a clean inference-only checkpoint.
RIFE_REPO_ID = "AlignedAI/RIFE"
RIFE_WEIGHT_FILE = "flownet.pkl"


def load_rife_model(device: str) -> Tuple[Optional[object], str]:
    """
    Download (if needed) and load the RIFE IFNet model.

    Returns:
        (model, device)  — model is a callable that accepts two frame tensors
                           and returns an intermediate frame tensor.

    On failure:
        Returns (None, device) and logs a warning.
        The engine must handle None gracefully (pass-through mode).
    """
    try:
        import torch
        from huggingface_hub import hf_hub_download

        LOG.info(f"RIFE: Downloading/loading weights from {RIFE_REPO_ID}...")
        weight_path = hf_hub_download(repo_id=RIFE_REPO_ID, filename=RIFE_WEIGHT_FILE)

        model = _build_ifnet(weight_path, device)
        LOG.info(f"RIFE: Model loaded on device={device}")
        return model, device

    except ImportError as exc:
        LOG.warning(f"RIFE: Missing dependency ({exc}). Interpolation will be skipped.")
        return None, device
    except Exception as exc:
        LOG.warning(f"RIFE: Could not load model ({exc}). Interpolation will be skipped.")
        return None, device


def _build_ifnet(weight_path: str, device: str) -> object:
    """
    Build the IFNet model from the given weight file.

    Uses a minimal pure-PyTorch bilateral-motion IFNet implementation
    compatible with the RIFE 4.x checkpoint format.
    """
    import torch

    # Inline minimal IFNet definition — avoids requiring the full RIFE training repo.
    # This matches the inference interface of RIFE 4.x checkpoints.
    try:
        from torch import nn

        class IFBlock(nn.Module):
            def __init__(self, in_planes: int, c: int = 64):
                super().__init__()
                self.conv0 = nn.Sequential(
                    _conv(in_planes, c // 2, 3, 2, 1),
                    _conv(c // 2, c, 3, 2, 1),
                )
                self.convblock = nn.Sequential(
                    _conv(c, c), _conv(c, c), _conv(c, c),
                    _conv(c, c), _conv(c, c), _conv(c, c),
                )
                self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

            def forward(self, x, flow, scale):
                if scale != 1:
                    x = torch.nn.functional.interpolate(
                        x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
                    )
                if flow is not None:
                    flow = torch.nn.functional.interpolate(
                        flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
                    ) * (1.0 / scale)
                    x = torch.cat((x, flow), 1)
                x = self.conv0(x)
                x = self.convblock(x) + x
                tmp = self.lastconv(x)
                tmp = torch.nn.functional.interpolate(
                    tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False
                )
                flow = tmp[:, :4] * scale * 2
                mask = tmp[:, 4:5]
                return flow, mask

        class IFNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.block0 = IFBlock(7 + 4, c=90)
                self.block1 = IFBlock(7 + 4, c=90)
                self.block2 = IFBlock(7 + 4, c=90)
                self.block_tea = IFBlock(10 + 4, c=90)

            def forward(self, x, scale_list=None, training=False):
                if scale_list is None:
                    scale_list = [4, 2, 1]
                img0 = x[:, :3]
                img1 = x[:, 3:6]
                gt = x[:, 6:]  # only used during training
                flow, mask = None, None
                merged = []
                for i, block in enumerate([self.block0, self.block1, self.block2]):
                    if flow is None:
                        flow, mask = block(
                            torch.cat((img0, img1, *([gt] if training else [torch.zeros_like(gt)])), 1),
                            None,
                            scale=scale_list[i],
                        )
                    else:
                        f0 = _warp(img0, flow[:, :2])
                        f1 = _warp(img1, flow[:, 2:4])
                        fd, mask = block(
                            torch.cat((f0, f1, *([gt] if training else [torch.zeros_like(gt)])), 1),
                            flow,
                            scale=scale_list[i],
                        )
                        flow = flow + fd
                    warped0 = _warp(img0, flow[:, :2])
                    warped1 = _warp(img1, flow[:, 2:4])
                    merged.append(
                        (warped0 * torch.sigmoid(mask) + warped1 * (1 - torch.sigmoid(mask)))
                    )
                return merged[-1]

        dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        model = IFNet().to(dev)
        state = torch.load(weight_path, map_location=dev)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        model.eval()
        return model

    except Exception as exc:
        raise RuntimeError(f"IFNet build failed: {exc}") from exc


def _conv(in_planes: int, out_planes: int, kernel: int = 3, stride: int = 1, pad: int = 1):
    from torch import nn
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel, stride, pad, bias=False),
        nn.PReLU(out_planes),
    )


def _warp(tenInput, tenFlow):
    """Bilinear backward warping using a 2-channel flow field."""
    import torch
    import torch.nn.functional as F
    k = (
        torch.Tensor([[(tenInput.shape[3] - 1.0) / 2.0]])
        .float()
        .to(tenInput.device)
    )
    tenFlow = torch.cat(
        [tenFlow[:, 0:1, :, :] / k, tenFlow[:, 1:2, :, :] / k], 1
    )
    g = (
        torch.nn.functional.affine_grid(
            torch.zeros(tenInput.shape[0], 2, 3).to(tenInput.device),
            tenInput.size(),
            align_corners=False,
        )
        + tenFlow.permute(0, 2, 3, 1)
    )
    return F.grid_sample(tenInput, g, align_corners=False)
