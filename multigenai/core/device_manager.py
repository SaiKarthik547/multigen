"""
DeviceManager — Hardware detection and selection for MultiGenAI OS.

Provides:
  - Best available torch.device detection (CUDA → DirectML → CPU)
  - VRAM reporting (free, total in GB)
  - Safe operation when PyTorch is not installed (CPU-only environments)
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from typing import Optional

from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)


@dataclass
class VRAMInfo:
    """VRAM statistics in gigabytes."""
    free_gb: float
    total_gb: float
    device_name: str


class DeviceManager:
    """
    Detects and manages compute device selection.

    Usage:
        dm = DeviceManager()
        device = dm.get_device()      # "cuda" | "directml" | "cpu"
        vram   = dm.get_vram_info()   # VRAMInfo | None
    """

    def __init__(self, preferred: str = "auto") -> None:
        """
        Args:
            preferred: "auto" (default) selects best available.
                       Can be "cuda", "directml", or "cpu" to force.
        """
        self._preferred = preferred
        self._device: Optional[str] = None
        self._torch = None
        self._load_torch()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_device(self) -> str:
        """Return the selected device string ("cuda", "directml", or "cpu")."""
        if self._device is None:
            self._device = self._detect_device()
        return self._device

    def get_vram_info(self) -> Optional[VRAMInfo]:
        """
        Return VRAM statistics for the selected GPU, or None if on CPU.

        Returns:
            VRAMInfo with free_gb / total_gb / device_name, or None.
        """
        torch = self._torch
        if torch is None:
            return None
        device = self.get_device()
        if device == "cuda":
            try:
                free_b, total_b = torch.cuda.mem_get_info()
                name = torch.cuda.get_device_name(0)
                return VRAMInfo(
                    free_gb=free_b / 1024 ** 3,
                    total_gb=total_b / 1024 ** 3,
                    device_name=name,
                )
            except Exception as exc:
                LOG.warning(f"Could not query CUDA VRAM: {exc}")
        return None

    def clear_cache(self) -> None:
        """Release unused CUDA memory (no-op on CPU / DirectML)."""
        torch = self._torch
        if torch is None:
            return
        if self.get_device() == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            LOG.debug("CUDA cache cleared.")

    def summary(self) -> dict:
        """Return a serialisable summary for CapabilityReport."""
        device = self.get_device()
        vram = self.get_vram_info()
        return {
            "device": device,
            "torch_available": self._torch is not None,
            "cuda_available": self._torch is not None and self._torch.cuda.is_available(),
            "vram_free_gb": round(vram.free_gb, 2) if vram else None,
            "vram_total_gb": round(vram.total_gb, 2) if vram else None,
            "gpu_name": vram.device_name if vram else None,
            "python": sys.version,
            "platform": platform.platform(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_torch(self) -> None:
        """Import torch lazily — never at module level."""
        try:
            import torch  # noqa: F401 — intentional lazy import
            self._torch = torch
        except ImportError:
            self._torch = None
            LOG.debug("PyTorch not installed — device set to CPU.")

    def _detect_device(self) -> str:
        """Determine the best available device."""
        if self._preferred != "auto":
            LOG.info(f"Device forced to '{self._preferred}' by configuration.")
            return self._preferred

        torch = self._torch
        if torch is None:
            LOG.info("PyTorch unavailable — running on CPU (no AI generation).")
            return "cpu"

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            LOG.info(f"CUDA GPU detected: {name}")
            return "cuda"

        # DirectML (AMD / Intel on Windows)
        try:
            import torch_directml  # noqa: F401
            LOG.info("DirectML device detected (AMD/Intel GPU on Windows).")
            return "directml"
        except ImportError:
            pass

        LOG.info("No GPU found — falling back to CPU.")
        return "cpu"
