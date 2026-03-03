"""
EnvironmentDetector — Platform and resource detection for adaptive execution.

Detects:
  - Platform: kaggle | local | unknown
  - Device:   cuda | directml | cpu
  - VRAM (MB): live query via torch if available
  - RAM (MB):  via psutil if available (fallback: 0)
  - CI environment: GitHub Actions / generic CI env vars
  - Python version string

All detection is lazy and safe — no crash on missing libraries.
torch and psutil are imported only when needed, inside try/except.

resolve_auto_mode() maps an EnvironmentProfile to an MGOS mode string
so ExecutionContext.build() can pick the right behaviour table.
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Literal

from pydantic import BaseModel

from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)

# Mode strings used throughout MGOS
_MODE_KAGGLE = "kaggle"
_MODE_DEV = "dev"
_MODE_PRODUCTION = "production"


# ---------------------------------------------------------------------------
# Behaviour matrix per mode
# (consumed by ImageEngine, VideoEngine, ModelRegistry)
# ---------------------------------------------------------------------------

class BehaviourProfile(BaseModel):
    """
    Computed capability limits for a given mode + device combination.

    Attributes:
        max_image_resolution:  Hard cap on H and W for image generation.
        max_video_frames:      Hard cap on video frame count.
        max_controlnets:       Max simultaneous ControlNets allowed.
        ip_adapter_allowed:    Whether IP-Adapter can be used.
        auto_unload_after_gen: Unload model + empty CUDA cache after each generation.
        batch_size:            Default batch size for generation.
    """
    max_image_resolution: int = 512
    max_video_frames: int = 8
    max_controlnets: int = 0
    ip_adapter_allowed: bool = False
    auto_unload_after_gen: bool = False
    batch_size: int = 1


# ---------------------------------------------------------------------------
# EnvironmentProfile
# ---------------------------------------------------------------------------

class EnvironmentProfile(BaseModel):
    """
    Immutable snapshot of the detected runtime environment.

    Built once in ExecutionContext.build() and attached to the context.

    Attributes:
        platform:       "local" | "kaggle" | "unknown"
        device_type:    "cpu" | "cuda" | "directml"
        vram_mb:        Total VRAM in MB (0 if CPU-only or undetectable)
        ram_mb:         Total system RAM in MB (0 if psutil unavailable)
        is_ci:          True if running in a CI environment
        python_version: e.g. "3.12.5"
        mode:           Resolved MGOS mode (set after auto-mode resolution)
        behaviour:      Capability limits derived from mode + device
    """
    platform: Literal["local", "kaggle", "unknown"] = "unknown"
    device_type: Literal["cpu", "cuda", "directml"] = "cpu"
    vram_mb: int = 0
    ram_mb: int = 0
    is_ci: bool = False
    python_version: str = ""
    mode: str = "dev"
    behaviour: BehaviourProfile = BehaviourProfile()


# ---------------------------------------------------------------------------
# EnvironmentDetector
# ---------------------------------------------------------------------------

class EnvironmentDetector:
    """
    Detects platform, device, VRAM, and RAM to produce EnvironmentProfile.

    Usage:
        detector = EnvironmentDetector()
        profile = detector.detect()

    The returned profile is immutable (Pydantic model).
    """

    def detect(self) -> EnvironmentProfile:
        """
        Run all detection heuristics and return a frozen EnvironmentProfile.

        Never raises. All per-library detection is wrapped in try/except.
        """
        detected_platform = self._detect_platform()
        device_type = self._detect_device()
        vram_mb = self._detect_vram(device_type)
        ram_mb = self._detect_ram()
        is_ci = self._detect_ci()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        LOG.info(
            f"EnvironmentDetector: platform={detected_platform} device={device_type} "
            f"vram={vram_mb}MB ram={ram_mb}MB ci={is_ci}"
        )

        return EnvironmentProfile(
            platform=detected_platform,
            device_type=device_type,
            vram_mb=vram_mb,
            ram_mb=ram_mb,
            is_ci=is_ci,
            python_version=python_version,
            # mode and behaviour are filled in by resolve_auto_mode()
            mode="dev",
            behaviour=BehaviourProfile(),
        )

    # ------------------------------------------------------------------
    # Platform detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_platform() -> Literal["local", "kaggle", "unknown"]:
        """Kaggle exposes KAGGLE_KERNEL_RUN_TYPE or the /kaggle/input folder."""
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            return "kaggle"
        if os.path.exists("/kaggle/input"):
            return "kaggle"
        return "local"

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device() -> Literal["cpu", "cuda", "directml"]:
        """
        Probe for CUDA, then DirectML.  Falls back to CPU.
        torch import is lazy so Kaggle CPU kernels don't fail.
        """
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            # Apple MPS not in the enum — treat as CPU for now
        except ImportError:
            pass

        # DirectML (Windows GPU without CUDA)
        try:
            import torch_directml  # type: ignore[import]
            return "directml"
        except ImportError:
            pass

        return "cpu"

    # ------------------------------------------------------------------
    # VRAM detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_vram(device_type: str) -> int:
        """
        Return total VRAM in MB.  Returns 0 for CPU or when undetectable.
        """
        if device_type == "cpu":
            return 0

        if device_type == "cuda":
            try:
                import torch
                idx = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(idx)
                return props.total_memory // (1024 * 1024)
            except Exception:
                return 0

        if device_type == "directml":
            # torch_directml doesn't expose VRAM easily — return 0 and let
            # the user cap manually via config
            return 0

        return 0

    # ------------------------------------------------------------------
    # RAM detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_ram() -> int:
        """Return total system RAM in MB via psutil.  Returns 0 if unavailable."""
        try:
            import psutil
            return psutil.virtual_memory().total // (1024 * 1024)
        except ImportError:
            return 0

    # ------------------------------------------------------------------
    # CI detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_ci() -> bool:
        """Detect common CI environments."""
        ci_vars = ("CI", "GITHUB_ACTIONS", "TRAVIS", "CIRCLECI", "GITLAB_CI", "JENKINS_URL")
        return any(os.environ.get(v) for v in ci_vars)


# ---------------------------------------------------------------------------
# Mode resolution
# ---------------------------------------------------------------------------

def resolve_auto_mode(env: EnvironmentProfile) -> str:
    """
    Map a detected environment to an MGOS mode string.

    Decision table:
        Kaggle (any device) → "kaggle"
        CI                  → "dev"
        local GPU           → "dev"
        local CPU           → "dev"
        unknown             → "dev"

    The granular capability differences within "dev" are driven by
    the BehaviourProfile, not the mode name.
    """
    if env.platform == "kaggle":
        return _MODE_KAGGLE
    return _MODE_DEV


def build_behaviour(mode: str, env: EnvironmentProfile, performance_mode: str = "balanced") -> BehaviourProfile:
    """
    Build a BehaviourProfile from mode + environment + performance intent.

    This is the capability matrix that scales with hardware VRAM and the
    user-selected performance_mode (max-speed | balanced | max-quality).

    Platform    Device   VRAM        max_res  max_frames  controlnets  auto_unload
    ---------   ------   --------    -------  ----------  -----------  -----------
    any         CPU      —           512      8           0            False
    kaggle      CUDA     ≥14 GB      1024     24          2            True
    local       CUDA     <7 GB       512      8           0            False
    local       CUDA     7–13 GB     768      16          1            False
    local       CUDA     ≥14 GB      1024     24          2            False
    production  any      any         2048     48          2            False
    """
    device = env.device_type
    vram = env.vram_mb

    # 1. Start with mode-based defaults
    if mode == _MODE_PRODUCTION:
        res, frames, cn = 2048, 48, 2
    elif device == "cpu":
        res, frames, cn = 512, 8, 0
    elif mode == _MODE_KAGGLE:
        res, frames, cn = 1024, 24, 2
    else:
        # Local GPU tiering
        if vram < 7000:
            res, frames, cn = 512, 8, 0
        elif vram < 14000:
            res, frames, cn = 768, 16, 1
        else:
            res, frames, cn = 1024, 24, 2

    # 2. Performance Mode scaling (Impacts resolution and frame count only)
    if performance_mode == "max-speed":
        res = min(res, 512)
        frames = min(frames, 8)
    elif performance_mode == "max-quality":
        # Boost caps slightly if hardware allows
        if res >= 1024:
            res = 1280
        if frames >= 24:
            frames = 32

    return BehaviourProfile(
        max_image_resolution=res,
        max_video_frames=frames,
        max_controlnets=cn,
        ip_adapter_allowed=(cn > 0),
        auto_unload_after_gen=(mode == _MODE_KAGGLE),
        batch_size=1,
    )
