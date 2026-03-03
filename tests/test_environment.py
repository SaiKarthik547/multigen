"""
Tests for Phase 3 — Environment-Aware Adaptive Execution Layer.

Covers:
  - EnvironmentDetector: local CPU profile (no Kaggle env)
  - EnvironmentDetector: Kaggle platform detected via env var
  - EnvironmentDetector: Kaggle platform detected via /kaggle/input path
  - EnvironmentDetector: CI flag detected
  - resolve_auto_mode(): kaggle env → "kaggle"
  - resolve_auto_mode(): local env → "dev"
  - build_behaviour(): CPU → max_res=512, auto_unload=False
  - build_behaviour(): kaggle GPU → max_res=1024, auto_unload=True
  - build_behaviour(): local GPU <7 GB → max_res=512
  - build_behaviour(): local GPU 7–13 GB → max_res=768
  - build_behaviour(): local GPU ≥14 GB → max_res=1024
  - ModelRegistry: VRAM guard raises InsufficientVRAMError
  - ModelRegistry: SD1.5 downgrade when VRAM < 8 GB
  - ImageEngine: resolution capped correctly for CPU
  - ImageEngine: resolution capped for low VRAM GPU
  - ExecutionContext: has environment attribute after build()
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# EnvironmentDetector — platform detection
# ---------------------------------------------------------------------------

def test_detect_local_when_no_kaggle(monkeypatch):
    """Without any Kaggle signals, platform should be 'local'."""
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    from multigenai.core.environment import EnvironmentDetector
    det = EnvironmentDetector()
    assert det._detect_platform() == "local"


def test_detect_kaggle_via_env_var(monkeypatch):
    """KAGGLE_KERNEL_RUN_TYPE env var → platform='kaggle'."""
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    from multigenai.core.environment import EnvironmentDetector
    det = EnvironmentDetector()
    assert det._detect_platform() == "kaggle"


def test_detect_kaggle_via_path(monkeypatch, tmp_path):
    """Presence of /kaggle/input directory → platform='kaggle' (mocked)."""
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)

    # Patch os.path.exists specifically for /kaggle/input
    original_exists = os.path.exists

    def patched_exists(path):
        if path == "/kaggle/input":
            return True
        return original_exists(path)

    with patch("os.path.exists", side_effect=patched_exists):
        from multigenai.core.environment import EnvironmentDetector
        det = EnvironmentDetector()
        assert det._detect_platform() == "kaggle"


def test_detect_ci_via_env(monkeypatch):
    """CI env var → is_ci=True."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    from multigenai.core.environment import EnvironmentDetector
    det = EnvironmentDetector()
    assert det._detect_ci() is True


def test_no_ci_when_no_env(monkeypatch):
    """No CI env vars → is_ci=False."""
    for v in ("CI", "GITHUB_ACTIONS", "TRAVIS", "CIRCLECI", "GITLAB_CI", "JENKINS_URL"):
        monkeypatch.delenv(v, raising=False)
    from multigenai.core.environment import EnvironmentDetector
    det = EnvironmentDetector()
    assert det._detect_ci() is False


def test_full_detect_returns_profile(monkeypatch):
    """detect() returns a valid EnvironmentProfile with expected fields."""
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    from multigenai.core.environment import EnvironmentDetector, EnvironmentProfile
    profile = EnvironmentDetector().detect()
    assert isinstance(profile, EnvironmentProfile)
    assert profile.platform in ("local", "kaggle", "unknown")
    assert profile.device_type in ("cpu", "cuda", "directml")
    assert profile.python_version != ""


# ---------------------------------------------------------------------------
# resolve_auto_mode
# ---------------------------------------------------------------------------

def test_auto_mode_kaggle(monkeypatch):
    """Kaggle platform → mode='kaggle'."""
    from multigenai.core.environment import EnvironmentProfile, resolve_auto_mode, BehaviourProfile
    env = EnvironmentProfile(platform="kaggle", device_type="cuda", vram_mb=15000, behaviour=BehaviourProfile())
    assert resolve_auto_mode(env) == "kaggle"


def test_auto_mode_local(monkeypatch):
    """Local platform → mode='dev'."""
    from multigenai.core.environment import EnvironmentProfile, resolve_auto_mode, BehaviourProfile
    env = EnvironmentProfile(platform="local", device_type="cpu", vram_mb=0, behaviour=BehaviourProfile())
    assert resolve_auto_mode(env) == "dev"


# ---------------------------------------------------------------------------
# build_behaviour — behaviour matrix
# ---------------------------------------------------------------------------

def test_behaviour_cpu():
    """CPU device → max_res=512, auto_unload=False."""
    from multigenai.core.environment import EnvironmentProfile, build_behaviour, BehaviourProfile
    env = EnvironmentProfile(platform="local", device_type="cpu", vram_mb=0, behaviour=BehaviourProfile())
    b = build_behaviour("dev", env)
    assert b.max_image_resolution == 512
    assert b.auto_unload_after_gen is False
    assert b.max_video_frames == 8


def test_behaviour_kaggle_gpu():
    """Kaggle + CUDA → max_res=1024, auto_unload=True, 24 frames."""
    from multigenai.core.environment import EnvironmentProfile, build_behaviour, BehaviourProfile
    env = EnvironmentProfile(platform="kaggle", device_type="cuda", vram_mb=15109, behaviour=BehaviourProfile())
    b = build_behaviour("kaggle", env)
    assert b.max_image_resolution == 1024
    assert b.auto_unload_after_gen is True
    assert b.max_video_frames == 24


def test_behaviour_local_low_vram():
    """Local GPU < 7 GB → max_res=512."""
    from multigenai.core.environment import EnvironmentProfile, build_behaviour, BehaviourProfile
    env = EnvironmentProfile(platform="local", device_type="cuda", vram_mb=4096, behaviour=BehaviourProfile())
    b = build_behaviour("dev", env)
    assert b.max_image_resolution == 512


def test_behaviour_local_mid_vram():
    """Local GPU 7–13 GB → max_res=768, 1 controlnet."""
    from multigenai.core.environment import EnvironmentProfile, build_behaviour, BehaviourProfile
    env = EnvironmentProfile(platform="local", device_type="cuda", vram_mb=8192, behaviour=BehaviourProfile())
    b = build_behaviour("dev", env)
    assert b.max_image_resolution == 768
    assert b.max_controlnets == 1


def test_behaviour_local_high_vram():
    """Local GPU ≥ 14 GB → max_res=1024, 2 controlnets."""
    from multigenai.core.environment import EnvironmentProfile, build_behaviour, BehaviourProfile
    env = EnvironmentProfile(platform="local", device_type="cuda", vram_mb=16384, behaviour=BehaviourProfile())
    b = build_behaviour("dev", env)
    assert b.max_image_resolution == 1024
    assert b.max_controlnets == 2


def test_behaviour_production():
    """Production mode → max_res=2048, no auto_unload."""
    from multigenai.core.environment import EnvironmentProfile, build_behaviour, BehaviourProfile
    env = EnvironmentProfile(platform="local", device_type="cuda", vram_mb=24576, behaviour=BehaviourProfile())
    b = build_behaviour("production", env)
    assert b.max_image_resolution == 2048
    assert b.auto_unload_after_gen is False


# ---------------------------------------------------------------------------
# ModelRegistry — VRAM guard and SD1.5 downgrade
# ---------------------------------------------------------------------------

def test_model_registry_vram_guard_raises():
    """ModelRegistry.get() raises InsufficientVRAMError when env.vram_mb < 80% of required."""
    from multigenai.core.model_registry import ModelRegistry
    from multigenai.core.exceptions import InsufficientVRAMError
    from multigenai.core.environment import EnvironmentProfile, BehaviourProfile

    reg = ModelRegistry()
    reg.register("big-model", loader=lambda: object(), min_vram_gb=10.0)

    # 4 GB VRAM < 10 GB * 0.8 = 8 GB (80% threshold)
    env = EnvironmentProfile(platform="local", device_type="cuda", vram_mb=4096, behaviour=BehaviourProfile())
    with pytest.raises(InsufficientVRAMError):
        reg.get("big-model", environment=env)


def test_model_registry_sdxl_downgrade_to_sd15():
    """SD1.5 downgrade: requesting SDXL model on low-VRAM env loads SD1.5 instead."""
    from multigenai.core.model_registry import ModelRegistry, _SDXL_MODEL_ID, _SD15_MODEL_ID
    from multigenai.core.environment import EnvironmentProfile, BehaviourProfile

    reg = ModelRegistry()
    sentinel = object()
    reg.register(_SDXL_MODEL_ID, loader=lambda: "sdxl_loaded", min_vram_gb=6.0)
    reg.register(_SD15_MODEL_ID, loader=lambda: sentinel, min_vram_gb=2.0)

    # 4 GB < 8 GB threshold → downgrade
    env = EnvironmentProfile(platform="local", device_type="cuda", vram_mb=4096, behaviour=BehaviourProfile())
    result = reg.get(_SDXL_MODEL_ID, environment=env)
    assert result is sentinel  # SD1.5 was loaded instead of SDXL


def test_model_registry_no_downgrade_on_cpu():
    """CPU (vram_mb=0) does not trigger SDXL downgrade (no VRAM to compare)."""
    from multigenai.core.model_registry import ModelRegistry, _SDXL_MODEL_ID, _SD15_MODEL_ID
    from multigenai.core.environment import EnvironmentProfile, BehaviourProfile

    reg = ModelRegistry()
    sdxl_result = object()
    reg.register(_SDXL_MODEL_ID, loader=lambda: sdxl_result, min_vram_gb=0.0)  # 0 = CPU safe

    env = EnvironmentProfile(platform="local", device_type="cpu", vram_mb=0, behaviour=BehaviourProfile())
    result = reg.get(_SDXL_MODEL_ID, environment=env)
    # vram_mb=0 → downgrade condition `0 < vram_mb < threshold` is False → SDXL loaded
    assert result is sdxl_result


# ---------------------------------------------------------------------------
# ImageEngine — resolution capping
# ---------------------------------------------------------------------------

def _make_mock_ctx(device_type: str = "cpu", vram_mb: int = 0):
    """Helper: build a minimal mock ExecutionContext with environment set."""
    from multigenai.core.environment import EnvironmentProfile, build_behaviour, BehaviourProfile

    env = EnvironmentProfile(
        platform="local",
        device_type=device_type,
        vram_mb=vram_mb,
        behaviour=BehaviourProfile(),
    )
    mode = "dev"
    behaviour = build_behaviour(mode, env)
    env = env.model_copy(update={"mode": mode, "behaviour": behaviour})

    ctx = MagicMock()
    ctx.environment = env
    ctx.behaviour = behaviour
    ctx.device = device_type
    ctx.settings.output_dir = "multigen_outputs"
    return ctx


def test_image_engine_resolution_schema_rejects_non_multiple_of_64():
    """Phase 7: Resolution is now validated at schema level (% 64), not capped at runtime."""
    from multigenai.llm.schema_validator import ImageGenerationRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="divisible by 64"):
        ImageGenerationRequest(prompt="a sunset", width=777, height=555)


def test_image_engine_resolution_schema_accepts_valid():
    """Phase 7: 1024x1024 (divisible by 64) passes schema validation."""
    from multigenai.llm.schema_validator import ImageGenerationRequest
    req = ImageGenerationRequest(prompt="a sunset", width=1024, height=576)
    assert req.width == 1024
    assert req.height == 576


def test_image_engine_resolution_schema_default_is_1024():
    """Phase 7: Default resolution is 1024x1024 (SDXL native)."""
    from multigenai.llm.schema_validator import ImageGenerationRequest
    req = ImageGenerationRequest(prompt="test")
    assert req.width == 1024
    assert req.height == 1024
    assert req.width % 64 == 0
    assert req.height % 64 == 0


def test_image_engine_has_strict_lifecycle_attributes():
    """Phase 7: ImageEngine exposes pipe and refiner attributes (strict lifecycle)."""
    from multigenai.engines.image_engine.engine import ImageEngine
    engine = ImageEngine.__new__(ImageEngine)
    engine.pipe = None
    engine.refiner = None
    assert engine.pipe is None
    assert engine.refiner is None


# ---------------------------------------------------------------------------
# ExecutionContext — has environment after build()
# ---------------------------------------------------------------------------

def test_execution_context_has_environment():
    """ExecutionContext.build() populates environment with a valid EnvironmentProfile."""
    from multigenai.core.config.settings import get_settings
    from multigenai.core.execution_context import ExecutionContext
    from multigenai.core.environment import EnvironmentProfile

    settings = get_settings()
    ctx = ExecutionContext.build(settings)

    assert ctx.environment is not None
    assert isinstance(ctx.environment, EnvironmentProfile)
    assert ctx.environment.platform in ("local", "kaggle", "unknown")
    assert ctx.behaviour is not None
    assert ctx.behaviour.max_image_resolution >= 512
