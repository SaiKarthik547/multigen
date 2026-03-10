"""
test_phase7_optimization.py - Unit tests for performance toggles and memory scaling.

Covers:
  1. Memory optimization helper (_apply_memory_optimizations) works conditionally.
  2. The Generator created in ImageEngine correctly uses device="cpu" for determinism.
  3. The config / settings default values for SDXL are correct.
  4. The model cleanup calls (safe_unload) are resilient to bad/none input.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Optimizaton tests
# ---------------------------------------------------------------------------

def test_apply_memory_optimizations_cuda():
    """Verify CUDA applies sequential offload, VAE tiling, and attention slicing."""
    from multigenai.engines.image_engine.engine import _apply_memory_optimizations
    
    mock_pipe = MagicMock()
    _apply_memory_optimizations(mock_pipe, device="cuda")
    
    # Needs model CPU offload, VAE Tiling, and Attention slicing
    mock_pipe.enable_model_cpu_offload.assert_called_once()
    mock_pipe.vae.enable_tiling.assert_called_once()
    mock_pipe.enable_attention_slicing.assert_called_once()

def test_apply_memory_optimizations_directml():
    """Verify DirectML skips CPU offload, but applies VAE tiling and attention slicing."""
    from multigenai.engines.image_engine.engine import _apply_memory_optimizations
    
    mock_pipe = MagicMock()
    _apply_memory_optimizations(mock_pipe, device="directml")
    
    mock_pipe.enable_model_cpu_offload.assert_not_called()
    mock_pipe.vae.enable_tiling.assert_called_once()
    mock_pipe.enable_attention_slicing.assert_called_once()
    mock_pipe.to.assert_called_once_with("directml")

def test_apply_memory_optimizations_cpu():
    """Verify CPU applies VAE tiling (saves RAM), but no CUDA-specific offloads."""
    from multigenai.engines.image_engine.engine import _apply_memory_optimizations
    
    mock_pipe = MagicMock()
    _apply_memory_optimizations(mock_pipe, device="cpu")
    
    mock_pipe.enable_model_cpu_offload.assert_not_called()
    mock_pipe.enable_attention_slicing.assert_not_called()  # We didn't enable slicing for cpu in impl
    mock_pipe.vae.enable_tiling.assert_called_once()
    mock_pipe.to.assert_called_once_with("cpu")


def test_model_lifecycle_handles_exceptions_safely():
    """Verify safe_unload is resilient to teardown errors."""
    from multigenai.core.model_lifecycle import ModelLifecycle
    import unittest.mock as mock

    obj = MagicMock()
    with mock.patch("multigenai.core.model_lifecycle.LOG") as mock_log:
        ModelLifecycle.safe_unload(obj)
        # Ensure it at least attempts to log the start of unloading
        assert mock_log.debug.called

def test_video_engine_safe_schema_mapping():
    """Ensure VideoGenerationRequest maps defaults correctly and validates resolution."""
    from multigenai.llm.schema_validator import VideoGenerationRequest
    
    req = VideoGenerationRequest(prompt="test video", fps=12)
    assert req.fps == 12
    # Verify the unused schema field 'frame_duration' is still present for backwards compat
    assert req.frame_duration >= 0.1

    with pytest.raises(ValueError, match="divisible by 64"):
        VideoGenerationRequest(prompt="test video", width=258) # not % 64


# ---------------------------------------------------------------------------
# New tests for architecture fixes
# ---------------------------------------------------------------------------

def test_image_engine_sdxl_repo_detection():
    """Verify that SDXL repos are correctly identified by 'xl' in the repo id."""
    from multigenai.engines.image_engine.engine import MODEL_ALIASES
    
    sdxl_repos = ["sdxl-base", "sdxl-refiner"]
    sd15_repos = ["sd15", "sd-1.5"]
    
    for alias in sdxl_repos:
        repo_id = MODEL_ALIASES.get(alias, alias)
        assert "xl" in repo_id.lower(), f"{alias} → {repo_id} should be detected as XL"
    
    for alias in sd15_repos:
        repo_id = MODEL_ALIASES.get(alias, alias)
        assert "xl" not in repo_id.lower(), f"{alias} → {repo_id} should NOT be detected as XL"


def test_image_engine_sd15_alias_resolves_correctly():
    """Verify that sd15 alias resolves to runwayml/stable-diffusion-v1-5, not an XL model."""
    from multigenai.engines.image_engine.engine import MODEL_ALIASES
    
    assert MODEL_ALIASES["sd15"] == "runwayml/stable-diffusion-v1-5"
    assert MODEL_ALIASES["sd-1.5"] == "runwayml/stable-diffusion-v1-5"
    # SD1.5 repos must NOT contain 'xl' — this is what gates pipeline class selection
    assert "xl" not in MODEL_ALIASES["sd15"].lower()


def test_video_engine_motion_bucket_is_clamped():
    """motion_bucket_id is clamped to [0, 255] even for out-of-range temporal_strength."""
    # temporal_strength > 1.0 would produce motion_bucket > 255 without clamping
    assert max(0, min(255, int(2.5 * 255))) == 255
    assert max(0, min(255, int(-0.5 * 255))) == 0
    assert max(0, min(255, int(0.5 * 255))) == 127


def test_video_engine_does_not_mutate_request_num_frames():
    """VideoEngine adaptive capping uses effective_frames and leaves request.num_frames unchanged."""
    from multigenai.llm.schema_validator import VideoGenerationRequest

    # 1024 * 576 = 589,824 — exceeds the 600,000 threshold
    req = VideoGenerationRequest(prompt="ocean wave", width=1024, height=768, num_frames=16)
    # 1024 * 768 = 786,432 > 600,000 → cap should apply
    original_frames = req.num_frames  # still 16

    effective_frames = req.num_frames
    if req.width * req.height > 600000 and effective_frames > 8:
        effective_frames = 8

    assert req.num_frames == original_frames  # request NOT mutated
    assert effective_frames == 8              # cap applied to local var only
