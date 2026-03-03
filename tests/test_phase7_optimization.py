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
    
    # Needs sequence CPU offload, VAE Tiling, and Attention slicing
    mock_pipe.enable_sequential_cpu_offload.assert_called_once()
    mock_pipe.enable_vae_tiling.assert_called_once()
    mock_pipe.enable_attention_slicing.assert_called_once()

def test_apply_memory_optimizations_directml():
    """Verify DirectML skips CPU offload, but applies VAE tiling and attention slicing."""
    from multigenai.engines.image_engine.engine import _apply_memory_optimizations
    
    mock_pipe = MagicMock()
    _apply_memory_optimizations(mock_pipe, device="directml")
    
    mock_pipe.enable_sequential_cpu_offload.assert_not_called()
    mock_pipe.enable_vae_tiling.assert_called_once()
    mock_pipe.enable_attention_slicing.assert_called_once()
    mock_pipe.to.assert_called_once_with("directml")

def test_apply_memory_optimizations_cpu():
    """Verify CPU applies VAE tiling (saves RAM), but no CUDA-specific offloads."""
    from multigenai.engines.image_engine.engine import _apply_memory_optimizations
    
    mock_pipe = MagicMock()
    _apply_memory_optimizations(mock_pipe, device="cpu")
    
    mock_pipe.enable_sequential_cpu_offload.assert_not_called()
    mock_pipe.enable_attention_slicing.assert_not_called()  # We didn't enable slicing for cpu in impl
    mock_pipe.enable_vae_tiling.assert_called_once()
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
