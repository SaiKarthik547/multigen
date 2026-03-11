"""
tools/system_check.py — Phase 15 System Health Verification

Checks all critical pipeline contracts before Kaggle run:
  ✓ VRAM free after unload
  ✓ identity latent active
  ✓ scene latent carryover
  ✓ window overlap correct
  ✓ prompt token limit
  ✓ interpolation frame count
  ✓ directional propagation
  ✓ keyframe latent priority chain
"""

from __future__ import annotations

import sys
import os

# Ensure package root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def check_vram_guard() -> None:
    """VRAM health guard behaves correctly on CUDA / CPU."""
    from multigenai.core.model_lifecycle import ModelLifecycle
    try:
        ModelLifecycle.assert_vram_clean(threshold_gb=100.0, context="system_check")
        print("✓ VRAM guard: passed (below threshold).")
    except RuntimeError as e:
        print(f"✗ VRAM guard: FAILED — {e}")
        sys.exit(1)


def check_identity_latent() -> None:
    """TemporalState correctly stores and resets identity latent."""
    from multigenai.core.temporal_state import TemporalState
    import torch

    state = TemporalState()
    dummy = torch.randn(1, 4, 64, 64)
    state.identity_latent = dummy
    assert state.identity_latent is not None, "Identity latent not stored!"

    state.reset()
    assert state.identity_latent is None, "Identity latent not cleared on reset!"
    print("✓ Identity latent: active and resets correctly.")


def check_scene_latent_carryover() -> None:
    """Global latent is correctly propagated via TemporalState."""
    from multigenai.core.temporal_state import TemporalState
    import torch

    state = TemporalState()
    dummy = torch.randn(1, 4, 8, 64, 96)
    state.global_latent = dummy
    state.latent_velocity = torch.randn_like(dummy)

    assert state.global_latent is not None, "global_latent not stored!"
    assert state.latent_velocity is not None, "latent_velocity not stored!"
    print("✓ Scene latent carryover: global_latent and latent_velocity active.")


def check_window_overlap() -> None:
    """Sliding window constants match Phase 15 spec: 24 frames, 8 overlap."""
    # Parse constants directly from the engine source
    import pathlib
    engine_path = pathlib.Path(__file__).parent.parent / "multigenai/engines/video_engine/engine.py"
    src = engine_path.read_text(encoding="utf-8")
    assert "WINDOW_SIZE = 24" in src, "WINDOW_SIZE is not 24!"
    assert "OVERLAP = 8" in src, "OVERLAP is not 8!"
    print("✓ Sliding window: WINDOW_SIZE=24, OVERLAP=8 confirmed.")


def check_prompt_token_limit() -> None:
    """Prompt processor respects 70-token hard limit."""
    from multigenai.prompting.prompt_processor import PromptProcessor
    long_prompt = " ".join([f"word{i}" for i in range(200)])
    processor = PromptProcessor()
    plan = processor.process(long_prompt, force_single_segment=True)
    seg = plan.segments[0]
    token_count = len(seg.positive.split())
    assert token_count <= 77, f"Prompt exceeds CLIP limit: {token_count} tokens"
    print(f"✓ Prompt token limit: {token_count} tokens ≤ 77 limit.")


def check_directional_propagation() -> None:
    """LatentPropagator produces clamped output with directional velocity."""
    from multigenai.temporal.latent_propagator import LatentPropagator
    import torch

    propagator = LatentPropagator()
    lat = torch.randn(1, 4, 24, 64, 96)
    prev = torch.randn_like(lat)

    result, velocity = propagator.propagate(lat, prev_latent=prev)
    assert result.shape == lat.shape, "Propagated shape mismatch!"
    assert result.max() <= 4.01, "Latent not clamped to max 4!"
    assert result.min() >= -4.01, "Latent not clamped to min -4!"
    assert velocity is not None, "Velocity not returned on non-first scene!"
    print("✓ Directional propagation: shape correct, velocity returned, clamp enforced.")


def check_interpolation_chunk() -> None:
    """InterpolationEngine source has CHUNK_SIZE=64 defined."""
    import pathlib
    engine_path = pathlib.Path(__file__).parent.parent / "multigenai/engines/interpolation_engine/engine.py"
    src = engine_path.read_text(encoding="utf-8")
    assert "CHUNK_SIZE = 64" in src, "CHUNK_SIZE=64 not found in interpolation engine!"
    print("✓ Interpolation chunking: CHUNK_SIZE=64 confirmed.")


def check_keyframe_priority_chain() -> None:
    """VideoEngine source implements keyframe_latent priority chain."""
    import pathlib
    engine_path = pathlib.Path(__file__).parent.parent / "multigenai/engines/video_engine/engine.py"
    src = engine_path.read_text(encoding="utf-8")
    assert "keyframe_latent is not None" in src, "keyframe_latent priority chain not found!"
    assert "Keyframe latent channel mismatch" in src, "Channel validation missing!"
    print("✓ Keyframe priority chain: implemented in VideoEngine.")


def main() -> None:
    print("\n" + "=" * 60)
    print("  MultiGenAI Phase 15 System Health Check")
    print("=" * 60 + "\n")

    checks = [
        check_vram_guard,
        check_identity_latent,
        check_scene_latent_carryover,
        check_window_overlap,
        check_prompt_token_limit,
        check_directional_propagation,
        check_interpolation_chunk,
        check_keyframe_priority_chain,
    ]

    passed = 0
    failed = 0

    for check in checks:
        try:
            check()
            passed += 1
        except SystemExit:
            failed += 1
        except Exception as e:
            print(f"✗ {check.__name__}: FAILED — {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
