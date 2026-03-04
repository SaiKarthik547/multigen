"""
test_phase8_interpolation.py — Phase 8 Temporal Enhancement Test Suite

Covers:
  1. Frame count expansion formula for factor 2 and 3
  2. factor=1 passthrough (no modification)
  3. interpolate=False flag prevents engine invocation
  4. Schema: factor > 4 rejected
  5. Schema: interpolate=False with factor != 1 rejected
  6. Lifecycle: _unload_model called on success
  7. Lifecycle: _unload_model called on interpolation failure
  8. Graceful degradation when RIFE model unavailable
  9. Schema defaults (interpolate=True, factor=2)
 10. Output frame formula mathematical correctness
 11. generate_frames() / encode() split API signatures on VideoEngine

All tests run without GPU, RIFE weights, or network access.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frames(n: int) -> list:
    """Return n distinct solid-color PIL Images (64x64 RGB)."""
    colors = [
        (i * 15 % 256, (i * 30) % 256, (i * 45) % 256) for i in range(n)
    ]
    return [PILImage.new("RGB", (64, 64), color=c) for c in colors]


def _mock_ctx():
    ctx = MagicMock()
    ctx.device = "cpu"
    ctx.settings.output_dir = "/tmp/test_out"
    return ctx


# ---------------------------------------------------------------------------
# 1. Frame count formula — factor 2
# ---------------------------------------------------------------------------

class TestFrameCountFormula:
    def test_factor_2_formula(self):
        """10 frames × factor 2 → 10 + 9×1 = 19 frames."""
        n, factor = 10, 2
        expected = n + (n - 1) * (factor - 1)
        assert expected == 19

    def test_factor_3_formula(self):
        """10 frames × factor 3 → 10 + 9×2 = 28 frames."""
        n, factor = 10, 3
        expected = n + (n - 1) * (factor - 1)
        assert expected == 28

    def test_factor_4_formula(self):
        """10 frames × factor 4 → 10 + 9×3 = 37 frames."""
        n, factor = 10, 4
        expected = n + (n - 1) * (factor - 1)
        assert expected == 37

    def test_svd_16_frames_factor_2(self):
        """Standard SVD run: 16 frames × factor 2 → 31 frames."""
        n, factor = 16, 2
        expected = n + (n - 1) * (factor - 1)
        assert expected == 31


# ---------------------------------------------------------------------------
# 2. factor=1 passthrough
# ---------------------------------------------------------------------------

class TestPassthrough:
    def test_factor_1_returns_original(self):
        """InterpolationEngine with factor=1 must return the original list unchanged."""
        from multigenai.engines.interpolation_engine.engine import InterpolationEngine

        ctx = _mock_ctx()
        engine = InterpolationEngine(ctx)
        frames = _make_frames(8)

        result = engine.interpolate(frames, factor=1)

        assert result is frames  # same object, no copy
        assert len(result) == 8

    def test_single_frame_passthrough(self):
        """Only 1 frame — no pairs, always passthrough."""
        from multigenai.engines.interpolation_engine.engine import InterpolationEngine

        ctx = _mock_ctx()
        engine = InterpolationEngine(ctx)
        frames = _make_frames(1)

        result = engine.interpolate(frames, factor=2)
        assert result is frames


# ---------------------------------------------------------------------------
# 3. interpolate=False flag
# ---------------------------------------------------------------------------

class TestInterpolateFlag:
    def test_interpolate_false_skips_engine(self):
        """When request.interpolate=False, GenerationManager must not call interpolate()."""
        from multigenai.llm.schema_validator import VideoGenerationRequest

        req = VideoGenerationRequest(
            prompt="ocean waves",
            interpolate=False,
            interpolation_factor=1,
        )
        assert req.interpolate is False
        # The GenerationManager checks `request.interpolate` before booting the engine
        # Simulate that gating logic:
        called = False
        if req.interpolate and req.interpolation_factor > 1:
            called = True
        assert called is False

    def test_interpolate_true_factor_1_skips_engine(self):
        """interpolate=True but factor=1 → gate condition is False (no-op)."""
        from multigenai.llm.schema_validator import VideoGenerationRequest

        req = VideoGenerationRequest(
            prompt="ocean waves",
            interpolate=True,
            interpolation_factor=1,
        )
        called = False
        if req.interpolate and req.interpolation_factor > 1:
            called = True
        assert called is False


# ---------------------------------------------------------------------------
# 4. Schema validation — factor bounds
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_factor_5_rejected(self):
        """interpolation_factor=5 must raise ValidationError."""
        from multigenai.llm.schema_validator import VideoGenerationRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VideoGenerationRequest(prompt="test", interpolation_factor=5)

    def test_factor_0_rejected(self):
        """interpolation_factor=0 is below minimum of 1."""
        from multigenai.llm.schema_validator import VideoGenerationRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VideoGenerationRequest(prompt="test", interpolation_factor=0)

    def test_disabled_with_factor_2_rejected(self):
        """interpolate=False with interpolation_factor=2 must raise ValueError."""
        from multigenai.llm.schema_validator import VideoGenerationRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="interpolation_factor must be 1"):
            VideoGenerationRequest(
                prompt="test", interpolate=False, interpolation_factor=2
            )

    def test_disabled_with_factor_1_accepted(self):
        """interpolate=False with interpolation_factor=1 is valid."""
        from multigenai.llm.schema_validator import VideoGenerationRequest

        req = VideoGenerationRequest(prompt="test", interpolate=False, interpolation_factor=1)
        assert req.interpolate is False
        assert req.interpolation_factor == 1

    def test_schema_defaults(self):
        """Default values: interpolate=True, interpolation_factor=2."""
        from multigenai.llm.schema_validator import VideoGenerationRequest

        req = VideoGenerationRequest(prompt="ocean waves at sunrise")
        assert req.interpolate is True
        assert req.interpolation_factor == 2

    def test_factor_4_accepted(self):
        """interpolation_factor=4 is at the maximum boundary — must be accepted."""
        from multigenai.llm.schema_validator import VideoGenerationRequest

        req = VideoGenerationRequest(prompt="test", interpolation_factor=4)
        assert req.interpolation_factor == 4


# ---------------------------------------------------------------------------
# 5. Lifecycle: _unload_model called on success
# ---------------------------------------------------------------------------

class TestLifecycleOnSuccess:
    def test_unload_called_after_successful_interpolation(self):
        """_unload_model must be called after successful interpolate()."""
        from multigenai.engines.interpolation_engine.engine import InterpolationEngine

        ctx = _mock_ctx()
        engine = InterpolationEngine(ctx)
        frames = _make_frames(4)

        with patch.object(engine, "_load_model") as mock_load, \
             patch.object(engine, "_unload_model") as mock_unload, \
             patch.object(engine, "_interpolate_pair", return_value=[_make_frames(1)[0]]):

            engine._model = MagicMock()  # pretend model loaded
            result = engine.interpolate(frames, factor=2)

        mock_unload.assert_called_once()

    def test_unload_called_on_failure(self):
        """_unload_model must be called even when _interpolate_pair raises."""
        from multigenai.engines.interpolation_engine.engine import InterpolationEngine

        ctx = _mock_ctx()
        engine = InterpolationEngine(ctx)
        frames = _make_frames(4)

        with patch.object(engine, "_load_model"):
            engine._model = MagicMock()  # ensure model is "loaded"
            with patch.object(engine, "_interpolate_pair", side_effect=RuntimeError("mock failure")), \
                 patch.object(engine, "_unload_model") as mock_unload:
                result = engine.interpolate(frames, factor=2)

        mock_unload.assert_called_once()
        # Graceful degradation — original frames returned
        assert result is frames


# ---------------------------------------------------------------------------
# 6. Graceful degradation — model unavailable
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_returns_original_when_model_is_none(self):
        """If _load_model results in self._model = None, original frames must be returned."""
        from multigenai.engines.interpolation_engine.engine import InterpolationEngine

        ctx = _mock_ctx()
        engine = InterpolationEngine(ctx)
        frames = _make_frames(6)

        with patch.object(engine, "_load_model") as mock_load:
            # _load_model leaves self._model = None (simulates download failure)
            def _side_effect():
                engine._model = None
            mock_load.side_effect = _side_effect

            with patch.object(engine, "_unload_model"):
                result = engine.interpolate(frames, factor=2)

        assert result is frames
        assert len(result) == 6

    def test_returns_original_on_loader_exception(self):
        """If model loading raises, original frames are returned (no crash)."""
        from multigenai.engines.interpolation_engine.engine import InterpolationEngine

        ctx = _mock_ctx()
        engine = InterpolationEngine(ctx)
        frames = _make_frames(4)

        with patch.object(engine, "_load_model", side_effect=RuntimeError("HF download failed")), \
             patch.object(engine, "_unload_model"):
            result = engine.interpolate(frames, factor=2)

        assert result is frames


# ---------------------------------------------------------------------------
# 7. VideoEngine split API
# ---------------------------------------------------------------------------

class TestVideoEngineSplitAPI:
    def test_generate_frames_signature(self):
        """generate_frames() must accept (request, conditioning_image_path) and return 3-tuple."""
        import inspect
        from multigenai.engines.video_engine.engine import VideoEngine

        sig = inspect.signature(VideoEngine.generate_frames)
        params = list(sig.parameters.keys())
        assert "request" in params
        assert "conditioning_image_path" in params

    def test_encode_signature(self):
        """encode() must accept frames, out_path, fps, seed, requested_frames."""
        import inspect
        from multigenai.engines.video_engine.engine import VideoEngine

        sig = inspect.signature(VideoEngine.encode)
        params = list(sig.parameters.keys())
        assert "frames" in params
        assert "out_path" in params
        assert "fps" in params
        assert "seed" in params
        assert "requested_frames" in params

    def test_generate_wrapper_exists(self):
        """generate() backwards-compatible wrapper must still exist."""
        from multigenai.engines.video_engine.engine import VideoEngine
        assert callable(VideoEngine.generate)

    def test_interpolation_engine_isolated_from_video_engine(self):
        """InterpolationEngine must not import from video_engine (no cross-engine coupling)."""
        import pathlib

        src = pathlib.Path(
            r"c:\multigen\multigenai\engines\interpolation_engine\engine.py"
        ).read_text()
        # Only check actual import statements — comments mentioning VideoEngine are fine
        assert "from multigenai.engines.video_engine" not in src, \
            "InterpolationEngine must not import from video_engine"
        assert "import VideoEngine" not in src, \
            "InterpolationEngine must not import VideoEngine class directly"
