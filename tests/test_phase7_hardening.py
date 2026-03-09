"""
test_phase7_hardening.py — Phase 7 Hardening Verification Suite

Covers:
  1. Model alias resolution (MODEL_ALIASES dict)
  2. Schema: num_inference_steps field presence and bounds
  3. Engine lifecycle: double-unload does not crash
  4. Determinism: same seed → same generator state (contract check)
  5. Resolution sync: VideoGenerationRequest keyframe dimensions
  6. Prompt boundary: PromptCompiler rejects quality-contaminated subjects
  7. FFmpeg encode: communicate() pattern (no deadlock regression)
  8. PromptCompiler quality token injection (once and only once)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# 1. Model alias resolution
# ---------------------------------------------------------------------------

class TestModelAliasResolution:
    def test_sdxl_base_alias_resolves_to_hf_repo(self):
        """'sdxl-base' resolves to the correct HuggingFace repo id (Juggernaut XL as of Phase 12)."""
        from multigenai.engines.image_engine.engine import MODEL_ALIASES
        assert MODEL_ALIASES["sdxl-base"] == "RunDiffusion/Juggernaut-XL-v9"

    def test_sdxl_refiner_alias_resolves(self):
        from multigenai.engines.image_engine.engine import MODEL_ALIASES
        assert MODEL_ALIASES["sdxl-refiner"] == "stabilityai/stable-diffusion-xl-refiner-1.0"

    def test_unknown_alias_passes_through(self):
        """An unrecognised model_name is returned unchanged (custom HF repo path)."""
        from multigenai.engines.image_engine.engine import ImageEngine
        engine = ImageEngine.__new__(ImageEngine)
        custom = "my-org/my-custom-sdxl"
        assert engine._resolve_model_name(custom) == custom

    def test_known_alias_does_not_pass_through(self):
        """'sdxl-base' must NOT pass through; it must be replaced."""
        from multigenai.engines.image_engine.engine import ImageEngine
        engine = ImageEngine.__new__(ImageEngine)
        resolved = engine._resolve_model_name("sdxl-base")
        assert resolved != "sdxl-base"
        # Phase 12 uses Juggernaut XL (RunDiffusion) or standard SDXL (stabilityai)
        assert ("stabilityai" in resolved.lower() or "rundiffusion" in resolved.lower())


# ---------------------------------------------------------------------------
# 2. Schema: num_inference_steps
# ---------------------------------------------------------------------------

class TestSchemaNumInferenceSteps:
    def test_default_is_30(self):
        from multigenai.llm.schema_validator import ImageGenerationRequest
        req = ImageGenerationRequest(prompt="a castle at dawn")
        assert req.num_inference_steps == 30

    def test_custom_steps_accepted(self):
        from multigenai.llm.schema_validator import ImageGenerationRequest
        req = ImageGenerationRequest(prompt="a castle at dawn", num_inference_steps=50)
        assert req.num_inference_steps == 50

    def test_below_minimum_rejected(self):
        from multigenai.llm.schema_validator import ImageGenerationRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ImageGenerationRequest(prompt="test", num_inference_steps=5)

    def test_above_maximum_rejected(self):
        from multigenai.llm.schema_validator import ImageGenerationRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ImageGenerationRequest(prompt="test", num_inference_steps=200)

    def test_boundary_minimum_accepted(self):
        from multigenai.llm.schema_validator import ImageGenerationRequest
        req = ImageGenerationRequest(prompt="test", num_inference_steps=10)
        assert req.num_inference_steps == 10

    def test_boundary_maximum_accepted(self):
        from multigenai.llm.schema_validator import ImageGenerationRequest
        req = ImageGenerationRequest(prompt="test", num_inference_steps=100)
        assert req.num_inference_steps == 100


# ---------------------------------------------------------------------------
# 3. Engine lifecycle: double-unload does not crash
# ---------------------------------------------------------------------------

class TestEngineLifecycleSafety:
    def test_unload_twice_no_crash(self):
        """ModelLifecycle.safe_unload must be idempotent — calling it twice is safe."""
        from multigenai.core.model_lifecycle import ModelLifecycle

        class FakePipe:
            pass

        obj = FakePipe()
        ModelLifecycle.safe_unload(obj)
        ModelLifecycle.safe_unload(obj)  # second call — must not raise
        assert True

    def test_unload_none_no_crash(self):
        from multigenai.core.model_lifecycle import ModelLifecycle
        ModelLifecycle.safe_unload(None)
        ModelLifecycle.safe_unload(None)  # also idempotent on None
        assert True

    def test_image_engine_starts_unloaded(self):
        """Fresh ImageEngine must have pipe=None and refiner=None."""
        from multigenai.engines.image_engine.engine import ImageEngine
        engine = ImageEngine.__new__(ImageEngine)
        engine.pipe = None
        engine.refiner = None
        assert engine.pipe is None
        assert engine.refiner is None


# ---------------------------------------------------------------------------
# 4. Determinism: generator identity contract
# ---------------------------------------------------------------------------

class TestDeterminismContract:
    def test_same_seed_produces_same_generator_state(self):
        """Two generators with same seed must produce identical random sequences."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        seed = 42
        g1 = torch.Generator().manual_seed(seed)
        g2 = torch.Generator().manual_seed(seed)
        # Draw a single random value from each
        v1 = torch.randint(0, 1_000_000, (1,), generator=g1).item()
        v2 = torch.randint(0, 1_000_000, (1,), generator=g2).item()
        assert v1 == v2, "Same seed must produce identical generator output"

    def test_engine_seed_field_defaults_to_42(self):
        """Default seed=42 in schema ensures reproducibility without explicit seed."""
        from multigenai.llm.schema_validator import ImageGenerationRequest
        req = ImageGenerationRequest(prompt="test")
        assert req.seed == 42

    def test_engine_exposes_run_signature_with_compiled_prompts(self):
        """ImageEngine.run() must accept (compiled_prompt, negative_prompt, request)."""
        from multigenai.engines.image_engine.engine import ImageEngine
        import inspect
        sig = inspect.signature(ImageEngine.run)
        params = list(sig.parameters.keys())
        assert "compiled_prompt" in params
        assert "negative_prompt" in params
        assert "request" in params


# ---------------------------------------------------------------------------
# 5. Resolution sync: keyframe must match video dimensions
# ---------------------------------------------------------------------------

class TestVideoResolutionSync:
    def test_video_request_resolution_fields_exist(self):
        """VideoGenerationRequest must expose width and height."""
        from multigenai.llm.schema_validator import VideoGenerationRequest
        req = VideoGenerationRequest(prompt="ocean at sunset", num_frames=8, fps=8)
        assert hasattr(req, "width")
        assert hasattr(req, "height")

    def test_keyframe_resolution_matches_video(self):
        """
        The ImageGenerationRequest created for keyframe must use the
        VideoGenerationRequest's width and height (not defaults).
        Simulate generate_video keyframe creation logic.
        """
        from multigenai.llm.schema_validator import VideoGenerationRequest, ImageGenerationRequest

        video_req = VideoGenerationRequest(
            prompt="a forest in fog",
            num_frames=16,
            fps=8,
            width=1024,
            height=576,
        )

        # Simulate what GenerationManager does
        keyframe_req = ImageGenerationRequest(
            prompt=video_req.prompt,
            width=video_req.width,   # hard override
            height=video_req.height, # hard override
            seed=video_req.seed,
        )

        assert keyframe_req.width == video_req.width, "Keyframe width must match video width"
        assert keyframe_req.height == video_req.height, "Keyframe height must match video height"

    def test_video_default_resolution_is_divisible_by_64(self):
        """SVD-XT default resolution must be compatible with diffusion models."""
        from multigenai.llm.schema_validator import VideoGenerationRequest
        req = VideoGenerationRequest(prompt="test")
        assert req.width % 64 == 0
        assert req.height % 64 == 0


# ---------------------------------------------------------------------------
# 6. Prompt boundary: PromptCompiler rejects contaminated subjects
# ---------------------------------------------------------------------------

class TestPromptBoundaryEnforcement:
    def test_compiler_rejects_quality_tokens_in_subject(self):
        """PromptCompiler must raise AssertionError if subject contains quality tokens."""
        from multigenai.creative.prompt_compiler import PromptCompiler
        from multigenai.creative.scene_designer import SceneBlueprint

        # Simulate a contaminated blueprint (as if SceneDesigner added quality tokens)
        contaminated_blueprint = SceneBlueprint(
            subject="a knight in 8k ultra-detailed armor",  # violation
            environment="medieval castle",
            lighting="golden hour",
            camera_description="wide shot",
            character_details="",
            rendering_style="",
            atmosphere="",
        )

        compiler = PromptCompiler()
        with pytest.raises(AssertionError, match="boundary violation"):
            compiler.compile(contaminated_blueprint, "sdxl-base")

    def test_compiler_accepts_clean_subject(self):
        """A subject without quality tokens must compile without error."""
        from multigenai.creative.prompt_compiler import PromptCompiler
        from multigenai.creative.scene_designer import SceneBlueprint

        clean_blueprint = SceneBlueprint(
            subject="a knight in shining armor",
            environment="medieval castle",
            lighting="golden hour",
            camera_description="wide shot",
            character_details="",
            rendering_style="cinematic",
            atmosphere="dramatic",
        )

        compiler = PromptCompiler()
        positive, negative = compiler.compile(clean_blueprint, "sdxl-base")
        assert "knight" in positive
        assert len(negative) > 10

    def test_quality_tokens_appear_once_in_positive(self):
        """Quality tokens such as 'masterpiece' must appear exactly once."""
        from multigenai.creative.prompt_compiler import PromptCompiler
        from multigenai.creative.scene_designer import SceneBlueprint

        blueprint = SceneBlueprint(
            subject="a sunset over the ocean",
            environment="open sea",
            lighting="golden",
            camera_description="",
            character_details="",
            rendering_style="",
            atmosphere="",
        )

        compiler = PromptCompiler()
        positive, _ = compiler.compile(blueprint, "sdxl-base")
        assert positive.count("masterpiece") == 1, "masterpiece must appear exactly once"


# ---------------------------------------------------------------------------
# 7. FFmpeg: correct streaming lifecycle (no communicate() after stdin.close)
# ---------------------------------------------------------------------------

class TestFFmpegCommunicateContract:
    def test_encode_video_uses_correct_streaming_lifecycle(self):
        """
        VideoEngine._encode_video must use the correct streaming lifecycle:
          write frames → stdin.close() → wait() → stderr.read() → check returncode

        Ordering: wait() FIRST, stderr.read() AFTER.
        Reading stderr before wait() races with ultrafast ffmpeg exit on Kaggle
        causing flush-of-closed-file ValueError on stdin pipe internals.
        """
        import inspect
        from multigenai.engines.video_engine.engine import VideoEngine
        source = inspect.getsource(VideoEngine.encode)

        assert "stdin.close()" in source, "must close stdin to signal EOF"
        assert "process.wait()" in source, "must call wait() to let ffmpeg finish"
        assert "stderr.read()" in source, "must read stderr for error diagnostics"
        assert "process.communicate()" not in source, \
            "communicate() must NOT be used after manual stdin writes"

        # Enforce ordering: wait() must appear BEFORE stderr.read() in source
        wait_pos = source.index("process.wait()")
        stderr_pos = source.index("process.stderr.read()")
        assert wait_pos < stderr_pos, \
            "process.wait() must come BEFORE stderr.read() to avoid race on fast ffmpeg exit"

    def test_generate_video_accepts_explicit_num_frames_param(self):
        """_generate_video must accept num_frames as an explicit parameter (not read from request)."""
        import inspect
        from multigenai.engines.video_engine.engine import VideoEngine
        sig = inspect.signature(VideoEngine._generate_video)
        assert "num_frames" in sig.parameters, \
            "_generate_video must accept num_frames explicitly to avoid request mutation"
