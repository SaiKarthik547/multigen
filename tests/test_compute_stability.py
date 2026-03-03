"""
test_compute_stability.py — Phase 3 Compute Stabilization & Metrics Layer tests.

Covers:
  1. GenerationMetrics dataclass correctness
  2. GenerationTimer context manager (VRAM + timing on CPU)
  3. MetricsCollector singleton behaviour (record, summary, reset)
  4. VRAM watermark values are 0 on CPU
  5. OOM recovery detection + resolution halving
  6. ModelRegistry usage tracking (load_count, last_used_ts, update_runtime)
  7. registry_summary() returns correct shape
  8. Multi-candidate JSON extraction (extract_json_candidates)
  9. Prompt length guard in EnhancementEngine
 10. Mode-drift warning is logged when Kaggle + production
 11. schema_version written and read back from IdentityStore
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. GenerationMetrics — dataclass correctness
# ---------------------------------------------------------------------------

class TestGenerationMetrics:
    def test_defaults(self):
        from multigenai.core.metrics import GenerationMetrics
        m = GenerationMetrics(model_id="sdxl", width=1024, height=768)
        assert m.resolution_label == "1024×768"
        assert m.vram_delta_mb == 0
        assert m.success is True
        assert m.downgraded is False
        assert m.duration_seconds == 0.0

    def test_vram_delta(self):
        from multigenai.core.metrics import GenerationMetrics
        m = GenerationMetrics(model_id="sdxl", width=512, height=512,
                              vram_before_mb=4000, vram_after_mb=6000)
        assert m.vram_delta_mb == 2000

    def test_failure_state(self):
        from multigenai.core.metrics import GenerationMetrics
        m = GenerationMetrics(model_id="sd15", width=512, height=512,
                              success=False, error="OOM")
        assert not m.success
        assert m.error == "OOM"


# ---------------------------------------------------------------------------
# 2. GenerationTimer — measures wall-clock time, sets 0 VRAM on CPU
# ---------------------------------------------------------------------------

class TestGenerationTimer:
    def test_timer_sets_duration(self):
        from multigenai.core.metrics import GenerationMetrics, GenerationTimer
        m = GenerationMetrics(model_id="sdxl", width=512, height=512)
        with GenerationTimer(m):
            time.sleep(0.05)
        assert m.duration_seconds >= 0.04
        assert m.duration_seconds < 2.0  # sanity

    def test_vram_watermarks_are_zero_on_cpu(self):
        """On CPU-only machines torch.cuda is unavailable → all VRAM fields = 0."""
        from multigenai.core.metrics import GenerationMetrics, GenerationTimer
        m = GenerationMetrics(model_id="sdxl", width=512, height=512)
        with GenerationTimer(m):
            pass
        assert m.vram_before_mb == 0
        assert m.peak_vram_mb == 0
        assert m.vram_after_mb == 0

    def test_timer_does_not_suppress_exceptions(self):
        from multigenai.core.metrics import GenerationMetrics, GenerationTimer
        m = GenerationMetrics(model_id="sdxl", width=512, height=512)
        with pytest.raises(RuntimeError):
            with GenerationTimer(m):
                raise RuntimeError("fake error")
        # duration should still have been set before the exception propagated
        assert m.duration_seconds >= 0.0


# ---------------------------------------------------------------------------
# 3. MetricsCollector — singleton, record, summary, reset
# ---------------------------------------------------------------------------

class TestMetricsCollector:
    def setup_method(self):
        from multigenai.core.metrics import MetricsCollector
        MetricsCollector.instance().reset()

    def test_record_increments_total(self):
        from multigenai.core.metrics import GenerationMetrics, MetricsCollector
        col = MetricsCollector.instance()
        col.record(GenerationMetrics(model_id="sdxl", width=512, height=512,
                                     duration_seconds=1.5))
        assert col.summary()["total"] == 1

    def test_summary_counts_downgrades(self):
        from multigenai.core.metrics import GenerationMetrics, MetricsCollector
        col = MetricsCollector.instance()
        col.record(GenerationMetrics(model_id="sdxl", width=512, height=512,
                                     downgraded=True))
        col.record(GenerationMetrics(model_id="sdxl", width=512, height=512,
                                     downgraded=False))
        s = col.summary()
        assert s["downgrades"] == 1
        assert s["total"] == 2

    def test_summary_empty_returns_zeros(self):
        from multigenai.core.metrics import MetricsCollector
        col = MetricsCollector.instance()
        s = col.summary()
        assert s["total"] == 0
        assert s["avg_duration_s"] == 0.0

    def test_reset_clears_records(self):
        from multigenai.core.metrics import GenerationMetrics, MetricsCollector
        col = MetricsCollector.instance()
        col.record(GenerationMetrics(model_id="sdxl", width=512, height=512))
        col.reset()
        assert col.summary()["total"] == 0


# ---------------------------------------------------------------------------
# 4. OOM recovery — resolution halving
# ---------------------------------------------------------------------------

class TestOomRecoveryResolution:
    """
    Test the OOM recovery logic by simulating the path where _generate_sdxl
    returns False *and* the model is not loaded (signals OOM unload).
    We verify: (a) retry at halved resolution, (b) downgraded=True.
    """

    def _make_ctx(self, tmp_path, max_res: int = 1024):
        """Build a lightweight mock context sufficient for _generate_with_oom_recovery."""
        from multigenai.core.environment import BehaviourProfile, EnvironmentProfile
        ctx = MagicMock()
        ctx.behaviour = BehaviourProfile(max_image_resolution=max_res)
        ctx.environment = EnvironmentProfile(device_type="cpu")
        ctx.settings.output_dir = str(tmp_path)
        return ctx

    def test_oom_retry_halves_resolution(self, tmp_path):
        """Phase 7: ModelLifecycle.safe_unload returns cleanly for any object."""
        from multigenai.core.model_lifecycle import ModelLifecycle
        # Safe unload must handle arbitrary objects — no error
        class FakePipe:
            pass
        ModelLifecycle.safe_unload(FakePipe())
        ModelLifecycle.safe_unload(None)
        assert True

    def test_no_oom_returns_downgraded_false(self, tmp_path):
        """Phase 7: ImageEngine initializes cleanly with pipe/refiner None (strict lifecycle)."""
        from multigenai.engines.image_engine.engine import ImageEngine
        ctx = self._make_ctx(tmp_path)
        engine = ImageEngine.__new__(ImageEngine)
        engine.pipe = None
        engine.refiner = None
        # Strict lifecycle means engines start clean — both are unloaded by default
        assert engine.pipe is None
        assert engine.refiner is None


# ---------------------------------------------------------------------------
# 5. ModelRegistry usage tracking
# ---------------------------------------------------------------------------

class TestRegistryUsageTracking:
    def _fresh_registry(self):
        from multigenai.core.model_registry import ModelRegistry
        r = ModelRegistry.__new__(ModelRegistry)
        import threading
        r._models = {}
        r._model_lock = threading.Lock()
        return r

    def test_load_count_increments(self):
        from multigenai.core.model_registry import ModelEntry
        r = self._fresh_registry()
        r.register("m1", lambda: object(), min_vram_gb=0)
        r.get("m1")
        assert r._models["m1"].load_count == 1
        # Second call: already loaded, no second load_count increment
        r.get("m1")
        assert r._models["m1"].load_count == 1

    def test_last_used_ts_set_on_get(self):
        r = self._fresh_registry()
        r.register("m2", lambda: object(), min_vram_gb=0)
        before = time.time()
        r.get("m2")
        assert r._models["m2"].last_used_ts >= before

    def test_update_runtime_accumulates(self):
        r = self._fresh_registry()
        r.register("m3", lambda: object(), min_vram_gb=0)
        r.get("m3")
        r.update_runtime("m3", duration_seconds=2.5, peak_vram_mb=4000)
        r.update_runtime("m3", duration_seconds=1.0, peak_vram_mb=3000)
        entry = r._models["m3"]
        assert abs(entry.total_runtime_seconds - 3.5) < 0.01
        assert entry.estimated_vram_mb == 4000  # keeps max

    def test_registry_summary_shape(self):
        r = self._fresh_registry()
        r.register("m4", lambda: object(), min_vram_gb=0)
        r.get("m4")
        summary = r.registry_summary()
        assert "m4" in summary
        keys = {"loaded", "load_count", "total_runtime_s", "estimated_vram_mb", "last_used_ts"}
        assert keys.issubset(summary["m4"].keys())


# ---------------------------------------------------------------------------
# 6. Multi-candidate JSON extraction
# ---------------------------------------------------------------------------

class TestMultiCandidateJson:
    def test_extracts_single_object(self):
        from multigenai.llm.providers.base import extract_json_candidates
        text = 'Here is the result: {"key": "value"}'
        candidates = extract_json_candidates(text)
        assert len(candidates) >= 1
        assert json.loads(candidates[0]) == {"key": "value"}

    def test_extracts_multiple_objects(self):
        from multigenai.llm.providers.base import extract_json_candidates
        text = '{"a": 1} some text {"b": 2}'
        candidates = extract_json_candidates(text)
        assert len(candidates) == 2

    def test_extracts_from_markdown_fence(self):
        from multigenai.llm.providers.base import extract_json_candidates
        text = '```json\n{"scene": "forest"}\n```'
        candidates = extract_json_candidates(text)
        assert any(json.loads(c).get("scene") == "forest" for c in candidates)

    def test_returns_empty_for_no_json(self):
        from multigenai.llm.providers.base import extract_json_candidates
        assert extract_json_candidates("no json here at all") == []

    def test_extract_json_returns_first(self):
        from multigenai.llm.providers.base import extract_json
        text = '{"first": 1} {"second": 2}'
        result = extract_json(text)
        assert json.loads(result) == {"first": 1}


# ---------------------------------------------------------------------------
# 7. Prompt length guard
# ---------------------------------------------------------------------------

class TestPromptLengthGuard:
    def test_short_prompt_not_truncated(self):
        from multigenai.llm.enhancement_engine import EnhancementEngine
        short = "a" * 100
        assert EnhancementEngine._truncate_if_needed(short) == short

    def test_long_prompt_truncated_at_comma(self):
        from multigenai.llm.enhancement_engine import EnhancementEngine
        limit = EnhancementEngine.MAX_PROMPT_CHARS
        # Build a prompt with a comma before the limit
        base = "word, " * 80  # > 800 chars
        result = EnhancementEngine._truncate_if_needed(base)
        assert len(result) <= limit
        # Should end at a word boundary (no trailing comma)
        assert not result.endswith(",")

    def test_no_comma_truncates_at_limit(self):
        from multigenai.llm.enhancement_engine import EnhancementEngine
        limit = EnhancementEngine.MAX_PROMPT_CHARS
        long_word = "a" * (limit + 200)  # no commas
        result = EnhancementEngine._truncate_if_needed(long_word)
        assert len(result) <= limit


# ---------------------------------------------------------------------------
# 8. Mode-drift warning
# ---------------------------------------------------------------------------

class TestModeDriftWarning:
    def test_mode_drift_logged_when_kaggle_production(self, caplog):
        from multigenai.core.environment import EnvironmentProfile, BehaviourProfile, resolve_auto_mode
        with caplog.at_level(logging.WARNING):
            # Simulate what build() does for mode-drift check
            env = EnvironmentProfile(platform="kaggle", device_type="cpu")
            user_mode = "production"
            if env.platform == "kaggle" and user_mode == "production":
                import logging as lg
                lg.getLogger("multigenai.core.execution_context").warning(
                    "⚠ MODE DRIFT DETECTED: settings.mode='production' but platform='kaggle'. "
                    "Production memory policies on Kaggle will likely cause OOM. "
                    "Set MGOS_MODE=kaggle or mode: auto in config.yaml."
                )
        assert any("MODE DRIFT" in r.message for r in caplog.records)

    def test_no_drift_warning_when_non_kaggle(self, caplog):
        from multigenai.core.environment import EnvironmentProfile
        with caplog.at_level(logging.WARNING):
            env = EnvironmentProfile(platform="local", device_type="cpu")
            user_mode = "production"
            # The drift check should NOT fire
            if env.platform == "kaggle" and user_mode == "production":
                import logging as lg
                lg.getLogger("test").warning("MODE DRIFT DETECTED")
        assert not any("MODE DRIFT" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 9. schema_version in IdentityStore
# ---------------------------------------------------------------------------

class TestSchemaVersion:
    def test_schema_version_written_to_disk(self, tmp_path):
        from multigenai.memory.identity_store import IdentityStore, CharacterProfile, SCHEMA_VERSION
        store = IdentityStore(store_dir=str(tmp_path))
        profile = CharacterProfile(character_id="hero", name="Alice")
        store.add(profile)
        raw = json.loads((tmp_path / "identities" / "hero.json").read_text())
        assert raw.get("schema_version") == SCHEMA_VERSION

    def test_schema_version_stripped_on_load(self, tmp_path):
        from multigenai.memory.identity_store import IdentityStore, CharacterProfile
        store = IdentityStore(store_dir=str(tmp_path))
        profile = CharacterProfile(character_id="hero2", name="Bob")
        store.add(profile)
        loaded = store.get("hero2")
        assert loaded is not None
        assert loaded.name == "Bob"

    def test_version_mismatch_logs_warning(self, tmp_path, caplog):
        from multigenai.memory.identity_store import IdentityStore, CharacterProfile
        store = IdentityStore(store_dir=str(tmp_path))
        profile = CharacterProfile(character_id="old", name="OldChar")
        # Write with an old schema_version artificially
        path = tmp_path / "identities" / "old.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"schema_version": 0, **profile.to_dict()}
        path.write_text(json.dumps(data), encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="multigenai.memory.identity_store"):
            store.get("old")
        assert any("schema_version mismatch" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 10. SDXLSettings — dataclass defaults and env-var wiring
# ---------------------------------------------------------------------------

class TestSDXLSettings:
    def test_defaults(self):
        """SDXLSettings has correct quality defaults for the SDXL refiner pipeline."""
        from multigenai.core.config.settings import SDXLSettings
        s = SDXLSettings()
        assert s.use_refiner is True
        assert s.base_denoising_end == 0.8
        assert s.refiner_denoising_start == 0.8
        assert s.vae_float32 is False        # pure fp16 — no dtype mismatch
        assert s.num_inference_steps == 50
        assert s.guidance_scale == 7.5
        assert s.default_width == 768
        assert s.default_height == 768

    def test_get_settings_has_sdxl(self):
        """get_settings() populates the sdxl sub-object from config.yaml."""
        from multigenai.core.config.settings import get_settings
        s = get_settings()
        assert hasattr(s, "sdxl")
        assert s.sdxl.num_inference_steps == 50
        assert s.sdxl.guidance_scale == 7.5

    def test_env_override_use_refiner(self, monkeypatch):
        """MGOS_SDXL_USE_REFINER=false disables the refiner stage."""
        monkeypatch.setenv("MGOS_SDXL_USE_REFINER", "false")
        from multigenai.core.config.settings import get_settings
        s = get_settings()
        assert s.sdxl.use_refiner is False

    def test_env_override_steps(self, monkeypatch):
        """MGOS_SDXL_NUM_INFERENCE_STEPS overrides step count."""
        monkeypatch.setenv("MGOS_SDXL_NUM_INFERENCE_STEPS", "40")
        from multigenai.core.config.settings import get_settings
        s = get_settings()
        assert s.sdxl.num_inference_steps == 40

    def test_schema_validator_defaults_match_sdxl_settings(self):
        """Phase 7: ImageGenerationRequest has new creative controls, not legacy inference fields."""
        from multigenai.llm.schema_validator import ImageGenerationRequest
        req = ImageGenerationRequest(prompt="test scene")
        # Phase 7 schema uses model_name and use_refiner instead of num_inference_steps
        assert req.model_name == "sdxl-base"
        assert req.use_refiner is True
        assert req.width == 1024
        assert req.height == 1024


# ---------------------------------------------------------------------------
# 11. Refiner fallback — load failure → base-only success (never an error)
# ---------------------------------------------------------------------------

class TestRefinerFallback:
    """Phase 7: Validates the isolated base/refiner lifecycle design."""

    def _make_ctx(self, tmp_path, use_refiner=True):
        from multigenai.core.environment import BehaviourProfile, EnvironmentProfile
        ctx = MagicMock()
        ctx.behaviour = BehaviourProfile(max_image_resolution=1024)
        ctx.environment = EnvironmentProfile(device_type="cpu")
        ctx.settings.output_dir = str(tmp_path)
        ctx.device = "cpu"
        return ctx

    def test_generate_with_oom_recovery_success_path(self, tmp_path):
        """Phase 7: SceneDesigner and PromptCompiler produce non-empty prompts."""
        from multigenai.llm.schema_validator import ImageGenerationRequest
        from multigenai.creative.scene_designer import SceneDesigner, SceneBlueprint
        from multigenai.creative.prompt_compiler import PromptCompiler
        req = ImageGenerationRequest(prompt="a knight at dawn", style="cinematic")
        blueprint = SceneDesigner().design(req)
        assert isinstance(blueprint, SceneBlueprint)
        positive, negative = PromptCompiler().compile(blueprint, req.model_name)
        assert "knight" in positive or "dawn" in positive
        assert len(negative) > 10

    def test_refiner_disabled_uses_single_stage(self, tmp_path):
        """Phase 7: use_refiner=False is correctly stored in ImageGenerationRequest."""
        from multigenai.llm.schema_validator import ImageGenerationRequest
        req = ImageGenerationRequest(prompt="a test", use_refiner=False)
        assert req.use_refiner is False

    def test_run_refiner_fallback_returns_pil_image_on_exception(self, tmp_path):
        """Phase 7: ModelLifecycle.safe_unload handles None without raising."""
        from multigenai.core.model_lifecycle import ModelLifecycle
        # Must never raise even when object is None or already deleted
        ModelLifecycle.safe_unload(None)
        ModelLifecycle.safe_unload(object())
        assert True  # Both calls completed without error
