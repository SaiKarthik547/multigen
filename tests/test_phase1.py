"""
Phase 1 smoke tests — import safety and basic interface verification.

All tests must pass on a CPU-only machine with no GPU and no large models.
Tests skip gracefully when optional dependencies (torch, PIL, etc.) are absent.
"""

from __future__ import annotations

import importlib
import sys
import pathlib
import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _skip_if_missing(*packages: str):
    """Return a pytest.mark.skipif marker if any of the packages are missing."""
    missing = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        return pytest.mark.skip(reason=f"Missing packages: {missing}")
    return pytest.mark.skipif(False, reason="")


# ---------------------------------------------------------------------------
# Package import safety (all modules must import without GPU / heavy deps)
# ---------------------------------------------------------------------------

MODULES_TO_IMPORT = [
    "multigenai",
    "multigenai.core",
    "multigenai.core.config.settings",
    "multigenai.core.logging.logger",
    "multigenai.core.exceptions",
    "multigenai.core.device_manager",
    "multigenai.core.model_registry",
    "multigenai.core.capability_report",
    "multigenai.core.lifecycle",
    "multigenai.core.execution_context",
    "multigenai.memory.identity_store",
    "multigenai.memory.world_state",
    "multigenai.memory.style_registry",
    "multigenai.memory.embedding_store",
    "multigenai.llm.schema_validator",
    "multigenai.llm.enhancement_engine",
    "multigenai.llm.scene_planner",
    "multigenai.engines.image_engine.engine",
    "multigenai.engines.video_engine.engine",
    "multigenai.engines.audio_engine.engine",
    "multigenai.engines.document_engine.engine",
    "multigenai.engines.presentation_engine.engine",
    "multigenai.engines.code_engine.engine",
    "multigenai.control.controlnet_manager",
    "multigenai.control.guidance_manager",
    "multigenai.control.consistency_enforcer",
    "multigenai.temporal.motion_engine",
    "multigenai.temporal.optical_flow",
    "multigenai.temporal.latent_propagator",
    "multigenai.orchestration.dag_engine",
    "multigenai.orchestration.task_scheduler",
    "multigenai.orchestration.job_queue",
    "multigenai.api.rest_api",
    "multigenai.api.websocket",
]


@pytest.mark.parametrize("module_path", MODULES_TO_IMPORT)
def test_module_imports_safely(module_path: str):
    """Every module must import without error on a CPU-only machine."""
    mod = importlib.import_module(module_path)
    assert mod is not None


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def test_settings_defaults():
    from multigenai.core.config.settings import get_settings
    s = get_settings()
    assert s.output_dir == "multigen_outputs"
    assert s.log_level in ("DEBUG", "INFO", "WARNING", "ERROR")
    assert s.device in ("auto", "cuda", "directml", "cpu")


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("MGOS_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MGOS_DEVICE", "cpu")
    from multigenai.core.config import settings as settings_mod
    import importlib
    importlib.reload(settings_mod)
    from multigenai.core.config.settings import get_settings
    s = get_settings()
    assert s.log_level == "DEBUG"
    assert s.device == "cpu"


def test_settings_missing_yaml(tmp_path):
    """Settings must return defaults when YAML file is missing."""
    from multigenai.core.config.settings import get_settings
    missing = tmp_path / "nonexistent.yaml"
    s = get_settings(config_path=missing)
    assert isinstance(s.output_dir, str)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def test_logger_returns_logger():
    from multigenai.core.logging.logger import get_logger
    log = get_logger("test")
    assert log is not None


def test_logger_correlation_id():
    from multigenai.core.logging.logger import new_correlation_id, get_correlation_id
    cid = new_correlation_id()
    assert get_correlation_id() == cid
    assert len(cid) == 12


# ---------------------------------------------------------------------------
# DeviceManager
# ---------------------------------------------------------------------------

def test_device_manager_returns_valid_device():
    from multigenai.core.device_manager import DeviceManager
    dm = DeviceManager()
    device = dm.get_device()
    assert device in ("cuda", "directml", "cpu")


def test_device_manager_summary_keys():
    from multigenai.core.device_manager import DeviceManager
    dm = DeviceManager()
    summary = dm.summary()
    assert "device" in summary
    assert "torch_available" in summary
    assert "cuda_available" in summary


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

def test_model_registry_singleton():
    from multigenai.core.model_registry import ModelRegistry
    r1 = ModelRegistry.instance()
    r2 = ModelRegistry.instance()
    assert r1 is r2


def test_model_registry_register_and_get():
    from multigenai.core.model_registry import ModelRegistry
    r = ModelRegistry.instance()
    r.register("test_dummy", loader=lambda: {"model": "loaded"}, min_vram_gb=0.0)
    result = r.get("test_dummy")
    assert result == {"model": "loaded"}


def test_model_registry_unload():
    from multigenai.core.model_registry import ModelRegistry
    r = ModelRegistry.instance()
    r.register("test_unload", loader=lambda: "data", min_vram_gb=0.0)
    r.get("test_unload")
    assert r.is_loaded("test_unload")
    r.unload("test_unload")
    assert not r.is_loaded("test_unload")


def test_model_registry_not_found():
    from multigenai.core.model_registry import ModelRegistry
    from multigenai.core.exceptions import ModelNotFoundError
    r = ModelRegistry.instance()
    with pytest.raises(ModelNotFoundError):
        r.get("definitely_does_not_exist_xyz")


# ---------------------------------------------------------------------------
# CapabilityReport
# ---------------------------------------------------------------------------

def test_capability_report_returns_dict():
    from multigenai.core.capability_report import CapabilityReport
    cap = CapabilityReport()
    data = cap.to_dict()
    assert "system" in data
    assert "cuda" in data
    assert "libraries" in data


def test_capability_report_required_keys():
    from multigenai.core.capability_report import CapabilityReport
    data = CapabilityReport().to_dict()
    assert "os" in data["system"]
    assert "python" in data["system"]
    assert "available" in data["cuda"]


def test_capability_report_libraries_present():
    from multigenai.core.capability_report import CapabilityReport
    data = CapabilityReport().to_dict()
    libs = data["libraries"]
    assert "PyTorch" in libs
    assert "Pillow" in libs
    assert "Pydantic" in libs


# ---------------------------------------------------------------------------
# Memory: IdentityStore
# ---------------------------------------------------------------------------

def test_identity_store_add_get_round_trip(tmp_path):
    from multigenai.memory.identity_store import IdentityStore, CharacterProfile
    store = IdentityStore(store_dir=str(tmp_path))
    p = CharacterProfile(character_id="hero", name="Alice", description="Main character")
    store.add(p)
    retrieved = store.get("hero")
    assert retrieved is not None
    assert retrieved.name == "Alice"
    assert retrieved.character_id == "hero"


def test_identity_store_list_all(tmp_path):
    from multigenai.memory.identity_store import IdentityStore, CharacterProfile
    store = IdentityStore(store_dir=str(tmp_path))
    store.add(CharacterProfile(character_id="a", name="A"))
    store.add(CharacterProfile(character_id="b", name="B"))
    ids = store.list_all()
    assert "a" in ids
    assert "b" in ids


def test_identity_store_delete(tmp_path):
    from multigenai.memory.identity_store import IdentityStore, CharacterProfile
    store = IdentityStore(store_dir=str(tmp_path))
    store.add(CharacterProfile(character_id="temp", name="Temp"))
    assert store.delete("temp") is True
    assert store.get("temp") is None


# ---------------------------------------------------------------------------
# Memory: StyleRegistry
# ---------------------------------------------------------------------------

def test_style_registry_builtins(tmp_path):
    from multigenai.memory.style_registry import StyleRegistry
    sr = StyleRegistry(store_dir=str(tmp_path))
    profile = sr.get("cinematic-dark")
    assert profile is not None
    assert profile.style_id == "cinematic-dark"


def test_style_registry_register_get(tmp_path):
    from multigenai.memory.style_registry import StyleRegistry, StyleProfile
    sr = StyleRegistry(store_dir=str(tmp_path))
    sp = StyleProfile(style_id="my-style", name="My Style", atmosphere_tags=["moody"])
    sr.register(sp)
    retrieved = sr.get("my-style")
    assert retrieved is not None
    assert retrieved.name == "My Style"


def test_style_registry_prompt_fragment(tmp_path):
    from multigenai.memory.style_registry import StyleRegistry
    sr = StyleRegistry(store_dir=str(tmp_path))
    profile = sr.get("cinematic-dark")
    fragment = profile.to_prompt_fragment()
    assert isinstance(fragment, str)
    assert len(fragment) > 0


# ---------------------------------------------------------------------------
# Memory: WorldStateEngine
# ---------------------------------------------------------------------------

def test_world_state_update_snapshot(tmp_path):
    from multigenai.memory.world_state import WorldStateEngine, WorldState
    wse = WorldStateEngine(store_dir=str(tmp_path))
    wse.update(WorldState(scene_id="s01", weather="rain", time_of_day="dusk"))
    snap = wse.snapshot()
    assert snap is not None
    assert snap.scene_id == "s01"
    assert snap.weather == "rain"


def test_world_state_reset(tmp_path):
    from multigenai.memory.world_state import WorldStateEngine, WorldState
    wse = WorldStateEngine(store_dir=str(tmp_path))
    wse.update(WorldState(scene_id="s01"))
    wse.reset()
    assert wse.snapshot() is None


# ---------------------------------------------------------------------------
# Memory: EmbeddingStore
# ---------------------------------------------------------------------------

def test_embedding_store_store_retrieve():
    from multigenai.memory.embedding_store import EmbeddingStore
    store = EmbeddingStore()
    vec = [0.1, 0.9, 0.4]
    store.store("key1", vec)
    retrieved = store.retrieve("key1")
    assert retrieved == vec


def test_embedding_store_cosine_similarity():
    from multigenai.memory.embedding_store import EmbeddingStore
    store = EmbeddingStore()
    store.store("identical", [1.0, 0.0, 0.0])
    store.store("orthogonal", [0.0, 1.0, 0.0])
    results = store.similarity_search([1.0, 0.0, 0.0], top_k=2)
    assert results[0][0] == "identical"
    assert abs(results[0][1] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# LLM: Schema validation
# ---------------------------------------------------------------------------

def test_image_request_valid():
    from multigenai.llm.schema_validator import ImageGenerationRequest
    req = ImageGenerationRequest(prompt="a knight at dawn")
    assert req.prompt == "a knight at dawn"
    assert req.width == 768


def test_image_request_invalid_prompt():
    from multigenai.llm.schema_validator import ImageGenerationRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ImageGenerationRequest(prompt="")


def test_video_request_phase5_defaults():
    from multigenai.llm.schema_validator import VideoGenerationRequest
    req = VideoGenerationRequest(prompt="a knight at dawn")
    assert req.temporal_strength == 0.25
    assert req.motion_hint == ""
    assert req.num_inference_steps == 20
    assert req.num_frames == 4
    assert req.width == 640
    assert req.height == 640


def test_video_request_temporal_strength_range():
    from multigenai.llm.schema_validator import VideoGenerationRequest
    from pydantic import ValidationError
    
    # Valid explicit override
    req = VideoGenerationRequest(prompt="test", temporal_strength=0.35)
    assert req.temporal_strength == 0.35
    
    # Above max
    with pytest.raises(ValidationError):
        VideoGenerationRequest(prompt="test", temporal_strength=0.9)
        
    # Below min
    with pytest.raises(ValidationError):
        VideoGenerationRequest(prompt="test", temporal_strength=0.05)


def test_latent_propagator_inject_noise_signature():
    from multigenai.temporal.latent_propagator import LatentPropagator
    lp = LatentPropagator()
    assert hasattr(lp, "inject_noise")
    assert callable(lp.inject_noise)


def test_image_engine_run_from_previous_frame_signature(tmp_path):
    from multigenai.engines.image_engine.engine import ImageEngine
    from multigenai.core.execution_context import ExecutionContext
    from multigenai.core.config.settings import get_settings
    
    s = get_settings()
    s.output_dir = str(tmp_path)
    s.memory.store_dir = str(tmp_path / ".memory")
    ctx = ExecutionContext.build(s)
    
    engine = ImageEngine(ctx)
    assert hasattr(engine, "run_from_previous_frame")
    assert callable(engine.run_from_previous_frame)


def test_prompt_engine_process_image(tmp_path):
    from multigenai.llm.schema_validator import ImageGenerationRequest
    from multigenai.llm.prompt_engine import PromptEngine
    from multigenai.memory.style_registry import StyleRegistry
    sr = StyleRegistry(store_dir=str(tmp_path))
    engine = PromptEngine(style_registry=sr)
    req = ImageGenerationRequest(prompt="a stormy sea", style_id="cinematic-dark")
    enhanced = engine.process_image(req)
    assert "stormy sea" in enhanced.enhanced
    assert len(enhanced.negative) > 10


def test_enhancement_engine_adds_quality_tokens():
    from multigenai.llm.enhancement_engine import EnhancementEngine
    engine = EnhancementEngine()
    result = engine.enhance("a simple cat")
    assert "masterpiece" in result


def test_enhancement_engine_idempotent():
    from multigenai.llm.enhancement_engine import EnhancementEngine
    engine = EnhancementEngine()
    result1 = engine.enhance("a cat, masterpiece, best quality")
    result2 = engine.enhance(result1)
    # Should not double-add quality tokens
    assert result1.count("masterpiece") == 1


def test_scene_planner_splits_scenes():
    from multigenai.llm.scene_planner import ScenePlanner
    planner = ScenePlanner()
    script = "A knight walks into the forest. He discovers a glowing sword. A dragon appears."
    scenes = planner.plan(script)
    assert len(scenes) == 3
    assert scenes[0].scene_id == "s01"


# ---------------------------------------------------------------------------
# DAGEngine
# ---------------------------------------------------------------------------

def test_dag_engine_simple():
    from multigenai.orchestration.dag_engine import DAGEngine
    dag = DAGEngine()
    dag.add_node("a", fn=lambda: 10)
    dag.add_node("b", fn=lambda a: a * 2, deps=["a"])
    results = dag.run()
    assert results["a"] == 10
    assert results["b"] == 20


# ---------------------------------------------------------------------------
# ExecutionContext wiring
# ---------------------------------------------------------------------------

def test_execution_context_builds(tmp_path):
    from multigenai.core.config.settings import get_settings
    from multigenai.core.execution_context import ExecutionContext
    s = get_settings()
    s.output_dir = str(tmp_path)
    s.memory.store_dir = str(tmp_path / ".memory")
    ctx = ExecutionContext.build(s)
    assert ctx.device in ("cuda", "directml", "cpu")
    assert ctx.registry is not None
    assert ctx.identity_store is not None
    assert ctx.style_registry is not None
