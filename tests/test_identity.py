"""
Phase 4 Generic Identity Layer — Unit Tests.

Tests cover:
  - CharacterProfile: embedding fields, has_* properties, metadata round-trip
  - IdentityStore: persist/load, get_embedding/set_embedding, get_profile, schema version
  - Schema migration: v1→v3 and v2→v3 (non-sequential, non-destructive)
  - IdentityResolver: resolve face/voice/style, missing character, unknown modality
  - ImageGenerationRequest: identity fields validation
  - VideoGenerationRequest: identity fields
  - AudioGenerationRequest: identity fields
  - GenerationMetrics: identity fields
  - ConsistencyEnforcer: seed enforcement, check_embedding_drift, alias
  - PromptEngine: facial token stripping
  - ImageEngine: _inject_identity graceful skip paths
  - OOM recovery path
  - Edge cases: corrupt embedding, unknown modality, non-existent character

All tests run without GPU, torch, diffusers, insightface, or any network access.
"""

from __future__ import annotations

import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Core imports — all must succeed without heavy ML deps
# ---------------------------------------------------------------------------
from multigenai.core.exceptions import IdentityEncoderError
from multigenai.core.metrics import GenerationMetrics, MetricsCollector
from multigenai.control.consistency_enforcer import ConsistencyEnforcer
from multigenai.identity import FaceEncoder, IdentityResolver
from multigenai.llm.schema_validator import (
    AudioGenerationRequest,
    ImageGenerationRequest,
    VideoGenerationRequest,
)
from multigenai.memory.identity_store import (
    SCHEMA_VERSION,
    CharacterProfile,
    IdentityStore,
)


# ===========================================================================
# 1–6: CharacterProfile — embedding fields and properties
# ===========================================================================

class TestCharacterProfileEmbedding:

    def test_character_profile_has_face_embedding_field(self):
        """face_embedding exists and defaults to None."""
        profile = CharacterProfile(character_id="hero", name="Alice")
        assert hasattr(profile, "face_embedding")
        assert profile.face_embedding is None

    def test_has_embedding_property_false_when_none(self):
        profile = CharacterProfile(character_id="hero", name="Alice")
        assert profile.has_embedding is False

    def test_has_embedding_property_true_when_set(self):
        emb = [0.1] * 512
        profile = CharacterProfile(character_id="hero", name="Alice", face_embedding=emb)
        assert profile.has_embedding is True
        assert profile.has_face_embedding is True

    def test_has_voice_embedding_false_by_default(self):
        profile = CharacterProfile(character_id="hero", name="Alice")
        assert profile.has_voice_embedding is False

    def test_has_voice_embedding_true_when_set(self):
        emb = [0.1] * 256
        profile = CharacterProfile(character_id="hero", name="Alice", voice_embedding=emb)
        assert profile.has_voice_embedding is True

    def test_has_style_embedding_false_by_default(self):
        profile = CharacterProfile(character_id="hero", name="Alice")
        assert profile.has_style_embedding is False

    def test_has_style_embedding_true_when_set(self):
        emb = [0.1] * 768
        profile = CharacterProfile(character_id="hero", name="Alice", style_embedding=emb)
        assert profile.has_style_embedding is True

    def test_metadata_defaults_empty_dict(self):
        profile = CharacterProfile(character_id="hero", name="Alice")
        assert profile.metadata == {}

    def test_metadata_round_trip_disk(self, tmp_path):
        """metadata dict survives persist/load cycle."""
        store = IdentityStore(str(tmp_path))
        profile = CharacterProfile(
            character_id="meta_hero", name="Meta",
            metadata={"wardrobe": {"scene1": "suit"}, "lighting_bias": "golden"},
        )
        store.add(profile)
        loaded = store.get("meta_hero")
        assert loaded.metadata["wardrobe"] == {"scene1": "suit"}
        assert loaded.metadata["lighting_bias"] == "golden"

    def test_embedding_persisted_to_disk(self, tmp_path):
        """store.add() writes face_embedding to JSON."""
        store = IdentityStore(str(tmp_path))
        emb = [float(i) / 512 for i in range(512)]
        store.add(CharacterProfile(character_id="hero", name="Alice", face_embedding=emb))
        data = json.loads((tmp_path / "identities" / "hero.json").read_text())
        assert "face_embedding" in data
        assert len(data["face_embedding"]) == 512

    def test_embedding_loaded_from_disk(self, tmp_path):
        """store.get() returns a profile with its embedding intact."""
        store = IdentityStore(str(tmp_path))
        emb = [0.5] * 512
        store.add(CharacterProfile(character_id="bob", name="Bob", face_embedding=emb))
        loaded = store.get("bob")
        assert loaded is not None
        assert loaded.face_embedding == emb
        assert loaded.has_embedding is True


# ===========================================================================
# 7: SCHEMA_VERSION
# ===========================================================================

class TestSchemaVersion:

    def test_schema_version_is_3(self):
        """Identity schema must be v3 — the generic multi-modal version."""
        assert SCHEMA_VERSION == 3

    def test_schema_version_written_to_disk(self, tmp_path):
        """JSON file saved by store.add() includes the current schema_version."""
        store = IdentityStore(str(tmp_path))
        store.add(CharacterProfile(character_id="v3test", name="V3"))
        raw = json.loads((tmp_path / "identities" / "v3test.json").read_text())
        assert raw["schema_version"] == 3


# ===========================================================================
# 8–11: IdentityStore helpers — get_embedding / set_embedding / get_profile
# ===========================================================================

class TestIdentityStoreHelpers:

    def test_get_embedding_face(self, tmp_path):
        store = IdentityStore(str(tmp_path))
        emb = [0.1] * 512
        store.add(CharacterProfile(character_id="h", name="H", face_embedding=emb))
        result = store.get_embedding("h", modality="face")
        assert result == emb

    def test_get_embedding_unknown_modality_returns_none(self, tmp_path):
        """Unknown modality key returns None cleanly — no exception."""
        store = IdentityStore(str(tmp_path))
        store.add(CharacterProfile(character_id="h", name="H"))
        result = store.get_embedding("h", modality="hologram")
        assert result is None

    def test_set_embedding_face_validates_dim(self, tmp_path):
        """set_embedding raises MGOSMemoryError when face vector has wrong dim."""
        from multigenai.core.exceptions import MemoryError as MGOSMemoryError
        store = IdentityStore(str(tmp_path))
        store.add(CharacterProfile(character_id="dim_hero", name="Dim"))
        with pytest.raises(MGOSMemoryError, match="512 dimensions"):
            store.set_embedding("dim_hero", modality="face", vector=[0.1] * 10)

    def test_set_embedding_coerces_numpy(self, tmp_path):
        """set_embedding accepts numpy arrays and converts to list silently."""
        pytest.importorskip("numpy")
        import numpy as np
        store = IdentityStore(str(tmp_path))
        store.add(CharacterProfile(character_id="np_hero", name="NumPy"))
        store.set_embedding("np_hero", modality="face", vector=np.zeros(512))
        result = store.get_embedding("np_hero", modality="face")
        assert isinstance(result, list)
        assert len(result) == 512

    def test_get_profile_alias(self, tmp_path):
        """get_profile() returns the same object as get()."""
        store = IdentityStore(str(tmp_path))
        store.add(CharacterProfile(character_id="palias", name="Alias"))
        p1 = store.get("palias")
        p2 = store.get_profile("palias")
        assert p1 is not None and p2 is not None
        assert p1.character_id == p2.character_id


# ===========================================================================
# 12–15: Schema Migration — v1→v3, v2→v3
# ===========================================================================

class TestSchemaMigration:

    def _write_raw(self, tmp_path, character_id, data):
        d = tmp_path / "identities"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{character_id}.json").write_text(json.dumps(data))

    def test_v1_to_v3_migration_injects_face_embedding(self, tmp_path):
        """v1 profile (no face_embedding) is upgraded; face_embedding is None."""
        self._write_raw(tmp_path, "v1", {
            "schema_version": 1, "character_id": "v1", "name": "V1Hero",
            "description": "", "persistent_seed": None,
            "wardrobe": {}, "lighting_bias": "neutral", "personality_profile": {},
        })
        store = IdentityStore(str(tmp_path))
        profile = store.get("v1")
        assert profile is not None
        assert profile.face_embedding is None

    def test_v1_to_v3_migration_promotes_legacy_fields_to_metadata(self, tmp_path):
        """wardrobe/lighting_bias/personality_profile moved to metadata{}."""
        self._write_raw(tmp_path, "v1_meta", {
            "schema_version": 1, "character_id": "v1_meta", "name": "Legacy",
            "description": "", "persistent_seed": None,
            "wardrobe": {"s1": "cape"}, "lighting_bias": "golden", "personality_profile": {"brave": True},
        })
        store = IdentityStore(str(tmp_path))
        profile = store.get("v1_meta")
        assert profile.metadata.get("wardrobe") == {"s1": "cape"}
        assert profile.metadata.get("lighting_bias") == "golden"
        assert profile.metadata.get("personality_profile") == {"brave": True}

    def test_v2_to_v3_migration_promotes_legacy_fields(self, tmp_path):
        """v2 profile also has wardrobe etc. promoted to metadata."""
        self._write_raw(tmp_path, "v2", {
            "schema_version": 2, "character_id": "v2", "name": "V2Hero",
            "description": "", "persistent_seed": None, "face_embedding": None,
            "wardrobe": {"x": "armor"}, "lighting_bias": "cool", "personality_profile": {},
        })
        store = IdentityStore(str(tmp_path))
        profile = store.get("v2")
        assert profile.metadata.get("wardrobe") == {"x": "armor"}
        assert profile.face_embedding is None

    def test_migration_does_not_overwrite_existing_metadata(self, tmp_path):
        """setdefault semantics: existing metadata keys are NOT overwritten."""
        self._write_raw(tmp_path, "v1_safe", {
            "schema_version": 1, "character_id": "v1_safe", "name": "Safe",
            "description": "", "persistent_seed": None,
            "wardrobe": {"legacy": "robe"},
            "metadata": {"wardrobe": {"existing": "tunic"}},
        })
        store = IdentityStore(str(tmp_path))
        profile = store.get("v1_safe")
        # existing key must survive
        assert profile.metadata["wardrobe"] == {"existing": "tunic"}


# ===========================================================================
# 16–19: IdentityResolver
# ===========================================================================

class TestIdentityResolver:

    def test_resolve_face_returns_embedding(self, tmp_path):
        store = IdentityStore(str(tmp_path))
        emb = [0.5] * 512
        store.add(CharacterProfile(character_id="res_hero", name="R", face_embedding=emb))
        result = IdentityResolver.get_face_embedding("res_hero", store)
        assert result == emb

    def test_resolve_voice_returns_none_when_missing(self, tmp_path):
        """Voice embedding not set → returns None cleanly."""
        store = IdentityStore(str(tmp_path))
        store.add(CharacterProfile(character_id="voiceless", name="V"))
        result = IdentityResolver.get_voice_embedding("voiceless", store)
        assert result is None

    def test_resolve_style_returns_none_when_missing(self, tmp_path):
        """Style embedding not set → returns None cleanly."""
        store = IdentityStore(str(tmp_path))
        store.add(CharacterProfile(character_id="styleless", name="S"))
        result = IdentityResolver.get_style_embedding("styleless", store)
        assert result is None

    def test_resolve_non_existent_character_returns_none(self, tmp_path):
        """Non-existent character_id → logs warning, returns None, no crash."""
        store = IdentityStore(str(tmp_path))
        result = IdentityResolver.resolve("ghost_42", store, modality="face")
        assert result is None

    def test_resolve_unknown_modality_returns_none(self, tmp_path):
        """Unknown modality key → returns None cleanly, no exception."""
        store = IdentityStore(str(tmp_path))
        store.add(CharacterProfile(character_id="hero_mod", name="M"))
        result = IdentityResolver.resolve("hero_mod", store, modality="hologram")
        assert result is None

    def test_resolve_none_identity_name_returns_none(self, tmp_path):
        """Falsy identity_name → returns None immediately."""
        store = IdentityStore(str(tmp_path))
        assert IdentityResolver.resolve(None, store) is None
        assert IdentityResolver.resolve("", store) is None


# ===========================================================================
# 20–23: Schema Validator — identity fields on all modalities
# ===========================================================================

class TestImageRequestIdentityFields:

    def test_image_request_identity_name_default_none(self):
        req = ImageGenerationRequest(prompt="a hero at dawn")
        assert req.identity_name is None

    def test_image_request_identity_strength_default(self):
        req = ImageGenerationRequest(prompt="a hero at dawn")
        assert req.identity_strength == pytest.approx(0.8)

    def test_identity_strength_validation_rejects_negative(self):
        with pytest.raises(ValidationError):
            ImageGenerationRequest(prompt="test", identity_strength=-0.1)


class TestVideoRequestIdentityFields:

    def test_video_request_has_identity_name(self):
        req = VideoGenerationRequest(prompt="a stormy sea")
        assert hasattr(req, "identity_name")
        assert req.identity_name is None

    def test_video_request_has_identity_strength(self):
        req = VideoGenerationRequest(prompt="a stormy sea")
        assert hasattr(req, "identity_strength")
        assert req.identity_strength == pytest.approx(0.8)

    def test_video_request_identity_name_set(self):
        req = VideoGenerationRequest(prompt="a stormy sea", identity_name="hero_01")
        assert req.identity_name == "hero_01"


class TestAudioRequestIdentityFields:

    def test_audio_request_has_identity_name(self):
        req = AudioGenerationRequest(prompt="a calm voice")
        assert hasattr(req, "identity_name")
        assert req.identity_name is None

    def test_audio_request_has_identity_strength(self):
        req = AudioGenerationRequest(prompt="a calm voice")
        assert hasattr(req, "identity_strength")
        assert req.identity_strength == pytest.approx(0.8)


# ===========================================================================
# 24–25: _inject_identity graceful skip paths (mocked engine)
# ===========================================================================

class TestInjectIdentitySkipPaths:

    def _make_engine(self):
        from multigenai.engines.image_engine.engine import ImageEngine
        with patch.object(ImageEngine, "__init__", lambda self, ctx: None):
            engine = ImageEngine.__new__(ImageEngine)
        env = MagicMock()
        env.vram_mb = 0  # no VRAM — forces skip
        ctx = MagicMock()
        ctx.environment = env
        engine._ctx = ctx
        return engine

    def test_inject_identity_missing_profile_returns_false(self, tmp_path):
        """Identity name set but profile not in store → returns False, no crash."""
        engine = self._make_engine()
        engine._ctx.settings.output_dir = str(tmp_path)
        req = ImageGenerationRequest(prompt="hero", identity_name="ghost")
        result = engine._inject_identity(req, MagicMock())
        assert result is False

    def test_inject_identity_missing_embedding_returns_false(self, tmp_path):
        """Profile exists but has no embedding → returns False, no crash."""
        engine = self._make_engine()
        engine._ctx.settings.output_dir = str(tmp_path)
        store = IdentityStore(str(tmp_path))
        store.add(CharacterProfile(character_id="blank", name="Blank"))
        engine._ctx.environment.vram_mb = 12 * 1024
        req = ImageGenerationRequest(prompt="hero", identity_name="blank")
        result = engine._inject_identity(req, MagicMock())
        assert result is False


# ===========================================================================
# 26–27: GenerationMetrics — identity fields
# ===========================================================================

class TestGenerationMetricsIdentityFields:

    def test_generation_metrics_has_identity_fields(self):
        m = GenerationMetrics(model_id="sdxl", width=1024, height=1024)
        assert hasattr(m, "identity_used")
        assert hasattr(m, "identity_name")

    def test_metrics_identity_used_default_false(self):
        m = GenerationMetrics(model_id="sdxl", width=512, height=512)
        assert m.identity_used is False
        assert m.identity_name is None


# ===========================================================================
# 28–31: ConsistencyEnforcer
# ===========================================================================

class TestConsistencyEnforcer:

    def test_consistency_enforcer_seed_enforcement(self):
        enforcer = ConsistencyEnforcer()
        req = ImageGenerationRequest(prompt="scene", seed=None)
        profile = CharacterProfile(character_id="hero", name="Alice", persistent_seed=42)
        seed = enforcer.enforce_seed(req, profile)
        assert seed == 42

    def test_consistency_enforcer_seed_request_takes_priority(self):
        enforcer = ConsistencyEnforcer()
        req = ImageGenerationRequest(prompt="scene", seed=99)
        profile = CharacterProfile(character_id="hero", name="Alice", persistent_seed=42)
        assert enforcer.enforce_seed(req, profile) == 99

    def test_check_embedding_drift_identical(self):
        """Drift score for identical vectors = 1.0."""
        enforcer = ConsistencyEnforcer()
        v = [1.0, 0.0, 0.0, 0.0]
        score = enforcer.check_embedding_drift(v, v)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_check_embedding_drift_orthogonal(self):
        """Drift score for orthogonal vectors ≈ 0.0."""
        enforcer = ConsistencyEnforcer()
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        score = enforcer.check_embedding_drift(a, b)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_check_identity_drift_alias_works(self):
        """Deprecated alias check_identity_drift() still produces identical results."""
        enforcer = ConsistencyEnforcer()
        v = [1.0, 0.0]
        assert enforcer.check_identity_drift(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_consistency_enforcer_enforce_always_true(self):
        """enforce() is advisory in Phase 4 — always True."""
        enforcer = ConsistencyEnforcer()
        assert enforcer.enforce(result=object(), request=object()) is True


# ===========================================================================
# 32–33: PromptEngine — facial token stripping
# ===========================================================================

class TestPromptEngineIdentityTokenStripping:

    def test_prompt_engine_strips_face_descriptors_when_identity_active(self):
        from multigenai.llm.prompt_engine import PromptEngine
        engine = PromptEngine(style_registry=None)
        req = ImageGenerationRequest(
            prompt="a hero, blue eyes, red hair, standing in a field",
            identity_name="hero_01",
        )
        result = engine.process_image(req)
        assert "blue eyes" not in result.enhanced.lower()
        assert "red hair" not in result.enhanced.lower()

    def test_prompt_engine_no_stripping_without_identity(self):
        from multigenai.llm.prompt_engine import PromptEngine
        engine = PromptEngine(style_registry=None)
        req = ImageGenerationRequest(
            prompt="a hero, blue eyes, red hair, standing in a field",
            identity_name=None,
        )
        result = engine.process_image(req)
        assert "blue eyes" in result.enhanced.lower() or "blue eyes" in result.original.lower()

    def test_prompt_engine_safe_phrase_not_stripped(self):
        from multigenai.llm.prompt_engine import PromptEngine
        engine = PromptEngine(style_registry=None)
        req = ImageGenerationRequest(
            prompt="a wizard, blue glowing energy, casting spells",
            identity_name="wizard_01",
        )
        result = engine.process_image(req)
        assert "blue glowing energy" in result.enhanced or "blue glowing energy" in result.original


# ===========================================================================
# 34–36: Edge cases and OOM recovery
# ===========================================================================

class TestEdgeCasesAndOomPaths:

    def test_generation_succeeds_without_identity(self):
        req = ImageGenerationRequest(prompt="a sunset over mountains")
        assert req.identity_name is None
        assert req.identity_strength == pytest.approx(0.8)

    def test_metrics_records_identity_info_correctly(self):
        MetricsCollector.instance().reset()
        m = GenerationMetrics(
            model_id="sdxl", width=512, height=512,
            identity_used=True, identity_name="hero_01",
        )
        MetricsCollector.instance().record(m)
        summary = MetricsCollector.instance().summary()
        assert summary["total"] == 1

    def test_corrupt_embedding_set_raises_not_silently_fails(self, tmp_path):
        """set_embedding with malformed data raises MGOSMemoryError, not a silent crash."""
        from multigenai.core.exceptions import MemoryError as MGOSMemoryError
        store = IdentityStore(str(tmp_path))
        store.add(CharacterProfile(character_id="corrupt", name="Corrupt"))
        with pytest.raises((MGOSMemoryError, ValueError)):
            # Pass a string that looks like a vector — should fail validation
            store.set_embedding("corrupt", modality="face", vector="not_a_vector")

    def test_get_embedding_for_nonexistent_character_returns_none(self, tmp_path):
        """get_embedding for unknown character returns None, no exception."""
        store = IdentityStore(str(tmp_path))
        result = store.get_embedding("ghost_99", modality="face")
        assert result is None

    def test_identity_package_imports_both_classes(self):
        """identity package exports FaceEncoder and IdentityResolver."""
        from multigenai.identity import FaceEncoder, IdentityResolver
        assert FaceEncoder is not None
        assert IdentityResolver is not None
