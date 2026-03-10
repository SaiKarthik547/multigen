"""
Tests for the Phase 2 LLM Provider layer.

Covers:
  - LLMProvider ABC cannot be instantiated
  - LocalLLMProvider: ProviderUnavailableError when Ollama offline
  - APILLMProvider: ProviderAuthError when API key env var absent
  - EnhancementEngine: rule-based when provider=None
  - EnhancementEngine: uses provider result when provider mocked
  - EnhancementEngine: post-LLM idempotence guard
  - ScenePlanner: heuristic fallback when provider=None
  - ScenePlanner: uses structured_generate when provider set (mock)
  - ScenePlanner: malformed provider response → fallback (not crash)
  - structured_generate: JSON validation failure → ProviderResponseFormatError
  - ExecutionContext: llm_provider=None when llm.enabled=False
  - Settings: LLMSettings defaults load correctly
  - Settings: MGOS_LLM_* env overrides work symmetrically
"""

from __future__ import annotations

import os
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Settings tests
# ---------------------------------------------------------------------------

def test_llm_settings_defaults():
    """LLMSettings has sane defaults without any env vars."""
    from multigenai.core.config.settings import get_settings
    s = get_settings()
    assert s.llm.enabled is False
    assert s.llm.provider == "local"
    assert s.llm.model == "mistral"
    assert s.llm.timeout_seconds == 30.0
    assert s.mode == "auto"


def test_llm_settings_env_override(monkeypatch):
    """All MGOS_LLM_* env vars override their corresponding settings field."""
    monkeypatch.setenv("MGOS_LLM_ENABLED", "true")
    monkeypatch.setenv("MGOS_LLM_PROVIDER", "api")
    monkeypatch.setenv("MGOS_LLM_API_MODE", "openai")
    monkeypatch.setenv("MGOS_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("MGOS_LLM_TIMEOUT", "60")

    from multigenai.core.config.settings import get_settings
    s = get_settings()

    assert s.llm.enabled is True
    assert s.llm.provider == "api"
    assert s.llm.api_mode == "openai"
    assert s.llm.model == "gpt-4o-mini"
    assert s.llm.timeout_seconds == 60.0


# ---------------------------------------------------------------------------
# LLMProvider ABC
# ---------------------------------------------------------------------------

def test_llm_provider_is_abstract():
    """LLMProvider cannot be instantiated directly."""
    from multigenai.llm.providers.base import LLMProvider
    with pytest.raises(TypeError):
        LLMProvider()


# ---------------------------------------------------------------------------
# LocalLLMProvider
# ---------------------------------------------------------------------------

def test_local_provider_raises_when_ollama_offline():
    """LocalLLMProvider raises ProviderUnavailableError gracefully when Ollama unreachable."""
    from multigenai.llm.providers.local_provider import LocalLLMProvider
    from multigenai.core.exceptions import ProviderUnavailableError

    provider = LocalLLMProvider(
        endpoint="http://localhost:19999/nonexistent",  # guaranteed offline
        timeout_seconds=1.0,
    )

    with pytest.raises(ProviderUnavailableError):
        provider.generate("test prompt")


def test_local_provider_raises_on_missing_requests(monkeypatch):
    """LocalLLMProvider raises ProviderUnavailableError if requests not installed."""
    from multigenai.core.exceptions import ProviderUnavailableError
    from multigenai.llm.providers import local_provider as lp_module

    # Simulate requests not being installed
    original = lp_module.__builtins__
    with patch.dict("sys.modules", {"requests": None}):
        provider = lp_module.LocalLLMProvider()
        with pytest.raises((ProviderUnavailableError, ImportError)):
            provider.generate("test")


# ---------------------------------------------------------------------------
# APILLMProvider
# ---------------------------------------------------------------------------

def test_api_provider_raises_auth_when_no_key(monkeypatch):
    """APILLMProvider raises ProviderAuthError when API key env var is absent."""
    from multigenai.llm.providers.api_provider import APILLMProvider
    from multigenai.core.exceptions import ProviderAuthError

    # Make sure the key env var is not set
    monkeypatch.delenv("MGOS_LLM_API_KEY", raising=False)

    provider = APILLMProvider(api_mode="gemini", api_key_env="MGOS_LLM_API_KEY")
    with pytest.raises(ProviderAuthError):
        provider.generate("test prompt")


def test_api_provider_raises_auth_when_no_key_openai(monkeypatch):
    """APILLMProvider raises ProviderAuthError for OpenAI when key is absent."""
    from multigenai.llm.providers.api_provider import APILLMProvider
    from multigenai.core.exceptions import ProviderAuthError

    monkeypatch.delenv("MGOS_LLM_API_KEY", raising=False)
    provider = APILLMProvider(api_mode="openai", api_key_env="MGOS_LLM_API_KEY")
    with pytest.raises(ProviderAuthError):
        provider.generate("test prompt")


def test_api_provider_invalid_mode():
    """APILLMProvider raises ValueError for unknown api_mode."""
    from multigenai.llm.providers.api_provider import APILLMProvider
    with pytest.raises(ValueError, match="Unknown api_mode"):
        APILLMProvider(api_mode="unknown_vendor")


# ---------------------------------------------------------------------------
# EnhancementEngine
# ---------------------------------------------------------------------------

def test_enhancement_engine_rule_based_no_provider():
    """Without a provider, EnhancementEngine uses rule-based path."""
    from multigenai.llm.enhancement_engine import EnhancementEngine
    engine = EnhancementEngine(provider=None)
    result = engine.enhance("a stormy ocean")
    assert "masterpiece" in result
    assert "stormy ocean" in result


def test_enhancement_engine_idempotent_rule_based():
    """Rule-based path: calling enhance() twice doesn't double-add tokens."""
    from multigenai.llm.enhancement_engine import EnhancementEngine
    engine = EnhancementEngine()
    first = engine.enhance("a dragon")
    second = engine.enhance(first)
    assert second.count("masterpiece") == first.count("masterpiece")


def test_enhancement_engine_uses_provider():
    """With a mock provider, EnhancementEngine uses its output."""
    from multigenai.llm.enhancement_engine import EnhancementEngine
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "a majestic dragon soaring over mountains"

    engine = EnhancementEngine(provider=mock_provider)
    result = engine.enhance("a dragon")

    mock_provider.generate.assert_called_once()
    assert "majestic dragon" in result
    assert "masterpiece" in result  # quality tokens still appended


def test_enhancement_engine_post_llm_idempotence():
    """Post-LLM idempotence guard: if LLM already added 'masterpiece', don't re-add."""
    from multigenai.llm.enhancement_engine import EnhancementEngine
    mock_provider = MagicMock()
    # LLM returns a result that already has quality markers
    mock_provider.generate.return_value = (
        "a dragon, masterpiece, best quality, ultra-detailed"
    )

    engine = EnhancementEngine(provider=mock_provider)
    result = engine.enhance("a dragon")

    # Should NOT have duplicated quality tokens
    assert result.count("masterpiece") == 1


def test_enhancement_engine_fallback_on_provider_error():
    """EnhancementEngine falls back to rule-based if provider raises."""
    from multigenai.llm.enhancement_engine import EnhancementEngine
    from multigenai.core.exceptions import ProviderUnavailableError

    mock_provider = MagicMock()
    mock_provider.generate.side_effect = ProviderUnavailableError("offline")

    engine = EnhancementEngine(provider=mock_provider)
    result = engine.enhance("a castle")

    # Fallback must still produce an enhanced result
    assert "masterpiece" in result
    assert "castle" in result


# ---------------------------------------------------------------------------
# ScenePlanner
# ---------------------------------------------------------------------------

def test_scene_planner_heuristic_no_provider():
    """ScenePlanner with no provider uses heuristic sentence split."""
    from multigenai.llm.scene_planner import ScenePlanner
    planner = ScenePlanner()
    script = "A knight rides through a forest. He finds a glowing sword. Dawn breaks."
    plan = planner.plan(script)
    assert len(plan.scenes) == 3
    assert plan.scenes[0].scene_id == "s01"
    assert plan.scenes[2].time_of_day == "dawn"


def test_scene_planner_uses_structured_generate():
    """ScenePlanner routes to plan_with_llm() when provider set (mock)."""
    from multigenai.llm.scene_planner import ScenePlanner, _SceneListResponse, _SceneItem

    mock_provider = MagicMock()
    mock_scene = _SceneItem(
        title="Forest Ride",
        description="A knight rides through a dark forest at night.",
        time_of_day="night",
        location="forest",
        characters=["knight"],
        duration_hint=4.0,
    )
    mock_provider.structured_generate.return_value = _SceneListResponse(
        scenes=[mock_scene]
    )

    planner = ScenePlanner(provider=mock_provider)
    plan = planner.plan("A knight rides through a dark forest.")

    mock_provider.structured_generate.assert_called_once()
    assert len(plan.scenes) == 1
    assert plan.scenes[0].time_of_day == "night"
    assert "knight" in plan.scenes[0].characters


def test_scene_planner_malformed_response_falls_back():
    """ScenePlanner falls back to heuristic if structured_generate raises."""
    from multigenai.llm.scene_planner import ScenePlanner
    from multigenai.core.exceptions import ProviderResponseFormatError

    mock_provider = MagicMock()
    mock_provider.structured_generate.side_effect = ProviderResponseFormatError(
        "bad json"
    )

    planner = ScenePlanner(provider=mock_provider)
    script = "A knight rides. He fights a dragon."
    plan = planner.plan(script)

    # Must fall back gracefully — 2 heuristic scenes
    assert len(plan.scenes) == 2
    assert "knight" in plan.scenes[0].description.lower()


# ---------------------------------------------------------------------------
# structured_generate — JSON failure paths
# ---------------------------------------------------------------------------

def test_structured_generate_raises_format_error_on_invalid_json():
    """structured_generate raises ProviderResponseFormatError when LLM returns garbage."""
    from multigenai.llm.providers.local_provider import LocalLLMProvider
    from multigenai.core.exceptions import ProviderResponseFormatError
    from pydantic import BaseModel

    class MySchema(BaseModel):
        value: str

    provider = LocalLLMProvider.__new__(LocalLLMProvider)
    provider._model = "test"
    provider._endpoint = "http://x"
    provider._timeout = 1.0

    # Monkey-patch generate() to always return invalid JSON
    provider.generate = MagicMock(return_value="this is not json at all!!!")

    with pytest.raises(ProviderResponseFormatError):
        provider.structured_generate("generate something", MySchema)


# ---------------------------------------------------------------------------
# ExecutionContext
# ---------------------------------------------------------------------------

def test_execution_context_llm_provider_none_when_disabled():
    """ExecutionContext.build() sets llm_provider=None when llm.enabled=False."""
    from multigenai.core.config.settings import get_settings
    from multigenai.core.execution_context import ExecutionContext

    settings = get_settings()
    assert settings.llm.enabled is False  # default

    ctx = ExecutionContext.build(settings)
    assert ctx.llm_provider is None
    assert ctx.llm is None  # property alias
