"""
ExecutionContext — Dependency container for all MGOS engines and services.

Holds all initialized subsystem references in one object.
Pass this to engines instead of relying on global state.

Benefits:
  - Testability: inject mocks/stubs in tests
  - No global singletons passed implicitly
  - Clear dependency graph
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from multigenai.core.config.settings import Settings
    from multigenai.core.device_manager import DeviceManager
    from multigenai.core.environment import EnvironmentProfile
    from multigenai.core.model_registry import ModelRegistry
    from multigenai.llm.providers.base import LLMProvider
    from multigenai.memory.identity_store import IdentityStore
    from multigenai.memory.world_state import WorldStateEngine
    from multigenai.memory.style_registry import StyleRegistry
    from multigenai.memory.embedding_store import EmbeddingStore


@dataclass
class ExecutionContext:
    """
    Immutable runtime context passed to all engines and handlers.

    Create via `ExecutionContext.build(settings)` after LifecycleManager.startup().

    Attributes:
        settings:         Loaded application settings.
        device:           Active compute device string ("cuda" | "directml" | "cpu").
        device_manager:   DeviceManager instance for VRAM queries.
        registry:         ModelRegistry singleton.
        identity_store:   Character identity memory.
        world_state:      Scene/world state engine.
        style_registry:   Style profile registry.
        embedding_store:  Vector embedding store.
        capability:       Capability report data dict.
        llm_provider:     Optional LLM backend (None → rule-based fallback).
        environment:      Detected EnvironmentProfile (platform, vram, behaviour).
    """
    settings: "Settings"
    device: str
    device_manager: "DeviceManager"
    registry: "ModelRegistry"
    identity_store: "IdentityStore"
    world_state: "WorldStateEngine"
    style_registry: "StyleRegistry"
    embedding_store: "EmbeddingStore"
    capability: dict = field(default_factory=dict)
    llm_provider: Optional["LLMProvider"] = field(default=None)
    environment: Optional["EnvironmentProfile"] = field(default=None)

    # ------------------------------------------------------------------
    # Convenience aliases
    # ------------------------------------------------------------------

    @property
    def llm(self) -> Optional["LLMProvider"]:
        """Alias for llm_provider. Use this in engines and prompt layer."""
        return self.llm_provider

    @property
    def behaviour(self):
        """Capability limits for this run (max_image_resolution, max_video_frames, etc.)."""
        if self.environment is not None:
            return self.environment.behaviour
        from multigenai.core.environment import BehaviourProfile
        return BehaviourProfile()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, settings: "Optional[Settings]" = None) -> "ExecutionContext":
        """
        Construct a fully-wired ExecutionContext from a Settings object.

        If settings is None, get_settings() is called automatically.
        This allows zero-arg usage: ExecutionContext.build()

        LLM provider is instantiated lazily inside a branch — only one
        provider module is ever imported per run. If instantiation fails
        or llm.enabled is False, llm_provider is set to None so callers
        fall back to rule-based logic.

        Args:
            settings: Loaded Settings (from LifecycleManager.startup()).

        Returns:
            Populated ExecutionContext ready to pass to engines.
        """
        from multigenai.core.capability_report import CapabilityReport
        from multigenai.core.device_manager import DeviceManager
        from multigenai.core.environment import EnvironmentDetector, build_behaviour
        from multigenai.core.model_registry import ModelRegistry
        from multigenai.memory.identity_store import IdentityStore
        from multigenai.memory.world_state import WorldStateEngine
        from multigenai.memory.style_registry import StyleRegistry
        from multigenai.memory.embedding_store import EmbeddingStore
        from multigenai.core.logging.logger import get_logger

        LOG = get_logger(__name__)

        # --- Settings fallback (allows zero-arg usage) ---
        if settings is None:
            from multigenai.core.config.settings import get_settings
            settings = get_settings()

        # --- Environment detection ---
        env = EnvironmentDetector().detect()
        if settings.mode == "auto":
            if env.platform == "kaggle":
                settings.mode = "kaggle"
            else:
                settings.mode = "dev"
        behaviour = build_behaviour(settings.mode, env)
        environment = env.model_copy(update={"mode": settings.mode, "behaviour": behaviour})
        LOG.info(
            f"MGOS execution profile | platform={environment.platform} "
            f"device={environment.device_type} vram={environment.vram_mb}MB "
            f"ram={environment.ram_mb}MB mode={settings.mode} "
            f"max_res={behaviour.max_image_resolution} "
            f"auto_unload={behaviour.auto_unload_after_gen}"
        )

        # --- Mode-drift guard ---
        if environment.platform == "kaggle" and settings.mode == "production":
            LOG.warning(
                "⚠ MODE DRIFT DETECTED: settings.mode='production' but platform='kaggle'. "
                "Production memory policies on Kaggle will likely cause OOM. "
                "Set MGOS_MODE=kaggle or mode: auto in config.yaml."
            )

        dm = DeviceManager(preferred=settings.device)
        cap = CapabilityReport().to_dict()

        # Lazy branch import — only the chosen provider module is imported
        llm_provider: Optional["LLMProvider"] = None
        if settings.llm.enabled:
            try:
                if settings.llm.provider == "local":
                    from multigenai.llm.providers.local_provider import LocalLLMProvider
                    llm_provider = LocalLLMProvider(
                        model=settings.llm.model,
                        endpoint=settings.llm.endpoint,
                        timeout_seconds=settings.llm.timeout_seconds,
                    )
                elif settings.llm.provider == "api":
                    from multigenai.llm.providers.api_provider import APILLMProvider
                    llm_provider = APILLMProvider(
                        api_mode=settings.llm.api_mode,
                        model=settings.llm.model,
                        api_key_env=settings.llm.api_key_env,
                        timeout_seconds=settings.llm.timeout_seconds,
                    )
                else:
                    LOG.warning(
                        f"Unknown llm.provider '{settings.llm.provider}' — "
                        "falling back to rule-based."
                    )
            except Exception as exc:
                LOG.warning(
                    f"LLM provider instantiation failed ({exc}) — "
                    "falling back to rule-based."
                )
                llm_provider = None

        return cls(
            settings=settings,
            device=dm.get_device(),
            device_manager=dm,
            registry=ModelRegistry.instance(),
            identity_store=IdentityStore(store_dir=settings.memory.store_dir),
            world_state=WorldStateEngine(store_dir=settings.memory.store_dir),
            style_registry=StyleRegistry(store_dir=settings.memory.store_dir),
            embedding_store=EmbeddingStore(),
            capability=cap,
            llm_provider=llm_provider,
            environment=environment,
        )
