"""
Settings loader for MultiGenAI OS.

Priority (highest → lowest):
  1. Environment variables  (MGOS_<KEY>=value)
  2. config.yaml in the package config/ directory
  3. Built-in defaults

LLM env overrides follow the same pattern, all prefixed MGOS_LLM_:
  MGOS_LLM_ENABLED=true
  MGOS_LLM_PROVIDER=api
  MGOS_LLM_API_MODE=gemini
  MGOS_LLM_MODEL=gemini-1.5-flash
  MGOS_LLM_ENDPOINT=https://...
  MGOS_LLM_API_KEY_ENV=MY_KEY_VAR
  MGOS_LLM_TIMEOUT=60
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

_DEFAULT_CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"


# ---------------------------------------------------------------------------
# Nested settings dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelRegistrySettings:
    lazy_load: bool = True
    cache_dir: str = "~/.cache/mgos"


@dataclass
class MemorySettings:
    backend: str = "json"
    store_dir: str = "multigen_outputs/.memory"


@dataclass
class OrchestrationSettings:
    max_workers: int = 1
    job_timeout: int = 1800


@dataclass
class SDXLSettings:
    """
    Quality-tuning parameters for the SDXL two-stage pipeline.

    All fields are overridable by MGOS_SDXL_<FIELD> environment variables.

    Attributes:
        use_refiner:              Enable the refiner second-stage pass.
        base_denoising_end:       Fraction of denoising steps for the base model.
        refiner_denoising_start:  Fraction where the refiner takes over.
        vae_float32:              Cast VAE to float32 for improved texture detail.
        num_inference_steps:      Default inference step count (base + refiner total).
        guidance_scale:           Classifier-free guidance scale.
    """
    use_refiner: bool = True
    base_denoising_end: float = 0.8
    refiner_denoising_start: float = 0.8
    vae_float32: bool = False          # False = pure fp16; no dtype mismatch with refiner
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    # Kaggle-safe default resolution (768 drops peak VRAM ~30% vs 1024)
    default_width: int = 768
    default_height: int = 768


@dataclass
class LLMSettings:
    """
    Configuration for the LLM intelligence layer.

    All fields are overridable by MGOS_LLM_<FIELD> environment variables.

    Attributes:
        enabled:       Master switch — if False, rule-based fallback is used.
        provider:      "local" (Ollama) | "api" (Gemini/OpenAI)
        api_mode:      "gemini" | "openai" — only relevant when provider="api"
        model:         Ollama model name or API model ID
        endpoint:      Full URL for the provider's generate endpoint
        api_key_env:   Name of the env var that holds the actual API key
        timeout_seconds: Per-request timeout
    """
    enabled: bool = False
    provider: str = "local"
    api_mode: str = "gemini"
    model: str = "mistral"
    endpoint: str = "http://localhost:11434/api/generate"
    api_key_env: str = "MGOS_LLM_API_KEY"
    timeout_seconds: float = 30.0


@dataclass
class PromptSettings:
    """
    Configuration for the Phase 9 Advanced Prompt Processing Engine.

    All fields overridable via MGOS_PROMPT_<FIELD> environment variables.

    Attributes:
        max_tokens:        CLIP token hard limit (75 = BOS/EOS-safe headroom from 77).
        negative_reserve:  Tokens reserved for negative prompt per segment.
        segmentation_mode: "semantic" (default) | "sentence" | "word"
        expansion:         Enrich sparse segments with context tokens (True default).
    """
    max_tokens: int = 75
    negative_reserve: int = 25
    segmentation_mode: str = "semantic"    # semantic | sentence | word
    expansion: bool = True

    @property
    def positive_budget(self) -> int:
        """Derived: tokens available for positive prompt per segment."""
        return self.max_tokens - self.negative_reserve


# ---------------------------------------------------------------------------
# Top-level Settings
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    mode: str = "dev"              # dev | production | kaggle
    output_dir: str = "multigen_outputs"
    log_level: str = "INFO"
    log_mode: str = "pretty"      # "pretty" | "json"
    log_file: Optional[str] = None
    device: str = "auto"
    performance_mode: str = "balanced"  # max-speed | balanced | max-quality
    model_registry: ModelRegistrySettings = field(default_factory=ModelRegistrySettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    orchestration: OrchestrationSettings = field(default_factory=OrchestrationSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    sdxl: SDXLSettings = field(default_factory=SDXLSettings)
    prompt: PromptSettings = field(default_factory=PromptSettings)


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: pathlib.Path) -> dict:
    """Load a YAML file; return empty dict if unavailable."""
    try:
        import yaml  # lazy — optional at core level
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _env(key: str, default: str) -> str:
    """Read MGOS_<KEY> environment variable, falling back to default."""
    return os.environ.get(f"MGOS_{key.upper()}", default)


def _env_bool(key: str, default: bool) -> bool:
    """Read a boolean env var (MGOS_<KEY>=true|false|1|0)."""
    raw = os.environ.get(f"MGOS_{key.upper()}")
    if raw is None:
        return default
    return raw.strip().lower() in ("true", "1", "yes")


def _env_float(key: str, default: float) -> float:
    """Read a float env var."""
    raw = os.environ.get(f"MGOS_{key.upper()}")
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_settings(config_path: Optional[pathlib.Path] = None) -> Settings:
    """
    Build and return a Settings instance.

    Args:
        config_path: Override path to config.yaml. Defaults to the
                     package-bundled config/config.yaml.

    Returns:
        Populated Settings dataclass (env vars take highest priority).
    """
    raw = _load_yaml(config_path or _DEFAULT_CONFIG_PATH)

    mr_raw = raw.get("model_registry", {})
    mem_raw = raw.get("memory", {})
    orch_raw = raw.get("orchestration", {})
    llm_raw = raw.get("llm", {})
    sdxl_raw = raw.get("sdxl", {})
    prompt_raw = raw.get("prompt", {})

    return Settings(
        mode=_env("mode", raw.get("mode", "dev")),
        output_dir=_env("output_dir", raw.get("output_dir", "multigen_outputs")),
        log_level=_env("log_level", raw.get("log_level", "INFO")),
        log_mode=_env("log_mode", raw.get("log_mode", "pretty")),
        log_file=_env("log_file", raw.get("log_file") or "") or None,
        device=_env("device", raw.get("device", "auto")),
        performance_mode=_env("performance_mode", raw.get("performance_mode", "balanced")),
        model_registry=ModelRegistrySettings(
            lazy_load=bool(mr_raw.get("lazy_load", True)),
            cache_dir=str(mr_raw.get("cache_dir", "~/.cache/mgos")),
        ),
        memory=MemorySettings(
            backend=str(mem_raw.get("backend", "json")),
            store_dir=str(mem_raw.get("store_dir", "multigen_outputs/.memory")),
        ),
        orchestration=OrchestrationSettings(
            max_workers=int(orch_raw.get("max_workers", 1)),
            job_timeout=int(orch_raw.get("job_timeout", 1800)),
        ),
        llm=LLMSettings(
            # Symmetric env overrides: MGOS_LLM_<FIELD>
            enabled=_env_bool(
                "llm_enabled", bool(llm_raw.get("enabled", False))
            ),
            provider=_env(
                "llm_provider", llm_raw.get("provider", "local")
            ),
            api_mode=_env(
                "llm_api_mode", llm_raw.get("api_mode", "gemini")
            ),
            model=_env(
                "llm_model", llm_raw.get("model", "mistral")
            ),
            endpoint=_env(
                "llm_endpoint",
                llm_raw.get("endpoint", "http://localhost:11434/api/generate"),
            ),
            api_key_env=_env(
                "llm_api_key_env", llm_raw.get("api_key_env", "MGOS_LLM_API_KEY")
            ),
            timeout_seconds=_env_float(
                "llm_timeout",
                float(llm_raw.get("timeout_seconds", 30.0)),
            ),
        ),
        sdxl=SDXLSettings(
            use_refiner=_env_bool(
                "sdxl_use_refiner", bool(sdxl_raw.get("use_refiner", True))
            ),
            base_denoising_end=_env_float(
                "sdxl_base_denoising_end",
                float(sdxl_raw.get("base_denoising_end", 0.8)),
            ),
            refiner_denoising_start=_env_float(
                "sdxl_refiner_denoising_start",
                float(sdxl_raw.get("refiner_denoising_start", 0.8)),
            ),
            vae_float32=_env_bool(
                "sdxl_vae_float32", bool(sdxl_raw.get("vae_float32", True))
            ),
            num_inference_steps=int(
                _env_float(
                    "sdxl_num_inference_steps",
                    float(sdxl_raw.get("num_inference_steps", 50)),
                )
            ),
            guidance_scale=_env_float(
                "sdxl_guidance_scale",
                float(sdxl_raw.get("guidance_scale", 7.5)),
            ),
            default_width=int(
                _env_float(
                    "sdxl_default_width",
                    float(sdxl_raw.get("default_width", 768)),
                )
            ),
            default_height=int(
                _env_float(
                    "sdxl_default_height",
                    float(sdxl_raw.get("default_height", 768)),
                )
            ),
        ),
        prompt=PromptSettings(
            max_tokens=int(
                _env_float(
                    "prompt_max_tokens",
                    float(prompt_raw.get("max_tokens", 75)),
                )
            ),
            negative_reserve=int(
                _env_float(
                    "prompt_negative_reserve",
                    float(prompt_raw.get("negative_reserve", 25)),
                )
            ),
            segmentation_mode=_env(
                "prompt_segmentation_mode",
                prompt_raw.get("segmentation_mode", "semantic"),
            ),
            expansion=_env_bool(
                "prompt_expansion",
                bool(prompt_raw.get("expansion", True)),
            ),
        ),
    )
