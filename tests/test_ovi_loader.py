"""Ovi loader-contract tests — checkpoint resolution + latent contract.

These cover the two P0 loader guarantees that can be verified without weights:

1. The variant checkpoint resolves to the upstream layout
   ``<ckpt_dir>/Ovi/<basename>`` and its absence is a HARD error (never a
   silent fall back to base Wan/MMAudio weights).
2. The temporal latent contract (31/157 for 5s, 61/314 for 10s) is attached
   only through ``attach_upstream_contract``, which records provenance;
   ``get_variant`` stays metadata-only.

No torch and no weights required.
"""

import pathlib

import pytest

from multigenai.core.model_registry import ModelRegistry
from multigenai.ovi.contracts import (
    FP8_CKPT_BASENAME,
    KNOWN_VARIANTS,
    OviContractError,
    UPSTREAM_MODEL_SPECS,
    attach_upstream_contract,
    get_variant,
    resolve_ovi_checkpoint,
)
from multigenai.ovi.loader import build_runtime_spec, resolve_ovi_settings


def _registry(tmp_path, **config):
    reg = ModelRegistry()
    reg._config = {
        "ovi_checkpoint_dir": str(tmp_path),
        "wan_model_path": str(tmp_path / "Wan2.2-TI2V-5B"),
        "mmaudio_model_path": str(tmp_path / "MMAudio"),
        "shared_t5_path": str(tmp_path / "t5.pth"),
        "t5_tokenizer_path": str(tmp_path / "t5tok"),
        **config,
    }
    return reg


# ---------------------------------------------------------------------------
# Checkpoint resolution (upstream layout)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "variant,basename",
    [
        ("720x720_5s", "model.safetensors"),
        ("960x960_5s", "model_960x960.safetensors"),
        ("960x960_10s", "model_960x960_10s.safetensors"),
    ],
)
def test_checkpoint_resolves_to_upstream_layout(tmp_path, variant, basename):
    (tmp_path / "Ovi").mkdir()
    target = tmp_path / "Ovi" / basename
    target.write_bytes(b"x")
    assert resolve_ovi_checkpoint(tmp_path, variant) == target
    assert KNOWN_VARIANTS[variant]["checkpoint"] == basename


def test_missing_checkpoint_is_a_hard_error(tmp_path):
    """Upstream treats the fusion checkpoint as REQUIRED; we must too."""
    (tmp_path / "Ovi").mkdir()
    with pytest.raises(FileNotFoundError) as exc:
        resolve_ovi_checkpoint(tmp_path, "720x720_5s")
    msg = str(exc.value)
    assert "model.safetensors" in msg
    assert "REQUIRED" in msg


def test_unknown_variant_rejected(tmp_path):
    with pytest.raises(OviContractError):
        resolve_ovi_checkpoint(tmp_path, "nope_5s")


# ---------------------------------------------------------------------------
# Upstream temporal latent contract
# ---------------------------------------------------------------------------

def test_upstream_latent_lengths_are_the_vendor_contract():
    assert UPSTREAM_MODEL_SPECS["720x720_5s"]["video_latent_length"] == 31
    assert UPSTREAM_MODEL_SPECS["720x720_5s"]["audio_latent_length"] == 157
    assert UPSTREAM_MODEL_SPECS["960x960_5s"]["video_latent_length"] == 31
    assert UPSTREAM_MODEL_SPECS["960x960_10s"]["video_latent_length"] == 61
    assert UPSTREAM_MODEL_SPECS["960x960_10s"]["audio_latent_length"] == 314


def test_get_variant_stays_metadata_only():
    """Raw registry must not smuggle in unprovenanced latents."""
    spec = get_variant("720x720_5s")
    assert spec.video_latent_length is None
    assert spec.audio_latent_length is None
    assert spec.observed_video is None
    assert spec.observed_audio is None
    assert not spec.is_fully_observed


def test_attach_upstream_contract_records_provenance():
    spec = attach_upstream_contract(get_variant("720x720_5s"), 48, 20)
    assert spec.video_latent_length == 31
    assert spec.audio_latent_length == 157
    assert spec.video_in_dim == 48
    assert spec.audio_in_dim == 20
    assert spec.latent_provenance
    assert spec.observed_audio.is_provenance_backed
    assert spec.observed_audio.latent_shape[0] == 157
    # Only the video shape still needs measuring per-scene.
    assert spec.observed_video is None


def test_attach_upstream_contract_rejects_bad_in_dims():
    with pytest.raises(OviContractError):
        attach_upstream_contract(get_variant("720x720_5s"), 0, 20)


def test_audio_tag_style_per_variant():
    assert UPSTREAM_MODEL_SPECS["720x720_5s"]["audio_tag_style"] == "audcap"
    assert UPSTREAM_MODEL_SPECS["960x960_5s"]["audio_tag_style"] == "audio"
    assert build_runtime_spec("960x960_10s").audio_tag_style == "audio"


def test_runtime_spec_reads_channels_from_dit_configs():
    """Channel counts must come from the DiT configs, never literals."""
    spec = build_runtime_spec("720x720_5s")
    assert spec.video_in_dim == 48
    assert spec.audio_in_dim == 20


# ---------------------------------------------------------------------------
# Kaggle-aware settings resolution
# ---------------------------------------------------------------------------

class _Env:
    def __init__(self, platform):
        self.platform = platform


def test_ckpt_dir_falls_back_to_wan_parent(tmp_path):
    reg = ModelRegistry()
    reg._config = {"wan_model_path": str(tmp_path / "Wan2.2-TI2V-5B")}
    settings = resolve_ovi_settings(reg, _Env("local"))
    assert pathlib.Path(settings["ckpt_dir"]) == tmp_path


def test_local_defaults_no_offload_and_no_fp8(tmp_path):
    settings = resolve_ovi_settings(_registry(tmp_path), _Env("local"))
    assert settings["cpu_offload"] is False
    assert settings["fp8"] is False


def test_kaggle_enables_offload_and_fp8_when_present(tmp_path):
    (tmp_path / "Ovi").mkdir()
    (tmp_path / "Ovi" / FP8_CKPT_BASENAME).write_bytes(b"x")
    settings = resolve_ovi_settings(_registry(tmp_path), _Env("kaggle"))
    assert settings["cpu_offload"] is True
    assert settings["fp8"] is True


def test_kaggle_without_fp8_checkpoint_falls_back(tmp_path):
    settings = resolve_ovi_settings(_registry(tmp_path), _Env("kaggle"))
    assert settings["fp8"] is False


def test_kaggle_960_variant_never_auto_selects_fp8(tmp_path):
    (tmp_path / "Ovi").mkdir()
    (tmp_path / "Ovi" / FP8_CKPT_BASENAME).write_bytes(b"x")
    settings = resolve_ovi_settings(
        _registry(tmp_path, ovi_model_name="960x960_5s"), _Env("kaggle")
    )
    assert settings["fp8"] is False


def test_explicit_config_overrides_auto(tmp_path):
    settings = resolve_ovi_settings(
        _registry(tmp_path, ovi_cpu_offload=False, ovi_fp8=False),
        _Env("kaggle"),
    )
    assert settings["cpu_offload"] is False
    assert settings["fp8"] is False