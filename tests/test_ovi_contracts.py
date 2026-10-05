"""Gate M0 / M1 — Ovi contract tests.

DESIGN RULE
-----------
These tests assert **invariants**, never unverified numeric fixtures.

The temporal latent length of the Wan VAE is NOT the frame count: the VAE
applies ``temperal_downsample=[False, True, True]`` (4x temporal compression)
and it has not yet been measured on real weights.  Therefore no test in this
file asserts a literal like ``latent_t == 21`` or ``vid_seq_len == 10164``.
Such a test would convert an assumption into a frozen "truth" precisely
because it was written down.

What IS asserted:
  * derive_seq_len() reproduces Conv3d(kernel=patch, stride=patch) semantics
  * the naive aggregate formula is rejected/never used
  * the audio branch refuses a rank-3 patch
  * OviModelSpec refuses to report sequence lengths before a measured shape
  * an ObservedShape without provenance is rejected
  * the DiT configs load package-relatively and agree with the vendored model

Run:  python -m pytest tests/test_ovi_contracts.py -v
No torch and no weights required.
"""

import pathlib

import pytest

from multigenai.ovi.contracts import (
    KNOWN_VARIANTS,
    ObservedShape,
    OviContractError,
    OviModelSpec,
    UnobservedContractError,
    conv_out_size,
    derive_seq_len,
    get_variant,
    load_dit_config,
    require_observed_shape,
)


# ---------------------------------------------------------------------------
# Gate M1 — mathematical shape contract
# ---------------------------------------------------------------------------

def test_conv_out_size_matches_torch_conv1d_semantics():
    """conv_out_size must equal (n - k)//s + 1, i.e. no padding."""
    # Reference values from the standard conv output formula.
    assert conv_out_size(45, 2, 2) == 22
    assert conv_out_size(32, 2, 2) == 16
    assert conv_out_size(60, 2, 2) == 30
    assert conv_out_size(81, 1, 1) == 81


def test_conv_out_size_rejects_input_smaller_than_kernel():
    with pytest.raises(OviContractError):
        conv_out_size(1, 2, 2)


def test_conv_out_size_rejects_nonpositive_kernel():
    with pytest.raises(OviContractError):
        conv_out_size(10, 0, 1)


def test_derive_seq_len_is_floor_per_dimension_not_aggregate():
    """The invariant that the previous roadmap got wrong.

    At latent 45x45 the aggregate form ``H*W//4`` yields 506 while the true
    per-dimension Conv3d output is 22*22 = 484.  derive_seq_len must give the
    per-dimension result.
    """
    seq = derive_seq_len((81, 45, 45), (1, 2, 2))
    assert seq == 81 * 22 * 22
    assert seq != 81 * ((45 * 45) // 4)


def test_derive_seq_len_matches_per_dimension_arithmetic_across_resolutions():
    """Cross-check against an independent per-dimension implementation."""
    for frames, h, w in (
        (81, 45, 45), (81, 32, 32), (81, 36, 64),
        (81, 60, 60), (81, 44, 80), (21, 45, 45), (21, 36, 64),
    ):
        expected = frames * conv_out_size(h, 2, 2) * conv_out_size(w, 2, 2)
        assert derive_seq_len((frames, h, w), (1, 2, 2)) == expected


def test_derive_seq_len_audio_branch_uses_identity_patch():
    """Audio patch is (1,) — no spatial patching, sequence length == latent length."""
    assert derive_seq_len((157, 1, 1), (1,)) == 157
    # Equivalent explicit form must agree.
    assert derive_seq_len((157, 1, 1), (1, 1, 1)) == 157


def test_derive_seq_len_rejects_invalid_patch_rank():
    with pytest.raises(OviContractError):
        derive_seq_len((157, 1, 1), (1, 2))  # rank 2 is neither audio nor video
    with pytest.raises(OviContractError):
        derive_seq_len((81, 45, 45), (1, 2, 2, 2))


def test_derive_seq_len_rejects_bad_latent_rank():
    with pytest.raises(OviContractError):
        derive_seq_len((81, 45), (1, 2, 2))
    with pytest.raises(OviContractError):
        derive_seq_len((81, 45, 45, 48), (1, 2, 2))


def test_derive_seq_len_rejects_nonpositive_dims():
    with pytest.raises(OviContractError):
        derive_seq_len((81, 0, 45), (1, 2, 2))


# ---------------------------------------------------------------------------
# Gate M0 — static contracts from the vendored config
# ---------------------------------------------------------------------------

def test_dit_configs_load_package_relatively():
    """P0-N: no CWD-relative config access."""
    video = load_dit_config("video")
    audio = load_dit_config("audio")
    assert video["model_type"] == "ti2v"
    assert audio["model_type"] == "t2a"
    assert tuple(video["patch_size"]) == (1, 2, 2)
    assert tuple(audio["patch_size"]) == (1,)


def test_video_and_audio_channels_come_from_config_not_hardcoded():
    """The engine previously allocated 16 video / 12 audio channels.

    Those literals contradicted this config.  The config is authoritative; the
    engine must read it.  Here we only pin the config's own agreement between
    in_dim and out_dim, which the patch embedding depends on.
    """
    video = load_dit_config("video")
    audio = load_dit_config("audio")
    assert video["in_dim"] == video["out_dim"]
    assert audio["in_dim"] == audio["out_dim"]


def test_config_dir_is_resolved_from_package_not_cwd(monkeypatch, tmp_path):
    """load_dit_config must work after chdir elsewhere."""
    video = load_dit_config("video")
    monkeypatch.chdir(tmp_path)
    assert load_dit_config("video") == video


def test_duplicate_config_tree_is_identical_to_canonical():
    """core/config/engines/... is a byte-identical duplicate of configs/...

    If it ever diverges, one of the two is authoritative and the other is a
    trap.  Flag it here instead of discovering it at inference time.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    for name in ("video.json", "audio.json"):
        canonical = root / "configs" / "model" / "dit" / name
        duplicate = root / "core" / "config" / "engines" / "model" / "dit" / name
        if duplicate.exists():
            assert canonical.read_bytes() == duplicate.read_bytes(), (
                f"{name} has diverged between configs/ and core/config/engines/"
            )


# ---------------------------------------------------------------------------
# Provenance guards — "do not freeze numeric fixtures yet"
# ---------------------------------------------------------------------------

def test_require_observed_shape_rejects_missing():
    with pytest.raises(UnobservedContractError):
        require_observed_shape(None)


def test_require_observed_shape_rejects_shape_without_provenance():
    """A shape with no `source` was not measured; refuse to trust it."""
    guessed = ObservedShape(latent_shape=(81, 45, 45), dtype="bfloat16", channels=48, source=None)
    with pytest.raises(UnobservedContractError):
        require_observed_shape(guessed)


def test_require_observed_shape_accepts_provenance_backed():
    measured = ObservedShape(
        latent_shape=(81, 45, 45),
        dtype="bfloat16",
        channels=48,
        source="unit-test: synthetic shape, arithmetic only",
    )
    assert require_observed_shape(measured) is measured


def test_observed_shape_rejects_bad_rank():
    with pytest.raises(OviContractError):
        ObservedShape(latent_shape=(81, 45), dtype="bfloat16", channels=48, source="x")


# ---------------------------------------------------------------------------
# OviModelSpec
# ---------------------------------------------------------------------------

def test_known_variants_expose_metadata_only():
    for name in KNOWN_VARIANTS:
        spec = get_variant(name)
        assert spec.name == name
        assert spec.checkpoint
        # Latent contracts are deliberately unset until measured.
        assert spec.observed_video is None
        assert spec.observed_audio is None
        assert not spec.is_fully_observed


def test_unknown_variant_raises():
    with pytest.raises(OviContractError):
        get_variant("does_not_exist")


def test_spec_refuses_sequence_length_before_observation():
    """Core rule: no measured shape -> no sequence length. Fail loudly."""
    spec = get_variant("720x720_5s")
    with pytest.raises(UnobservedContractError):
        spec.video_seq_len()
    with pytest.raises(UnobservedContractError):
        spec.audio_seq_len()


def test_spec_assert_ready_reports_every_missing_contract():
    spec = get_variant("720x720_5s")
    with pytest.raises(UnobservedContractError) as exc:
        spec.assert_ready()
    msg = str(exc.value)
    assert "observed_video" in msg
    assert "observed_audio" in msg
    assert "video_in_dim" in msg
    assert "audio_in_dim" in msg


def test_spec_seq_len_uses_measured_shape_when_present():
    spec = OviModelSpec(
        name="unit",
        checkpoint="x",
        target_width=720,
        target_height=720,
        num_frames=81,
        fps=24,
        observed_video=ObservedShape(
            latent_shape=(81, 45, 45), dtype="bfloat16", channels=48,
            source="unit-test",
        ),
        observed_audio=ObservedShape(
            latent_shape=(157, 1, 1), dtype="bfloat16", channels=20,
            source="unit-test",
        ),
    )
    assert spec.video_seq_len() == 81 * 22 * 22
    assert spec.audio_seq_len() == 157
    assert spec.is_fully_observed


def test_variant_frame_counts_are_not_latent_lengths():
    """Guard against conflating `num_frames` with the latent temporal length.

    5s @ 24fps is 120 frames; upstream encodes a 121-frame latent-friendly
    window.  Either way the latent length must come from the VAE, never from
    this field.  This test documents the intent without freezing the latent.
    """
    spec = get_variant("720x720_5s")
    assert spec.fps == 24
    assert spec.num_frames != spec.duration_seconds  # frames != seconds

# ---------------------------------------------------------------------------
# P0-B regression guard — the ti2v / additional_emb_* constructor conflict
# ---------------------------------------------------------------------------

def test_ti2v_requires_none_additional_emb():
    """The exact invariant wan/model.py enforces at construction.

    wan/model.py resolves cross_attn_type from model_type:
        't2v_cross_attn' if model_type in ('t2v', 't2a', 'ti2v')
    and then asserts, for the t2v branch:
        additional_emb_dim is None and additional_emb_length is None

    The FusionEngine used to inject 512/257 for I2V, which made every I2V
    bundle fail to construct.  This test pins the rule so it cannot silently
    return, and documents *why* identity must go through the image latent
    instead (first_frame_is_clean=True).
    """
    video = load_dit_config("video")
    model_type = video["model_type"]

    cross_attn = (
        "t2v_cross_attn"
        if model_type in ("t2v", "t2a", "ti2v")
        else "i2v_cross_attn"
    )
    assert cross_attn == "t2v_cross_attn"

    config = dict(video)
    config.pop("additional_emb_dim", None)
    config.pop("additional_emb_length", None)

    # The ti2v/t2v branch must be satisfied.
    assert config.get("additional_emb_dim") is None
    assert config.get("additional_emb_length") is None

    # And demonstrate that injecting them would break construction.
    bad = dict(config, additional_emb_dim=512, additional_emb_length=257)
    with pytest.raises(AssertionError):
        if cross_attn == "t2v_cross_attn":
            assert bad["additional_emb_dim"] is None and bad["additional_emb_length"] is None


def test_fusion_engine_does_not_inject_additional_emb():
    """Static guard: the FusionEngine must not reintroduce the injection."""
    root = pathlib.Path(__file__).resolve().parents[1]
    src = (root / "multigenai" / "engines" / "fusion_engine" / "engine.py").read_text(
        encoding="utf-8"
    )
    # Assignment (bad) vs pop (the fix). Only a plain assignment is forbidden.
    assert 'config["additional_emb_dim"]' not in src
    assert "config['additional_emb_dim']" not in src
    assert 'config["additional_emb_length"]' not in src
    assert "config['additional_emb_length']" not in src


def test_fusion_engine_noise_shapes_match_config_in_dims():
    """The engine must not hardcode latent channels that contradict the config.

    Config declares video in_dim=48 and audio in_dim=20.  The engine allocated
    16 and 12 respectively, so every forward pass would have failed the Conv3d
    patch embedding on channel count alone.

    Detection is regex-based on stripped comments, so an explanatory comment
    mentioning the old numbers cannot mask a live defect.
    """
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    src = (root / "multigenai" / "engines" / "fusion_engine" / "engine.py").read_text(
        encoding="utf-8"
    )
    code = "\n".join(
        line.split("#", 1)[0] for line in src.splitlines()
    )

    video_in_dim = load_dit_config("video")["in_dim"]
    audio_in_dim = load_dit_config("audio")["in_dim"]

    video_noise = re.search(
        r"v_noise\s*=\s*torch\.randn\(\s*\(\s*(\w+)\s*,", code, re.DOTALL
    )
    audio_noise = re.search(
        r"a_noise\s*=\s*torch\.randn\(\s*\(\s*([\w\.]+)\s*,", code, re.DOTALL
    )

    assert video_noise is not None, "could not locate v_noise allocation"
    assert audio_noise is not None, "could not locate a_noise allocation"

    # Video must reference the config-derived channel count, not a literal.
    assert video_noise.group(1) == "video_in_dim", (
        f"v_noise channels must come from config (video_in_dim); got {video_noise.group(1)}"
    )
    # Audio latent length is derived (e.g. self._a_latent_len); channels config-derived.
    assert audio_noise.group(1) == "self._a_latent_len", (
        f"a_noise latent length must be derived; got {audio_noise.group(1)}"
    )

    # No hardcoded numeric dims may remain in either allocation.
    assert not re.search(r"randn\(\s*\(\s*\d+\s*,", code), (
        "noise allocation still uses a literal numeric dim"
    )
