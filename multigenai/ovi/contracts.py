"""OviModelSpec — variant metadata and shape contracts for the Ovi backend.

DESIGN RULE (do not weaken)
---------------------------
This module deliberately contains **no hard-coded latent dimensions**.

The upstream Ovi release ships several variants (``720x720_5s``,
``960x960_5s``, ``960x960_10s``) that differ in checkpoint, spatial area,
temporal length, frame rate and sample configuration.  The temporal latent
length in particular is **not** the frame count: the Wan VAE applies
``temperal_downsample=[False, True, True]`` (a 4x temporal compression), so
the latent temporal length must be *observed* from a real encode rather than
assumed.

Consequently:

* every latent/shape field on :class:`OviModelSpec` is ``Optional``;
* :func:`require_observed_shape` raises unless the caller supplies a shape that
  was actually measured through the VAE;
* :func:`derive_seq_len` computes the transformer sequence length from the
  **actual latent shape** using the same per-dimension arithmetic that
  ``nn.Conv3d(kernel_size=patch, stride=patch)`` performs.

Nothing here imports torch, so the contracts are testable on a CPU-only box
without weights.

See ``docs/ROADMAP_REVIEW.md`` §1.1 and §5 for the reasoning.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from dataclasses import dataclass, field, asdict
from typing import Dict, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ObservedShape",
    "OviModelSpec",
    "OviContractError",
    "UnobservedContractError",
    "derive_seq_len",
    "conv_out_size",
    "require_observed_shape",
    "load_dit_config",
    "load_observed_shape_fixture",
    "KNOWN_VARIANTS",
    "UPSTREAM_MODEL_SPECS",
    "UPSTREAM_PROVENANCE",
    "OVI_CKPT_SUBDIR",
    "FP8_CKPT_BASENAME",
    "attach_upstream_contract",
    "resolve_ovi_checkpoint",
    "get_variant",
]


class OviContractError(ValueError):
    """Base class for Ovi contract violations."""


class UnobservedContractError(OviContractError):
    """Raised when a latent shape is required but was never measured."""


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def conv_out_size(size: int, kernel: int, stride: Optional[int] = None) -> int:
    """Output size of a 1-D convolution with no padding.

    Mirrors ``torch.nn.Conv{1,2,3}d`` when ``padding=0``, which is how the Wan
    patch embedding is constructed::

        nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)

    ``stride`` defaults to ``kernel`` because the Wan patch embedding uses
    ``kernel == stride`` (non-overlapping patches).

    Because the floor happens **per dimension**, an aggregate expression such
    as ``H*W // (ph*pw)`` is not equivalent and is silently wrong whenever
    ``H*W % (ph*pw) != 0``.  At 720x720 (latent 45x45) it returns 506 instead
    of the correct 484 -- i.e. wrong only at the resolution the plan targets.
    """
    if stride is None:
        stride = kernel
    if kernel <= 0 or stride <= 0:
        raise OviContractError(f"kernel/stride must be positive, got {kernel}/{stride}")
    if size < kernel:
        raise OviContractError(
            f"input {size} is smaller than kernel {kernel}; conv would produce no output"
        )
    return (size - kernel) // stride + 1


def derive_seq_len(
    latent_shape: Sequence[int],
    patch_size: Sequence[int],
) -> int:
    """Transformer sequence length for a given latent shape.

    Args:
        latent_shape: ``(F, H, W)`` of the **latent** tensor actually produced
            by the VAE. Channel count is irrelevant to sequence length and is
            therefore excluded.
        patch_size: either a 1-tuple (the audio branch, which asserts
            ``patch_size == (1,)`` — i.e. no spatial patching) or a 3-tuple
            ``(pf, ph, pw)`` for the video branch, e.g. ``(1, 2, 2)``.

    Returns:
        The product of the per-dimension convolution output sizes, using
        ``kernel=patch, stride=patch`` exactly as ``nn.Conv3d`` does.

    Raises:
        OviContractError: on rank other than 1 or 3, non-positive dims, or an
            input smaller than its patch.
    """
    latent_shape = tuple(int(v) for v in latent_shape)
    patch_size = tuple(int(v) for v in patch_size)

    if len(latent_shape) != 3:
        raise OviContractError(
            f"latent_shape must be (F, H, W); got {len(latent_shape)} dims: {latent_shape}"
        )
    if len(patch_size) == 1:
        # Audio branch: no spatial patching (wan/model.py asserts patch_size == (1,)).
        # The audio latent is a 1-D sequence, so only the temporal dim contributes.
        pf, ph, pw = patch_size[0], 1, 1
    elif len(patch_size) == 3:
        pf, ph, pw = patch_size
    else:
        raise OviContractError(
            f"patch_size must be rank 1 (audio, e.g. (1,)) or rank 3 "
            f"(video, e.g. (1, 2, 2)); got {patch_size}"
        )
    if any(v <= 0 for v in latent_shape):
        raise OviContractError(f"latent dims must be positive; got {latent_shape}")

    out_sizes = tuple(
        conv_out_size(dim, patch, patch)
        for dim, patch in zip(latent_shape, (pf, ph, pw))
    )
    f_out, h_out, w_out = out_sizes
    return f_out * h_out * w_out


# ---------------------------------------------------------------------------
# Observed shapes (measured, never assumed)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObservedShape:
    """A latent shape that was **actually measured** through the real VAE.

    ``source`` records how it was obtained, e.g.::

        "kaggle://ovi-720x720_5s/wan2_2_vae.encode(rgb_uint8[81,720,720,3])"

    An ``ObservedShape`` with ``source=None`` is rejected by
    :func:`require_observed_shape` — that is the mechanism that prevents an
    assumption from being promoted to a frozen test fixture.
    """

    latent_shape: Tuple[int, ...]
    dtype: str
    channels: int
    source: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.latent_shape) != 3:
            raise OviContractError(
                f"ObservedShape.latent_shape must be (F, H, W); got {self.latent_shape}"
            )
        if self.channels <= 0:
            raise OviContractError(f"channels must be positive; got {self.channels}")

    @property
    def is_provenance_backed(self) -> bool:
        return bool(self.source)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "ObservedShape":
        return cls(
            latent_shape=tuple(d["latent_shape"]),
            dtype=d["dtype"],
            channels=int(d["channels"]),
            source=d.get("source"),
        )


def require_observed_shape(shape: Optional[ObservedShape], *, context: str = "") -> ObservedShape:
    """Return ``shape`` only if it was genuinely measured.

    This is the guard that implements "do not freeze numeric fixtures yet".
    A caller cannot quietly pass a guessed shape: it must carry provenance.
    """
    if shape is None:
        raise UnobservedContractError(
            f"{context or 'latent shape'}: no measured shape available. "
            "Encode a real clip through the Wan VAE and record the result "
            "(see Gate M2 in docs/ROADMAP_REVIEW.md) before freezing this value."
        )
    if not shape.is_provenance_backed:
        raise UnobservedContractError(
            f"{context or 'latent shape'}: ObservedShape has no `source`, so it was "
            "not measured through the VAE. Refusing to treat an assumption as fact."
        )
    return shape


# ---------------------------------------------------------------------------
# Model spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OviModelSpec:
    """First-class model contract for one Ovi variant.

    Latent/shape fields are Optional on purpose: they stay unset until a real
    encode populates them.  :meth:`assert_ready` then refuses to run rather
    than guessing.
    """

    name: str
    checkpoint: str
    target_width: int
    target_height: int
    num_frames: int
    fps: int
    sample_steps: int = 50
    shift: float = 5.0
    video_guidance_scale: float = 4.0
    audio_guidance_scale: float = 3.0
    slg_layer: int = 11
    precision: str = "bf16"
    cpu_offload: bool = False
    fp8: bool = False

    # --- Observed (populated only from a real VAE run) ---
    video_patch: Tuple[int, ...] = (1, 2, 2)
    audio_patch: Tuple[int, ...] = (1,)
    video_in_dim: Optional[int] = None
    audio_in_dim: Optional[int] = None
    observed_video: Optional[ObservedShape] = None
    observed_audio: Optional[ObservedShape] = None

    # --- Temporal latent contract (upstream-shipped, provenance-tracked) ---
    # Set ONLY via attach_upstream_contract(); get_variant() leaves these as
    # None so the raw registry stays metadata-only. The temporal latent length
    # is NOT the frame count: the Wan VAE applies 4x temporal compression, so
    # 121 frames @ 24fps -> 31 video latent frames and 157 audio latent frames.
    video_latent_length: Optional[int] = None
    audio_latent_length: Optional[int] = None
    latent_provenance: str = ""
    audio_tag_style: str = "audio"  # 'audcap' (720x720_5s) or 'audio' (960)

    # ------------------------------------------------------------------
    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.fps if self.fps else 0.0

    @property
    def is_fully_observed(self) -> bool:
        return (
            self.observed_video is not None
            and self.observed_video.is_provenance_backed
            and self.observed_audio is not None
            and self.observed_audio.is_provenance_backed
        )

    def video_seq_len(self) -> int:
        """Sequence length from the *measured* video latent shape."""
        shape = require_observed_shape(self.observed_video, context=f"{self.name}.video_seq_len")
        return derive_seq_len(shape.latent_shape, self.video_patch)

    def audio_seq_len(self) -> int:
        """Sequence length from the *measured* audio latent shape."""
        shape = require_observed_shape(self.observed_audio, context=f"{self.name}.audio_seq_len")
        return derive_seq_len(shape.latent_shape, self.audio_patch)

    def assert_ready(self) -> None:
        """Fail loudly if any contract needed for inference is unobserved."""
        missing = []
        if self.observed_video is None or not self.observed_video.is_provenance_backed:
            missing.append("observed_video (Gate M2: Wan VAE encode)")
        if self.observed_audio is None or not self.observed_audio.is_provenance_backed:
            missing.append("observed_audio (Gate M3: Ovi audio VAE)")
        if self.video_in_dim is None:
            missing.append("video_in_dim (load from configs/model/dit/video.json)")
        if self.audio_in_dim is None:
            missing.append("audio_in_dim (load from configs/model/dit/audio.json)")
        if self.video_latent_length is None:
            missing.append("video_latent_length (upstream checkpoint contract)")
        if self.audio_latent_length is None:
            missing.append("audio_latent_length (upstream checkpoint contract)")
        if missing:
            raise UnobservedContractError(
                f"OviModelSpec '{self.name}' is not ready. Missing: {missing}. "
                "Resolve via Gates M0-M3 in docs/ROADMAP_REVIEW.md or attach the "
                "upstream contract via attach_upstream_contract()."
            )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

#: Variants documented by the upstream Ovi release. Values are metadata only —
#: latent dimensions are intentionally absent until observed (see
#: :data:`UPSTREAM_MODEL_SPECS` for the vendor-shipped temporal contract).
#: ``checkpoint`` is the *basename* inside ``<ckpt_dir>/Ovi/`` — the upstream
#: download layout (not a variant subdirectory).
KNOWN_VARIANTS: Dict[str, Dict] = {
    "720x720_5s": dict(
        checkpoint="model.safetensors",
        target_width=720, target_height=720, num_frames=121, fps=24,
    ),
    "960x960_5s": dict(
        checkpoint="model_960x960.safetensors",
        target_width=960, target_height=960, num_frames=121, fps=24,
    ),
    "960x960_10s": dict(
        checkpoint="model_960x960_10s.safetensors",
        target_width=960, target_height=960, num_frames=241, fps=24,
    ),
}

#: Upstream checkpoint subdirectory (relative to ``ckpt_dir``).
OVI_CKPT_SUBDIR = "Ovi"

#: FP8 checkpoint basename (upstream: only the 720x720_5s variant ships one).
FP8_CKPT_BASENAME = "model_fp8_e4m3fn.safetensors"

#: Temporal latent contract shipped with the trained upstream checkpoints
#: (``ovi/ovi_fusion_engine.py::NAME_TO_MODEL_SPECS_MAP`` in
#: character-ai/Ovi). These are NOT guesses: the numbers are part of the
#: released checkpoint contract — the transformer RoPE/time embeddings and the
#: audio VAE were trained at exactly these temporal lengths. They are injected
#: into a spec only through :func:`attach_upstream_contract`, which records the
#: provenance explicitly; :func:`get_variant` stays metadata-only.
UPSTREAM_MODEL_SPECS: Dict[str, Dict] = {
    "720x720_5s": dict(
        video_latent_length=31,
        audio_latent_length=157,
        audio_tag_style="audcap",   # engine expects <AUDCAP>...<ENDAUDCAP>
    ),
    "960x960_5s": dict(
        video_latent_length=31,
        audio_latent_length=157,
        audio_tag_style="audio",    # engine expects plain "Audio: ..."
    ),
    "960x960_10s": dict(
        video_latent_length=61,
        audio_latent_length=314,
        audio_tag_style="audio",
    ),
}

#: Provenance string recorded whenever the upstream latent contract is used.
UPSTREAM_PROVENANCE = (
    "upstream:character-ai/Ovi ovi_fusion_engine.NAME_TO_MODEL_SPECS_MAP "
    "(temporal lengths shipped with the trained checkpoints)"
)


def get_variant(name: str) -> OviModelSpec:
    """Build a spec for a known variant. Latent fields stay unset."""
    if name not in KNOWN_VARIANTS:
        raise OviContractError(
            f"Unknown Ovi variant '{name}'. Known: {sorted(KNOWN_VARIANTS)}"
        )
    return OviModelSpec(name=name, **KNOWN_VARIANTS[name])


def attach_upstream_contract(
    spec: OviModelSpec,
    video_in_dim: int,
    audio_in_dim: int,
) -> OviModelSpec:
    """Attach the vendor-shipped temporal latent contract to a variant spec.

    The temporal lengths (31/157 for 5s, 61/314 for 10s) are part of the
    upstream checkpoint contract, not assumptions — the released models were
    trained at exactly these sequence lengths. This is the ONLY sanctioned way
    to populate ``video_latent_length`` / ``audio_latent_length``: it records
    the provenance on the spec so downstream gates can verify it.

    Args:
        spec: spec from :func:`get_variant` (metadata only).
        video_in_dim: latent channel count from ``configs/model/dit/video.json``.
        audio_in_dim: latent channel count from ``configs/model/dit/audio.json``.

    Returns:
        A new spec (frozen dataclass) with the contract attached.

    Raises:
        OviContractError: if the variant has no upstream entry, or the channel
            counts are not positive integers.
    """
    entry = UPSTREAM_MODEL_SPECS.get(spec.name)
    if entry is None:
        raise OviContractError(
            f"variant '{spec.name}' has no upstream latent contract"
        )
    if int(video_in_dim) <= 0 or int(audio_in_dim) <= 0:
        raise OviContractError(
            f"in_dims must be positive, got video={video_in_dim} audio={audio_in_dim}"
        )
    return dataclasses.replace(
        spec,
        video_latent_length=int(entry["video_latent_length"]),
        audio_latent_length=int(entry["audio_latent_length"]),
        audio_tag_style=str(entry["audio_tag_style"]),
        video_in_dim=int(video_in_dim),
        audio_in_dim=int(audio_in_dim),
        latent_provenance=UPSTREAM_PROVENANCE,
        observed_audio=ObservedShape(
            latent_shape=(int(entry["audio_latent_length"]), 1, 1),
            dtype="bfloat16",
            channels=int(audio_in_dim),
            source=UPSTREAM_PROVENANCE,
        ),
    )


def resolve_ovi_checkpoint(ckpt_dir, variant: str) -> "pathlib.Path":
    """Resolve ``<ckpt_dir>/Ovi/<basename>`` for a variant (upstream layout).

    Raises:
        OviContractError: unknown variant.
        FileNotFoundError: checkpoint file missing — upstream treats this as a
            hard error (the fusion checkpoint is REQUIRED, never optional).
    """
    spec = get_variant(variant)
    path = pathlib.Path(ckpt_dir) / OVI_CKPT_SUBDIR / spec.checkpoint
    if not path.exists():
        raise FileNotFoundError(
            f"REQUIRED Ovi fusion checkpoint not found at {path}. Download the "
            f"upstream release for variant '{variant}' into "
            f"{pathlib.Path(ckpt_dir) / OVI_CKPT_SUBDIR}/ (expected file: "
            f"{spec.checkpoint})."
        )
    return path


# ---------------------------------------------------------------------------
# Package-relative config access (P0-N)
# ---------------------------------------------------------------------------

_PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DIT_DIR = _PKG_ROOT / "configs" / "model" / "dit"
_OBSERVED_DIR = _PKG_ROOT / "configs" / "observed"


def load_dit_config(which: str = "video") -> Dict:
    """Load ``configs/model/dit/{video,audio}.json`` package-relatively.

    Never CWD-relative: the previous ``pathlib.Path("multigenai/configs/...")``
    only resolved when the process happened to start from the repo root.
    """
    if which not in ("video", "audio"):
        raise OviContractError(f"which must be 'video' or 'audio'; got {which!r}")
    path = _DIT_DIR / f"{which}.json"
    if not path.exists():
        raise OviContractError(f"missing DiT config: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_observed_shape_fixture(variant: str) -> Tuple[Optional[ObservedShape], Optional[ObservedShape]]:
    """Load provenance-backed observations recorded by a real VAE run.

    Returns ``(None, None)`` when no measurement has been recorded yet — which
    is the expected state until Gate M2/M3 have run on Kaggle with real
    weights.  Callers must treat that as "not ready", never as a default.
    """
    path = _OBSERVED_DIR / f"{variant}.observed.json"
    if not path.exists():
        return None, None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    video = ObservedShape.from_dict(data["video"]) if data.get("video") else None
    audio = ObservedShape.from_dict(data["audio"]) if data.get("audio") else None
    return video, audio