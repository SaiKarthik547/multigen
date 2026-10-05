"""OviModelLoader — the single authoritative loader for the Ovi fusion stack.

Upstream contract (ovi/ovi_fusion_engine.py in character-ai/Ovi):

    FusionModel (meta or eager init)
        -> ONE variant checkpoint from <ckpt_dir>/Ovi/<basename>, strict=True
        -> bf16 (or fp8) + eval
        -> set_rope_params() AFTER the checkpoint load
    Wan2.2 VAE + MMAudio VAE + T5 encoder alongside

The previous MultiGen FusionEngine loaded separate Wan/MMAudio weights with
``strict=False`` and an optional non-upstream "fusion adapter".  That meant the
cross-modal fusion projections (``k_fusion`` / ``v_fusion`` /
``pre_attn_norm_fusion`` / ``norm_k_fusion``) were left randomly initialised —
the engine would "run" without executing the trained Ovi model.  This loader is
the fix: the fusion checkpoint is mandatory and loaded strictly; there is no
adapter fallback.

Kaggle free-tier policy (16 GB T4/P100):
    - upstream bf16 needs ~32 GB peak, fp8 ~24 GB — neither fits a 16 GB card
      with everything resident;
    - therefore on Kaggle the default is the fp8 checkpoint (720x720_5s only,
      as upstream) + whole-model CPU offload around the sampling loop + T5/VAE
      offload around their use, matching the upstream ``cpu_offload`` mode.
"""

from __future__ import annotations

import dataclasses
import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from multigenai.core.logging.logger import get_logger
from multigenai.ovi.contracts import (
    OviContractError,
    OviModelSpec,
    FP8_CKPT_BASENAME,
    attach_upstream_contract,
    get_variant,
    load_dit_config,
    resolve_ovi_checkpoint,
)

if TYPE_CHECKING:
    from multigenai.core.model_registry import ModelRegistry

LOG = get_logger(__name__)

#: Registry model id for the joint bundle.
OVI_BUNDLE_ID = "ovi_fusion_bundle"

# VRAM floors (GB) per precision/offload configuration. These are admission
# guards, not guarantees: upstream documents ~24 GB peak for fp8 and ~32 GB
# for bf16; the cpu-offload floor assumes fp8 weights + activation headroom.
_MIN_VRAM = {
    "fp8": 14.0,
    "bf16_offload": 24.0,
    "bf16": 32.0,
}


@dataclass
class OviBundle:
    """Everything needed for one joint audio-video generation."""

    model: Any                      # FusionModel
    v_vae: Any                      # Wan2_2_VAE
    a_vae: Any                      # FeaturesUtils (MMAudio VAE + vocoder)
    t5: Any                         # T5EncoderModel
    spec: OviModelSpec
    device: Any
    cpu_offload: bool = False
    fp8: bool = False

    # -- offload helpers (upstream pattern) --------------------------------

    def model_to_device(self) -> None:
        if self.cpu_offload:
            self.model = self.model.to(self.device)

    def model_offload(self) -> None:
        if self.cpu_offload:
            self._offload(self.model)

    def t5_to_device(self) -> None:
        if self.cpu_offload:
            self.t5.model = self.t5.model.to(self.device)

    def t5_offload(self) -> None:
        if self.cpu_offload:
            self._offload(self.t5.model)

    def vae_video_to_device(self) -> None:
        if self.cpu_offload:
            self.v_vae.model = self.v_vae.model.to(self.device)

    def vae_video_offload(self) -> None:
        if self.cpu_offload:
            self._offload(self.v_vae.model)

    def vae_audio_to_device(self) -> None:
        if self.cpu_offload:
            self.a_vae = self.a_vae.to(self.device)

    def vae_audio_offload(self) -> None:
        if self.cpu_offload:
            self._offload(self.a_vae)

    @staticmethod
    def _offload(module: Any) -> None:
        import torch

        module = module.cpu()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()


# ---------------------------------------------------------------------------
# Configuration resolution
# ---------------------------------------------------------------------------


def resolve_ovi_settings(registry: "ModelRegistry", environment=None) -> dict:
    """Resolve Ovi configuration with Kaggle-aware defaults.

    Keys read from model_config.yaml (all optional):
        ovi_checkpoint_dir: root dir containing ``Ovi/`` (falls back to the
            parent of ``wan_model_path``).
        ovi_model_name:     variant name (default ``720x720_5s``).
        ovi_cpu_offload:    bool (default: True on Kaggle, False elsewhere).
        ovi_fp8:            bool (default: True on Kaggle when the fp8
            checkpoint exists, else False).
    """
    wan_path = registry.get_config_value("wan_model_path", "")
    ckpt_dir = registry.get_config_value("ovi_checkpoint_dir")
    if not ckpt_dir:
        if wan_path:
            ckpt_dir = str(pathlib.Path(str(wan_path)).parent)
        else:
            raise OviContractError(
                "model_config.yaml defines neither ovi_checkpoint_dir nor "
                "wan_model_path; cannot locate the Ovi checkpoint directory."
            )

    variant = registry.get_config_value("ovi_model_name", "720x720_5s")
    on_kaggle = bool(
        environment is not None and getattr(environment, "platform", "") == "kaggle"
    )

    cpu_offload = registry.get_config_value("ovi_cpu_offload")
    if cpu_offload is None:
        cpu_offload = on_kaggle

    fp8 = registry.get_config_value("ovi_fp8")
    if fp8 is None:
        # Auto: Kaggle (16 GB cards) wants the fp8 checkpoint when present.
        fp8 = False
        if on_kaggle:
            if variant != "720x720_5s":
                LOG.warning(
                    "Ovi loader: fp8 is only shipped for 720x720_5s; variant "
                    f"'{variant}' will use bf16 (needs >= 24 GB VRAM)."
                )
            else:
                fp8_path = (
                    pathlib.Path(str(ckpt_dir)) / "Ovi" / FP8_CKPT_BASENAME
                )
                fp8 = fp8_path.exists()
                if not fp8:
                    LOG.warning(
                        "Ovi loader: Kaggle detected but fp8 checkpoint not "
                        f"found at {fp8_path}; falling back to bf16 "
                        "(requires >= 24 GB VRAM)."
                    )

    return {
        "ckpt_dir": str(ckpt_dir),
        "variant": str(variant),
        "cpu_offload": bool(cpu_offload),
        "fp8": bool(fp8),
    }


# ---------------------------------------------------------------------------
# Bundle construction
# ---------------------------------------------------------------------------


def build_runtime_spec(variant: str) -> OviModelSpec:
    """Metadata spec + upstream temporal contract + DiT channel counts."""
    video_cfg = load_dit_config("video")
    audio_cfg = load_dit_config("audio")
    spec = attach_upstream_contract(
        get_variant(variant),
        video_in_dim=int(video_cfg["in_dim"]),
        audio_in_dim=int(audio_cfg["in_dim"]),
    )
    return spec


def _resolve_component_paths(registry: "ModelRegistry") -> dict:
    """Component weight paths. Prefers explicit config keys; the Ovi fusion
    checkpoint itself follows the upstream ``<ckpt_dir>/Ovi/`` layout."""
    wan_path = registry.get_config_value("wan_model_path")
    mmaudio_path = registry.get_config_value("mmaudio_model_path")
    t5_path = registry.get_config_value("shared_t5_path")
    t5_tokenizer_path = registry.get_config_value("t5_tokenizer_path")
    if not (wan_path and mmaudio_path and t5_path and t5_tokenizer_path):
        raise OviContractError(
            "model_config.yaml must define wan_model_path, mmaudio_model_path, "
            "shared_t5_path and t5_tokenizer_path for the Ovi stack."
        )
    return {
        "wan_path": pathlib.Path(str(wan_path)),
        "mmaudio_path": pathlib.Path(str(mmaudio_path)),
        "t5_path": str(t5_path),
        "t5_tokenizer_path": str(t5_tokenizer_path),
    }


def load_ovi_bundle(
    registry: "ModelRegistry",
    environment=None,
    spec: Optional[OviModelSpec] = None,
) -> OviBundle:
    """Construct and load the full Ovi stack.

    Raises:
        FileNotFoundError: the variant's fusion checkpoint is missing — the
            upstream contract treats this as fatal (never degrade to base
            Wan/MMAudio weights).
        InsufficientVRAMError: (from the caller's registry pre-check) when the
            admission floor for the resolved precision is not met.
    """
    import torch

    settings = resolve_ovi_settings(registry, environment)
    variant = settings["variant"]
    fp8 = settings["fp8"] and variant == "720x720_5s"
    if settings["fp8"] and variant != "720x720_5s":
        LOG.warning(
            "Ovi loader: fp8 requested for variant '%s' but upstream ships fp8 "
            "only for 720x720_5s — using the bf16 checkpoint instead.", variant
        )
    cpu_offload = settings["cpu_offload"]

    if spec is None:
        spec = build_runtime_spec(variant)

    paths = _resolve_component_paths(registry)
    ckpt_dir = settings["ckpt_dir"]

    if fp8:
        checkpoint = pathlib.Path(ckpt_dir) / "Ovi" / FP8_CKPT_BASENAME
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"fp8 checkpoint not found at {checkpoint}. Download "
                f"{FP8_CKPT_BASENAME} from the upstream Ovi release, or set "
                "ovi_fp8: false in model_config.yaml (requires >= 24 GB VRAM)."
            )
    else:
        checkpoint = resolve_ovi_checkpoint(ckpt_dir, variant)

    from multigenai.models.fusion.fusion import FusionModel
    from multigenai.models.wan.vae2_2 import Wan2_2_VAE
    from multigenai.models.mmaudio.mmaudio_core.features_utils import FeaturesUtils
    from multigenai.models.shared.t5 import T5EncoderModel
    from multigenai.utils.model_loading_utils import (
        init_fusion_score_model_multigenai,
        load_fusion_checkpoint,
    )

    target_dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # meta_init is required for the fp8 checkpoint (upstream uses assign=True
    # so the fp8 tensors land in the module without a dtype cast).
    meta_init = fp8

    model, video_config, audio_config = init_fusion_score_model_multigenai(
        rank=device, meta_init=meta_init
    )

    # THE load: one trained fusion checkpoint, strictly validated.
    load_fusion_checkpoint(model, checkpoint_path=str(checkpoint), from_meta=meta_init)

    if not fp8:
        model = model.to(dtype=target_dtype)
    model = model.to(device=device if not cpu_offload else "cpu").eval()
    # Upstream order: set_rope_params AFTER the checkpoint load.
    model.set_rope_params()

    # --- Wan2.2 video VAE ---
    vae_pth = paths["wan_path"] / "Wan2.2_VAE.pth"
    v_vae = Wan2_2_VAE(vae_pth=str(vae_pth), device=device)
    v_vae.model.requires_grad_(False).eval()
    v_vae.model = v_vae.model.bfloat16()
    if cpu_offload:
        OviBundle._offload(v_vae.model)

    # --- MMAudio audio VAE + BigVGAN vocoder ---
    a_vae = FeaturesUtils(
        mode="16k",
        need_vae_encoder=True,
        tod_vae_ckpt=str(paths["mmaudio_path"] / "ext_weights" / "v1-16.pth"),
        bigvgan_vocoder_ckpt=str(
            paths["mmaudio_path"] / "ext_weights" / "best_netG.pt"
        ),
    ).to(device)
    a_vae.requires_grad_(False).eval()
    a_vae = a_vae.bfloat16()
    if cpu_offload:
        OviBundle._offload(a_vae)

    # --- T5 text encoder ---
    t5 = T5EncoderModel(
        text_len=512,
        dtype=target_dtype,
        device=device,
        checkpoint_path=paths["t5_path"],
        tokenizer_path=paths["t5_tokenizer_path"],
    )
    if cpu_offload:
        OviBundle._offload(t5.model)

    LOG.info(
        "Ovi bundle ready: variant=%s ckpt=%s fp8=%s cpu_offload=%s "
        "video_latent=%s audio_latent=%s",
        variant,
        checkpoint.name,
        fp8,
        cpu_offload,
        spec.video_latent_length,
        spec.audio_latent_length,
    )

    return OviBundle(
        model=model,
        v_vae=v_vae,
        a_vae=a_vae,
        t5=t5,
        spec=spec,
        device=device,
        cpu_offload=cpu_offload,
        fp8=fp8,
    )


def register_ovi_bundle(registry: "ModelRegistry", environment=None) -> OviModelSpec:
    """Register the lazy loader in the ModelRegistry and return the spec.

    The registry pre-checks the admission floor before the loader runs, so an
    under-powered card fails BEFORE any weights are touched.
    """
    settings = resolve_ovi_settings(registry, environment)
    spec = build_runtime_spec(settings["variant"])
    fp8 = settings["fp8"] and settings["variant"] == "720x720_5s"
    if fp8:
        min_vram = _MIN_VRAM["fp8"]
    elif settings["cpu_offload"]:
        min_vram = _MIN_VRAM["bf16_offload"]
    else:
        min_vram = _MIN_VRAM["bf16"]

    def _loader():
        return load_ovi_bundle(registry, environment, spec=spec)

    registry.register(OVI_BUNDLE_ID, loader=_loader, min_vram_gb=min_vram)
    return spec
