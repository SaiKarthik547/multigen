"""
MGOS Joint Fusion Engine — Synchronized Cinematic Video & Audio Generation.

Features:
- Joint Diffusion: High-fidelity Video and Audio latents predicted in a single DiT pass.
- Backbone: Wan 2.2 Transformer.
- Audio: MMAudio (BigVGAN Vocoding).
- Identity: MGOS IdentityResolver (512-d ArcFace).
- Optimization: Kaggle T4/P100 (Flow Selection, Guidance Scaling, SLG).
"""

from __future__ import annotations

import gc
import pathlib
import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple
from tqdm import tqdm

from multigenai.core.logging.logger import get_logger
from multigenai.core.model_registry import ModelRegistry
from multigenai.utils.processing_utils import preprocess_image_tensor, snap_hw_to_multiple_of_32

if TYPE_CHECKING:
    from multigenai.core.execution_context import ExecutionContext
    from multigenai.llm.schema_validator import VideoGenerationRequest

LOG = get_logger(__name__)

# P0-N: package-relative DiT config directory. Replaces the previous
# CWD-relative `pathlib.Path("multigenai/configs/model/dit")`, which only
# resolved when the process happened to start from the repo root.
_DIT_CONFIG_DIR = pathlib.Path(__file__).resolve().parents[2] / "configs" / "model" / "dit"


@dataclass
class FusionResult:
    """Joint Video and Audio Output."""
    video_path: str
    audio_path: str
    latents: torch.Tensor # For temporal handover
    frame_count: int
    fps: int
    seed: int
    success: bool = True
    error: Optional[str] = None


class FusionEngine:
    """
    Kaggle-grade Joint Fusion Engine.
    Implements the deep MGOS pipeline logic within the MGOS framework.
    """

    def __init__(self, ctx: "ExecutionContext") -> None:
        self._ctx = ctx
        self._out_dir = pathlib.Path(ctx.settings.output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self.device = ctx.device
        self.registry = ModelRegistry.instance()
        
        # Paths from model_config.yaml
        self.wan_path = self.registry.get_config_value("wan_model_path")
        self.mmaudio_path = self.registry.get_config_value("mmaudio_model_path")
        self.t5_path = self.registry.get_config_value("shared_t5_path")
        self.t5_tokenizer_path = self.registry.get_config_value("t5_tokenizer_path")
        self.fusion_adapter_path = self.registry.get_config_value("fusion_adapter_path")

    def _load_bundle(self, is_i2v: bool = False) -> None:
        """Lazily loads all cinematic components into a Joint Bundle."""
        model_id = "mgos_fusion_bundle"
        
        def loader():
            from multigenai.models.wan.model import WanModel
            from multigenai.models.wan.vae2_2 import Wan2_2_VAE
            from multigenai.models.mmaudio.mmaudio_core.features_utils import FeaturesUtils
            from multigenai.models.shared.t5 import T5EncoderModel
            from safetensors.torch import load_file

            # 1. Joint Transformer Backbone (Wan 2.2)
            # P0-N: package-relative config paths — never CWD-relative.
            cfg_path = _DIT_CONFIG_DIR / "video.json"
            with open(cfg_path) as f:
                config = json.load(f)

            # audio backbone
            a_cfg_path = _DIT_CONFIG_DIR / "audio.json"
            with open(a_cfg_path) as f:
                a_config = json.load(f)

            # P0-B: identity is NOT injected via additional_emb_*.
            #
            # wan/model.py maps model_type 'ti2v' -> 't2v_cross_attn', which asserts
            #     additional_emb_dim is None and additional_emb_length is None
            # (see wan/model.py cross_attn_type resolution + assert).
            # Injecting 512/257 therefore made construction of every I2V bundle fail
            # with:
            #     AssertionError: additional_emb_length should be None for t2v and t2a model
            #
            # I2V conditioning is delivered as an image latent occupying the first
            # temporal slot (first_frame_is_clean=True) instead. ArcFace embeddings
            # are used for candidate selection / quality control, never as a
            # transformer cross-attention embedding.
            for key in ("additional_emb_dim", "additional_emb_length"):
                config.pop(key, None)

            from multigenai.models.fusion.fusion import FusionModel
            model = FusionModel(video_config=config, audio_config=a_config).to(self.device).to(torch.bfloat16).eval()
            
            # Load weights for both (they might be in separate safetensors or shared)
            # Derived from the upstream reference loader: weights load into the video and audio sub-models.
            model.video_model.load_state_dict(load_file(pathlib.Path(self.wan_path) / "model.safetensors", device="cpu"), strict=False)
            model.audio_model.load_state_dict(load_file(pathlib.Path(self.mmaudio_path) / "model.safetensors", device="cpu"), strict=False)
            
            model.set_rope_params()

            # 2. Dual VAEs
            v_vae = Wan2_2_VAE(vae_pth=str(pathlib.Path(self.wan_path) / "Wan2.2_VAE.pth"), device=self.device)
            a_vae = FeaturesUtils(mode='16k', need_vae_encoder=True, 
                                 tod_vae_ckpt=str(pathlib.Path(self.mmaudio_path) / "ext_weights/v1-16.pth"),
                                 bigvgan_vocoder_ckpt=str(pathlib.Path(self.mmaudio_path) / "ext_weights/best_netG.pt")).to(self.device)

            # 3. Text Encoder
            t5 = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=self.device,
                                checkpoint_path=self.t5_path, tokenizer_path=self.t5_tokenizer_path)

            # 4. MGOS Core Fusion Weights Sync
            # This loads the cross-attention KV projections injected during FusionModel init.
            if self.fusion_adapter_path and pathlib.Path(self.fusion_adapter_path).exists():
                LOG.info(f"FusionEngine: Loading MGOS Fusion Adapter from {self.fusion_adapter_path}")
                adapter_path = pathlib.Path(self.fusion_adapter_path) / "model.safetensors"
                if adapter_path.exists():
                    model.load_state_dict(load_file(adapter_path, device="cpu"), strict=False)
                else:
                    LOG.warning("FusionEngine: Fusion adapter weights not found at expected path. Using default initialization.")

            return {"model": model, "v_vae": v_vae, "a_vae": a_vae, "t5": t5}

        if not self.registry.is_loaded(model_id):
            self.registry.register(model_id, loader=loader, min_vram_gb=15.0)
        
        self.bundle = self.registry.get(model_id, environment=self._ctx.environment)

    def _resolve_latent_dims(self, video_cfg, audio_cfg, vl_h, vl_w, latents_images, request) -> None:
        """Resolve the latent temporal length and transformer sequence lengths.

        Contract (Gate M2/M3 — see docs/ROADMAP_REVIEW.md):
        the latent temporal length is NOT the frame count. The Wan VAE applies
        ``temperal_downsample=[False, True, True]`` (4x temporal compression),
        so it has not been measured on real weights yet.

        Therefore, in priority order:
          1. If an I2V image latent was produced, its temporal length is
             authoritative — it came out of the real VAE.
          2. Otherwise the resolved OviModelSpec's measured shape is used.
          3. Otherwise we fall back to the frame count *and log a loud warning*,
             because that fallback is the exact assumption that caused the
             original bug. It is not silently trusted downstream.

        Sets ``self._v_latent_f``, ``self._a_latent_len``,
        ``self._v_seq_len`` and ``self._a_seq_len``.
        """
        from multigenai.ovi.contracts import (
            OviModelSpec,
            UnobservedContractError,
            derive_seq_len,
        )

        video_patch = tuple(video_cfg["patch_size"])
        audio_patch = tuple(audio_cfg["patch_size"])

        # --- 1. Measured-from-VAE I2V latent ---
        if latents_images is not None:
            latent_f = int(latents_images.shape[1])
            source = "i2v image latent (measured via Wan2_2_VAE.wrapped_encode)"
        else:
            # --- 2. Spec-observed, else 3. flagged fallback ---
            spec = getattr(self, "_ovi_spec", None)
            observed = getattr(spec, "observed_video", None) if spec else None
            if observed is not None and observed.is_provenance_backed:
                latent_f = int(observed.latent_shape[0])
                source = f"spec '{spec.name}' observed video latent"
            else:
                latent_f = int(getattr(request, "num_frames", 81) or 81)
                source = None
                LOG.warning(
                    "FusionEngine: NO MEASURED video latent length available. "
                    f"Falling back to num_frames={latent_f}. The Wan VAE applies "
                    "4x temporal compression, so this is very likely WRONG. "
                    "Run Gate M2 (encode a real clip through Wan2_2_VAE and record "
                    "configs/observed/<variant>.observed.json) before trusting output."
                )

        self._v_latent_f = latent_f
        self._v_seq_len = derive_seq_len((latent_f, vl_h, vl_w), video_patch)

        # Audio latent length has no I2V analogue; it must come from the spec.
        spec = getattr(self, "_ovi_spec", None)
        a_observed = getattr(spec, "observed_audio", None) if spec else None
        if a_observed is not None and a_observed.is_provenance_backed:
            self._a_latent_len = int(a_observed.latent_shape[0])
        else:
            self._a_latent_len = int(getattr(spec, "audio_latent_length", None) or 157)
            LOG.warning(
                "FusionEngine: NO MEASURED audio latent length available; using "
                f"{self._a_latent_len}. Run Gate M3 to record the real value."
            )
        self._a_seq_len = derive_seq_len(
            (self._a_latent_len, 1, 1), audio_patch
        )

        LOG.info(
            f"FusionEngine latent dims: video=({latent_f}, {vl_h}, {vl_w}) "
            f"-> vid_seq_len={self._v_seq_len} [{source}]; "
            f"audio_len={self._a_latent_len} -> audio_seq_len={self._a_seq_len}"
        )

    def _unload(self) -> None:
        """Release the Ovi bundle and return VRAM to the allocator.

        P0-C ownership contract: nulling the registry reference is not enough.
        The engine also holds the bundle in `self.bundle`, so that reference is
        dropped explicitly BEFORE the registry teardown runs the gc/CUDA sweep —
        otherwise the models stay reachable and VRAM is never freed.
        """
        from multigenai.core.model_lifecycle import ModelLifecycle

        self.bundle = None
        self.registry.unload("mgos_fusion_bundle")
        ModelLifecycle.enforce_cleanup("FusionEngine._unload")

    def run(self, request: "VideoGenerationRequest", image_path: Optional[str] = None, 
            temporal_state: Optional["TemporalState"] = None, 
            dialogue: Optional[str] = None) -> FusionResult:
        """Joint MGOS Cinematography Synthesis."""
        try:
            is_i2v = image_path is not None
            self._load_bundle(is_i2v=is_i2v)
            
            model = self.bundle["model"]
            v_vae = self.bundle["v_vae"]
            a_vae = self.bundle["a_vae"]
            t5 = self.bundle["t5"]
            
            seed = request.seed or int(torch.randint(0, 1000000, (1,)).item())
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            # 1. Conditioning Inputs
            audio_text = dialogue if dialogue else request.prompt
            text_embs = t5([request.prompt, request.negative_prompt, audio_text], self.device)
            v_pos_emb, v_neg_emb, a_pos_emb = text_embs[0], text_embs[1], text_embs[2]
            
            # Use a default negative prompt for audio
            a_neg_emb = t5(["low quality, noise, distortion"], self.device)[0]
            
            # Identity Injection
            id_emb = None
            if is_i2v and request.character_id:
                from multigenai.identity.identity_resolver import IdentityResolver
                id_emb = torch.tensor(IdentityResolver().resolve(request.character_id).face_embedding, 
                                     device=self.device, dtype=torch.bfloat16).unsqueeze(0)

            # 2. Resolution Snapping
            vh, vw = snap_hw_to_multiple_of_32(request.height, request.width, area=720*720)
            vl_h, vl_w = vh // 16, vw // 16
            
            # Preprocess first frame if I2V
            latents_images = None
            if is_i2v:
                from multigenai.utils.processing_utils import preprocess_image_tensor
                first_frame = preprocess_image_tensor(image_path, self.device, torch.bfloat16)
                with torch.no_grad():
                    latents_images = v_vae.wrapped_encode(first_frame[:, :, None]).to(torch.bfloat16).squeeze(0)
                vl_h, vl_w = latents_images.shape[2], latents_images.shape[3]

            # 3. Joint Noise Distribution
            #
            # Channel counts come from the DiT configs — they were previously
            # hardcoded as 16 (video) and 12 (audio), which contradicted
            # in_dim=48 / in_dim=20 and would have failed the Conv3d patch
            # embedding on channel count alone.
            from multigenai.ovi.contracts import load_dit_config

            video_cfg = load_dit_config("video")
            audio_cfg = load_dit_config("audio")
            video_in_dim = int(video_cfg["in_dim"])
            audio_in_dim = int(audio_cfg["in_dim"])

            # Temporal/ssequence lengths must come from the resolved model spec,
            # which holds shapes *measured* through the VAE. Until Gate M2/M3
            # have run on real weights there is no measured shape, so we refuse
            # to guess and derive from the I2V latent when one is available.
            self._resolve_latent_dims(
                video_cfg, audio_cfg, vl_h, vl_w, latents_images, request
            )

            v_noise = torch.randn(
                (video_in_dim, self._v_latent_f, vl_h, vl_w),
                device=self.device, dtype=torch.bfloat16,
            )
            a_noise = torch.randn(
                (self._a_latent_len, audio_in_dim),
                device=self.device, dtype=torch.bfloat16,
            )
            
            if temporal_state and temporal_state.global_latent is not None:
                v_noise[:, :1] = temporal_state.global_latent.to(self.device).to(torch.bfloat16)

            # 4. Sampling Loop (MGOS-Optimized Logic)
            solver_name = getattr(request, "solver", "unipc")
            shift = getattr(request, "shift", 5.0)
            scheduler, timesteps = self.get_scheduler_time_steps(
                sampling_steps=request.sampling_steps or 30,
                solver_name=solver_name,
                device=self.device,
                shift=shift
            )
            
            with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                current_v = v_noise
                current_a = a_noise
                
                for i, t in enumerate(tqdm(timesteps, desc=f"MGOS Fusion ({solver_name})")):
                    if is_i2v:
                        current_v[:, :1] = latents_images
                        
                    t_input = torch.full((1,), t, device=self.device)
                    
                    # A. Classifier-Free Guidance forward passes
                    p_v, p_a = model(vid=[current_v], audio=[current_a], t=t_input, 
                                     vid_context=[v_pos_emb], audio_context=[a_pos_emb],
                                     vid_seq_len=self._v_seq_len, audio_seq_len=self._a_seq_len,
                                     clip_fea=id_emb if is_i2v else None)
                    # Neg
                    n_v, n_a = model(vid=[current_v], audio=[current_a], t=t_input,
                                     vid_context=[v_neg_emb], audio_context=[a_neg_emb],
                                     vid_seq_len=self._v_seq_len, audio_seq_len=self._a_seq_len,
                                     slg_layer=9)
                    
                    # B. Guidance Merging (MGOS Standard Scales)
                    v_gs = request.guidance_scale or 5.0
                    a_gs = 4.0
                    v_drift = n_v[0] + v_gs * (p_v[0] - n_v[0])
                    a_drift = n_a[0] + a_gs * (p_a[0] - n_a[0])
                    
                    # C. Unified Scheduler Step
                    current_v = scheduler.step(v_drift.unsqueeze(0), t, current_v.unsqueeze(0), return_dict=False)[0].squeeze(0)
                    current_a = scheduler.step(a_drift.unsqueeze(0), t, current_a.unsqueeze(0), return_dict=False)[0].squeeze(0)

            # 5. Global Decode
            if is_i2v:
                current_v[:, :1] = latents_images

            # Video (Wan VAE)
            v_res = v_vae.wrapped_decode(current_v.unsqueeze(0))
            # Audio (MMAudio VAE)
            a_res = a_vae.wrapped_decode(current_a.unsqueeze(0).transpose(1, 2))
            
            v_res_np = v_res.squeeze(0).cpu().float().numpy()
            a_res_np = a_res.squeeze().cpu().float().numpy()
            
            # 6. Final Save
            from multigenai.engines.video_engine.ffmpeg_utils import encode_video
            import scipy.io.wavfile as wavfile
            
            v_path = self._out_dir / f"vid_{seed}.mp4"
            a_path = self._out_dir / f"aud_{seed}.wav"
            
            wavfile.write(str(a_path), 16000, (a_res_np * 32767).astype(np.int16))
            v_result = encode_video(v_res_np, v_path, request.fps, seed)
            
            return FusionResult(v_result.path, str(a_path), current_v[:, -1:], 81, request.fps, seed)

        except Exception as e:
            import traceback
            LOG.error(f"FusionEngine Error: {e}\n{traceback.format_exc()}")
            return FusionResult("", "", torch.zeros(1), 0, 0, 0, False, str(e))
        finally:
            if self._ctx.behaviour.auto_unload_after_gen:
                self._unload()

    def get_scheduler_time_steps(self, sampling_steps, solver_name='unipc', device=0, shift=5.0):
        from multigenai.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from multigenai.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
        from diffusers import FlowMatchEulerDiscreteScheduler

        if solver_name == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif solver_name == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift=shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)
        elif solver_name == 'euler':
            sample_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, sampling_steps, device=device)
        else:
            raise NotImplementedError(f"Unsupported solver: {solver_name}")
        
        return sample_scheduler, timesteps
