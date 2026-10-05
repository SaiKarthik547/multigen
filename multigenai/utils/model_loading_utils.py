"""Upstream-derived Ovi loading helpers.

This module is the reference implementation for Ovi model construction and
strict checkpoint loading. Engines should delegate here rather than growing a
second, divergent loader.

Repairs applied:
  * ``multigenai.modules.*`` imports pointed at a package layout that no longer
    exists; the modules live under ``multigenai.models.*``.
  * DiT config paths were CWD-relative (``"multigenai/configs/..."``) and only
    resolved when the process started from the repo root. They are now loaded
    package-relatively via :func:`multigenai.ovi.contracts.load_dit_config`.
"""
import torch
import os
import json
import pathlib

from safetensors.torch import load_file

from multigenai.models.fusion import FusionModel
from multigenai.models.shared.t5 import T5EncoderModel
from multigenai.models.wan.vae2_2 import Wan2_2_VAE
from multigenai.models.mmaudio.mmaudio_core.features_utils import FeaturesUtils
    
def init_wan_vae_2_2(ckpt_dir, rank=0):
    vae_config = {}
    vae_config['device'] = rank
    vae_pth = os.path.join(ckpt_dir, "Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    vae_config['vae_pth'] = vae_pth
    vae_model = Wan2_2_VAE(**vae_config)

    return vae_model

def init_mmaudio_vae(ckpt_dir, rank=0):
    vae_config = {}
    vae_config['mode'] = '16k'
    vae_config['need_vae_encoder'] = True

    tod_vae_ckpt = os.path.join(ckpt_dir, "MMAudio/ext_weights/v1-16.pth")
    bigvgan_vocoder_ckpt = os.path.join(ckpt_dir, "MMAudio/ext_weights/best_netG.pt")

    vae_config['tod_vae_ckpt'] = tod_vae_ckpt
    vae_config['bigvgan_vocoder_ckpt'] = bigvgan_vocoder_ckpt

    vae = FeaturesUtils(**vae_config).to(rank)

    return vae

def init_fusion_score_model_multigenai(rank: int = 0, meta_init=False):
    # Package-relative config loading (no CWD dependence).
    from multigenai.ovi.contracts import load_dit_config

    video_config = load_dit_config("video")
    audio_config = load_dit_config("audio")

    if meta_init:
        with torch.device("meta"):
            fusion_model = FusionModel(video_config, audio_config)
    else:
        fusion_model = FusionModel(video_config, audio_config)
    
    params_all = sum(p.numel() for p in fusion_model.parameters())
    
    if rank == 0:
        print(
            f"Score model (Fusion) all parameters:{params_all}"
        )

    return fusion_model, video_config, audio_config

def init_text_model(ckpt_dir, rank):
    wan_dir = os.path.join(ckpt_dir, "Wan2.2-TI2V-5B")
    text_encoder_path = os.path.join(wan_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    text_tokenizer_path = os.path.join(wan_dir, "google/umt5-xxl")

    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=rank,
        checkpoint_path=text_encoder_path,
        tokenizer_path=text_tokenizer_path,
        shard_fn=None)


    return text_encoder


def load_fusion_checkpoint(model, checkpoint_path, from_meta=False):
    """Load the Ovi fusion checkpoint STRICTLY (upstream contract).

    strict=True is load-bearing: FusionModel adds cross-modal fusion
    projections (k_fusion / v_fusion / pre_attn_norm_fusion / norm_k_fusion)
    that exist ONLY in a trained Ovi checkpoint. Silently skipping missing
    keys (strict=False) would run a model whose fusion path was randomly
    initialised — a wrong-video/wrong-audio result with no error.
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        if checkpoint_path.endswith(".safetensors"): 
            df = load_file(checkpoint_path, device="cpu")
        elif checkpoint_path.endswith(".pt"):
            try:
                df = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                df = df['module'] if 'module' in df else df
            except Exception:
                df = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                df = df['app']['model']
        else: 
            raise RuntimeError("We only support .safetensors and .pt checkpoints")

        missing, unexpected = model.load_state_dict(df, strict=True, assign=from_meta)

        del df
        import gc
        gc.collect()
        print(f"Successfully loaded fusion checkpoint from {checkpoint_path}")
    else: 
        raise RuntimeError(f"{checkpoint_path=} does not exist")
