"""
MGOS VideoEngine — High-performance Cinematic Video Generation using Wan 2.2.

Features:
- Powered by Wan 2.2 DiT (Diffusion Transformer).
- Natively supports Text-to-Video (T2V) and Image-to-Video (I2V).
- Deep integration with MGOS Character Identity (IdentityLayer).
- Optimized for Kaggle T4/P100 (Flash Attention, CPU Offloading, VRAM Isolation).
"""

from __future__ import annotations

import gc
import pathlib
import torch
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

from multigenai.core.exceptions import EngineExecutionError
from multigenai.core.logging.logger import get_logger
from multigenai.core.model_registry import ModelRegistry

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from multigenai.core.execution_context import ExecutionContext
    from multigenai.core.temporal_state import TemporalState
    from multigenai.llm.schema_validator import VideoGenerationRequest

LOG = get_logger(__name__)

# Global Scaling Constants for Wan 2.2
VAE_SCALE_FACTOR = 16


@dataclass
class VideoResult:
    """Output from the VideoEngine."""
    path: str
    frame_count: int
    fps: int
    seed: int
    success: bool = True
    error: Optional[str] = None


class VideoEngine:
    """
    Kaggle-grade Video Generation Engine using Wan 2.2.
    Overwrites the old AnimateDiff implementation for superior cinematic results.
    """

    def __init__(self, ctx: "ExecutionContext") -> None:
        self._ctx = ctx
        self._out_dir = pathlib.Path(ctx.settings.output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self.device = ctx.device
        self.registry = ModelRegistry.instance()
        
        # Paths from model_config.yaml
        self.wan_path = self.registry.get_config_value("wan_model_path")
        self.t5_path = self.registry.get_config_value("shared_t5_path")
        self.t5_tokenizer_path = self.registry.get_config_value("t5_tokenizer_path")

    def _load_model(self, is_i2v: bool = False) -> None:
        """
        Lazily loads Wan 2.2 model and VAE into MGOS ModelRegistry.
        """
        model_id = "wan2_2_i2v" if is_i2v else "wan2_2_t2v"
        
        def loader():
            from multigenai.models.wan.model import WanModel
            from multigenai.models.wan.vae2_2 import Wan2_2_VAE
            from multigenai.models.shared.t5 import T5EncoderModel
            import json
            import os
            from safetensors.torch import load_file

            # 1. Load DiT Backbone
            config_path = pathlib.Path(self.wan_path) / ("video_i2v.json" if is_i2v else "video_t2v.json")
            with open(config_path) as f:
                config = json.load(f)
            
            # Injection Point: MGOS Character Identity (512-d ArcFace)
            if is_i2v:
                config['additional_emb_dim'] = 512
                config['additional_emb_length'] = 257  # Matches multigenai CLIP proxy length
            
            model = WanModel(**config).to(self.device).to(torch.bfloat16).eval()
            
            # Load weights
            ckpt_path = pathlib.Path(self.wan_path) / "model.safetensors"
            state_dict = load_file(str(ckpt_path), device="cpu")
            model.load_state_dict(state_dict, strict=False)
            del state_dict

            # 2. VAE
            vae = Wan2_2_VAE(vae_pth=str(pathlib.Path(self.wan_path) / "Wan2.2_VAE.pth"), device=self.device)
            
            # 3. Text Encoder (T5)
            text_encoder = T5EncoderModel(
                text_len=512,
                dtype=torch.bfloat16,
                device=self.device,
                checkpoint_path=self.t5_path,
                tokenizer_path=self.t5_tokenizer_path
            )

            return {"dit": model, "vae": vae, "t5": text_encoder}

        # Register and get from MGOS lifecycle manager (Phase 13 VRAM isolation)
        if not self.registry.is_loaded(model_id):
            self.registry.register(model_id, loader=loader, min_vram_gb=14.0)
        
        self.bundle = self.registry.get(model_id, environment=self._ctx.environment)

    def _unload_model(self) -> None:
        """Flushes VRAM via MGOS Registry."""
        self.registry.unload("wan2_2_i2v")
        self.registry.unload("wan2_2_t2v")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _map_identity_to_wan(self, character_id: str) -> Optional[torch.Tensor]:
        """
        Hooks into MgosIdentityResolver to fetch face embeddings.
        Converts MGOS identity into Wan-compatible cross-attention seeds.
        """
        if not character_id:
            return None
            
        from multigenai.identity.identity_resolver import IdentityResolver
        resolver = IdentityResolver()
        identity = resolver.resolve(character_id)
        
        if identity and identity.face_embedding:
            # ArcFace 512-d vector
            emb = torch.tensor(identity.face_embedding, device=self.device, dtype=torch.bfloat16)
            return emb.unsqueeze(0)  # [1, 512]
        return None

    def generate_frames(
        self,
        request: "VideoGenerationRequest",
        conditioning_image_path: Optional[str],
        temporal_state: "TemporalState",
        scene_index: int = 0
    ) -> Tuple[List["PILImage"], Optional[torch.Tensor], pathlib.Path, int]:
        """
        Wan 2.2 Frame Generation Pass.
        Supports both T2V and I2V (via conditioning_image_path).
        """
        is_i2v = conditioning_image_path is not None
        self._load_model(is_i2v=is_i2v)
        
        dit = self.bundle["dit"]
        vae = self.bundle["vae"]
        t5 = self.bundle["t5"]
        
        seed = request.seed if request.seed is not None else int(torch.randint(0, 1000000, (1,)).item())
        torch.manual_seed(seed)
        
        # 1. Identity Injection Pass (Wan Heart)
        identity_embeds = self._map_identity_to_wan(request.character_id)
        
        # 2. Text Embeddings
        prompts = [request.prompt, request.negative_prompt]
        text_embs = t5(prompts, self.device)
        pos_emb, neg_emb = text_embs[0], text_embs[1]
        
        # 3. Latent Initialization (Phase 15 Memory Reuse)
        width, height = request.width, request.height
        frames = 81  # Wan 2.2 standard cinematic sequence length
        
        lat_h, lat_w = height // VAE_SCALE_FACTOR, width // VAE_SCALE_FACTOR
        noise = torch.randn((16, frames, lat_h, lat_w), device=self.device, dtype=torch.bfloat16)
        
        # I2V Image Latents
        if is_i2v:
            from PIL import Image
            from multigenai.utils.processing_utils import preprocess_image_tensor
            img = Image.open(conditioning_image_path).convert("RGB")
            img_tensor = preprocess_image_tensor(img, self.device, torch.bfloat16)
            with torch.no_grad():
                img_latents = vae.wrapped_encode(img_tensor[:, :, None]).squeeze(0)
            noise[:, 0] = img_latents  # First frame anchoring
            
        # 4. Sampling Loop (Wan 2.2 Optimized)
        from multigenai.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from multigenai.utils.fm_solvers import retrieve_timesteps
        
        sampling_steps = request.sampling_steps or 30
        shift = getattr(request, "shift", 5.0)
        
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
        timesteps = scheduler.timesteps
        
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            current_latents = noise
            
            for i, t in enumerate(timesteps):
                if is_i2v:
                    current_latents[:, 0] = img_latents
                
                t_input = torch.full((1,), t, device=self.device)
                
                # Forward Pass (Positive)
                p_out = dit(vid=[current_latents], t=t_input, context=[pos_emb], clip_fea=identity_embeds)[0]
                
                # Forward Pass (Negative / SLG)
                n_out = dit(vid=[current_latents], t=t_input, context=[neg_emb], slg_layer=9)[0]
                
                # Guidance Merging
                gs = request.guidance_scale or 5.0
                model_output = n_out + gs * (p_out - n_out)
                
                # Scheduler Step
                current_latents = scheduler.step(model_output.unsqueeze(0), t, current_latents.unsqueeze(0), return_dict=False)[0].squeeze(0)
            
            sampled_latents = current_latents
            if is_i2v:
                sampled_latents[:, 0] = img_latents

            # 5. Decode
            LOG.info(f"VideoEngine: Decoding {frames} frames via Wan VAE...")
            final_frames_tensor = vae.wrapped_decode(sampled_latents.unsqueeze(0))
            final_frames_np = final_frames_tensor.squeeze(0).cpu().float().numpy() # [C, F, H, W]
            
        # Convert to PIL
        pil_frames = []
        for f in range(frames):
            frame = final_frames_np[:, f, :, :]
            frame = ((frame + 1.0) * 127.5).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
            pil_frames.append(Image.fromarray(frame))
            
        out_path = self._out_dir / f"output_{seed}.mp4"
        return pil_frames, sampled_latents.cpu(), out_path, seed

    def generate(self, request: "VideoGenerationRequest", conditioning_image_path: str) -> "VideoResult":
        """Main entry point called by GenerationManager."""
        try:
            frames, _, out_path, seed = self.generate_frames(request, conditioning_image_path, None)
            
            # Encode via ffmpeg
            from multigenai.engines.video_engine.ffmpeg_utils import encode_video
            result = encode_video(frames, out_path, request.fps, seed)
            
            if self._ctx.settings.video.auto_unload:
                self._unload_model()
                
            return result
        except Exception as e:
            LOG.error(f"VideoEngine Error: {e}")
            return VideoResult("", 0, 0, 0, False, str(e))
