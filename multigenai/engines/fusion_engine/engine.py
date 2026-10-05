"""
MGOS Joint Fusion Engine — Synchronized Cinematic Video & Audio Generation.

Upstream-aligned Ovi inference contract:

- ONE trained fusion checkpoint (``<ckpt_dir>/Ovi/<basename>``) loaded
  STRICTLY via :mod:`multigenai.ovi.loader` — never separate Wan/MMAudio
  weights, never a non-upstream "adapter".
- Variant drives the temporal contract: 31/157 video/audio latent frames for
  the 5s checkpoints, 61/314 for 10s (shipped with the upstream checkpoints).
- I2V conditioning = first-frame latent + ``first_frame_is_clean=True`` on
  BOTH guided and unguided passes (the Wan time-embedding is zeroed on the
  first frame's patches). Identity is NOT injected as ``clip_fea`` —
  ``FusionModel.forward`` hard-asserts ``clip_fea is None``, and ArcFace is a
  recognition embedding, not an Ovi conditioning vector.
- Separate video and audio schedulers (multistep solvers carry per-modality
  history; sharing one instance corrupts it).
- Prompt contract: ``<S>dialogue<E>`` + ``Audio: description`` (with the
  per-variant ``<AUDCAP>`` rewrite) — see :mod:`multigenai.ovi.prompt`.
- Output is native 24 FPS per the variant; frame counts come from the decoded
  tensor, never from a hardcoded 81.
"""

from __future__ import annotations

import dataclasses
import pathlib
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from multigenai.core.logging.logger import get_logger
from multigenai.core.model_registry import ModelRegistry
from multigenai.ovi.contracts import (
    ObservedShape,
    OviModelSpec,
    derive_seq_len,
)
from multigenai.ovi.prompt import build_ovi_prompt
from multigenai.utils.processing_utils import (
    preprocess_image_tensor,
    snap_hw_to_multiple_of_32,
)

if TYPE_CHECKING:
    from multigenai.core.execution_context import ExecutionContext
    from multigenai.llm.schema_validator import VideoGenerationRequest

LOG = get_logger(__name__)


@dataclass
class FusionResult:
    """Joint Video and Audio Output."""

    video_path: str
    audio_path: str
    latents: torch.Tensor          # last-frame latent (kept for diagnostics)
    frame_count: int
    fps: int
    seed: int
    success: bool = True
    error: Optional[str] = None
    #: Decoded last frame saved as PNG — the I2V continuation reference for
    #: the next scene (image handoff, not latent poking).
    last_frame_path: Optional[str] = None
    duration_seconds: float = 0.0


class FusionEngine:
    """Joint Ovi cinematography engine (upstream-contract inference)."""

    def __init__(self, ctx: "ExecutionContext") -> None:
        self._ctx = ctx
        self._out_dir = pathlib.Path(ctx.settings.output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self.device = ctx.device
        self.registry = ModelRegistry.instance()
        self.bundle = None
        self._spec: Optional[OviModelSpec] = None
        # When the orchestrator runs several scenes back-to-back it sets this
        # flag so the (expensive) checkpoint stays resident between scenes;
        # the orchestrator then calls release() explicitly after the phase.
        self._hold_bundle = False

    def hold_bundle(self) -> None:
        """Keep the loaded bundle across consecutive run() calls."""
        self._hold_bundle = True

    # ------------------------------------------------------------------
    # Loading (delegates to the single authoritative Ovi loader)
    # ------------------------------------------------------------------

    def _load_bundle(self) -> OviModelSpec:
        from multigenai.ovi.loader import OVI_BUNDLE_ID, register_ovi_bundle

        if not self.registry.is_loaded(OVI_BUNDLE_ID):
            self._spec = register_ovi_bundle(self.registry, self._ctx.environment)
        self.bundle = self.registry.get(
            OVI_BUNDLE_ID, environment=self._ctx.environment
        )
        self._spec = self.bundle.spec
        return self._spec

    # ------------------------------------------------------------------
    # Latent geometry — measured/contract-driven, never frame-count fallback
    # ------------------------------------------------------------------

    def _resolve_latent_dims(self, vl_h: int, vl_w: int) -> Tuple[int, int]:
        """Resolve transformer sequence lengths from the variant contract.

        The temporal latent lengths come from the upstream checkpoint contract
        (attached by the loader with provenance); the spatial latent size comes
        from the measured I2V image latent or from variant-snapped geometry for
        T2V.  There is deliberately NO frame-count fallback: if the contract is
        incomplete we refuse to generate rather than guess.

        Sets ``self._v_latent_f``, ``self._a_latent_len``, ``self._v_seq_len``,
        ``self._a_seq_len`` and returns ``(video_seq_len, audio_seq_len)``.
        """
        assert self._spec is not None, "bundle must be loaded before geometry"
        self._spec.assert_ready()

        self._v_latent_f = int(self._spec.video_latent_length)
        self._v_seq_len = derive_seq_len(
            (self._v_latent_f, vl_h, vl_w), tuple(self._spec.video_patch)
        )
        self._a_latent_len = int(self._spec.audio_latent_length)
        self._a_seq_len = derive_seq_len(
            (self._a_latent_len, 1, 1), tuple(self._spec.audio_patch)
        )

        LOG.info(
            "FusionEngine latent dims: video=(%d, %d, %d) -> vid_seq_len=%d; "
            "audio_len=%d -> audio_seq_len=%d [provenance: %s]",
            self._v_latent_f, vl_h, vl_w, self._v_seq_len,
            self._a_latent_len, self._a_seq_len, self._spec.latent_provenance,
        )
        return self._v_seq_len, self._a_seq_len

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def _unload(self) -> None:
        """Release the Ovi bundle and return VRAM to the allocator.

        Ownership contract: null the engine's reference BEFORE the registry
        teardown runs the gc/CUDA sweep — otherwise the models stay reachable
        and VRAM is never freed.
        """
        from multigenai.core.model_lifecycle import ModelLifecycle

        self.bundle = None
        self.registry.unload("ovi_fusion_bundle")
        ModelLifecycle.enforce_cleanup("FusionEngine._unload")

    def release(self) -> None:
        """Public teardown used by the orchestrator between engine phases."""
        if self.registry.is_loaded("ovi_fusion_bundle"):
            self._unload()
        else:
            self.bundle = None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def run(
        self,
        request: "VideoGenerationRequest",
        image_path: Optional[str] = None,
        dialogue: Optional[str] = None,
        audio_description: Optional[str] = None,
        temporal_state: Optional[object] = None,
    ) -> FusionResult:
        """Joint Ovi synthesis for one scene.

        Continuity between scenes is handled by the orchestrator through the
        image channel (previous segment's last frame becomes this segment's
        I2V reference). ``temporal_state`` is accepted for API compatibility
        but is intentionally unused: poking a latent into the first temporal
        slot without ``first_frame_is_clean`` corrupts the time embedding.
        """
        if temporal_state is not None:
            LOG.debug(
                "FusionEngine: temporal_state ignored — continuation uses the "
                "last-frame image handoff (upstream I2V mechanism)."
            )
        try:
            spec = self._load_bundle()
            bundle = self.bundle

            model = bundle.model
            v_vae = bundle.v_vae
            a_vae = bundle.a_vae
            t5 = bundle.t5

            seed = request.seed if request.seed is not None else int(
                torch.randint(0, 1_000_000, (1,)).item()
            )
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            is_i2v = image_path is not None
            target_area = spec.target_width * spec.target_height

            # 1. Prompt contract (<S>speech<E> / Audio: / <AUDCAP>)
            ovi_prompt = build_ovi_prompt(
                scene_description=request.prompt,
                dialogue=dialogue,
                audio_description=audio_description,
                video_negative_prompt=getattr(
                    request, "video_negative_prompt", ""
                ) or None,
                audio_negative_prompt=getattr(
                    request, "audio_negative_prompt", ""
                ) or None,
                audio_tag_style=spec.audio_tag_style,
            )
            final_prompt = ovi_prompt.render()
            if ovi_prompt.speech or ovi_prompt.audio_description:
                LOG.info("FusionEngine prompt: %s", final_prompt)

            # 2. Text encodings (upstream: prompt -> joint video+audio pos,
            #    then separate video/audio negatives).
            bundle.t5_to_device()
            text_embeddings = t5(
                [
                    final_prompt,
                    ovi_prompt.video_negative_prompt,
                    ovi_prompt.audio_negative_prompt,
                ],
                t5.device,
            )
            text_embeddings = [
                emb.to(torch.bfloat16).to(self.device) for emb in text_embeddings
            ]
            emb_joint_pos = text_embeddings[0]
            emb_video_neg = text_embeddings[1]
            emb_audio_neg = text_embeddings[2]
            bundle.t5_offload()

            # 3. Spatial latent geometry
            if is_i2v:
                bundle.vae_video_to_device()
                with torch.no_grad():
                    first_frame = preprocess_image_tensor(
                        image_path, self.device, torch.bfloat16,
                        resize_total_area=target_area,
                    )
                    latents_images = v_vae.wrapped_encode(
                        first_frame[:, :, None]
                    ).to(torch.bfloat16).squeeze(0)
                bundle.vae_video_offload()
                vl_h, vl_w = int(latents_images.shape[2]), int(latents_images.shape[3])
                spatial_source = "i2v image latent (measured via Wan2_2_VAE)"
            else:
                vh, vw = snap_hw_to_multiple_of_32(
                    request.height, request.width, area=target_area
                )
                vl_h, vl_w = vh // 16, vw // 16
                latents_images = None
                spatial_source = "t2v area-snapped geometry"

            # Observed video shape: upstream temporal length + measured/snapped
            # spatial dims, provenance recorded for the readiness gate.
            spec = dataclasses.replace(
                spec,
                observed_video=ObservedShape(
                    latent_shape=(int(spec.video_latent_length), vl_h, vl_w),
                    dtype="bfloat16",
                    channels=int(spec.video_in_dim),
                    source=f"{spatial_source}; temporal from {spec.latent_provenance}",
                ),
            )
            self._spec = spec
            self._resolve_latent_dims(vl_h, vl_w)

            # 4. Joint noise (channels from the DiT configs via the spec;
            #    temporal lengths from the upstream checkpoint contract).
            video_in_dim = int(spec.video_in_dim)
            audio_in_dim = int(spec.audio_in_dim)
            device = self.device if torch.cuda.is_available() else "cpu"

            v_noise = torch.randn(
                (video_in_dim, self._v_latent_f, vl_h, vl_w),
                device=device, dtype=torch.bfloat16,
                generator=torch.Generator(device=device).manual_seed(seed),
            )
            a_noise = torch.randn(
                (self._a_latent_len, audio_in_dim),
                device=device, dtype=torch.bfloat16,
                generator=torch.Generator(device=device).manual_seed(seed),
            )

            # 5. Sampling configuration (upstream defaults, variant-driven)
            solver_name = getattr(request, "solver", "unipc") or "unipc"
            shift = float(getattr(request, "shift", 5.0) or 5.0)
            sample_steps = int(
                getattr(request, "sample_steps", None)
                or getattr(request, "sampling_steps", None)
                or spec.sample_steps
            )
            video_gs = float(
                getattr(request, "video_guidance_scale", None)
                or spec.video_guidance_scale
            )
            audio_gs = float(
                getattr(request, "audio_guidance_scale", None)
                or spec.audio_guidance_scale
            )
            slg_layer = int(
                getattr(request, "slg_layer", None)
                if getattr(request, "slg_layer", None) is not None
                else spec.slg_layer
            )

            # Upstream: independent schedulers per modality — multistep
            # solvers carry state, so sharing one instance corrupts history.
            scheduler_video, timesteps_video = self.get_scheduler_time_steps(
                sampling_steps=sample_steps, solver_name=solver_name,
                device=device, shift=shift,
            )
            scheduler_audio, timesteps_audio = self.get_scheduler_time_steps(
                sampling_steps=sample_steps, solver_name=solver_name,
                device=device, shift=shift,
            )

            # 6. Sampling loop
            bundle.model_to_device()
            with torch.inference_mode(), torch.amp.autocast(
                "cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16
            ):
                for i, (t_v, t_a) in enumerate(
                    tqdm(
                        list(zip(timesteps_video, timesteps_audio)),
                        desc=f"Ovi fusion ({solver_name}, {sample_steps} steps)",
                    )
                ):
                    timestep_input = torch.full((1,), t_v, device=device)

                    if is_i2v:
                        v_noise[:, :1] = latents_images

                    # Positive (conditional) pass
                    pred_vid_pos, pred_audio_pos = model(
                        vid=[v_noise],
                        audio=[a_noise],
                        t=timestep_input,
                        vid_context=[emb_joint_pos],
                        audio_context=[emb_joint_pos],
                        vid_seq_len=self._v_seq_len,
                        audio_seq_len=self._a_seq_len,
                        first_frame_is_clean=is_i2v,
                    )
                    # Negative (unconditional) pass — SLG on this branch only.
                    pred_vid_neg, pred_audio_neg = model(
                        vid=[v_noise],
                        audio=[a_noise],
                        t=timestep_input,
                        vid_context=[emb_video_neg],
                        audio_context=[emb_audio_neg],
                        vid_seq_len=self._v_seq_len,
                        audio_seq_len=self._a_seq_len,
                        first_frame_is_clean=is_i2v,
                        slg_layer=slg_layer,
                    )

                    # Classifier-free guidance per modality
                    pred_video_guided = pred_vid_neg[0] + video_gs * (
                        pred_vid_pos[0] - pred_vid_neg[0]
                    )
                    pred_audio_guided = pred_audio_neg[0] + audio_gs * (
                        pred_audio_pos[0] - pred_audio_neg[0]
                    )

                    # Independent scheduler steps
                    v_noise = scheduler_video.step(
                        pred_video_guided.unsqueeze(0), t_v,
                        v_noise.unsqueeze(0), return_dict=False,
                    )[0].squeeze(0)
                    a_noise = scheduler_audio.step(
                        pred_audio_guided.unsqueeze(0), t_a,
                        a_noise.unsqueeze(0), return_dict=False,
                    )[0].squeeze(0)
            bundle.model_offload()

            # 7-8. Re-condition the first frame before decode (upstream does
            #      this unconditionally after the loop for I2V), then decode.
            #      NOTE: v_noise was produced INSIDE inference_mode by the
            #      scheduler step, so mutating it here requires the same mode —
            #      upstream gets this by decorating generate() wholesale.
            with torch.inference_mode():
                if is_i2v:
                    v_noise[:, :1] = latents_images

                # Decode (audio first, matching upstream ordering)
                bundle.vae_audio_to_device()
                audio_latents = a_noise.unsqueeze(0).transpose(1, 2)
                generated_audio = a_vae.wrapped_decode(audio_latents)
                generated_audio = generated_audio.squeeze().cpu().float().numpy()
                bundle.vae_audio_offload()

                bundle.vae_video_to_device()
                generated_video = v_vae.wrapped_decode(v_noise.unsqueeze(0))
                generated_video = generated_video.squeeze(0).cpu().float().numpy()
                bundle.vae_video_offload()

                # Copy the tail latent OUT of inference mode so callers get a
                # normal tensor they can freely mutate.
                with torch.inference_mode(False):
                    last_latent = v_noise[:, -1:].clone()

            frame_count = int(generated_video.shape[1])
            fps = int(spec.fps)

            # 9. Persist outputs. The tag includes a uuid because the same seed
            #    is legitimately reused across scenes (e.g. seed=42) — without
            #    it every scene would overwrite the previous segment's files.
            import scipy.io.wavfile as wavfile
            from multigenai.engines.video_engine.ffmpeg_utils import encode_video

            tag = f"{seed}_{uuid.uuid4().hex[:8]}"
            v_path = self._out_dir / f"vid_{tag}.mp4"
            a_path = self._out_dir / f"aud_{tag}.wav"
            wavfile.write(
                str(a_path), 16000,
                (np.clip(generated_audio, -1.0, 1.0) * 32767).astype(np.int16),
            )
            v_result = encode_video(
                list(generated_video.transpose(1, 2, 3, 0)), v_path, fps, seed
            )
            if not v_result.success:
                return FusionResult("", str(a_path), last_latent, 0, fps, seed,
                                    False, v_result.error)

            # 10. Last-frame extraction — the continuation reference for the
            #     next scene (image handoff, not latent poking).
            #     generated_video is (c, f, h, w) float in [-1, 1].
            last_frame_path: Optional[str] = None
            try:
                from PIL import Image

                last = generated_video[:, -1]                     # (c, h, w)
                arr = np.clip((last.transpose(1, 2, 0) + 1.0) * 127.5, 0, 255)
                lf_path = self._out_dir / f"last_frame_{tag}.png"
                Image.fromarray(arr.astype(np.uint8)).save(str(lf_path))
                last_frame_path = str(lf_path)
            except Exception as exc:  # non-fatal: video is already saved
                LOG.warning("FusionEngine: last-frame extraction failed: %s", exc)

            LOG.info(
                "FusionEngine done: %d frames @ %d fps (%.2fs) seed=%d",
                frame_count, fps, frame_count / fps if fps else 0.0, seed,
            )
            return FusionResult(
                video_path=v_result.path,
                audio_path=str(a_path),
                latents=last_latent,
                frame_count=frame_count,
                fps=fps,
                seed=seed,
                success=True,
                last_frame_path=last_frame_path,
                duration_seconds=frame_count / fps if fps else 0.0,
            )

        except Exception as e:
            import traceback

            LOG.error(
                "FusionEngine Error: %s\n%s", e, traceback.format_exc()
            )
            return FusionResult("", "", torch.zeros(1), 0, 0, 0, False, str(e))
        finally:
            if (
                not self._hold_bundle
                and self._ctx.behaviour.auto_unload_after_gen
            ):
                self._unload()

    # ------------------------------------------------------------------
    # Solvers (unchanged — upstream-identical construction)
    # ------------------------------------------------------------------

    def get_scheduler_time_steps(self, sampling_steps, solver_name="unipc", device=0, shift=5.0):
        from multigenai.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from multigenai.utils.fm_solvers import (
            FlowDPMSolverMultistepScheduler,
            get_sampling_sigmas,
            retrieve_timesteps,
        )
        from diffusers import FlowMatchEulerDiscreteScheduler

        if solver_name == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
            )
            sample_scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif solver_name == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift=shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler, device=device, sigmas=sampling_sigmas
            )
        elif solver_name == "euler":
            sample_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler, sampling_steps, device=device
            )
        else:
            raise NotImplementedError(f"Unsupported solver: {solver_name}")

        return sample_scheduler, timesteps
