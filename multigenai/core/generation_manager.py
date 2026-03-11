"""
GenerationManager — Orchestrates the safe, non-overlapping execution
of distinct generative engines (Image, Video, Audio).

Phase 6:  SDXL + SVD-XT strict VRAM isolation.
Phase 7:  SceneDesigner → PromptCompiler creative layer.
Phase 8:  RIFE InterpolationEngine.
Phase 9:  PromptProcessor — token-safe segmentation, no truncation.
         Multi-segment plans iterate engines per-segment and save each
         output independently to segmented_runs/{run_id}/.
"""

from __future__ import annotations

import gc
import pathlib
import copy
from typing import TYPE_CHECKING, List, Optional

from multigenai.core.logging.logger import get_logger

if TYPE_CHECKING:
    from multigenai.llm.schema_validator import (
        ImageGenerationRequest,
        VideoGenerationRequest,
        AudioGenerationRequest,
        DocumentGenerationRequest,
        CodeGenerationRequest
    )
    from multigenai.engines.image_engine.engine import ImageResult
    from multigenai.engines.video_engine.engine import VideoResult
    from multigenai.engines.audio_engine.engine import AudioResult
    from multigenai.engines.document_engine.engine import DocumentResult
    from multigenai.engines.presentation_engine.engine import PresentationResult
    from multigenai.engines.code_engine.engine import CodeResult
    from multigenai.prompting.prompt_plan import PromptPlan

LOG = get_logger(__name__)


class GenerationManager:
    """
    Central orchestrator for engine workflows.

    Phase 9 addition:
      All image and video flows now run the prompt through PromptProcessor
      BEFORE the creative layer.  For short prompts this is a zero-overhead
      pass-through.  For long scripts the processor returns multiple
      PromptSegments, each of which is generated independently.

    Multi-segment output layout:
      {output_dir}/segmented_runs/{run_id}/segment_{N:03d}.{ext}
    """

    def __init__(self, ctx) -> None:
        self._ctx = ctx
        from multigenai.engines.image_engine.engine import ImageEngine
        self.image_engine = ImageEngine(ctx)

    # ------------------------------------------------------------------
    # Phase 9 helper — build PromptProcessor from context settings
    # ------------------------------------------------------------------
    def _build_processor(self, model_name: str = "sdxl-base"):
        """Lazily construct a PromptProcessor from application settings."""
        from multigenai.prompting.prompt_processor import PromptProcessor
        return PromptProcessor.from_settings(self._ctx.settings, model_name=model_name)

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------

    def generate_image(self, request: "ImageGenerationRequest") -> "ImageResult":
        """
        Orchestrate SDXL image generation with Phase 9 prompt processing.

        Enforces single-segment generation for consistency and leverages
        SceneMemory for character identity persistence across sessions.
        """
        LOG.info("GenerationManager: Preparing ImageEngine pipeline (SDXL)...")
        from multigenai.creative.scene_designer import SceneDesigner
        from multigenai.creative.prompt_compiler import PromptCompiler
        from multigenai.core.model_lifecycle import ModelLifecycle
        from PIL import Image

        # --- Phase 9/12: process prompt (force single segment for images) ---
        processor = self._build_processor(model_name=request.model_name)
        plan = processor.process(
            prompt=request.prompt,
            negative_prompt=getattr(request, "negative_prompt", ""),
            model_name=request.model_name,
            force_single_segment=True,
        )

        original_unload = self._ctx.behaviour.auto_unload_after_gen
        self._ctx.behaviour.auto_unload_after_gen = False  # Prevent unload during loop iteration

        # Always start with a fresh memory if explicitly requested, otherwise reuse for consistency
        # In this implementation, we assume one call = one coherent task.
        # However, we don't reset() here to allow cross-call identity if pre-loaded.
        
        results: List["ImageResult"] = []
        seg_dir = self._segmented_dir(plan.run_id) if plan.is_multi_segment else None
        
        engine = self.image_engine
        
        try:
            for seg in plan.segments:
                scene_state = self._ctx.scene_memory.get()
                
                # Build a per-segment request with the processed prompts
                seg_request = request.model_copy(update={"prompt": seg.positive})

                scene = SceneDesigner().design(seg_request)
                compiled_pos, _ = PromptCompiler().compile(scene, request.model_name)
                # Use the processor's negative (already token-safe)
                compiled_neg = seg.negative

                # strict enforcement again because PromptCompiler may have appended _QUALITY_TOKENS
                compiled_pos = processor._budget_mgr.trim_positive(compiled_pos)

                try:
                    ref_img_obj = None
                    if scene_state.character_reference_path:
                        with Image.open(scene_state.character_reference_path) as img:
                            ref_img_obj = img.copy().convert("RGB")
                            
                    ctrl_img_obj = None
                    if scene_state.reference_frame_path:
                        with Image.open(scene_state.reference_frame_path) as img:
                            ctrl_img_obj = img.copy().convert("RGB")

                    result = engine.run(
                        compiled_pos, 
                        compiled_neg, 
                        seg_request,
                        ref_image=ref_img_obj,
                        control_image=ctrl_img_obj
                    )

                    if result.success:
                        if seg_dir:
                            result = self._relocate_result(result, seg_dir, seg.index, "png")
                        
                        # Phase 12: Capture identity for future consistency if not already anchored
                        if scene_state.character_reference_path is None:
                            LOG.info("GenerationManager: Capturing first image as Character Reference.")
                            self._ctx.scene_memory.update(character_reference_path=result.path)
                        
                        # Update structural reference
                        self._ctx.scene_memory.update(reference_frame_path=result.path)

                    results.append(result)
                except Exception as exc:
                    LOG.error(f"GenerationManager: Image segment {seg.index} failed: {exc}", exc_info=True)
        finally:
            self._ctx.behaviour.auto_unload_after_gen = original_unload
            if engine and original_unload:
                ModelLifecycle.safe_unload(engine)

        # Return the first successful result
        for r in results:
            if r.success:
                return r
        return results[-1] if results else self._image_fail("no segments generated")

    # ------------------------------------------------------------------
    # Video generation
    # ------------------------------------------------------------------

    def generate_video(
        self,
        request: "VideoGenerationRequest",
        conditioning_image_path: Optional[str] = None,
        character_reference_path: Optional[str] = None,
    ) -> "VideoResult":
        """
        Orchestrate SVD-XT/AnimateDiff video generation with Phase 13 architecture.

        Phase 13 Architecture Contract:
        1. Scene Planning: Use ScenePlanner to split narrative into scenes.
        2. Anchor Generation: Use SDXL (ImageEngine) to generate explicit Character
           and Environment anchors.
        3. Strict Isolation: Hard unload SDXL, gc.collect(), IPC collect. VRAM < 2GB.
        4. Video Generation: Boot VideoEngine (AnimateDiff) and iterate over scenes.
        """
        from multigenai.core.model_lifecycle import ModelLifecycle
        from multigenai.core.temporal_state import TemporalState
        from multigenai.llm.scene_planner import ScenePlanner
        from multigenai.prompting.prompt_analyzer import PromptAnalyzer
        from multigenai.llm.schema_validator import ImageGenerationRequest
        from PIL import Image
        import copy
        import torch
        import gc

        LOG.info("GenerationManager: Starting Phase 13 Video Generation pipeline.")

        # --- Phase 13: Scene Planning ---
        planner = ScenePlanner(getattr(self._ctx, "llm", None))
        video_plan = planner.plan(request.prompt)
        seg_dir = None # Prevent undefined variable during fallback
        
        # Reset scene memory at the start of a generation plan
        self._ctx.scene_memory.reset()

        if character_reference_path:
            LOG.info(f"GenerationManager: Loading user character reference from {character_reference_path}")
            self._ctx.scene_memory.update(character_reference_path=character_reference_path)

        # STEP 1: ANCHOR GENERATION (SDXL)
        structure = PromptAnalyzer().analyze(request.prompt)
        
        # Build anchor prompts
        char_subject = structure.subjects[0] if structure.subjects else request.prompt
        char_prompt = f"cinematic portrait, {char_subject}, highly detailed, 8k resolution"
        
        env_desc = ", ".join(structure.environment) if structure.environment else "cinematic background"
        env_style = ", ".join(structure.style) if structure.style else "photorealistic"
        env_prompt = f"wide establishing shot, {env_desc}, {env_style}, highly detailed"

        if conditioning_image_path:
            # If user provided a starting image, we use it directly as the environment/starting anchor
            LOG.info("GenerationManager: Using provided conditioning image as environment anchor.")
            self._ctx.scene_memory.update(reference_frame_path=conditioning_image_path)
        else:
            LOG.info("GenerationManager: Booting ImageEngine for Anchor Generation...")
            
            original_unload = self._ctx.behaviour.auto_unload_after_gen
            self._ctx.behaviour.auto_unload_after_gen = False
            
            image_engine = self.image_engine
            
            try:
                # 1A. Character Anchor
                if not character_reference_path:
                    LOG.info(f"GenerationManager: Generating Character Anchor: {char_prompt}")
                    char_req = ImageGenerationRequest(
                        prompt=char_prompt, width=request.width, height=request.height, seed=request.seed
                    )
                    char_res = image_engine.run(char_prompt, getattr(request, "negative_prompt", ""), char_req)
                    if char_res.success:
                        self._ctx.scene_memory.update(character_reference_path=char_res.path)
                    else:
                        LOG.error("GenerationManager: Character Anchor generation failed.")
                        return self._video_fail(request, char_res.error)

                # 1B. Environment Anchor
                LOG.info(f"GenerationManager: Generating Environment Anchor: {env_prompt}")
                env_req = ImageGenerationRequest(
                    prompt=env_prompt, width=request.width, height=request.height, seed=request.seed
                )
                env_res = image_engine.run(env_prompt, getattr(request, "negative_prompt", ""), env_req)
                if env_res.success:
                    self._ctx.scene_memory.update(reference_frame_path=env_res.path, environment_prompt=env_prompt)
                else:
                    LOG.error("GenerationManager: Environment Anchor generation failed.")
                    return self._video_fail(request, env_res.error)

                # Phase 14: Extract Identity Latent while ImageEngine is hot
                char_path = self._ctx.scene_memory.get().character_reference_path
                extracted_identity_latent = None
                if char_path is not None:
                    try:
                        char_img = Image.open(char_path).convert("RGB")
                        from multigenai.identity.identity_latent_encoder import IdentityLatentEncoder
                        encoder = IdentityLatentEncoder()
                        id_latent = encoder.encode(image_engine.pipe, char_img)
                        extracted_identity_latent = id_latent.detach().cpu().clone()
                        LOG.info("GenerationManager: Successfully extracted ILC Character Latent.")
                    except Exception as e:
                        LOG.warning(f"GenerationManager: Failed to extract identity latent: {e}")

            finally:
                self._ctx.behaviour.auto_unload_after_gen = original_unload
                
        # --- HARD UNLOAD PROTOCOL (PHASE 13) ---
        LOG.info("GenerationManager: Executing Hard Unload of ImageEngine to free VRAM...")
        if hasattr(self, "image_engine") and self.image_engine:
            ModelLifecycle.safe_unload(self.image_engine.pipe)
            # Nullify internal dict reference if generated dynamically, or clear explicitly
            self.image_engine.pipe = None

        if hasattr(self, "_engines") and "image" in self._engines:
            del self._engines["image"]

        ModelLifecycle.enforce_cleanup("GenerationManager (ImageEngine > VideoEngine)")
        if torch.cuda.is_available():
            LOG.info(f"GenerationManager VRAM Log (Pre-VideoBoot): {torch.cuda.memory_reserved()/1024**2:.0f} MB")

        # STEP 2: VIDEO GENERATION (AnimateDiff)
        LOG.info("GenerationManager: Booting isolated VideoEngine (AnimateDiff) for scenes...")
        from multigenai.engines.video_engine.engine import VideoEngine
        
        original_unload = self._ctx.behaviour.auto_unload_after_gen
        self._ctx.behaviour.auto_unload_after_gen = False
        
        video_engine = VideoEngine(self._ctx)
        temporal_state = TemporalState()
        
        # Inject Identity Latent
        if 'extracted_identity_latent' in locals() and extracted_identity_latent is not None:
            temporal_state.identity_latent = extracted_identity_latent
            
        seg_frames = []
        processor = self._build_processor(model_name="animatediff")
        total_frame_count = 0  # Phase 16: Track sequence length for safety reset
        
        try:
            for scene in video_plan.scenes:
                scene_state = self._ctx.scene_memory.get()

                # Enrich scene description with environment anchor
                raw_prompt = scene.description
                if scene_state.environment_prompt and scene_state.environment_prompt not in raw_prompt:
                    raw_prompt += ", " + scene_state.environment_prompt

                # Phase 14: Token Safety
                seg_plan = processor.process(raw_prompt, force_single_segment=True)
                seg_prompt = seg_plan.segments[0].positive if seg_plan.segments else raw_prompt

                # Phase 14: Seed differentiation
                scene_seed = request.seed + temporal_state.scene_index if request.seed is not None else None
                seg_request = request.model_copy(update={"prompt": seg_prompt, "seed": scene_seed})

                # Phase 15: Keyframe Anchor Strategy
                # Each scene generates a dedicated SDXL keyframe used to initialize
                # the AnimateDiff latent — this is the biggest consistency upgrade.
                keyframe_path = scene.keyframe_path or scene_state.reference_frame_path
                keyframe_latent = None

                if keyframe_path:
                    if video_engine.pipe is None:
                        video_engine._load_model()
                    
                    try:
                        with Image.open(keyframe_path) as kf_img:
                            kf_img_rgb = kf_img.copy().convert("RGB")
                        from multigenai.identity.identity_latent_encoder import IdentityLatentEncoder
                        keyframe_latent = IdentityLatentEncoder().encode(video_engine.pipe, kf_img_rgb)
                        LOG.info(f"GenerationManager: Keyframe latent extracted for scene {scene.scene_id}.")
                    except Exception as ke:
                        LOG.warning(f"GenerationManager: Keyframe encoding failed ({ke}), no latent seed.")

                # Phase 15: pull global latent forward as prior context
                if temporal_state.global_latent is not None:
                    temporal_state.previous_latent = temporal_state.global_latent

                # Extract starting frame path
                c_path = scene_state.reference_frame_path

                LOG.info(f"GenerationManager: Generating scene {scene.scene_id}: {seg_prompt}")
                frames, new_latents, out_path, seed = video_engine.generate_frames(
                    seg_request,
                    c_path,
                    temporal_state=temporal_state,
                    scene_index=temporal_state.scene_index,
                    keyframe_latent=keyframe_latent,
                )

                # Fallback object wrapping for compatibility with later stitching
                from types import SimpleNamespace
                seg_obj = SimpleNamespace(positive=seg_prompt, negative="", index=temporal_state.scene_index)
                seg_frames.append((seg_obj, frames, out_path, seed))

                # Update Temporal State
                if frames:
                    total_frame_count += len(frames)

                    # Phase 16: Long sequence safety to prevent accumulating drift
                    if total_frame_count > 600:
                        LOG.warning(f"GenerationManager: Sequence exceeded 600 frames ({total_frame_count}). Forcing keyframe flush to kill drift.")
                        temporal_state.global_latent = None
                        temporal_state.previous_latent = None
                        temporal_state.latent_velocity = None
                        total_frame_count = 0

                    last_frame = frames[-1]
                    if isinstance(last_frame, str):
                        with Image.open(last_frame) as img:
                            temporal_state.previous_frame = img.copy().convert("RGB")
                    else:
                        temporal_state.previous_frame = last_frame.copy().convert("RGB")

                    # Phase 15: persist global + previous latent for next scene
                    # VideoEngine already returns CPU-detached tensors — no .cpu() needed
                    if temporal_state.global_latent is not None:
                         temporal_state.previous_latent = new_latents  # already CPU
                         temporal_state.global_latent = temporal_state.previous_latent
                    else: 
                         # We just reset latents to kill drift, so start the chain fresh
                         temporal_state.previous_latent = new_latents
                         temporal_state.global_latent = temporal_state.previous_latent

                    temporal_state.scene_index += 1

                    ref_path = last_frame if isinstance(last_frame, str) else None
                    self._ctx.scene_memory.update(
                        reference_frame_path=ref_path,
                        temporal_state=copy.deepcopy(temporal_state)
                    )

            # Final destruction of video engine
            video_engine._unload_model()
            video_engine = None

            ModelLifecycle.enforce_cleanup("GenerationManager (VideoEngine > InterpolationEngine)")
            ModelLifecycle.assert_vram_clean(threshold_gb=2.5, context="post-VideoEngine")
            if torch.cuda.is_available():
                LOG.info(f"GenerationManager VRAM Log (Post-VideoBoot): {torch.cuda.memory_reserved()/1024**2:.0f} MB")
                    
        except Exception as exc:
            LOG.error(f"GenerationManager: Video generation failed: {exc}", exc_info=True)
            return self._video_fail(request, str(exc))
        finally:
            self._ctx.behaviour.auto_unload_after_gen = original_unload
            if video_engine:
                 video_engine._unload_model()
            
            ModelLifecycle.enforce_cleanup("GenerationManager (Video Pipeline Cleanup)")

        # STEP 3 & 4: Interpolation and mp4 Encoding
        results: List["VideoResult"] = []
        
        interp_engine = None
        if request.interpolate and request.interpolation_factor > 1:
            LOG.info(
                f"GenerationManager: Booting InterpolationEngine "
                f"(factor={request.interpolation_factor})..."
            )
            from multigenai.engines.interpolation_engine.engine import InterpolationEngine
            interp_engine = InterpolationEngine(self._ctx)
            
        try:
            # Phase 11/14: Scene Transition Blending & Global Interpolation
            # Strategy: Combine all segments into one global list, blend boundaries, then interpolate ONCE.
            if len(video_plan.scenes) > 1 and len(seg_frames) > 1:
                LOG.info("GenerationManager: Merging segments and blending boundaries...")
                global_frame_paths = []
                
                # Phase 14 Fix: Blend window must match VideoEngine overlap (4 frames)
                blend_window = 4
                from PIL import Image
                
                for i in range(len(seg_frames)):
                    seg, frames, op, s = seg_frames[i]
                    
                    if not global_frame_paths:
                        global_frame_paths = list(frames)
                    else:
                        # Perform temporal alpha-blending at the boundary
                        overlap_a_paths = global_frame_paths[-blend_window:]
                        overlap_b_paths = frames[:blend_window]
                        
                        # Load images for blending
                        img_a = []
                        for p in overlap_a_paths:
                            with Image.open(p) as img:
                                img_a.append(img.copy().convert("RGB"))
                                
                        img_b = []
                        for p in overlap_b_paths:
                            with Image.open(p) as img:
                                img_b.append(img.copy().convert("RGB"))
                        
                        from multigenai.engines.transition_engine.engine import TransitionEngine
                        blended_images = TransitionEngine.blend(img_a, img_b, window=blend_window)
                        
                        # Save blended frames to a "blends" subfolder in the first segment's temp dir
                        # This keeps them within a folder that VideoEngine.encode will clean up.
                        first_frame_path = pathlib.Path(global_frame_paths[0])
                        blend_dir = first_frame_path.parent / "blends"
                        blend_dir.mkdir(exist_ok=True)
                        
                        new_blend_paths = []
                        for idx, bimg in enumerate(blended_images):
                            bp = blend_dir / f"blend_{i}_{idx:02d}.png"
                            bimg.save(bp)
                            new_blend_paths.append(str(bp))
                        
                        # Replace the tail of A and head of B with the blended sequence
                        global_frame_paths = global_frame_paths[:-blend_window] + new_blend_paths + list(frames[blend_window:])

                # Global Interpolation (Interpolate once for consistent motion)
                if interp_engine:
                    LOG.info(f"GenerationManager: Interpolating global sequence (factor={request.interpolation_factor})...")
                    global_frame_paths = interp_engine.interpolate(global_frame_paths, request.interpolation_factor)
                
                # Treat as one giant sequence for encoding
                seg, _, out_path, seed = seg_frames[0]
                seg_result = VideoEngine.encode(
                    frames=global_frame_paths,
                    out_path=out_path,
                    fps=request.fps,
                    seed=seed,
                    requested_frames=len(global_frame_paths),
                )
                results.append(seg_result)
            else:
                # Single segment or simple linear processing
                for seg, frames, out_path, seed in seg_frames:
                    if interp_engine:
                        frames = interp_engine.interpolate(frames, request.interpolation_factor)

                    seg_result = VideoEngine.encode(
                        frames=frames,
                        out_path=out_path,
                        fps=request.fps,
                        seed=seed,
                        requested_frames=request.num_frames,
                    )
                    
                    if seg_dir is not None and seg_result.success:
                        seg_result = self._relocate_result(seg_result, seg_dir, seg.index, "mp4")
                        
                    results.append(seg_result)
                    LOG.info(
                        f"GenerationManager: Video segment {seg.index + 1}/{len(video_plan.scenes)} done. "
                        f"path={seg_result.path}"
                    )
        finally:
            if interp_engine:
                ModelLifecycle.safe_unload(interp_engine)

        # Return the first successful result (or last attempt error info)
        for r in results:
            if r.success:
                return r
        return results[-1] if results else self._video_fail(request, "no segments generated")

    def _generate_video_segment(self):
        # Deprecated: Video segment logic merged into generate_video to prevent VRAM thrashing
        pass

    # ------------------------------------------------------------------
    # Audio / Document / Presentation / Code — unchanged lifecycle
    # ------------------------------------------------------------------

    def generate_audio(self, request: "AudioGenerationRequest") -> "AudioResult":
        """Orchestrates Audio generation with strict lifecycle."""
        LOG.info("GenerationManager: Booting isolated AudioEngine...")
        from multigenai.engines.audio_engine.engine import AudioEngine
        from multigenai.core.model_lifecycle import ModelLifecycle
        try:
            engine = AudioEngine(self._ctx)
            return engine.run(request)
        finally:
            ModelLifecycle.safe_unload(engine)

    def generate_document(self, request: "DocumentGenerationRequest") -> "DocumentResult":
        """Orchestrates Document generation."""
        LOG.info("GenerationManager: Booting isolated DocumentEngine...")
        from multigenai.engines.document_engine.engine import DocumentEngine
        from multigenai.core.model_lifecycle import ModelLifecycle
        try:
            engine = DocumentEngine(self._ctx)
            return engine.run(request)
        finally:
            ModelLifecycle.safe_unload(engine)

    def generate_presentation(self, request: "DocumentGenerationRequest") -> "PresentationResult":
        """Orchestrates Presentation generation."""
        LOG.info("GenerationManager: Booting isolated PresentationEngine...")
        from multigenai.engines.presentation_engine.engine import PresentationEngine
        from multigenai.core.model_lifecycle import ModelLifecycle
        try:
            engine = PresentationEngine(self._ctx)
            return engine.run(request)
        finally:
            ModelLifecycle.safe_unload(engine)

    def generate_code(self, request: "CodeGenerationRequest") -> "CodeResult":
        """Orchestrates Code generation."""
        LOG.info("GenerationManager: Booting isolated CodeEngine...")
        from multigenai.engines.code_engine.engine import CodeEngine
        from multigenai.core.model_lifecycle import ModelLifecycle
        try:
            engine = CodeEngine(self._ctx)
            return engine.run(request.prompt)
        finally:
            ModelLifecycle.safe_unload(engine)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _segmented_dir(self, run_id: str) -> pathlib.Path:
        """Create and return the segmented-run output directory."""
        out = (
            pathlib.Path(self._ctx.settings.output_dir)
            / "segmented_runs"
            / run_id
        )
        out.mkdir(parents=True, exist_ok=True)
        LOG.info(f"GenerationManager: multi-segment output dir → {out}")
        return out

    def _relocate_result(self, result, dest_dir: pathlib.Path, index: int, ext: str):
        """
        Copy a result file into the segmented runs directory and update result.path.
        Returns the original result object with path mutated.
        """
        import shutil
        src = pathlib.Path(result.path)
        if src.exists():
            dst = dest_dir / f"segment_{index:03d}.{ext}"
            shutil.copy2(src, dst)
            # dataclass may be frozen — try attribute set, else return copy
            try:
                object.__setattr__(result, "path", str(dst))
            except (TypeError, AttributeError):
                pass  # frozen dataclass — caller will see original path
        return result

    @staticmethod
    def _image_fail(error: str):
        from multigenai.engines.image_engine.engine import ImageResult
        return ImageResult(path="", width=0, height=0, seed=0, success=False, error=error)

    @staticmethod
    def _video_fail(request, error: str):
        from multigenai.engines.video_engine.engine import VideoResult
        return VideoResult(
            path="", frame_count=0,
            fps=request.fps, seed=0,
            success=False, error=error,
        )
