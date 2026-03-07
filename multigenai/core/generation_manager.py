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

        For multi-segment plans the first successful segment's result is
        returned; all segments are generated and saved independently.
        """
        LOG.info("GenerationManager: Preparing ImageEngine pipeline (SDXL)...")
        from multigenai.creative.scene_designer import SceneDesigner
        from multigenai.creative.prompt_compiler import PromptCompiler
        from multigenai.core.model_lifecycle import ModelLifecycle

        # --- Phase 9: process prompt ---
        processor = self._build_processor(model_name=request.model_name)
        plan = processor.process(
            prompt=request.prompt,
            negative_prompt=getattr(request, "negative_prompt", ""),
            model_name=request.model_name,
        )

        original_unload = self._ctx.behaviour.auto_unload_after_gen
        self._ctx.behaviour.auto_unload_after_gen = False  # Prevent unload during loop iteration

        results: List["ImageResult"] = []
        seg_dir = self._segmented_dir(plan.run_id) if plan.is_multi_segment else None
        
        engine = self.image_engine
        
        try:
            for seg in plan.segments:
                # Build a per-segment request with the processed prompts
                seg_request = request.model_copy(update={"prompt": seg.positive})

                scene = SceneDesigner().design(seg_request)
                compiled_pos, _ = PromptCompiler().compile(scene, request.model_name)
                # Use the processor's negative (already token-safe)
                compiled_neg = seg.negative

                # strict enforcement again because PromptCompiler may have appended _QUALITY_TOKENS
                compiled_pos = processor._budget_mgr.trim_positive(compiled_pos)

                try:
                    result = engine.run(compiled_pos, compiled_neg, seg_request)

                    if seg_dir is not None and result.success:
                        result = self._relocate_result(result, seg_dir, seg.index, "png")

                    results.append(result)
                    LOG.info(
                        f"GenerationManager: Image segment {seg.index + 1}/{plan.segment_count} done. "
                        f"path={result.path}"
                    )
                except Exception as exc:
                    LOG.error(
                        f"GenerationManager: Image segment {seg.index} failed: {exc}",
                        exc_info=True,
                    )
        finally:
            self._ctx.behaviour.auto_unload_after_gen = original_unload
            if engine and original_unload:
                ModelLifecycle.safe_unload(engine)

        # Return the first successful result (or the last attempt for error info)
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
        Orchestrate SVD-XT video generation with Phase 9 prompt processing.

        Architecture contract (Phase 9):
        - PromptProcessor runs FIRST on the raw prompt.
        - Segments are grouped by pipeline stage to completely prevent VRAM thrashing:
          1. ImageEngine generates ALL conditioning frames (loaded once).
          2. VideoEngine generates ALL keyframes (loaded once).
          3. InterpolationEngine processes all output frames.
        """
        from multigenai.creative.scene_designer import SceneDesigner
        from multigenai.creative.prompt_compiler import PromptCompiler
        from multigenai.core.model_lifecycle import ModelLifecycle
        from multigenai.core.temporal_state import TemporalState
        from multigenai.engines.motion_engine.motion_estimator import MotionEstimator
        from multigenai.engines.transition_engine.engine import TransitionEngine
        from PIL import Image

        # --- Phase 9: process prompt ---
        processor = self._build_processor(model_name="svd-xt")
        plan = processor.process(
            prompt=request.prompt,
            negative_prompt=getattr(request, "negative_prompt", ""),
            model_name="svd-xt",
        )

        seg_dir = self._segmented_dir(plan.run_id) if plan.is_multi_segment else None

        # Reset scene memory at the start of a generation plan
        self._ctx.scene_memory.reset()

        if character_reference_path:
            LOG.info(f"GenerationManager: Loading user character reference from {character_reference_path}")
            ref = Image.open(character_reference_path).convert("RGB")
            self._ctx.scene_memory.update(character_reference=ref)

        # STEP 1: Generate all conditioning frames
        conditioning_paths = []
        if conditioning_image_path:
            conditioning_paths = [conditioning_image_path] * plan.segment_count
        else:
            LOG.info("GenerationManager: No initial conditioning image. Preparing ImageEngine for all segments...")
            from multigenai.llm.schema_validator import ImageGenerationRequest
            
            original_unload = self._ctx.behaviour.auto_unload_after_gen
            self._ctx.behaviour.auto_unload_after_gen = False
            
            image_engine = self.image_engine
            
            try:
                for seg in plan.segments:
                    scene_state = self._ctx.scene_memory.get()
                    
                    # Enrich segment prompt with remembered environment
                    seg_prompt = seg.positive
                    if scene_state.environment_prompt:
                        seg_prompt += ", " + scene_state.environment_prompt
                        
                    image_req = ImageGenerationRequest(
                        prompt=seg_prompt,
                        negative_prompt=seg.negative,
                        width=request.width,
                        height=request.height,
                        seed=request.seed,
                    )
                    scene = SceneDesigner().design(image_req)
                    compiled_pos, _ = PromptCompiler().compile(scene, image_req.model_name)
                    
                    # STRICT P9 token enforcement: ensure PromptCompiler additions fit
                    compiled_pos = processor._budget_mgr.trim_positive(compiled_pos)
                    
                    img_result = image_engine.run(
                        compiled_pos, 
                        seg.negative, 
                        image_req,
                        ref_image=scene_state.character_reference,
                        control_image=scene_state.reference_frame
                    )
                    
                    if not img_result.success:
                        LOG.error(f"GenerationManager: Conditioning frame failed: {img_result.error}")
                        return self._video_fail(request, img_result.error)
                    
                    # Store images in SceneMemory for successive segment consistency
                    img_obj = Image.open(img_result.path).convert("RGB")
                    
                    if scene_state.character_reference is None:
                        # FIRST FRAME IDENTITY CAPTURE
                        self._ctx.scene_memory.update(character_reference=img_obj)
                        self._ctx.scene_memory.update(environment_prompt=seg.positive)

                    # ALWAYS update spatial reference (ControlNet Depth)
                    self._ctx.scene_memory.update(reference_frame=img_obj)
                        
                    conditioning_paths.append(img_result.path)
            finally:
                self._ctx.behaviour.auto_unload_after_gen = original_unload
                if image_engine and original_unload:
                    ModelLifecycle.safe_unload(image_engine)

        # STEP 2: SVD-XT Keyframes (Load VideoEngine once)
        # CRITICAL: Unload image pipeline before video generation to free (6-8 GB) VRAM
        if self.image_engine:
            LOG.info("GenerationManager: Unloading image engine to free VRAM for video...")
            ModelLifecycle.safe_unload(self.image_engine)
            del self.image_engine
            self.image_engine = None

        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        LOG.info("GenerationManager: Booting isolated VideoEngine (SVD-XT) for all segments...")
        from multigenai.engines.video_engine.engine import VideoEngine
        
        original_unload = self._ctx.behaviour.auto_unload_after_gen
        self._ctx.behaviour.auto_unload_after_gen = False
        video_engine = None
        self.motion_estimator = MotionEstimator(device=self._ctx.device)
        motion_estimator = self.motion_estimator

        
        # Phase 11: Temporal State Tracking
        temporal_state = TemporalState()
        
        # Tuple of (segment, frames, out_path, seed)
        seg_frames = []
        try:
            video_engine = VideoEngine(self._ctx)
            for seg, c_path in zip(plan.segments, conditioning_paths):
                scene_state = self._ctx.scene_memory.get()
                
                # Enrich segment prompt with remembered environment
                seg_prompt = seg.positive
                if scene_state.environment_prompt:
                    seg_prompt += ", " + scene_state.environment_prompt
                    
                seg_request = request.model_copy(update={"prompt": seg_prompt})
                
                # Phase 11: Motion-guided warping
                if temporal_state.previous_frame is not None:
                    LOG.info(f"GenerationManager: Estimating motion for scene continuity...")
                    with Image.open(c_path) as img:
                        init_image = img.convert("RGB")
                    flow = motion_estimator.estimate(temporal_state.previous_frame, init_image)
                    if flow is not None:
                        warped_c = motion_estimator.warp_frame(temporal_state.previous_frame, flow)
                        # Save warped conditioning image to a separate file to avoid overwriting original
                        warp_path = str(pathlib.Path(c_path).with_name(f"warp_{pathlib.Path(c_path).name}"))
                        warped_c.save(warp_path)
                        c_path = warp_path
                
                frames, final_latent, out_path, seed = video_engine.generate_frames(
                    seg_request, 
                    c_path,
                    previous_latent=temporal_state.previous_latent
                )
                seg_frames.append((seg, frames, out_path, seed))
                
                # Phase 11: Update Temporal State
                if frames:
                    temporal_state.previous_frame = frames[-1]
                    # CRITICAL: detach and move to CPU to avoid VRAM leakage across scenes
                    temporal_state.previous_latent = final_latent.detach().cpu() if final_latent is not None else None
                    temporal_state.scene_index += 1
                    
                    # Push the last generated keyframe into SceneMemory to serve as the 
                    # semantic & structural 'reference_frame' for the next generated segment
                    # Use deepcopy to prevent pointer mutation issues across segments
                    self._ctx.scene_memory.update(
                        reference_frame=frames[-1],
                        temporal_state=copy.deepcopy(temporal_state)
                    )

                # Segment-level VRAM hardening: Clear cache after each segment
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
        except Exception as exc:
            LOG.error(f"GenerationManager: VideoEngine failed: {exc}", exc_info=True)
            return self._video_fail(request, str(exc))
        finally:
            self._ctx.behaviour.auto_unload_after_gen = original_unload
            if video_engine:
                if original_unload:
                    LOG.info("GenerationManager: auto-unloading VideoEngine...")
                    ModelLifecycle.safe_unload(video_engine)
                # Explicit nullification to assist GC
                video_engine = None
            
            # CRITICAL: unload RAFT after video generation loop to reclaim VRAM
            if hasattr(self, "motion_estimator") and self.motion_estimator:
                LOG.info("GenerationManager: Unloading MotionEstimator to free RAFT VRAM...")
                self.motion_estimator.unload()
                self.motion_estimator = None
            
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
            # Phase 11: Scene Transition Blending
            if plan.is_multi_segment and len(seg_frames) > 1:
                LOG.info("GenerationManager: Applying temporal alpha-blending across scene boundaries...")
                final_frames = []
                for i in range(len(seg_frames)):
                    seg, frames, out_path, seed = seg_frames[i]
                    if interp_engine:
                        frames = interp_engine.interpolate(frames, request.interpolation_factor)
                    
                    if not final_frames:
                        final_frames = frames
                    else:
                        final_frames = TransitionEngine.blend(final_frames, frames, window=4)
                
                # Treat as one giant sequence for encoding
                seg, _, out_path, seed = seg_frames[0]
                seg_result = VideoEngine.encode(
                    frames=final_frames,
                    out_path=out_path,
                    fps=request.fps,
                    seed=seed,
                    requested_frames=request.num_frames * plan.segment_count,
                )
                results.append(seg_result)
            else:
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
                        f"GenerationManager: Video segment {seg.index + 1}/{plan.segment_count} done. "
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
        return ImageResult(path="", seed=0, success=False, error=error)

    @staticmethod
    def _video_fail(request, error: str):
        from multigenai.engines.video_engine.engine import VideoResult
        return VideoResult(
            path="", frame_count=0,
            fps=request.fps, seed=0,
            success=False, error=error,
        )
