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
        character_id: Optional[str] = None,
    ) -> "VideoResult":
        """
        Orchestrate Wan 2.2 Unified Cinematic Pass.
        
        Workflow:
        1. Contextual Scene Planning (using MGOS ScenePlanner).
        2. Character Identity Retrieval (via IdentityResolver).
        3. Segmented Synthesis (Wan 2.2 for 81-frame cinematic segments).
        4. Synchronized Audio Pass (MMAudio).
        5. Temporal Handover (Latent persistence across shots).
        """
        from multigenai.core.temporal_state import TemporalState
        from multigenai.llm.scene_planner import ScenePlanner
        import torch
        
        LOG.info("GenerationManager: Initiating Wan 2.2 Unified Cinematic Pass.")
        
        # 1. Planning
        planner = ScenePlanner(getattr(self._ctx, "llm", None))
        video_plan = planner.plan(request.prompt)
        
        self._ctx.scene_memory.reset()
        temporal_state = TemporalState()
        
        # 2. Engines Boot (VRAM isolated)
        # 2. Joint Fusion Engine Boot (Phase 4: Synthesis Depth)
        from multigenai.engines.fusion_engine.engine import FusionEngine
        fusion_engine = FusionEngine(self._ctx)
        
        video_segments = []
        audio_segments = []
        
        try:
            for scene in video_plan.scenes:
                LOG.info(f"GenerationManager: Joint Fusion Pass - Scene {scene.scene_id}")
                
                # Scene-specific request
                seg_request = request.model_copy(update={
                    "prompt": scene.description,
                    "character_id": character_id or request.character_id
                })
                
                # A. Deep Synchronization (Wan 2.2 + MMAudio)
                # This performs a unified score prediction for visuals and sound
                fusion_res = fusion_engine.run(
                    request=seg_request,
                    image_path=scene.keyframe_path or conditioning_image_path,
                    temporal_state=temporal_state,
                    dialogue=scene.dialogue
                )
                
                if not fusion_res.success:
                    LOG.error(f"GenerationManager: Scene {scene.scene_id} failed: {fusion_res.error}")
                    continue
                
                # B. Result Aggregation
                video_segments.append(fusion_res.video_path)
                audio_segments.append(fusion_res.audio_path)
                
                # C. Temporal Handover (Update latent for next scene consistency)
                temporal_state.global_latent = fusion_res.latents
                temporal_state.scene_index += 1
                
            # D. Final Cinematic Movie Stitching
            from multigenai.utils.movie_utils import stitch_cinematic_movie
            final_movie_path = self._ctx.settings.output_dir / f"mgos_movie_{int(torch.randint(0, 1000, (1,)).item())}.mp4"
            stitch_cinematic_movie(video_segments, audio_segments, str(final_movie_path))
            
            from multigenai.engines.video_engine.engine import VideoResult
            return VideoResult(
                path=str(final_movie_path),
                frame_count=len(video_segments) * 81,
                fps=request.fps,
                seed=request.seed or 0,
                success=True
            )
            
            # Return result wrapping the final stitched movie
            from multigenai.engines.video_engine.engine import VideoResult
            return VideoResult(
                path=str(final_movie_path),
                frame_count=len(video_segments) * 81,
                fps=request.fps,
                seed=request.seed or 0,
                success=True
            )
            
        except Exception as e:
            LOG.error(f"GenerationManager: Cinematic pass failed: {e}")
            return self._video_fail(request, str(e))

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

            # Phase 16: Ensure all temp frame directories are fully swept from disk.
            # VideoEngine cleans the direct sequence it's fed, but if InterpolationEngine
            # was used, the raw `.temp_frames_` are orphaned on disk!
            import shutil
            for _, f_list, _, _ in seg_frames:
                if f_list and isinstance(f_list[0], (str, pathlib.Path)):
                    p = pathlib.Path(f_list[0])
                    temp_dir = next((pathlib.Path(*p.parts[:i+1]) for i, part in enumerate(p.parts) if ".temp_frames_" in part or ".interpolated_frames" in part), None)
                    if temp_dir is not None:
                        try: shutil.rmtree(temp_dir)
                        except: pass

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
