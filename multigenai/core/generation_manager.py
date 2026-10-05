"""
GenerationManager — Orchestrates safe, non-overlapping engine phases.

Cinematic workflow (Ovi era)
----------------------------
Phase IMAGE   — SDXL keyframes are generated first, then the image engine is
                FULLY released (release(), not a bare safe_unload of the
                engine object) so VRAM is empty before Ovi loads.
Phase VIDEO   — the Ovi fusion bundle loads once; scene 1 is conditioned on
                its SDXL keyframe (I2V), every later scene is conditioned on
                the previous segment's decoded LAST FRAME — that image
                handoff, plus first_frame_is_clean, is the upstream Ovi
                continuation mechanism (never latent poking).
Phase ASSEMBLE— segments are stitched with short cross-fades (xfade +
                acrossfade) so scene switches read as smooth dissolves.

Engines never share VRAM: image and video phases are strictly sequential
within Kaggle free-tier limits.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, List, Optional

from multigenai.core.logging.logger import get_logger

if TYPE_CHECKING:
    from multigenai.llm.schema_validator import (
        ImageGenerationRequest,
        VideoGenerationRequest,
        AudioGenerationRequest,
        DocumentGenerationRequest,
        CodeGenerationRequest,
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
    """Central orchestrator for engine workflows."""

    def __init__(self, ctx) -> None:
        self._ctx = ctx
        from multigenai.engines.image_engine.engine import ImageEngine
        self.image_engine = ImageEngine(ctx)

    # ------------------------------------------------------------------
    # Prompt processing helper
    # ------------------------------------------------------------------
    def _build_processor(self, model_name: str = "sdxl-base"):
        from multigenai.prompting.prompt_processor import PromptProcessor
        return PromptProcessor.from_settings(self._ctx.settings, model_name=model_name)

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------

    def generate_image(self, request: "ImageGenerationRequest") -> "ImageResult":
        """Orchestrate SDXL image generation with prompt processing."""
        LOG.info("GenerationManager: Preparing ImageEngine pipeline (SDXL)...")
        from multigenai.creative.scene_designer import SceneDesigner
        from multigenai.creative.prompt_compiler import PromptCompiler

        processor = self._build_processor(model_name=request.model_name)
        plan = processor.process(
            prompt=request.prompt,
            negative_prompt=getattr(request, "negative_prompt", ""),
            model_name=request.model_name,
            force_single_segment=True,
        )

        original_unload = self._ctx.behaviour.auto_unload_after_gen
        self._ctx.behaviour.auto_unload_after_gen = False

        results: List["ImageResult"] = []
        seg_dir = self._segmented_dir(plan.run_id) if plan.is_multi_segment else None
        engine = self.image_engine

        try:
            for seg in plan.segments:
                scene_state = self._ctx.scene_memory.get()
                seg_request = request.model_copy(update={"prompt": seg.positive})

                scene = SceneDesigner().design(seg_request)
                compiled_pos, _ = PromptCompiler().compile(scene, request.model_name)
                compiled_neg = seg.negative
                compiled_pos = processor._budget_mgr.trim_positive(compiled_pos)

                try:
                    ref_img_obj = None
                    if scene_state.character_reference_path:
                        from PIL import Image
                        with Image.open(scene_state.character_reference_path) as img:
                            ref_img_obj = img.copy().convert("RGB")

                    ctrl_img_obj = None
                    if scene_state.reference_frame_path:
                        from PIL import Image
                        with Image.open(scene_state.reference_frame_path) as img:
                            ctrl_img_obj = img.copy().convert("RGB")

                    result = engine.run(
                        compiled_pos,
                        compiled_neg,
                        seg_request,
                        ref_image=ref_img_obj,
                        control_image=ctrl_img_obj,
                    )

                    if result.success:
                        if seg_dir:
                            result = self._relocate_result(result, seg_dir, seg.index, "png")

                        if scene_state.character_reference_path is None:
                            LOG.info("GenerationManager: Capturing first image as Character Reference.")
                            self._ctx.scene_memory.update(character_reference_path=result.path)
                        self._ctx.scene_memory.update(reference_frame_path=result.path)

                    results.append(result)
                except Exception as exc:
                    LOG.error(f"GenerationManager: Image segment {seg.index} failed: {exc}", exc_info=True)
        finally:
            self._ctx.behaviour.auto_unload_after_gen = original_unload
            # P0 lifecycle: safe_unload(engine) cannot reach engine.pipe — use
            # the engine's own teardown so VRAM is actually returned.
            engine.release()

        for r in results:
            if r.success:
                return r
        return results[-1] if results else self._image_fail("no segments generated")

    # ------------------------------------------------------------------
    # Video generation — sequential keyframe -> Ovi -> assembly
    # ------------------------------------------------------------------

    def _keyframe_resolution(self, width: int, height: int) -> "tuple[int, int]":
        """SDXL keyframe size matching the video aspect at ~1024 area (/64)."""
        import math

        if width == height:
            return 1024, 1024
        scale = math.sqrt((1024 * 1024) / float(width * height))
        kw = max(64, int(round(width * scale / 64)) * 64)
        kh = max(64, int(round(height * scale / 64)) * 64)
        return kw, kh

    def generate_video(
        self,
        request: "VideoGenerationRequest",
        conditioning_image_path: Optional[str] = None,
        character_id: Optional[str] = None,
        dialogue_override: Optional[str] = None,
    ) -> "VideoResult":
        """Sequential cinematic pass: keyframes -> Ovi I2AV -> crossfade assembly."""
        from multigenai.llm.scene_planner import ScenePlanner

        LOG.info("GenerationManager: Ovi cinematic pass (%s).", request.variant)

        # 1. Planning. NOTE: scene_memory is deliberately NOT reset here —
        #    resetting it killed the image -> video reference handoff.
        planner = ScenePlanner(getattr(self._ctx, "llm", None))
        video_plan = planner.plan(request.prompt)
        if not video_plan.scenes:
            return self._video_fail(request, "scene planning produced no scenes")

        # Optional caller-level speech for the first scene (CLI --dialogue).
        if dialogue_override and video_plan.scenes:
            video_plan.scenes[0].dialogue = dialogue_override

        # 2. Phase IMAGE — keyframes for scene 1 (continuation frames come
        #    from Ovi's decoded last frames for every later scene).
        first_keyframe: Optional[str] = None
        if request.generate_keyframes:
            planner_scene = video_plan.scenes[0]
            scene_state = self._ctx.scene_memory.get()
            first_keyframe = (
                planner_scene.keyframe_path
                or scene_state.reference_frame_path
                or conditioning_image_path
            )
            if first_keyframe is None:
                first_keyframe = self._generate_keyframe(request, planner_scene)

        # IMAGE ENGINE FULL TEARDOWN — before any Ovi weight is touched.
        self.image_engine.release()

        # 3. Phase VIDEO — Ovi bundle resident; sequential scenes.
        from multigenai.engines.fusion_engine.engine import FusionEngine

        fusion_engine = FusionEngine(self._ctx)
        fusion_engine.hold_bundle()  # one checkpoint load for all scenes

        video_segments: List[str] = []
        audio_segments: List[str] = []
        total_frames = 0
        out_fps = 24
        continuation_frame: Optional[str] = first_keyframe

        try:
            for idx, scene in enumerate(video_plan.scenes):
                LOG.info("GenerationManager: Ovi scene %s (%d/%d).",
                         scene.scene_id, idx + 1, len(video_plan.scenes))

                reference = scene.keyframe_path or continuation_frame
                if reference is None:
                    reference = conditioning_image_path
                if reference is not None:
                    LOG.info("GenerationManager: scene %s I2V reference -> %s",
                             scene.scene_id, reference)

                seg_request = request.model_copy(update={
                    "prompt": scene.description,
                    "character_id": character_id or request.character_id,
                })

                fusion_res = fusion_engine.run(
                    request=seg_request,
                    image_path=reference,
                    dialogue=scene.dialogue,
                    audio_description=getattr(scene, "audio_description", None),
                )

                if not fusion_res.success:
                    LOG.error("GenerationManager: scene %s failed: %s",
                              scene.scene_id, fusion_res.error)
                    continue

                video_segments.append(fusion_res.video_path)
                audio_segments.append(fusion_res.audio_path)
                total_frames += fusion_res.frame_count
                out_fps = fusion_res.fps or out_fps

                # Last-frame handoff -> next scene's I2V reference.
                if fusion_res.last_frame_path:
                    continuation_frame = fusion_res.last_frame_path
                    self._ctx.scene_memory.update(
                        reference_frame_path=continuation_frame
                    )
        finally:
            fusion_engine.release()  # Ovi weights fully out before assembly

        if not video_segments:
            return self._video_fail(request, "no scenes generated successfully")

        # 4. Phase ASSEMBLE — crossfade transitions + mux.
        output_dir = pathlib.Path(self._ctx.settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        import uuid

        final_movie_path = output_dir / f"mgos_movie_{uuid.uuid4().hex[:8]}.mp4"

        from multigenai.utils.movie_utils import stitch_cinematic_movie

        stitched = stitch_cinematic_movie(
            video_segments, audio_segments, str(final_movie_path),
            transition_seconds=0.5,
        )
        if not stitched:
            return self._video_fail(
                request, "movie assembly failed (see logs for ffmpeg errors)"
            )

        from multigenai.engines.video_engine.engine import VideoResult

        return VideoResult(
            path=str(final_movie_path),
            frame_count=total_frames,
            fps=out_fps,
            seed=request.seed or 0,
            success=True,
        )

    def _generate_keyframe(
        self, request: "VideoGenerationRequest", scene
    ) -> Optional[str]:
        """Generate the SDXL keyframe anchor for the first scene."""
        from multigenai.llm.schema_validator import ImageGenerationRequest

        prompt = scene.keyframe_prompt or scene.description
        kw, kh = self._keyframe_resolution(request.width, request.height)
        img_request = ImageGenerationRequest(
            prompt=prompt,
            negative_prompt=request.negative_prompt,
            width=kw,
            height=kh,
            seed=request.seed,
            use_refiner=False,
        )
        LOG.info("GenerationManager: generating first-scene keyframe (%dx%d).", kw, kh)
        try:
            img_res = self.generate_image(img_request)
        except Exception as exc:
            LOG.error("GenerationManager: keyframe generation failed: %s", exc)
            return None
        if not img_res.success or not img_res.path:
            LOG.error(
                "GenerationManager: keyframe generation unsuccessful: %s",
                img_res.error,
            )
            return None
        scene.keyframe_path = img_res.path
        return img_res.path

    # ------------------------------------------------------------------
    # Audio / Document / Presentation / Code — unchanged lifecycle
    # ------------------------------------------------------------------

    def generate_audio(self, request: "AudioGenerationRequest") -> "AudioResult":
        LOG.info("GenerationManager: Booting isolated AudioEngine...")
        from multigenai.engines.audio_engine.engine import AudioEngine
        from multigenai.core.model_lifecycle import ModelLifecycle
        try:
            engine = AudioEngine(self._ctx)
            return engine.run(request)
        finally:
            ModelLifecycle.safe_unload(engine)

    def generate_document(self, request: "DocumentGenerationRequest") -> "DocumentResult":
        LOG.info("GenerationManager: Booting isolated DocumentEngine...")
        from multigenai.engines.document_engine.engine import DocumentEngine
        from multigenai.core.model_lifecycle import ModelLifecycle
        try:
            engine = DocumentEngine(self._ctx)
            return engine.run(request)
        finally:
            ModelLifecycle.safe_unload(engine)

    def generate_presentation(self, request: "DocumentGenerationRequest") -> "PresentationResult":
        LOG.info("GenerationManager: Booting isolated PresentationEngine...")
        from multigenai.engines.presentation_engine.engine import PresentationEngine
        from multigenai.core.model_lifecycle import ModelLifecycle
        try:
            engine = PresentationEngine(self._ctx)
            return engine.run(request)
        finally:
            ModelLifecycle.safe_unload(engine)

    def generate_code(self, request: "CodeGenerationRequest") -> "CodeResult":
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
        out = (
            pathlib.Path(self._ctx.settings.output_dir)
            / "segmented_runs"
            / run_id
        )
        out.mkdir(parents=True, exist_ok=True)
        LOG.info(f"GenerationManager: multi-segment output dir → {out}")
        return out

    def _relocate_result(self, result, dest_dir: pathlib.Path, index: int, ext: str):
        import shutil
        src = pathlib.Path(result.path)
        if src.exists():
            dst = dest_dir / f"segment_{index:03d}.{ext}"
            shutil.copy2(src, dst)
            try:
                object.__setattr__(result, "path", str(dst))
            except (TypeError, AttributeError):
                pass
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
