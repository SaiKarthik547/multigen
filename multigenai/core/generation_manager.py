"""
GenerationManager — Orchestrates the safe, non-overlapping execution
of distinct generative engines (Image, Video, Audio).

Phase 6: Ensures `ImageEngine` (SDXL) and `VideoEngine` (SVD-XT)
never overlap in VRAM by strictly enforcing load/unload lifecycles
during sequential generation flows.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Optional

from multigenai.core.logging.logger import get_logger

if TYPE_CHECKING:
    from multigenai.llm.schema_validator import (
        ImageGenerationRequest,
        VideoGenerationRequest,
        AudioGenerationRequest,
        DocumentGenerationRequest
    )
    from multigenai.engines.image_engine.engine import ImageResult
    from multigenai.engines.video_engine.engine import VideoResult
    from multigenai.engines.audio_engine.engine import AudioResult
    from multigenai.engines.document_engine.engine import DocumentResult
    from multigenai.engines.presentation_engine.engine import PresentationResult

LOG = get_logger(__name__)


class GenerationManager:
    """
    Central orchestrator for engine workflows.
    Ensures safe, isolated model execution and memory clearance
    between different models.
    """

    def __init__(self, ctx: "ExecutionContext") -> None:
        self._ctx = ctx

    def generate_video(
        self,
        request: "VideoGenerationRequest",
        conditioning_image_path: Optional[str] = None
    ) -> "VideoResult":
        """
        Orchestrates SVD-XT video generation.

        Architecture contract (Phase 7):
        - The video prompt ALWAYS passes through SceneDesigner → PromptCompiler
          regardless of whether a conditioning image is provided externally.
        - If no conditioning_image_path is given, ImageEngine runs first (SDXL),
          is fully unloaded, then VideoEngine (SVD-XT) boots.
        - No diffusion model is ever in VRAM during another engine's run.
        """
        import torch

        # ------------------------------------------------------------------
        # Creative layer: process video prompt ALWAYS — before any engine boots
        # ------------------------------------------------------------------
        from multigenai.creative.scene_designer import SceneDesigner
        from multigenai.creative.prompt_compiler import PromptCompiler

        video_blueprint  = SceneDesigner().design_video(request)
        compiled_vid_pos, compiled_vid_neg = PromptCompiler().compile(
            video_blueprint, model_name="svd-xt"
        )
        LOG.info(f"GenerationManager: Video creative layer compiled prompt.")

        # -------------------------------------------------------------
        # STEP 1: Generate Conditioning Frame (if needed)
        # -------------------------------------------------------------
        if not conditioning_image_path:
            LOG.info("GenerationManager: No conditioning image provided. Booting ImageEngine...")
            from multigenai.engines.image_engine.engine import ImageEngine  # bypasses ModelRegistry (Phase 7)
            from multigenai.llm.schema_validator import ImageGenerationRequest

            image_req = ImageGenerationRequest(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                # Hard-override resolution to match video dimensions
                # prevents resize distortion and composition drift in SVD-XT
                width=request.width,
                height=request.height,
                seed=request.seed,
            )

            scene = SceneDesigner().design(image_req)
            compiled_pos, compiled_neg = PromptCompiler().compile(scene, image_req.model_name)

            # Important: Force auto-unload so SDXL is erased from VRAM immediately
            original_unload = self._ctx.behaviour.auto_unload_after_gen
            self._ctx.behaviour.auto_unload_after_gen = True

            image_engine = None
            try:
                image_engine = ImageEngine(self._ctx)
                img_result = image_engine.run(compiled_pos, compiled_neg, image_req)

                if not img_result.success:
                    LOG.error(f"GenerationManager: Failed to generate conditioning frame: {img_result.error}")
                    from multigenai.engines.video_engine.engine import VideoResult
                    return VideoResult(path="", frame_count=0, fps=request.fps, seed=0, success=False, error=img_result.error)

                conditioning_image_path = img_result.path
                LOG.info(f"GenerationManager: Conditioning frame ready at {conditioning_image_path}")

            finally:
                self._ctx.behaviour.auto_unload_after_gen = original_unload

                # Absolute safety flush before booting SVD-XT
                from multigenai.core.model_lifecycle import ModelLifecycle
                ModelLifecycle.safe_unload(image_engine)

        # -------------------------------------------------------------
        # STEP 2: Generate Video via VideoEngine (SVD-XT)
        # -------------------------------------------------------------
        LOG.info("GenerationManager: Booting isolated VideoEngine (SVD-XT)...")
        from multigenai.engines.video_engine.engine import VideoEngine  # bypasses ModelRegistry (Phase 7)

        video_engine = None
        try:
            video_engine = VideoEngine(self._ctx)
            video_result = video_engine.generate(request, conditioning_image_path)
            return video_result
        finally:
            from multigenai.core.model_lifecycle import ModelLifecycle
            ModelLifecycle.safe_unload(video_engine)


    def generate_image(self, request: "ImageGenerationRequest") -> "ImageResult":
        """Orchestrates SDXL image generation with strict lifecycle."""
        LOG.info("GenerationManager: Booting isolated ImageEngine (SDXL)...")
        from multigenai.engines.image_engine.engine import ImageEngine
        from multigenai.creative.scene_designer import SceneDesigner
        from multigenai.creative.prompt_compiler import PromptCompiler
        
        scene = SceneDesigner().design(request)
        compiled_pos, compiled_neg = PromptCompiler().compile(scene, request.model_name)
        
        # Override context to force unload
        original_unload = self._ctx.behaviour.auto_unload_after_gen
        self._ctx.behaviour.auto_unload_after_gen = True
        
        try:
            engine = ImageEngine(self._ctx)
            return engine.run(compiled_pos, compiled_neg, request)
        finally:
            self._ctx.behaviour.auto_unload_after_gen = original_unload
            from multigenai.core.model_lifecycle import ModelLifecycle
            ModelLifecycle.safe_unload(engine)

    def generate_audio(self, request: "AudioGenerationRequest") -> "AudioResult":
        """Orchestrates Audio generation with strict lifecycle."""
        LOG.info("GenerationManager: Booting isolated AudioEngine...")
        from multigenai.engines.audio_engine.engine import AudioEngine
        try:
            engine = AudioEngine(self._ctx)
            return engine.run(request)
        finally:
            from multigenai.core.model_lifecycle import ModelLifecycle
            ModelLifecycle.safe_unload(engine)

    def generate_document(self, request: "DocumentGenerationRequest") -> "DocumentResult":
        """Orchestrates Document generation."""
        LOG.info("GenerationManager: Booting isolated DocumentEngine...")
        from multigenai.engines.document_engine.engine import DocumentEngine
        try:
            engine = DocumentEngine(self._ctx)
            return engine.run(request)
        finally:
            from multigenai.core.model_lifecycle import ModelLifecycle
            ModelLifecycle.safe_unload(engine)

    def generate_presentation(self, request: "DocumentGenerationRequest") -> "PresentationResult":
        """Orchestrates Presentation generation."""
        LOG.info("GenerationManager: Booting isolated PresentationEngine...")
        from multigenai.engines.presentation_engine.engine import PresentationEngine
        try:
            engine = PresentationEngine(self._ctx)
            return engine.run(request)
        finally:
            from multigenai.core.model_lifecycle import ModelLifecycle
            ModelLifecycle.safe_unload(engine)
