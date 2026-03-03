"""
Prompt Compiler — Phase 7 Creative Layer

Takes a SceneBlueprint and compiles it into optimized diffusion prompts.

Strict Prompt Boundary Contract
--------------------------------
Component       | May Add                      | Must NOT Add
----------------|------------------------------|---------------------------
SceneDesigner   | semantic descriptors          | quality tokens
PromptCompiler  | structural assembly           | 8k / masterpiece (moved to
                | quality/rendering tokens      |  SceneDesigner injection)
PromptEngine    | style quality tokens          | structural phrases
EnhancementEngine| optional LLM enrichment     | duplicate tokens

This prevents token accumulation, duplicate quality terms, and structural
contamination across layers.
"""

from typing import Tuple
from multigenai.creative.scene_designer import SceneBlueprint


class PromptCompiler:
    """
    Transforms a SceneBlueprint into a concrete diffusion positive + negative prompt pair.

    Boundary rules (enforced at runtime):
    - `subject` must NOT contain quality tokens like "8k" or "masterpiece";
      those belong in the rendering_style or atmosphere fields.
    - Quality tokens are appended here and ONLY here — never by SceneDesigner.
    """

    # Quality tokens injected once, here and nowhere else in the pipeline
    _QUALITY_TOKENS = "ultra-detailed, sharp focus, masterpiece, high fidelity"

    # Baseline negative prompt — stable across all SDXL-family models
    _BASE_NEGATIVE = (
        "low quality, worst quality, text, signature, watermark, blurry, "
        "distorted anatomy, extra fingers, malformed hands, overexposed, "
        "noise, jpeg artifacts, grainy, flat, ugly, oversaturated"
    )

    def compile(self, blueprint: SceneBlueprint, model_name: str) -> Tuple[str, str]:
        """
        Compile a SceneBlueprint into (positive_prompt, negative_prompt).

        Parameters
        ----------
        blueprint:
            Output of SceneDesigner.design(). Must NOT contain quality tokens
            in its subject field (enforced below).
        model_name:
            Resolved model alias (e.g. "sdxl-base"). Used for model-specific
            negative prompt extensions.

        Returns
        -------
        (positive_prompt, negative_prompt) tuple of strings.
        """
        # ---------------------------------------------------------------
        # P8 — Boundary enforcement: structural fields must stay clean
        # ---------------------------------------------------------------
        _quality_contamination = {"8k", "masterpiece", "ultra-detailed", "high resolution"}
        subject_lower = blueprint.subject.lower()
        contaminated = [tok for tok in _quality_contamination if tok in subject_lower]
        assert not contaminated, (
            f"PromptCompiler boundary violation: subject contains quality tokens {contaminated}. "
            "Quality tokens must only be added by PromptCompiler, not SceneDesigner."
        )

        # ---------------------------------------------------------------
        # Positive prompt assembly
        # ---------------------------------------------------------------
        parts = []

        if blueprint.camera_description:
            parts.append(f"{blueprint.camera_description} of {blueprint.subject}")
        else:
            parts.append(blueprint.subject)

        if blueprint.character_details:
            parts.append(blueprint.character_details)
        if blueprint.environment:
            parts.append(blueprint.environment)
        if blueprint.lighting:
            parts.append(blueprint.lighting)
        if blueprint.atmosphere:
            parts.append(blueprint.atmosphere)
        if blueprint.rendering_style:
            parts.append(f"style of {blueprint.rendering_style}")

        # Quality tokens appended ONCE, here and only here
        parts.append(self._QUALITY_TOKENS)

        positive_prompt = ", ".join(p.strip() for p in parts if p.strip())

        # ---------------------------------------------------------------
        # Negative prompt (model-aware)
        # ---------------------------------------------------------------
        negative_prompt = self._BASE_NEGATIVE
        _mn = model_name.lower()
        if "sdxl" in _mn or "stable-diffusion-xl" in _mn:
            # SDXL-family additions — matches both short alias and full HF repo id
            negative_prompt += ", deformed, disfigured, bad proportions"

        return positive_prompt, negative_prompt
