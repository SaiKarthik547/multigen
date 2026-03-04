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

    # Baseline negative prompt — heavily weighted against CGI, statues, and low-quality artifacts
    _BASE_NEGATIVE = (
        "low quality, worst quality, text, signature, watermark, blurry, "
        "distorted anatomy, extra fingers, malformed hands, overexposed, "
        "noise, jpeg artifacts, grainy, flat, ugly, oversaturated, "
        "statue, idol, static, CGI, 3D render, cartoon, artificial, "
        "lifeless, unmoving, plastic, painting, illustration, uncanney valley"
    )

    # CLIP text encoder hard limit is 77 tokens (including BOS/EOS).
    # We cap at 70 to leave headroom for the special tokens and avoid the
    # "Token indices sequence length … > 77" warning in SDXL/SVD pipelines.
    _MAX_TOKENS = 70

    def _truncate_prompt(self, text: str) -> str:
        """
        Truncate a comma-separated prompt to at most _MAX_TOKENS words.

        Uses whitespace-split word count as a fast approximation of CLIP token
        count (1 word ≈ 1–1.3 tokens for typical prompt vocabulary).
        Whole comma-delimited phrases are preserved where possible.

        A WARNING is emitted when truncation actually occurs so it shows up
        in Kaggle logs and CI output.
        """
        words = text.split()
        if len(words) <= self._MAX_TOKENS:
            return text

        # Build phrase list and drop trailing phrases until we're within budget
        phrases = [p.strip() for p in text.split(",") if p.strip()]
        kept: list[str] = []
        total = 0
        for phrase in phrases:
            phrase_words = len(phrase.split())
            if total + phrase_words > self._MAX_TOKENS:
                break
            kept.append(phrase)
            total += phrase_words

        truncated = ", ".join(kept)
        import logging
        logging.getLogger(__name__).warning(
            f"PromptCompiler: prompt truncated from {len(words)} → {total} words "
            f"(CLIP limit {self._MAX_TOKENS})."
        )
        return truncated

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
        Both are hard-capped at _MAX_TOKENS words to stay within CLIP limits.
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
        # Negative prompt (model-aware & user-aware)
        # ---------------------------------------------------------------
        negative_parts = [self._BASE_NEGATIVE]
        
        _mn = model_name.lower()
        if "sdxl" in _mn or "stable-diffusion-xl" in _mn:
            # SDXL-family additions — matches both short alias and full HF repo id
            negative_parts.append("deformed, disfigured, bad proportions")
            
        if blueprint.negative_prompt:
            negative_parts.append(blueprint.negative_prompt.strip())
            
        negative_prompt = ", ".join(negative_parts)

        # ---------------------------------------------------------------
        # Hard token cap — must be the final step before returning
        # ---------------------------------------------------------------
        positive_prompt = self._truncate_prompt(positive_prompt)
        negative_prompt = self._truncate_prompt(negative_prompt)

        return positive_prompt, negative_prompt

