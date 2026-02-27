"""
PromptEngine — Validates, enriches, and structures generation prompts.

Pipeline:
  User Prompt → Schema Validation → Style Injection → Enhancement
  → Identity Token Stripping (when identity active) → Negative Prompt Enforcement
  → Token Estimation → EnhancedPrompt

Phase 2 will connect EnhancementEngine to a real LLM (Gemini/Mistral).
Currently uses a rule-based quality token injector.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional

from multigenai.core.exceptions import InvalidPromptError
from multigenai.core.logging.logger import get_logger
from multigenai.llm.enhancement_engine import EnhancementEngine
from multigenai.llm.schema_validator import (
    EnhancedPrompt,
    ImageGenerationRequest,
)

if TYPE_CHECKING:
    from multigenai.memory.style_registry import StyleProfile, StyleRegistry

LOG = get_logger(__name__)

# Exact comma-delimited tokens that conflict with IP-Adapter identity conditioning.
# These are removed only when they appear as complete, standalone fragments in the
# prompt — never as substrings of unrelated phrases (e.g. "blue glowing energy" is safe).
_IDENTITY_CONFLICT_TOKENS: List[str] = [
    "blue eyes", "green eyes", "brown eyes", "hazel eyes", "grey eyes",
    "red hair", "blonde hair", "black hair", "white hair", "silver hair",
    "brown hair", "curly hair", "straight hair",
    "dark skin", "pale skin", "light skin", "olive skin",
    "freckles", "random face", "anonymous face", "unknown person",
    "different face", "varied appearance",
]


def _strip_identity_conflict_tokens(prompt: str, tokens: List[str]) -> tuple[str, int]:
    """
    Remove exact identity-conflicting tokens from a comma-delimited prompt.

    A token is removed only if it matches a complete comma-separated segment
    (after whitespace normalisation). Never performs substring deletion.

    Returns:
        (cleaned_prompt, n_removed)
    """
    # Split on commas, preserving order
    segments = [s.strip() for s in prompt.split(",")]
    tokens_lower = {t.lower() for t in tokens}

    kept = []
    removed = 0
    for seg in segments:
        if seg.lower() in tokens_lower:
            removed += 1
        else:
            kept.append(seg)

    return ", ".join(kept), removed


class PromptEngine:
    """
    Central prompt processing pipeline.

    Usage:
        engine = PromptEngine(style_registry=sr)
        enhanced = engine.process_image(ImageGenerationRequest(prompt="a hero at dawn"))
    """

    def __init__(self, style_registry: Optional["StyleRegistry"] = None) -> None:
        self._style_registry = style_registry
        self._enhancer = EnhancementEngine()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process_image(self, request: ImageGenerationRequest) -> EnhancedPrompt:
        """
        Run the full enhancement pipeline for an image request.

        Args:
            request: Validated ImageGenerationRequest.

        Returns:
            EnhancedPrompt ready for the image engine.

        Raises:
            InvalidPromptError: if prompt fails enrichment rules.
        """
        prompt = request.prompt

        # 1. Style injection
        style_fragment = ""
        if request.style_id and self._style_registry:
            profile = self._style_registry.get(request.style_id)
            if profile:
                style_fragment = profile.to_prompt_fragment()
                prompt = f"{prompt}, {style_fragment}" if style_fragment else prompt

        # 2. Camera / lighting injection
        cam = request.camera
        lit = request.lighting
        prompt = f"{prompt}, {cam.shot_type} shot, {cam.angle} angle, {lit.type} lighting"

        # 3. Identity-aware token stripping (Phase 4)
        #    When identity conditioning is active, remove facial descriptor tokens
        #    that would fight the IP-Adapter embedding. Uses exact segment matching
        #    to avoid destroying unrelated phrases.
        if request.identity_name:
            prompt, n_removed = _strip_identity_conflict_tokens(
                prompt, _IDENTITY_CONFLICT_TOKENS
            )
            LOG.debug(
                f"PromptEngine: identity active ('{request.identity_name}') — "
                f"suppressed {n_removed} facial descriptor token(s)."
            )

        # 4. LLM enhancement (rule-based now, real LLM in Phase 2)
        enhanced_text = self._enhancer.enhance(prompt)

        # 5. Negative prompt assembly
        base_negative = (
            "deformed, distorted, disfigured, poorly drawn, bad anatomy, "
            "wrong anatomy, extra limb, missing limb, floating limbs, "
            "mutated hands, blurry, low resolution, watermark, text, signature"
        )
        if self._style_registry and request.style_id:
            profile = self._style_registry.get(request.style_id)
            if profile and profile.to_negative_fragment():
                base_negative = f"{base_negative}, {profile.to_negative_fragment()}"
        negative = (
            f"{base_negative}, {request.negative_prompt}"
            if request.negative_prompt
            else base_negative
        )

        # 6. Token estimation (rough: 1 token ≈ 4 chars for SD-style prompts)
        tokens_estimated = len(enhanced_text) // 4

        LOG.debug(f"PromptEngine: original='{request.prompt[:60]}' → enhanced='{enhanced_text[:80]}'")

        return EnhancedPrompt(
            original=request.prompt,
            enhanced=enhanced_text,
            negative=negative,
            style_fragment=style_fragment,
            tokens_estimated=tokens_estimated,
            metadata={
                "style_id": request.style_id,
                "character_id": request.character_id,
                "scene_id": request.scene_id,
                "camera": cam.model_dump(),
                "lighting": lit.model_dump(),
                "identity_name": request.identity_name,
            },
        )
