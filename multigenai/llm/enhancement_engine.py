"""
EnhancementEngine — Prompt quality enhancement.

Phase 1: Rule-based quality token injection (no LLM dependency).
Phase 2: LLM-enhanced path via injected LLMProvider (with permanent fallback).

Design rules:
  - LLM is NEVER mandatory — rule-based fallback always present
  - Provider injected via constructor (DI) — no global imports
  - Idempotence guaranted: quality markers never double-injected,
    even after LLM rewrites the prompt
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from multigenai.core.logging.logger import get_logger

if TYPE_CHECKING:
    from multigenai.llm.providers.base import LLMProvider

LOG = get_logger(__name__)

# Quality tokens that consistently improve SD/SDXL output quality
_QUALITY_TOKENS = (
    "masterpiece, best quality, ultra-detailed, cinematic lighting, "
    "sharp focus, intricate details, 8k resolution, photorealistic"
)

# Markers that indicate an already-professional prompt (checked via substring)
_ALREADY_ENHANCED_MARKERS = (
    "masterpiece", "best quality", "ultra-detailed", "8k", "photorealistic",
    "cinematic lighting", "sharp focus",
)

# System prompt sent to the LLM for enhancement
_ENHANCEMENT_SYSTEM_PROMPT = (
    "You are a professional AI image generation prompt engineer. "
    "Rewrite the given prompt to be more vivid, cinematic, and detailed. "
    "Keep the core subject. "
    "Do NOT add 'masterpiece', 'best quality', or '8k' — those are added separately. "
    "Return only the improved prompt text, no explanations."
)


class EnhancementEngine:
    """
    Enhances raw prompts with quality-boosting tokens.

    When a LLMProvider is injected, it rewrites the prompt for richness
    before quality tokens are appended. Falls back to rule-based if
    the provider is unavailable or returns an error.

    Idempotent: calling enhance() twice on the same prompt is safe.

    Usage (rule-based, default):
        engine = EnhancementEngine()
        better = engine.enhance("a lone knight in a forest")

    Usage (LLM-backed):
        engine = EnhancementEngine(provider=ctx.llm)
        better = engine.enhance("a lone knight in a forest")
    """

    def __init__(self, provider: Optional["LLMProvider"] = None) -> None:
        """
        Args:
            provider: Optional LLM backend. If None, rule-based path is used.
        """
        self._provider = provider
        if provider:
            LOG.debug(f"EnhancementEngine: LLM provider set ({type(provider).__name__})")
        else:
            LOG.debug("EnhancementEngine: no provider — rule-based mode")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enhance(self, prompt: str) -> str:
        """
        Return an enhanced version of the prompt.

        Flow:
          1. If already enhanced (idempotence check) → return as-is
          2. If provider set → try LLM rewrite, catch ProviderUnavailableError
          3. Append quality tokens (with final idempotence guard)

        Args:
            prompt: Raw or partially-structured generation prompt.

        Returns:
            Enhanced prompt string (never raises).
        """
        if self._is_already_enhanced(prompt):
            LOG.debug("EnhancementEngine: prompt already enhanced — returning as-is")
            return prompt

        rewritten = prompt

        # --- LLM path (when provider is present) ---
        if self._provider is not None:
            try:
                from multigenai.core.exceptions import ProviderUnavailableError
                rewritten = self._provider.generate(
                    prompt, system_prompt=_ENHANCEMENT_SYSTEM_PROMPT
                )
                LOG.debug(
                    f"EnhancementEngine: LLM rewrite {len(prompt)}→{len(rewritten)} chars"
                )
                # --- Prompt length guard ---
                rewritten = self._truncate_if_needed(rewritten)
            except Exception as exc:
                LOG.warning(
                    f"EnhancementEngine: LLM enhancement failed ({exc}) "
                    "— falling back to rule-based"
                )
                rewritten = prompt  # restore original on failure

        # --- Quality token injection (with post-LLM idempotence guard) ---
        # Even if the LLM added quality terms, we check again before appending
        if self._is_already_enhanced(rewritten):
            return rewritten

        enhanced = f"{rewritten.rstrip(', ')}, {_QUALITY_TOKENS}"
        LOG.debug(f"EnhancementEngine: appended quality tokens")
        return enhanced

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Maximum allowed prompt character length before truncation
    MAX_PROMPT_CHARS: int = 800

    @staticmethod
    def _truncate_if_needed(prompt: str) -> str:
        """Truncate prompt to MAX_PROMPT_CHARS at the last comma boundary."""
        limit = EnhancementEngine.MAX_PROMPT_CHARS
        if len(prompt) <= limit:
            return prompt
        truncated = prompt[:limit]
        last_comma = truncated.rfind(",")
        if last_comma > limit // 2:
            truncated = truncated[: last_comma].rstrip()
        LOG.warning(
            f"EnhancementEngine: prompt truncated {len(prompt)} → {len(truncated)} chars "
            f"(limit={limit})"
        )
        return truncated

    @staticmethod
    def _is_already_enhanced(prompt: str) -> bool:
        """Substring check — safe against comma-adjacent tokens like 'masterpiece,'."""
        lowered = prompt.lower()
        return any(marker in lowered for marker in _ALREADY_ENHANCED_MARKERS)

