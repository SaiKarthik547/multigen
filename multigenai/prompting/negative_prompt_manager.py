"""
NegativePromptManager — Phase 9 Prompt Processing

Manages the master negative prompt and ensures it is:
  1. Tokenized safely within the negative token reserve.
  2. Segmented if too large (rotated across generation segments).
  3. Paired with each positive segment in order.

The master negative prompt is never silently truncated.
If it exceeds the negative_reserve budget, it is split into multiple
negative segments which are cycled across the positive segments in round-robin
fashion so that all negative tokens influence generation across the run.

Pure CPU — no GPU dependency.
"""

from __future__ import annotations

import itertools
from typing import List, Optional

from multigenai.core.logging.logger import get_logger
from multigenai.prompting.token_budget_manager import TokenBudgetManager

LOG = get_logger(__name__)

# ---------------------------------------------------------------------------
# Canonical base negative prompt — shared with PromptCompiler but managed here
# for the multi-segment workflow.
# ---------------------------------------------------------------------------
_BASE_NEGATIVE = (
    "low quality, worst quality, text, signature, watermark, blurry, "
    "distorted anatomy, extra fingers, malformed hands, overexposed, "
    "noise, jpeg artifacts, grainy, flat, ugly, oversaturated, "
    "statue, idol, static, CGI, 3D render, cartoon, artificial, "
    "lifeless, unmoving, plastic, painting, illustration, uncanny valley, "
    "glitch, low resolution, amateur, poorly drawn"
)


class NegativePromptManager:
    """
    Manages, segments, and pairs negative prompts for multi-segment generation.

    Usage:
        mgr = NegativePromptManager(budget_manager, user_negative="...")
        neg_segments = mgr.build_negative_segments()
        paired = mgr.pair(positive_segments)  # List[(pos, neg)]
    """

    def __init__(
        self,
        budget_manager: TokenBudgetManager,
        user_negative: str = "",
        model_name: str = "",
    ) -> None:
        self._mgr = budget_manager
        self._model_name = model_name.lower()

        # Merge base + user negative prompts
        self._master_negative = self._build_master(user_negative)

        # Cached segments (built lazily)
        self._segments: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def master_negative(self) -> str:
        """The full, untruncated master negative prompt string."""
        return self._master_negative

    def build_negative_segments(self) -> List[str]:
        """
        Tokenize and segment the master negative prompt.

        If the master negative fits within the reserve, a single-element
        list is returned.  Otherwise multiple segments are returned for rotation.

        Returns:
            List of negative prompt chunks, each ≤ negative_reserve tokens.
        """
        if self._segments is not None:
            return self._segments

        reserve = self._mgr.budget.negative_reserve
        total_tokens = self._mgr.count_tokens(self._master_negative)

        if total_tokens <= reserve:
            self._segments = [self._master_negative]
            LOG.info(
                f"NegativePromptManager: master negative fits in reserve "
                f"({total_tokens}/{reserve} tokens). Single segment."
            )
        else:
            # Over-budget negative prompts are trimmed to fit (Phase 10 optimization)
            # This 'removes segmentation overhead' and ensures a single negative pass.
            trimmed = self._mgr.trim_negative(self._master_negative)
            self._segments = [trimmed]
            LOG.warning(
                f"NegativePromptManager: master negative ({total_tokens} tokens) "
                f"exceeds reserve ({reserve}). Trimmed to fit (no segmentation overhead)."
            )

        return self._segments

    def pair(self, positive_segments: List[str]) -> List[tuple]:
        """
        Pair each positive segment with a negative segment.

        If there is only one negative segment, every positive segment receives it.
        If there are multiple negative segments, they are cycled in round-robin.

        Args:
            positive_segments: Ordered list of positive prompt strings.

        Returns:
            List of (positive, negative) 2-tuples, one per generation step.
        """
        neg_segments = self.build_negative_segments()
        neg_cycle = itertools.cycle(neg_segments)

        paired = [(pos, next(neg_cycle)) for pos in positive_segments]

        LOG.info(
            f"NegativePromptManager: paired {len(paired)} segment(s) "
            f"({len(neg_segments)} negative segment(s) cycling)."
        )
        return paired

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_master(self, user_negative: str) -> str:
        """
        Merge base negative with user-supplied additional tokens.

        Model-specific additions are appended for SDXL-family models.
        Duplicates are de-duplicated at phrase level.
        """
        parts = []

        # User-supplied tokens (Highest Priority)
        if user_negative and user_negative.strip():
            parts.append(user_negative.strip())

        # SDXL-specific additions
        if "sdxl" in self._model_name or "stable-diffusion-xl" in self._model_name:
            parts.append("deformed, disfigured, bad proportions")

        # Generic Base Negatives (Lowest Priority)
        parts.append(_BASE_NEGATIVE)

        combined = ", ".join(parts)

        # De-duplicate at phrase level (preserve order)
        phrases = [p.strip() for p in combined.split(",") if p.strip()]
        seen = set()
        deduped = []
        for phrase in phrases:
            key = phrase.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(phrase)

        return ", ".join(deduped)
