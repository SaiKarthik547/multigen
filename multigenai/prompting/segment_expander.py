"""
SegmentExpander — Phase 9 Prompt Processing

Enriches each raw positive segment with contextual detail from the
PromptStructure so that individual segments remain visually coherent
even when viewed independently.

Expansion strategy:
  - Append environment context if the segment lacks environmental keywords.
  - Append style token if the segment lacks a style descriptor.
  - Append camera qualifier from the analyzed camera list.
  - Append lighting descriptor from the analyzed lighting list.
  - Never exceed the positive token budget after expansion.
    If expansion would exceed budget, context tokens are dropped progressively
    (lighting → camera → environment → style) until the segment fits.

Pure CPU — no GPU dependency.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from multigenai.core.logging.logger import get_logger
from multigenai.prompting.token_budget_manager import TokenBudgetManager

if TYPE_CHECKING:
    from multigenai.prompting.prompt_analyzer import PromptStructure

LOG = get_logger(__name__)


class SegmentExpander:
    """
    Adds contextual tokens to segments that are too sparse to produce
    coherent visuals on their own.

    Usage:
        expander = SegmentExpander(budget_manager, structure)
        expanded_segments = expander.expand_all(segments)
    """

    def __init__(
        self,
        budget_manager: TokenBudgetManager,
        structure: "PromptStructure",
    ) -> None:
        self._mgr = budget_manager
        self._structure = structure

        # Pre-compute context strings once
        self._env_ctx: str = ", ".join(structure.environment[:2]) if structure.environment else ""
        self._lighting_ctx: str = structure.lighting[0] if structure.lighting else ""
        self._camera_ctx: str = structure.camera[0] if structure.camera else ""
        self._style_ctx: str = structure.style[0] if structure.style else "cinematic"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def expand_all(self, segments: List[str]) -> List[str]:
        """
        Expand all segments with contextual tokens.

        Args:
            segments: List of raw positive prompt segments from PromptSegmenter.

        Returns:
            List of expanded segments, each ≤ positive_budget tokens.
        """
        return [self.expand(seg) for seg in segments]

    def expand(self, segment: str) -> str:
        """
        Expand a single segment.

        Context tokens are appended only if:
          1. The segment doesn't already contain that context.
          2. Adding it keeps the segment within the positive budget.

        Args:
            segment: A single positive prompt string.

        Returns:
            Expanded segment string (still within positive_budget).
        """
        budget = self._mgr.budget.positive_budget
        seg_lower = segment.lower()

        # Determine which context tokens are missing and needed
        additions: List[str] = []

        # Environment — append if none of the detected env words are present
        if self._env_ctx and not any(e in seg_lower for e in self._env_ctx.split(", ")):
            additions.append(self._env_ctx)

        # Lighting — often critical for visual coherence
        if self._lighting_ctx and self._lighting_ctx not in seg_lower:
            additions.append(self._lighting_ctx)

        # Camera — orients the viewer
        if self._camera_ctx and self._camera_ctx not in seg_lower:
            additions.append(self._camera_ctx)

        # Style — ensures model-specific rendering coherence
        if self._style_ctx and self._style_ctx not in seg_lower:
            additions.append(self._style_ctx)

        # Greedily try to fit additions within budget, dropping from the back
        expanded = segment
        for i in range(len(additions), 0, -1):
            candidate_additions = additions[:i]
            candidate = segment + ", " + ", ".join(candidate_additions)
            if self._mgr.count_tokens(candidate) <= budget:
                expanded = candidate
                LOG.debug(
                    f"SegmentExpander: appended {i}/{len(additions)} context token(s) "
                    f"({self._mgr.count_tokens(expanded)}/{budget} tokens)."
                )
                break

        return expanded.strip(", ")
