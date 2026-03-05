"""
PromptSegmenter — Phase 9 Prompt Processing

Converts a PromptStructure (from PromptAnalyzer) into a list of
token-safe positive prompt segments using the TokenBudgetManager.

Segmentation strategy (in priority order):
  1. If narrative_blocks were detected, each block becomes a candidate segment.
  2. If a block exceeds the positive budget, it is split at sentence boundaries.
  3. If a sentence exceeds the positive budget, it is split at comma boundaries.
  4. If a comma phrase exceeds the positive budget, it is split at word boundaries
     (last-resort — semantically undesirable but never silently truncated).

Semantic boundary rules (enforced):
  - Segments NEVER split mid-sentence unless the sentence itself is budget-unsafe.
  - Segments preserve paragraph groupings when possible.

Pure CPU — no GPU dependency.
"""

from __future__ import annotations

import re
from typing import List, TYPE_CHECKING

from multigenai.core.logging.logger import get_logger
from multigenai.prompting.token_budget_manager import TokenBudgetManager

if TYPE_CHECKING:
    from multigenai.prompting.prompt_analyzer import PromptStructure

LOG = get_logger(__name__)

# Sentence-boundary split regex
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


class PromptSegmenter:
    """
    Splits a long prompt (represented as PromptStructure.narrative_blocks)
    into token-safe positive prompt segments.

    Usage:
        segmenter = PromptSegmenter(budget_manager)
        segments = segmenter.segment(structure)  # List[str]
    """

    def __init__(self, budget_manager: TokenBudgetManager) -> None:
        self._mgr = budget_manager

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def segment(self, structure: "PromptStructure") -> List[str]:
        """
        Convert narrative blocks into positive-budget-safe segments.

        Args:
            structure: PromptStructure output from PromptAnalyzer.

        Returns:
            List of non-empty segmented prompts, each ≤ positive_budget tokens.
        """
        positive_budget = self._mgr.budget.positive_budget
        segments: List[str] = []

        if not structure.narrative_blocks:
            return []

        for block in structure.narrative_blocks:
            block = block.strip()
            if not block:
                continue

            if self._mgr.count_tokens(block) <= positive_budget:
                segments.append(block)
            else:
                # Block too large — split at sentence boundaries first
                sub_segments = self._split_block(block, positive_budget)
                segments.extend(sub_segments)

        # Final safety pass: verify every segment is within budget
        verified = []
        for seg in segments:
            if self._mgr.count_tokens(seg) <= positive_budget:
                verified.append(seg)
            else:
                # Still too large — force-split at comma / word boundaries
                force_split = self._mgr.split_positive(seg)
                verified.extend(force_split)
                LOG.warning(
                    f"PromptSegmenter: forced secondary split on segment "
                    f"({self._mgr.count_tokens(seg)} tokens > {positive_budget} budget)."
                )

        final = [s.strip() for s in verified if s.strip()]
        LOG.info(
            f"PromptSegmenter: {len(structure.narrative_blocks)} blocks → "
            f"{len(final)} segment(s) "
            f"(budget={positive_budget} tokens/segment)"
        )
        return final

    def segment_raw(self, text: str) -> List[str]:
        """
        Directly segment a raw text string without a PromptStructure.

        Useful when the caller already has a simple string and doesn't
        need the full analysis pipeline.

        Args:
            text: Raw prompt string.

        Returns:
            List of budget-safe segments.
        """
        positive_budget = self._mgr.budget.positive_budget
        if self._mgr.count_tokens(text) <= positive_budget:
            return [text.strip()]
        return self._split_block(text.strip(), positive_budget)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_block(self, block: str, budget: int) -> List[str]:
        """
        Split a single block into sentence-group chunks that fit within budget.

        Groups consecutive sentences together until budget is reached, then
        starts a new chunk.
        """
        sentences = _SENTENCE_RE.split(block)
        chunks: List[str] = []
        current_sentences: List[str] = []
        current_count: int = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_tokens = self._mgr.count_tokens(sent)

            if current_count + sent_tokens > budget and current_sentences:
                # Flush current chunk
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_count = 0

            if sent_tokens > budget:
                # Single sentence is over budget — split by comma/word
                sub = self._mgr.split_to_budget(sent, budget)
                chunks.extend(sub)
                LOG.debug(
                    f"PromptSegmenter: sentence split into {len(sub)} sub-segments "
                    f"({sent_tokens} tokens > {budget})."
                )
            else:
                current_sentences.append(sent)
                current_count += sent_tokens

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return [c.strip() for c in chunks if c.strip()]
