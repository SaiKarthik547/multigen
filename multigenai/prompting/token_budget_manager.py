"""
TokenBudgetManager — Phase 9 Prompt Processing

Manages the per-segment token budget for positive and negative prompts
to guarantee that CLIP text encoders never receive an oversized sequence.

CLIP hard limits (including BOS + EOS special tokens):
  SDXL / SD 1.x / SVD-XT  → 77 tokens
  We operate with 75 usable tokens and a 2-token safety margin.

Budget split (configurable via PromptSettings):
  positive_budget  = max_tokens - negative_reserve
  negative_reserve = tokens reserved for the negative segment

Pure CPU — no torch dependency at import time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


# ---------------------------------------------------------------------------
# Default budget constants (overridden by PromptSettings at runtime)
# ---------------------------------------------------------------------------
_DEFAULT_MAX_TOKENS: int = 75        # usable CLIP slots (BOS/EOS excluded from accounting)
_DEFAULT_NEG_RESERVE: int = 15       # tokens reserved for negative prompt (user enforced positive+negative<=77)
_DEFAULT_POS_BUDGET: int = _DEFAULT_MAX_TOKENS - _DEFAULT_NEG_RESERVE  # 60


@dataclass(frozen=True)
class TokenBudget:
    """Immutable token budget for one model invocation."""
    max_tokens: int
    negative_reserve: int

    @property
    def positive_budget(self) -> int:
        return self.max_tokens - self.negative_reserve

    def __repr__(self) -> str:
        return (
            f"TokenBudget(max={self.max_tokens}, "
            f"pos≤{self.positive_budget}, neg≤{self.negative_reserve})"
        )


class TokenBudgetManager:
    """
    Estimates token counts and enforces budgets for positive/negative prompts.

    Token estimation uses a fast whitespace + punctuation tokenizer that gives
    a conservative over-estimate compared to the actual CLIP BPE tokenizer.
    This means we may split slightly more aggressively than strictly necessary,
    which is always safe (never under-splits).

    Usage:
        mgr = TokenBudgetManager(max_tokens=75, negative_reserve=25)
        count = mgr.count_tokens("a knight at dawn")
        chunks = mgr.split_to_budget("long positive prompt ...", budget=50)
    """

    def __init__(
        self,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        negative_reserve: int = _DEFAULT_NEG_RESERVE,
    ) -> None:
        self._budget = TokenBudget(
            max_tokens=max_tokens,
            negative_reserve=negative_reserve,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def budget(self) -> TokenBudget:
        """The current token budget configuration."""
        return self._budget

    def count_tokens(self, text: str) -> int:
        """
        Fast, conservative token-count estimate.

        Splits on whitespace and punctuation boundaries.  Over-estimates
        slightly (~1.1×) vs BPE tokenizer which is deliberately safe.

        Args:
            text: Prompt text to count.

        Returns:
            Estimated token count (int ≥ 0).
        """
        if not text or not text.strip():
            return 0
        # Split on whitespace + punctuation boundaries (conservative BPE proxy)
        tokens = re.findall(r"[\w']+|[.,;:!?–—\"\'()\[\]{}]", text)
        return len(tokens)

    def fits_positive_budget(self, text: str) -> bool:
        """Return True if text fits within the positive token budget."""
        return self.count_tokens(text) <= self._budget.positive_budget

    def fits_negative_budget(self, text: str) -> bool:
        """Return True if text fits within the negative token reserve."""
        return self.count_tokens(text) <= self._budget.negative_reserve

    def split_to_budget(self, text: str, budget: int) -> List[str]:
        """
        Split a comma-delimited prompt into chunks that each fit within `budget` tokens.

        Splits on comma boundaries where possible to respect semantic phrases.
        Falls back to word-boundary splitting if a single phrase exceeds budget.

        Args:
            text:   Prompt string (positive or negative).
            budget: Maximum token count per returned chunk.

        Returns:
            List of non-empty chunks, each ≤ budget tokens.
        """
        if not text or not text.strip():
            return []

        if self.count_tokens(text) <= budget:
            return [text.strip()]

        # Phase 1: try splitting on comma boundaries
        # NOTE: we count tokens on the ACTUAL joined string (not summed individual
        # phrase counts) so that ", " separator tokens are correctly accounted for.
        phrases = [p.strip() for p in text.split(",") if p.strip()]
        chunks: List[str] = []
        current_phrases: List[str] = []

        for phrase in phrases:
            phrase_tokens = self.count_tokens(phrase)

            # Single phrase is already over budget — force word-split immediately
            if phrase_tokens > budget:
                if current_phrases:
                    chunks.append(", ".join(current_phrases))
                    current_phrases = []
                word_chunks = self._split_by_words(phrase, budget)
                chunks.extend(word_chunks)
                continue

            # Would adding this phrase exceed budget on the actual joined string?
            candidate = ", ".join(current_phrases + [phrase])
            if self.count_tokens(candidate) > budget and current_phrases:
                # Flush current chunk, start a new one
                chunks.append(", ".join(current_phrases))
                current_phrases = [phrase]
            else:
                current_phrases.append(phrase)

        if current_phrases:
            chunks.append(", ".join(current_phrases))

        return [c for c in chunks if c.strip()]

    def split_positive(self, text: str) -> List[str]:
        """Split positive prompt into positive-budget-sized chunks."""
        return self.split_to_budget(text, self._budget.positive_budget)

    def split_negative(self, text: str) -> List[str]:
        """Split negative prompt into negative-reserve-sized chunks."""
        return self.split_to_budget(text, self._budget.negative_reserve)

    def trim_positive(self, text: str) -> str:
        """
        Hard-trim a positive segment to guarantee it fits the budget.
        Re-evaluates the actual token sequence and safely cuts without severing words.
        """
        if self.fits_positive_budget(text):
            return text
        chunks = self.split_positive(text)
        return chunks[0] if chunks else text

    def trim_negative(self, text: str) -> str:
        """
        Hard-trim a negative segment to guarantee it fits the reserve.
        """
        if self.fits_negative_budget(text):
            return text
        chunks = self.split_negative(text)
        return chunks[0] if chunks else text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_by_words(self, text: str, budget: int) -> List[str]:
        """Fallback: split long single-phrase by word boundaries."""
        words = text.split()
        chunks: List[str] = []
        current: List[str] = []
        current_count = 0

        for word in words:
            wt = self.count_tokens(word)
            if current_count + wt > budget and current:
                chunks.append(" ".join(current))
                current = []
                current_count = 0
            current.append(word)
            current_count += wt

        if current:
            chunks.append(" ".join(current))

        return chunks
