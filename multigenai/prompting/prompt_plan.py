"""
PromptPlan — Phase 9 Prompt Processing

The output contract of the entire prompting subsystem.

A PromptPlan holds:
  - The original user prompt (unmodified)
  - All paired (positive, negative) generation segments
  - Metadata about how the plan was constructed

GenerationManager iterates over `segments` to schedule each generation step.
Output files are saved to:
  {output_dir}/segmented_runs/{run_id}/segment_{N:03d}.{ext}
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PromptSegment:
    """
    A single token-safe (positive, negative) prompt pair.

    Attributes:
        index:    Zero-based position in the generation plan.
        positive: Positive prompt string (≤ positive_budget tokens).
        negative: Negative prompt string (≤ negative_reserve tokens).
    """
    index: int
    positive: str
    negative: str

    def __repr__(self) -> str:
        return (
            f"PromptSegment(index={self.index}, "
            f"pos={self.positive[:40]!r}..., "
            f"neg={self.negative[:30]!r}...)"
        )


@dataclass
class PromptPlan:
    """
    Complete generation plan produced by PromptProcessor.

    Attributes:
        original_prompt:  The raw unmodified user prompt.
        segments:         Ordered list of (positive, negative) PromptSegments.
        run_id:           Unique identifier for this plan execution.
        generation_mode:  Descriptive label (\"single\" | \"segmented\" | \"long-form\").
        positive_budget:  The positive token budget used when building this plan.
        negative_reserve: The negative token reserve used when building this plan.
    """
    original_prompt: str
    segments: List[PromptSegment] = field(default_factory=list)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    generation_mode: str = "single"          # single | segmented | long-form
    positive_budget: int = 50
    negative_reserve: int = 25

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def is_multi_segment(self) -> bool:
        return len(self.segments) > 1

    @property
    def positive_prompts(self) -> List[str]:
        return [s.positive for s in self.segments]

    @property
    def negative_prompts(self) -> List[str]:
        return [s.negative for s in self.segments]

    def as_pairs(self) -> List[Tuple[str, str]]:
        """Return list of (positive, negative) string tuples for engine use."""
        return [(s.positive, s.negative) for s in self.segments]

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pairs(
        cls,
        original_prompt: str,
        pairs: List[Tuple[str, str]],
        positive_budget: int = 50,
        negative_reserve: int = 25,
    ) -> "PromptPlan":
        """
        Construct a PromptPlan from a list of (positive, negative) pairs.

        Args:
            original_prompt: The unmodified user input.
            pairs:           List of (positive, negative) tuples.
            positive_budget: Token budget used for positive segments.
            negative_reserve: Token reserve used for negative segments.

        Returns:
            PromptPlan ready for GenerationManager consumption.
        """
        segments = [
            PromptSegment(index=i, positive=pos, negative=neg)
            for i, (pos, neg) in enumerate(pairs)
        ]
        mode = "single" if len(segments) == 1 else (
            "long-form" if len(segments) > 5 else "segmented"
        )
        return cls(
            original_prompt=original_prompt,
            segments=segments,
            generation_mode=mode,
            positive_budget=positive_budget,
            negative_reserve=negative_reserve,
        )

    def __repr__(self) -> str:
        return (
            f"PromptPlan(run_id={self.run_id!r}, mode={self.generation_mode!r}, "
            f"segments={self.segment_count}, "
            f"budget=pos≤{self.positive_budget}/neg≤{self.negative_reserve})"
        )
