"""
PromptProcessor — Phase 9 Prompt Processing

The single public entry point for the entire prompting subsystem.

Orchestrates:
  PromptAnalyzer → PromptSegmenter → SegmentExpander → NegativePromptManager
  → PromptPlan

Usage:
    from multigenai.prompting import PromptProcessor

    processor = PromptProcessor.from_settings(settings)
    plan = processor.process(
        prompt="A long script...",
        negative_prompt="blurry, distorted..."
    )
    for seg in plan.segments:
        engine.generate(seg.positive, seg.negative)

Design rules:
  - Zero GPU usage — fully CPU-bound.
  - Never truncates silently — every token is accounted for.
  - Short prompts (within budget) produce a single-segment PromptPlan
    with zero overhead compared to the old PromptCompiler path.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from multigenai.core.logging.logger import get_logger
from multigenai.prompting.prompt_analyzer import PromptAnalyzer
from multigenai.prompting.prompt_segmenter import PromptSegmenter
from multigenai.prompting.segment_expander import SegmentExpander
from multigenai.prompting.negative_prompt_manager import NegativePromptManager
from multigenai.prompting.token_budget_manager import TokenBudgetManager
from multigenai.prompting.prompt_plan import PromptPlan

if TYPE_CHECKING:
    from multigenai.core.config.settings import Settings

LOG = get_logger(__name__)


class PromptProcessor:
    """
    End-to-end prompt processing pipeline.

    For short prompts (≤ positive_budget tokens) this is a pass-through
    with a single-segment PromptPlan — no overhead.

    For long prompts / scripts the full segmentation pipeline fires:
      analyze → segment → expand → pair negatives → PromptPlan

    Args:
        max_tokens:       CLIP hard limit (75 by default, leaving BOS/EOS headroom).
        negative_reserve: Tokens reserved for the negative prompt (25 default).
        expand_segments:  Whether to enrich sparse segments with context (default True).
        model_name:       Model alias passed to NegativePromptManager for model-specific negatives.
    """

    def __init__(
        self,
        max_tokens: int = 75,
        negative_reserve: int = 25,
        expand_segments: bool = True,
        model_name: str = "sdxl-base",
    ) -> None:
        self._expand = expand_segments
        self._model_name = model_name
        self._budget_mgr = TokenBudgetManager(
            max_tokens=max_tokens,
            negative_reserve=negative_reserve,
        )
        LOG.debug(f"PromptProcessor initialised: {self._budget_mgr.budget}")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(
        cls,
        settings: "Settings",
        model_name: str = "sdxl-base",
    ) -> "PromptProcessor":
        """
        Construct a PromptProcessor from application Settings.

        Reads prompt.max_tokens, prompt.negative_reserve, prompt.expansion
        from settings.prompt (PromptSettings dataclass).

        Falls back to defaults if settings.prompt is not available (backwards
        compatibility with older Settings objects that lack the prompt section).
        """
        try:
            ps = settings.prompt
            return cls(
                max_tokens=ps.max_tokens,
                negative_reserve=ps.negative_reserve,
                expand_segments=ps.expansion,
                model_name=model_name,
            )
        except AttributeError:
            LOG.debug(
                "PromptProcessor.from_settings: settings.prompt not found — "
                "using built-in defaults."
            )
            return cls(model_name=model_name)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(
        self,
        prompt: str,
        negative_prompt: str = "",
        model_name: Optional[str] = None,
        force_single_segment: bool = False,
    ) -> PromptPlan:
        """
        Process a raw user prompt into a token-safe PromptPlan.

        Short prompts (≤ positive_budget) → 1 segment, zero analysis overhead.
        Long prompts / scripts → full analyze → segment → expand → pair pipeline.

        Args:
            prompt:          Raw user text (any length).
            negative_prompt: User-supplied negative additions (merged with master).
            model_name:      Override model for this call (uses init value if None).
            force_single_segment: If True, bypass segmentation even if prompt is long.

        Returns:
            PromptPlan ready for iteration by GenerationManager.
        """
        if not prompt or not prompt.strip():
            LOG.warning("PromptProcessor.process: empty prompt received — returning empty plan.")
            return PromptPlan(original_prompt="", segments=[])

        effective_model = model_name or self._model_name
        budget = self._budget_mgr.budget

        LOG.info(
            f"PromptProcessor: processing prompt "
            f"(~{self._budget_mgr.count_tokens(prompt)} tokens, "
            f"budget=pos≤{budget.positive_budget}/neg≤{budget.negative_reserve})"
        )

        # ------------------------------------------------------------------
        # FAST PATH: prompt fits in one segment — zero segmentation overhead
        # ------------------------------------------------------------------
        if force_single_segment or self._budget_mgr.fits_positive_budget(prompt):
            neg_mgr = NegativePromptManager(
                budget_manager=self._budget_mgr,
                user_negative=negative_prompt,
                model_name=effective_model,
            )
            pairs = neg_mgr.pair([prompt.strip()])

            # Enforce budget even on fast-path pairs
            enforced_pairs = [
                (self._budget_mgr.trim_positive(p), self._budget_mgr.trim_negative(n))
                for p, n in pairs
            ]

            plan = PromptPlan.from_pairs(
                original_prompt=prompt,
                pairs=enforced_pairs,
                positive_budget=budget.positive_budget,
                negative_reserve=budget.negative_reserve,
            )
            LOG.info(f"PromptProcessor: fast path — single segment. {plan}")
            return plan

        # ------------------------------------------------------------------
        # FULL PATH: multi-segment pipeline
        # ------------------------------------------------------------------

        # 1. Analyze
        structure = PromptAnalyzer().analyze(prompt)

        # 2. Segment
        segments = PromptSegmenter(self._budget_mgr).segment(structure)

        if not segments:
            # Fallback: treat full prompt as one segment (with budget warning)
            LOG.warning("PromptProcessor: segmenter returned no segments — using raw prompt.")
            segments = [prompt.strip()]

        # 3. Expand (conditional)
        if self._expand:
            segments = SegmentExpander(self._budget_mgr, structure).expand_all(segments)

        # 4. Negative pairing
        neg_mgr = NegativePromptManager(
            budget_manager=self._budget_mgr,
            user_negative=negative_prompt,
            model_name=effective_model,
        )
        pairs = neg_mgr.pair(segments)

        # 5. Finalize: strictly enforce budgets on all segments
        enforced_pairs = []
        for p, n in pairs:
            p_trim = self._budget_mgr.trim_positive(p)
            n_trim = self._budget_mgr.trim_negative(n)
            enforced_pairs.append((p_trim, n_trim))

        # 6. Build plan
        plan = PromptPlan.from_pairs(
            original_prompt=prompt,
            pairs=enforced_pairs,
            positive_budget=budget.positive_budget,
            negative_reserve=budget.negative_reserve,
        )

        LOG.info(
            f"PromptProcessor: full pipeline complete — {plan}. "
            f"Structure: blocks={structure.block_count}, "
            f"subjects={len(structure.subjects)}, "
            f"actions={len(structure.actions)}."
        )
        return plan
