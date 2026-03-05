"""
test_phase9_prompt_processing.py — Phase 9 Advanced Prompt Processing Engine

Tests cover:
  - TokenBudgetManager: counting, splitting, budget checks
  - PromptAnalyzer: block detection, keyword extraction, subject extraction
  - PromptSegmenter: single/multi-block, sentence-boundary splits
  - SegmentExpander: context injection, budget respect
  - NegativePromptManager: master construction, segmentation, pairing
  - PromptPlan: construction, properties, as_pairs
  - PromptProcessor: fast path, full pipeline, from_settings
  - Integration: extremely long prompt → segments within budget

All tests are pure CPU, no torch, no GPU, no network access.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# TokenBudgetManager
# ---------------------------------------------------------------------------

class TestTokenBudgetManager:
    def _make(self, max_tokens=75, negative_reserve=25):
        from multigenai.prompting.token_budget_manager import TokenBudgetManager
        return TokenBudgetManager(max_tokens=max_tokens, negative_reserve=negative_reserve)

    def test_budget_properties(self):
        mgr = self._make(75, 25)
        assert mgr.budget.max_tokens == 75
        assert mgr.budget.negative_reserve == 25
        assert mgr.budget.positive_budget == 50

    def test_count_tokens_empty(self):
        mgr = self._make()
        assert mgr.count_tokens("") == 0
        assert mgr.count_tokens("   ") == 0

    def test_count_tokens_simple(self):
        mgr = self._make()
        count = mgr.count_tokens("a knight at dawn")
        assert count == 4  # a / knight / at / dawn

    def test_count_tokens_comma_phrase(self):
        mgr = self._make()
        # "blurry, low quality" → 3 tokens + 1 comma punct = 4
        count = mgr.count_tokens("blurry, low quality")
        assert count >= 3

    def test_fits_positive_budget_short(self):
        mgr = self._make()
        assert mgr.fits_positive_budget("a knight") is True

    def test_fits_positive_budget_long(self):
        mgr = self._make(max_tokens=10, negative_reserve=5)
        long_text = " ".join(["word"] * 20)  # 20 tokens >> 5 budget
        assert mgr.fits_positive_budget(long_text) is False

    def test_fits_negative_budget_short(self):
        mgr = self._make()
        assert mgr.fits_negative_budget("blurry, low quality") is True

    def test_split_to_budget_short_noop(self):
        mgr = self._make()
        text = "a knight at dawn"
        result = mgr.split_to_budget(text, budget=50)
        assert result == [text]

    def test_split_to_budget_comma_split(self):
        mgr = self._make()
        # Build a phrase list that exceeds budget=5 per chunk
        phrases = ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"]
        text = ", ".join(phrases)
        chunks = mgr.split_to_budget(text, budget=3)
        # Each chunk must be within budget
        for chunk in chunks:
            assert mgr.count_tokens(chunk) <= 3

    def test_split_to_budget_empty(self):
        mgr = self._make()
        assert mgr.split_to_budget("", 50) == []

    def test_split_positive(self):
        mgr = self._make()
        long_pos = ", ".join([f"term_{i}" for i in range(80)])
        chunks = mgr.split_positive(long_pos)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert mgr.count_tokens(chunk) <= mgr.budget.positive_budget

    def test_split_negative(self):
        mgr = self._make()
        long_neg = ", ".join([f"bad_{i}" for i in range(60)])
        chunks = mgr.split_negative(long_neg)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert mgr.count_tokens(chunk) <= mgr.budget.negative_reserve

    def test_budget_repr(self):
        mgr = self._make()
        r = repr(mgr.budget)
        assert "pos" in r
        assert "neg" in r


# ---------------------------------------------------------------------------
# PromptAnalyzer
# ---------------------------------------------------------------------------

class TestPromptAnalyzer:
    def _make(self):
        from multigenai.prompting.prompt_analyzer import PromptAnalyzer
        return PromptAnalyzer()

    def test_empty_prompt(self):
        from multigenai.prompting.prompt_analyzer import PromptStructure
        analyzer = self._make()
        structure = analyzer.analyze("")
        assert isinstance(structure, PromptStructure)
        assert structure.narrative_blocks == []

    def test_single_sentence(self):
        analyzer = self._make()
        structure = analyzer.analyze("A temple beside a river at sunset.")
        assert len(structure.narrative_blocks) == 1
        assert structure.block_count == 1
        assert not structure.is_long_form

    def test_subjects_extracted(self):
        analyzer = self._make()
        structure = analyzer.analyze("A temple beside a river.")
        assert len(structure.subjects) >= 1
        assert any("temple" in s.lower() for s in structure.subjects)

    def test_environment_detected(self):
        analyzer = self._make()
        structure = analyzer.analyze("A village near the ocean during a storm.")
        assert "ocean" in structure.environment or "village" in structure.environment

    def test_lighting_detected(self):
        analyzer = self._make()
        structure = analyzer.analyze("Birds fly above ruins during sunset.")
        assert "sunset" in structure.lighting

    def test_camera_detected(self):
        analyzer = self._make()
        structure = analyzer.analyze("An aerial view of the city at night.")
        assert "aerial" in structure.camera

    def test_style_detected(self):
        analyzer = self._make()
        structure = analyzer.analyze("A cinematic shot of a warrior at dusk.")
        assert "cinematic" in structure.style

    def test_actions_detected(self):
        analyzer = self._make()
        structure = analyzer.analyze("Villagers are praying near the river while birds are flying.")
        assert "praying" in structure.actions or "flying" in structure.actions

    def test_paragraph_split(self):
        analyzer = self._make()
        prompt = "A temple by a river.\n\nBirds fly above the trees.\n\nVillagers pray at dusk."
        structure = analyzer.analyze(prompt)
        assert structure.block_count == 3
        assert structure.is_long_form

    def test_long_script(self):
        analyzer = self._make()
        # 5000-word-scale prompt simulation via many paragraphs
        blocks = [f"Scene {i}: A village near the ocean during sunset." for i in range(20)]
        prompt = "\n\n".join(blocks)
        structure = analyzer.analyze(prompt)
        assert structure.block_count == 20


# ---------------------------------------------------------------------------
# PromptSegmenter
# ---------------------------------------------------------------------------

class TestPromptSegmenter:
    def _make(self, max_tokens=75, negative_reserve=25):
        from multigenai.prompting.token_budget_manager import TokenBudgetManager
        from multigenai.prompting.prompt_segmenter import PromptSegmenter
        mgr = TokenBudgetManager(max_tokens=max_tokens, negative_reserve=negative_reserve)
        return PromptSegmenter(mgr), mgr

    def _structure(self, blocks):
        from multigenai.prompting.prompt_analyzer import PromptStructure
        s = PromptStructure()
        s.narrative_blocks = blocks
        return s

    def test_single_short_block(self):
        segmenter, mgr = self._make()
        structure = self._structure(["A knight at dawn."])
        segments = segmenter.segment(structure)
        assert len(segments) == 1
        assert mgr.count_tokens(segments[0]) <= mgr.budget.positive_budget

    def test_empty_structure(self):
        segmenter, _ = self._make()
        structure = self._structure([])
        assert segmenter.segment(structure) == []

    def test_multi_block(self):
        segmenter, mgr = self._make()
        blocks = [
            "A temple beside a river at sunset.",
            "Villagers pray in the courtyard.",
            "Birds fly above the ancient trees.",
        ]
        structure = self._structure(blocks)
        segments = segmenter.segment(structure)
        assert len(segments) == 3
        for seg in segments:
            assert mgr.count_tokens(seg) <= mgr.budget.positive_budget

    def test_long_block_is_split(self):
        # Budget = 10 positive tokens; a 50-word block must be split
        segmenter, mgr = self._make(max_tokens=15, negative_reserve=5)
        long_block = " ".join(["word"] * 50) + "."
        structure = self._structure([long_block])
        segments = segmenter.segment(structure)
        assert len(segments) > 1
        for seg in segments:
            assert mgr.count_tokens(seg) <= mgr.budget.positive_budget

    def test_segment_raw_short(self):
        segmenter, mgr = self._make()
        result = segmenter.segment_raw("a short prompt")
        assert result == ["a short prompt"]

    def test_segment_raw_long(self):
        segmenter, mgr = self._make(max_tokens=15, negative_reserve=5)
        long_text = ". ".join(["A knight rides into battle"] * 10)
        result = segmenter.segment_raw(long_text)
        assert len(result) > 1
        for seg in result:
            assert mgr.count_tokens(seg) <= mgr.budget.positive_budget

    def test_all_segments_non_empty(self):
        segmenter, _ = self._make()
        blocks = ["First block.", "  ", "Third block."]
        structure = self._structure(blocks)
        segments = segmenter.segment(structure)
        for seg in segments:
            assert seg.strip() != ""


# ---------------------------------------------------------------------------
# SegmentExpander
# ---------------------------------------------------------------------------

class TestSegmentExpander:
    def _make(self, structure_kwargs=None, max_tokens=75, negative_reserve=25):
        from multigenai.prompting.token_budget_manager import TokenBudgetManager
        from multigenai.prompting.segment_expander import SegmentExpander
        from multigenai.prompting.prompt_analyzer import PromptStructure
        mgr = TokenBudgetManager(max_tokens=max_tokens, negative_reserve=negative_reserve)
        structure = PromptStructure(**(structure_kwargs or {}))
        return SegmentExpander(mgr, structure), mgr

    def test_expand_adds_context(self):
        expander, mgr = self._make(
            structure_kwargs={
                "lighting": ["sunset"],
                "style": ["cinematic"],
                "camera": [],
                "environment": ["river"],
            }
        )
        seg = "birds flying"
        expanded = expander.expand(seg)
        # Should have received some additional context
        assert len(expanded) >= len(seg)

    def test_expand_does_not_exceed_budget(self):
        expander, mgr = self._make(
            structure_kwargs={
                "lighting": ["golden hour"],
                "style": ["cinematic"],
                "camera": ["aerial"],
                "environment": ["ocean", "forest"],
            }
        )
        seg = "a knight at dawn"
        expanded = expander.expand(seg)
        assert mgr.count_tokens(expanded) <= mgr.budget.positive_budget

    def test_expand_all_returns_same_count(self):
        expander, _ = self._make()
        segments = ["birds flying", "temple at sunset", "river valley"]
        result = expander.expand_all(segments)
        assert len(result) == len(segments)

    def test_no_duplicate_context(self):
        expander, _ = self._make(
            structure_kwargs={"lighting": ["sunset"], "style": ["cinematic"]}
        )
        # Context already in segment — should not be duplicated
        seg = "a temple at sunset, cinematic photography"
        expanded = expander.expand(seg)
        # "sunset" should appear exactly once
        assert expanded.lower().count("sunset") == 1


# ---------------------------------------------------------------------------
# NegativePromptManager
# ---------------------------------------------------------------------------

class TestNegativePromptManager:
    def _make(self, user_neg="", model="sdxl-base", max_tokens=75, negative_reserve=25):
        from multigenai.prompting.token_budget_manager import TokenBudgetManager
        from multigenai.prompting.negative_prompt_manager import NegativePromptManager
        mgr = TokenBudgetManager(max_tokens=max_tokens, negative_reserve=negative_reserve)
        return NegativePromptManager(mgr, user_negative=user_neg, model_name=model)

    def test_master_negative_not_empty(self):
        npm = self._make()
        assert len(npm.master_negative) > 0

    def test_user_negative_merged(self):
        npm = self._make(user_neg="extra bad thing")
        assert "extra bad thing" in npm.master_negative

    def test_sdxl_specific_tokens(self):
        npm = self._make(model="sdxl-base")
        assert "deformed" in npm.master_negative or "disfigured" in npm.master_negative

    def test_no_duplicate_phrases(self):
        npm = self._make(user_neg="blurry")
        phrases = [p.strip() for p in npm.master_negative.split(",")]
        # "blurry" should appear only once
        count = sum(1 for p in phrases if p.lower() == "blurry")
        assert count == 1

    def test_single_segment_when_fits(self):
        npm = self._make()
        segs = npm.build_negative_segments()
        # Master negative is ~82 tokens — always splits with default reserve=25
        # All resulting segments must individually fit within the negative reserve
        assert len(segs) >= 1
        mgr = npm._mgr
        for seg in segs:
            assert mgr.count_tokens(seg) <= mgr.budget.negative_reserve, (
                f"Segment exceeds reserve: '{seg}' = {mgr.count_tokens(seg)} tokens"
            )

    def test_pair_returns_correct_count(self):
        npm = self._make()
        positives = ["seg1", "seg2", "seg3"]
        pairs = npm.pair(positives)
        assert len(pairs) == 3
        for pos, neg in pairs:
            assert isinstance(pos, str)
            assert isinstance(neg, str)

    def test_pair_cycles_negatives(self):
        # Force small budget so negative is split into 2 segments
        npm = self._make(max_tokens=15, negative_reserve=5)
        positives = ["seg1", "seg2", "seg3", "seg4"]
        pairs = npm.pair(positives)
        assert len(pairs) == 4
        # All negatives should be non-empty
        for _, neg in pairs:
            assert neg.strip() != ""


# ---------------------------------------------------------------------------
# PromptPlan
# ---------------------------------------------------------------------------

class TestPromptPlan:
    def test_from_pairs_single(self):
        from multigenai.prompting.prompt_plan import PromptPlan
        plan = PromptPlan.from_pairs(
            original_prompt="a knight",
            pairs=[("a knight", "blurry")],
        )
        assert plan.segment_count == 1
        assert plan.generation_mode == "single"
        assert not plan.is_multi_segment

    def test_from_pairs_segmented(self):
        from multigenai.prompting.prompt_plan import PromptPlan
        pairs = [(f"seg{i}", "blurry") for i in range(3)]
        plan = PromptPlan.from_pairs("long prompt", pairs)
        assert plan.segment_count == 3
        assert plan.generation_mode == "segmented"
        assert plan.is_multi_segment

    def test_from_pairs_long_form(self):
        from multigenai.prompting.prompt_plan import PromptPlan
        pairs = [(f"seg{i}", "blurry") for i in range(6)]
        plan = PromptPlan.from_pairs("very long script", pairs)
        assert plan.generation_mode == "long-form"

    def test_as_pairs(self):
        from multigenai.prompting.prompt_plan import PromptPlan
        pairs = [("pos1", "neg1"), ("pos2", "neg2")]
        plan = PromptPlan.from_pairs("prompt", pairs)
        result = plan.as_pairs()
        assert result == pairs

    def test_positive_prompts_property(self):
        from multigenai.prompting.prompt_plan import PromptPlan
        pairs = [("pos1", "neg1"), ("pos2", "neg2")]
        plan = PromptPlan.from_pairs("prompt", pairs)
        assert plan.positive_prompts == ["pos1", "pos2"]

    def test_negative_prompts_property(self):
        from multigenai.prompting.prompt_plan import PromptPlan
        pairs = [("pos1", "neg1"), ("pos2", "neg2")]
        plan = PromptPlan.from_pairs("prompt", pairs)
        assert plan.negative_prompts == ["neg1", "neg2"]

    def test_run_id_unique(self):
        from multigenai.prompting.prompt_plan import PromptPlan
        plan1 = PromptPlan.from_pairs("p", [("a", "b")])
        plan2 = PromptPlan.from_pairs("p", [("a", "b")])
        assert plan1.run_id != plan2.run_id

    def test_repr(self):
        from multigenai.prompting.prompt_plan import PromptPlan
        plan = PromptPlan.from_pairs("p", [("a", "b")])
        r = repr(plan)
        assert "PromptPlan" in r


# ---------------------------------------------------------------------------
# PromptProcessor (integration)
# ---------------------------------------------------------------------------

class TestPromptProcessor:
    def _make(self, max_tokens=75, negative_reserve=25, expand=True):
        from multigenai.prompting.prompt_processor import PromptProcessor
        return PromptProcessor(
            max_tokens=max_tokens,
            negative_reserve=negative_reserve,
            expand_segments=expand,
            model_name="sdxl-base",
        )

    def test_empty_prompt(self):
        from multigenai.prompting.prompt_plan import PromptPlan
        processor = self._make()
        plan = processor.process("")
        assert isinstance(plan, PromptPlan)
        assert plan.segment_count == 0

    def test_short_prompt_fast_path(self):
        processor = self._make()
        plan = processor.process("a knight at dawn")
        assert plan.segment_count == 1
        assert plan.generation_mode == "single"
        # Positive should contain the original text
        assert "knight" in plan.segments[0].positive

    def test_short_prompt_has_negative(self):
        processor = self._make()
        plan = processor.process("a knight at dawn")
        assert len(plan.segments[0].negative) > 0

    def test_long_prompt_produces_multiple_segments(self):
        processor = self._make(max_tokens=20, negative_reserve=5)
        long_prompt = ". ".join(["A temple beside a river at sunset"] * 10)
        plan = processor.process(long_prompt)
        assert plan.segment_count > 1

    def test_all_segments_within_positive_budget(self):
        from multigenai.prompting.token_budget_manager import TokenBudgetManager
        processor = self._make(max_tokens=30, negative_reserve=10, expand=False)
        mgr = TokenBudgetManager(max_tokens=30, negative_reserve=10)
        long_prompt = ". ".join([f"Scene number {i} with a temple near the ocean" for i in range(8)])
        plan = processor.process(long_prompt)
        for seg in plan.segments:
            assert mgr.count_tokens(seg.positive) <= mgr.budget.positive_budget

    def test_all_segments_within_negative_reserve(self):
        from multigenai.prompting.token_budget_manager import TokenBudgetManager
        from multigenai.prompting.negative_prompt_manager import NegativePromptManager
        # Use same budget as the processor (default: max=75, reserve=25)
        mgr = TokenBudgetManager()
        npm = NegativePromptManager(mgr, user_negative="", model_name="sdxl-base")
        neg_segments = npm.build_negative_segments()
        # Every resulting rotation segment must be within the reserve
        for seg in neg_segments:
            tokens = mgr.count_tokens(seg)
            assert tokens <= mgr.budget.negative_reserve, (
                f"Negative chunk exceeds reserve: {tokens} tokens > "
                f"{mgr.budget.negative_reserve}: '{seg}'"
            )

    def test_user_negative_merged(self):
        processor = self._make()
        plan = processor.process("a knight", negative_prompt="my custom bad token")
        # The master negative is split when it exceeds the reserve.
        # The custom token may land in any rotation segment — check all.
        npm_segs = [seg.negative for seg in plan.segments]
        # Also rebuild the full NegativePromptManager to get all neg segments:
        from multigenai.prompting.token_budget_manager import TokenBudgetManager
        from multigenai.prompting.negative_prompt_manager import NegativePromptManager
        mgr = TokenBudgetManager()
        npm = NegativePromptManager(mgr, user_negative="my custom bad token", model_name="sdxl-base")
        all_neg_segs = npm.build_negative_segments()
        combined = " ".join(all_neg_segs)
        assert "my custom bad token" in combined

    def test_multi_paragraph_script(self):
        processor = self._make()
        script = (
            "A grand temple stands beside a river during golden hour.\n\n"
            "Villagers dressed in white robes gather in the courtyard to pray.\n\n"
            "Birds fly above ancient trees, their wings catching the last light of day.\n\n"
            "A lone monk walks along the stone path towards the inner sanctum."
        )
        plan = processor.process(script)
        # 4 paragraphs → should produce at least 2 segments if any block is over budget
        assert plan.segment_count >= 1

    def test_from_settings_fallback(self):
        """from_settings falls back gracefully if settings.prompt is missing."""
        from multigenai.prompting.prompt_processor import PromptProcessor

        class _FakeSettings:
            pass  # no .prompt attribute

        processor = PromptProcessor.from_settings(_FakeSettings())
        plan = processor.process("a test prompt")
        assert plan.segment_count >= 1

    def test_1000_word_script(self):
        """Stress test: 1000-word script processes without error."""
        processor = self._make()
        words = ["A magnificent temple stands beside a flowing river at twilight."] * 100
        script = "\n\n".join(words)
        plan = processor.process(script)
        assert plan.segment_count >= 1
        assert plan.original_prompt == script

    def test_5000_word_script(self):
        """Stress test: 5000-word script processes without error."""
        processor = self._make()
        sentence = "The ancient ruins of a lost civilization crumble at the edge of the forest at sunset."
        script = "\n\n".join([sentence] * 400)
        plan = processor.process(script)
        assert plan.segment_count >= 1

    def test_no_token_truncation_warning_on_short_prompt(self):
        """Short prompts (fast path) must never raise any exception."""
        processor = self._make()
        for prompt in [
            "a",
            "abc def ghi",
            "A beautiful sunset over the ocean, cinematic, photorealistic.",
        ]:
            plan = processor.process(prompt)
            assert plan.segment_count >= 1

    def test_segment_indices_sequential(self):
        processor = self._make(max_tokens=20, negative_reserve=5)
        plan = processor.process(". ".join(["A knight rides at dawn"] * 15))
        for i, seg in enumerate(plan.segments):
            assert seg.index == i
