"""
PromptAnalyzer — Phase 9 Prompt Processing

Extracts semantic structure from a raw user prompt or multi-paragraph script.

Outputs a PromptStructure dataclass containing:
  - subjects       : primary visual subjects
  - environment    : background, setting, world
  - actions        : motion/activity detected
  - camera         : camera-related phrases
  - lighting       : lighting descriptors
  - style          : rendering/art style clues
  - narrative_blocks : paragraph-level semantic groups for long scripts

Pure CPU — no GPU or model dependency. Uses rule-based NLP pattern matching
with optional nltk tokenization (graceful fallback if nltk unavailable).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Semantic keyword tables (used for rule-based classification)
# ---------------------------------------------------------------------------

_CAMERA_KEYWORDS = frozenset([
    "close-up", "closeup", "close up", "wide shot", "wide angle", "aerial",
    "bird's eye", "birds eye", "overhead", "low angle", "high angle",
    "dolly", "pan", "tilt", "tracking", "medium shot", "portrait",
    "establishing shot", "macro", "extreme close", "pov", "eye level",
])

_LIGHTING_KEYWORDS = frozenset([
    "sunset", "sunrise", "golden hour", "blue hour", "dusk", "dawn",
    "neon", "candlelight", "moonlight", "studio lighting", "backlit",
    "silhouette", "chiaroscuro", "volumetric", "rim light", "dramatic shadow",
    "soft light", "hard light", "overcast", "natural light", "night sky",
])

_STYLE_KEYWORDS = frozenset([
    "cinematic", "photorealistic", "anime", "watercolor", "oil painting",
    "sketch", "illustration", "digital art", "concept art", "8k", "4k",
    "hyperrealistic", "dark fantasy", "sci-fi", "gothic", "impressionist",
    "abstract", "surreal", "noir", "vintage", "cyberpunk", "steampunk",
])

_ACTION_VERBS = frozenset([
    "running", "walking", "flying", "swimming", "fighting", "dancing",
    "sitting", "standing", "praying", "crying", "laughing", "sleeping",
    "falling", "jumping", "rising", "floating", "hunting", "searching",
    "holding", "carrying", "playing", "working", "building", "fleeing",
])

_ENVIRONMENT_NOUNS = frozenset([
    "forest", "ocean", "sea", "river", "mountain", "desert", "city",
    "jungle", "temple", "castle", "village", "street", "alley", "sky",
    "space", "cave", "field", "meadow", "valley", "cliff", "beach",
    "swamp", "ruins", "battlefield", "garden", "market", "palace",
    "tower", "bridge", "courtyard", "kingdom", "island", "volcano",
])


@dataclass
class PromptStructure:
    """
    Semantic decomposition of a user prompt or script.

    All lists contain raw phrase strings exactly as extracted.
    narrative_blocks preserves the original paragraph boundaries
    for long-script segmentation by PromptSegmenter.
    """
    subjects: List[str] = field(default_factory=list)
    environment: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    camera: List[str] = field(default_factory=list)
    lighting: List[str] = field(default_factory=list)
    style: List[str] = field(default_factory=list)
    narrative_blocks: List[str] = field(default_factory=list)   # one per paragraph/scene

    @property
    def is_long_form(self) -> bool:
        """True if multiple narrative blocks were detected."""
        return len(self.narrative_blocks) > 1

    @property
    def block_count(self) -> int:
        return len(self.narrative_blocks)


class PromptAnalyzer:
    """
    Analyzes a raw user prompt and extracts its semantic structure.

    Works on prompts of any length — from a single phrase to a multi-page
    script.  Long scripts are split into narrative_blocks first, then each
    block is analyzed independently.

    Usage:
        analyzer = PromptAnalyzer()
        structure = analyzer.analyze("A temple beside a river at sunset ...")
    """

    # Sentence boundary regex — splits on . ! ? followed by whitespace or end
    _SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')
    # Paragraph boundary — two or more newlines
    _PARA_RE = re.compile(r'\n{2,}')
    # Scene transition markers common in scripts
    _SCENE_RE = re.compile(
        r'(?:^|\n)(?:INT\.|EXT\.|SCENE|CUT TO|FADE TO|DISSOLVE)', re.IGNORECASE
    )

    def analyze(self, prompt: str) -> PromptStructure:
        """
        Extract semantic structure from `prompt`.

        Args:
            prompt: Raw user prompt (any length).

        Returns:
            PromptStructure with classified fields populated.
        """
        if not prompt or not prompt.strip():
            return PromptStructure()

        structure = PromptStructure()

        # ---- Split into narrative blocks (paragraphs / scenes) ----
        blocks = self._split_into_blocks(prompt)
        structure.narrative_blocks = blocks
        LOG.debug(f"PromptAnalyzer: {len(blocks)} narrative block(s) detected.")

        # ---- Analyze the full text for semantic tags ----
        text_lower = prompt.lower()

        structure.camera    = self._extract_keywords(text_lower, _CAMERA_KEYWORDS)
        structure.lighting  = self._extract_keywords(text_lower, _LIGHTING_KEYWORDS)
        structure.style     = self._extract_keywords(text_lower, _STYLE_KEYWORDS)
        structure.actions   = self._extract_keywords(text_lower, _ACTION_VERBS)
        structure.environment = self._extract_keywords(text_lower, _ENVIRONMENT_NOUNS)

        # ---- Subject extraction: first meaningful noun phrase in each block ----
        structure.subjects = self._extract_subjects(blocks)

        LOG.info(
            f"PromptAnalyzer: blocks={len(blocks)}, "
            f"subjects={len(structure.subjects)}, "
            f"actions={len(structure.actions)}, "
            f"env={len(structure.environment)}"
        )
        return structure

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_into_blocks(self, text: str) -> List[str]:
        """
        Split text into narrative blocks using paragraph breaks and scene markers.

        Handles:
        - Double-newline paragraph boundaries
        - Screenplay scene headers (INT./EXT./SCENE/CUT TO)
        - Single long run-on paragraphs → split by sentence groups (≤3 sentences)
        """
        # Prefer explicit paragraph boundaries
        if self._PARA_RE.search(text):
            raw_blocks = self._PARA_RE.split(text)
        elif self._SCENE_RE.search(text):
            raw_blocks = self._SCENE_RE.split(text)
        else:
            # Single paragraph — split into sentence groups of ≤3
            sentences = self._SENTENCE_RE.split(text.strip())
            raw_blocks = [
                " ".join(sentences[i:i + 3])
                for i in range(0, len(sentences), 3)
            ]

        cleaned = [b.strip() for b in raw_blocks if b.strip()]
        return cleaned if cleaned else [text.strip()]

    def _extract_keywords(self, text_lower: str, keyword_set: frozenset) -> List[str]:
        """Return all keywords from `keyword_set` that appear in `text_lower`."""
        found = []
        for kw in keyword_set:
            if kw in text_lower:
                found.append(kw)
        return found

    def _extract_subjects(self, blocks: List[str]) -> List[str]:
        """
        Extract a representative subject phrase from each block.

        Heuristic: the first noun-like phrase (up to 5 words) before the
        first verb or preposition is used as the subject string.
        """
        subjects = []
        # Simple noun-phrase heuristic — no NLTK dependency required
        _SPLIT_RE = re.compile(
            r'\b(?:is|was|are|were|at|in|on|beside|near|above|below|with|and|'
            r'during|where|who|that|which|while|beneath|under|over)\b',
            re.IGNORECASE
        )
        for block in blocks:
            sentence = block.split(".")[0].strip()
            parts = _SPLIT_RE.split(sentence, maxsplit=1)
            candidate = parts[0].strip()
            # Keep only first 5 words
            words = candidate.split()[:5]
            subject = " ".join(words).strip(" ,")
            if subject:
                subjects.append(subject)
        return subjects
