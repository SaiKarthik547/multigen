"""
ScenePlanner — Breaks a script or narrative into ordered scene descriptors.

Phase 1: Rule-based sentence splitting + metadata tagging.
Phase 2: LLM-driven structured breakdown using structured_generate()
         with strict Pydantic validation — never free-text parsing.

Design rules:
  - LLM path uses structured_generate() with a JSON schema — deterministic
  - Falls back to heuristic split on any LLM failure
  - Provider injected via constructor (DI)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from multigenai.core.logging.logger import get_logger

if TYPE_CHECKING:
    from multigenai.llm.providers.base import LLMProvider

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------

@dataclass
class SceneDescriptor:
    """
    A single planned scene ready for the generation pipeline.

    Attributes:
        scene_id:        Unique scene index string (e.g. "s01").
        description:     Scene description (fed to image/video engine).
        characters:      Character IDs present in this scene.
        location:        Environment keyword (e.g. "forest", "city street").
        time_of_day:     "dawn" | "morning" | "noon" | "afternoon" | "dusk" | "night"
        duration_hint:   Approximate seconds of video this scene maps to.
        notes:           Extra context for the LLM or engine.
    """
    scene_id: str
    description: str
    characters: List[str] = field(default_factory=list)
    location: str = "unspecified"
    time_of_day: str = "noon"
    duration_hint: float = 3.0
    notes: str = ""

@dataclass
class VideoGenerationPlan:
    """
    Complete orchestrated plan for a multi-scene video.
    """
    scenes: List[SceneDescriptor] = field(default_factory=list)
    transitions: List[str] = field(default_factory=list)
    duration_estimate: float = 0.0


# ---------------------------------------------------------------------------
# Pydantic schemas for structured_generate()
# ---------------------------------------------------------------------------

class _SceneItem(BaseModel):
    """Schema for a single LLM-generated scene."""
    title: str = Field(description="Short scene title")
    description: str = Field(description="Detailed scene description for image generation")
    time_of_day: str = Field(
        default="noon",
        description="One of: dawn, morning, noon, afternoon, dusk, night"
    )
    location: str = Field(default="unspecified", description="Location or environment")
    characters: List[str] = Field(
        default_factory=list,
        description="List of character names present in this scene"
    )
    duration_hint: float = Field(
        default=3.0,
        description="Approximate duration in seconds"
    )


class _SceneListResponse(BaseModel):
    """Wrapper schema — LLM must output a JSON object with a scenes array."""
    scenes: List[_SceneItem]


# ---------------------------------------------------------------------------
# ScenePlanner
# ---------------------------------------------------------------------------

_SCENE_PLANNING_PROMPT_TEMPLATE = """Break the following script into individual scenes.

Script:
{script}

Return a JSON object with the key "scenes" containing an array of scene objects.
Each scene must have: title, description, time_of_day, location, characters (list), duration_hint (float seconds).
"""

_VALID_TIMES = {"dawn", "morning", "noon", "afternoon", "dusk", "night"}


class ScenePlanner:
    """
    Splits a narrative script into a sequence of SceneDescriptors.

    When a LLMProvider is injected, uses structured_generate() for
    rich, character-aware scene breakdown. Falls back to heuristic
    sentence-splitting on any LLM failure.

    Usage (heuristic, default):
        planner = ScenePlanner()
        scenes = planner.plan("A knight rides through a forest. He finds a sword.")

    Usage (LLM-backed):
        planner = ScenePlanner(provider=ctx.llm)
        scenes = planner.plan("A knight rides through a forest. He finds a sword.")
    """

    # Basic time-of-day keyword mapping (used in heuristic path)
    _TIME_KEYWORDS = {
        "dawn": ["dawn", "sunrise", "first light"],
        "morning": ["morning", "early"],
        "noon": ["noon", "midday", "day"],
        "afternoon": ["afternoon"],
        "dusk": ["dusk", "sunset", "twilight"],
        "night": ["night", "dark", "midnight", "evening"],
    }

    def __init__(self, provider: Optional["LLMProvider"] = None) -> None:
        """
        Args:
            provider: Optional LLM backend. If None, heuristic path is used.
        """
        self._provider = provider
        if provider:
            LOG.debug(f"ScenePlanner: LLM provider set ({type(provider).__name__})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, script: str, default_duration: float = 3.0) -> VideoGenerationPlan:
        """
        Parse a script string into a structured VideoGenerationPlan.

        Routes to plan_with_llm() when a provider is set; falls back to
        heuristic splitting on failure.

        Args:
            script:           Raw narrative / script text.
            default_duration: Seconds per scene (hint for video engine).

        Returns:
            VideoGenerationPlan containing the ordered scenes and metadata.
        """
        MAX_SCENES = 6
        scenes = []
        
        if self._provider is not None:
            try:
                scenes = self.plan_with_llm(script, default_duration)
            except Exception as exc:
                LOG.warning(
                    f"ScenePlanner: LLM planning failed ({exc}) "
                    "— falling back to heuristic"
                )
                
        if not scenes:
            scenes = self._heuristic_plan(script, default_duration)

        # Apply safety length limit (Phase 14)
        scenes = scenes[:MAX_SCENES]
            
        duration = sum(s.duration_hint for s in scenes)
        
        return VideoGenerationPlan(
            scenes=scenes,
            transitions=[],
            duration_estimate=duration
        )

    def plan_with_llm(
        self, script: str, default_duration: float = 3.0
    ) -> List[SceneDescriptor]:
        """
        LLM-driven structured scene planning.

        Uses structured_generate() with _SceneListResponse schema.
        Malformed JSON / validation failure → ProviderResponseFormatError
        (caught by plan() and falls back to heuristic).

        Args:
            script:           Raw script text.
            default_duration: Default duration for scenes without a hint.

        Returns:
            Ordered list of SceneDescriptor objects.

        Raises:
            ProviderResponseFormatError: if LLM output cannot be validated.
            ProviderUnavailableError: if provider is unreachable.
        """
        if self._provider is None:
            return self._heuristic_plan(script, default_duration)

        prompt = _SCENE_PLANNING_PROMPT_TEMPLATE.format(script=script)
        result: _SceneListResponse = self._provider.structured_generate(
            prompt, schema=_SceneListResponse
        )

        scenes: List[SceneDescriptor] = []
        for idx, item in enumerate(result.scenes):
            # Sanitize time_of_day — reject any free text the LLM might inject
            tod = item.time_of_day.lower().strip()
            if tod not in _VALID_TIMES:
                tod = "noon"

            scenes.append(SceneDescriptor(
                scene_id=f"s{idx + 1:02d}",
                description=item.description,
                characters=item.characters,
                location=item.location,
                time_of_day=tod,
                duration_hint=item.duration_hint or default_duration,
                notes=item.title,
            ))

        LOG.info(f"ScenePlanner (LLM): split script into {len(scenes)} scenes")
        return scenes

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    def _heuristic_plan(
        self, script: str, default_duration: float
    ) -> List[SceneDescriptor]:
        """Rule-based sentence-split fallback."""
        sentences = self._split_sentences(script)
        scenes: List[SceneDescriptor] = []
        for idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            scenes.append(SceneDescriptor(
                scene_id=f"s{idx + 1:02d}",
                description=sentence.strip(),
                time_of_day=self._detect_time_of_day(sentence),
                duration_hint=default_duration,
            ))
        LOG.info(f"ScenePlanner (heuristic): split script into {len(scenes)} scenes")
        return scenes

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split on sentence-ending punctuation."""
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    def _detect_time_of_day(self, text: str) -> str:
        lower = text.lower()
        for tod, keywords in self._TIME_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                return tod
        return "noon"
