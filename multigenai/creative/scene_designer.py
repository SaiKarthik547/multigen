"""
Scene Designer — Phase 7 Creative Layer

Converts user intent into a structured SceneBlueprint.
Isolates creative decisions from the diffusion engine.

Boundary contract (enforced by PromptCompiler)
----------------------------------------------
SceneDesigner MAY add:
  - Semantic subject descriptors
  - Environment/context words
  - Camera and lighting descriptions
  - Atmosphere/mood words

SceneDesigner MUST NOT add:
  - Quality tokens (8k, masterpiece, ultra-detailed, sharp focus)
  - Model-specific tuning tokens
  - Negative prompt content

Quality tokens are PromptCompiler's exclusive responsibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pydantic import BaseModel
from multigenai.llm.schema_validator import ImageGenerationRequest

if TYPE_CHECKING:
    from multigenai.llm.schema_validator import VideoGenerationRequest


class SceneBlueprint(BaseModel):
    """
    Structured scene description produced by SceneDesigner.
    All fields are semantic — no quality tokens allowed in subject.
    """
    subject: str
    character_details: str
    environment: str
    lighting: str
    atmosphere: str
    camera_description: str
    rendering_style: str
    negative_prompt: str = ""


# ---------------------------------------------------------------------------
# Style → lighting/atmosphere expansion table
# Must NOT include quality tokens — only semantic descriptors
# ---------------------------------------------------------------------------
_STYLE_MAP = {
    "cinematic":       ("dramatic directional lighting, volumetric shadows", "cinematic mood"),
    "photorealistic":  ("natural soft lighting, ambient occlusion", "realistic atmosphere"),
    "anime":           ("flat cel shading, bright highlights", "vibrant anime mood"),
    "watercolor":      ("diffuse soft lighting, pastel tones", "painterly atmosphere"),
    "sketch":          ("high-contrast line lighting", "artistic sketch atmosphere"),
    "dark-fantasy":    ("deep chiaroscuro lighting, rim light", "dark foreboding mood"),
    "sci-fi":          ("neon glow, blue-tinted ambient light", "futuristic atmosphere"),
}

_DEFAULT_LIGHTING  = "volumetric lighting, dramatic shadows"
_DEFAULT_ATMOSPHERE = "cinematic mood"


class SceneDesigner:
    """
    Translates raw generation requests into a semantic SceneBlueprint.

    Expands style/camera presets and scales environmental detail.
    Does NOT inject quality tokens (8k, masterpiece, etc.) — that is
    strictly PromptCompiler's responsibility.
    """

    def design(self, request: ImageGenerationRequest) -> SceneBlueprint:
        """
        Convert an ImageGenerationRequest into a SceneBlueprint.

        Args:
            request: Validated ImageGenerationRequest from schema_validator.

        Returns:
            SceneBlueprint with semantic fields populated.
        """
        # Camera
        cam_desc = (request.camera or "medium shot").strip()

        # Environment detail level → descriptive words
        detail_level = request.environment_detail_level
        if detail_level >= 0.8:
            detail_words = "intricate, highly detailed, rich background"
        elif detail_level <= 0.3:
            detail_words = "clean, focused, minimalist background"
        else:
            detail_words = "detailed background"

        # Style → lighting + atmosphere (NO quality tokens)
        style_key = (request.style or "cinematic").lower().strip()
        lighting, atmosphere = _STYLE_MAP.get(
            style_key, (_DEFAULT_LIGHTING, _DEFAULT_ATMOSPHERE)
        )

        # rendering_style carries the style name — semantic only, no quality tokens
        rendering_style = style_key  # e.g. "cinematic", "anime" — not "masterpiece"

        return SceneBlueprint(
            subject=request.prompt,          # raw user prompt — no tokens added here
            character_details="",            # Phase 9: identity hook
            environment=f"{detail_words} environment",
            lighting=lighting,
            atmosphere=atmosphere,
            camera_description=cam_desc,
            rendering_style=rendering_style,
            negative_prompt=request.negative_prompt,
        )

    def design_video(self, request: "VideoGenerationRequest", scene_index: int = 0) -> SceneBlueprint:
        """
        Produce a motion-aware SceneBlueprint from a VideoGenerationRequest.

        Used by GenerationManager to run the video's own prompt through the
        creative layer. Temporal lighting and motion phrasing are injected here.

        Args:
            request: Validated VideoGenerationRequest.
            scene_index: Current segment index (for camera trajectory).

        Returns:
            SceneBlueprint tuned for temporal/video generation.
        """
        # Motion Improvement 1: Motion token injection
        base_motion = "natural body movement, subtle environmental motion, walking animation, dynamic motion blur"
        
        # Motion Improvement 3: Camera trajectory system
        trajectories = {
            0: "slow tracking shot",
            1: "cinematic pan",
            2: "dolly forward"
        }
        cam_desc = trajectories.get(scene_index % 3, "cinematic medium shot")

        motion = request.motion_hint.strip() if request.motion_hint else ""
        lighting = "continuous natural lighting, soft temporal shadows"
        atmosphere = f"fluid motion, temporal coherence, {base_motion}"
        if motion:
            atmosphere = f"{atmosphere}, {motion}"

        return SceneBlueprint(
            subject=request.prompt,      # raw prompt — quality tokens added by PromptCompiler
            character_details="",        # Phase 9: identity hook
            environment="dynamic environment with motion",
            lighting=lighting,
            atmosphere=atmosphere,
            camera_description=cam_desc,
            rendering_style="cinematic",
            negative_prompt=request.negative_prompt,
        )

