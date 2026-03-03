"""
Scene Designer - Phase 7 Creative Layer

Converts user intent into a structured scene blueprint.
Abstracts the creative definition away from the diffusion logic.
"""

from pydantic import BaseModel
from multigenai.llm.schema_validator import ImageGenerationRequest

class SceneBlueprint(BaseModel):
    subject: str
    character_details: str
    environment: str
    lighting: str
    atmosphere: str
    camera_description: str
    rendering_style: str

class SceneDesigner:
    """
    Translates raw generation requests into a semantic blueprint
    by expanding style and camera presets, scaling environmental details,
    and isolating creative concerns from the diffusion engine.
    """

    def design(self, request: ImageGenerationRequest) -> SceneBlueprint:
        cam_desc = request.camera if request.camera else "cinematic medium shot"
        
        # Scale environment richness based on detail level
        if request.environment_detail_level >= 0.8:
            detail_words = "intricate, extremely detailed, highly complex, rich background"
        elif request.environment_detail_level <= 0.3:
            detail_words = "clean, focused, minimalist background, blank space"
        else:
            detail_words = "detailed background"

        # Expand basic style into descriptive lighting and atmosphere
        lighting = "volumetric lighting, dramatic shadows"
        atm = "cinematic mood"
        style = request.style if request.style else "photorealistic masterpiece"

        return SceneBlueprint(
            subject=request.prompt,
            character_details="",  # Phase 8 identity features hook
            environment=f"{detail_words} environment",
            lighting=lighting,
            atmosphere=atm,
            camera_description=cam_desc,
            rendering_style=style
        )
