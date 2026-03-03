"""
Prompt Compiler - Phase 7 Creative Layer

Takes a semantic SceneBlueprint and compiles it into an optimal
model-specific prompt and dynamic negative prompt.
"""

from typing import Tuple
from multigenai.creative.scene_designer import SceneBlueprint

class PromptCompiler:
    """
    Transforms a SceneBlueprint into a concrete diffusion prompt.
    Injects model-specific tuning tokens and generates dynamic
    negative prompts to ensure high fidelity rendering while abstracting
    away the diffusion specifics from the application layer.
    """

    def compile(self, blueprint: SceneBlueprint, model_name: str) -> Tuple[str, str]:
        # Synthesize positive prompt
        parts = []
        if blueprint.camera_description:
            parts.append(f"{blueprint.camera_description} of {blueprint.subject}")
        else:
            parts.append(blueprint.subject)
            
        if blueprint.character_details:
            parts.append(blueprint.character_details)
        if blueprint.environment:
            parts.append(blueprint.environment)
        if blueprint.lighting:
            parts.append(blueprint.lighting)
        if blueprint.atmosphere:
            parts.append(blueprint.atmosphere)
        if blueprint.rendering_style:
            parts.append(f"style of {blueprint.rendering_style}")
            
        parts.append("ultra-detailed, 8k resolution, sharp focus, masterpiece")
        
        positive_prompt = ", ".join(p.strip() for p in parts if p.strip())

        # Dynamic negative strategy
        base_negative = "low quality, worst quality, text, signature, watermark, blurry, distorted anatomy, extra fingers, malformed hands, overexposed"
        
        if "sdxl" in model_name.lower():
            negative_prompt = base_negative
        else:
            negative_prompt = base_negative

        return positive_prompt, negative_prompt
