"""
Pydantic v2 schemas for all generation request types.

All engines receive one of these validated request objects.
The schema enforces required/optional fields and provides
sane defaults so callers don't need to specify every field.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class CameraProfile(BaseModel):
    """Camera / shot parameters."""
    shot_type: str = "medium"           # close-up | medium | wide | aerial | macro
    angle: str = "eye-level"           # eye-level | low | high | bird | worm
    movement: str = "static"           # static | pan | tilt | dolly | handheld
    focal_length_mm: Optional[int] = None


class LightingProfile(BaseModel):
    """Lighting descriptor injected into prompts."""
    type: str = "natural"              # natural | studio | neon | candle | golden-hour
    direction: str = "front"           # front | side | back | overhead
    intensity: str = "medium"          # soft | medium | hard | dramatic


# ---------------------------------------------------------------------------
# Per-modality request schemas
# ---------------------------------------------------------------------------

class ImageGenerationRequest(BaseModel):
    """Validated request for the Image Engine."""
    prompt: str = Field(..., min_length=3, description="Base image prompt.")
    negative_prompt: str = ""
    character_id: Optional[str] = None
    scene_id: Optional[str] = None
    style_id: Optional[str] = None
    lighting: LightingProfile = Field(default_factory=LightingProfile)
    camera: CameraProfile = Field(default_factory=CameraProfile)
    width: int = Field(default=768, ge=64, le=4096)
    height: int = Field(default=768, ge=64, le=4096)
    num_inference_steps: int = Field(default=50, ge=10, le=150)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: Optional[int] = None
    # --- Phase 4: Identity conditioning ---
    identity_name: Optional[str] = None
    identity_strength: float = Field(
        default=0.8, ge=0.0, le=1.5,
        description=(
            "IP-Adapter identity conditioning strength. "
            "Range 0.0–1.5 (>1.0 for experimentation only; engine clamps to 1.0 for stability)."
        ),
    )

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be blank")
        return v.strip()



class VideoGenerationRequest(BaseModel):
    """Validated request for the Video Engine."""
    prompt: str = Field(..., min_length=3)
    negative_prompt: str = ""
    character_id: Optional[str] = None
    scene_id: Optional[str] = None
    style_id: Optional[str] = None
    num_frames: int = Field(default=4, ge=2, le=200)
    frame_duration: float = Field(default=0.5, ge=0.1)
    fps: int = Field(default=12, ge=8, le=60)
    width: int = Field(default=640, ge=64, le=1920)
    height: int = Field(default=640, ge=64, le=1080)
    seed: Optional[int] = None
    identity_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0,
        description="Cosine similarity threshold for identity drift detection."
    )
    # --- Phase 4: Identity conditioning (propagated to every frame) ---
    identity_name: Optional[str] = None
    identity_strength: float = Field(
        default=0.8, ge=0.0, le=1.5,
        description="IP-Adapter conditioning strength propagated to each frame."
    )


class AudioGenerationRequest(BaseModel):
    """Validated request for the Audio Engine (Phase 6 stub)."""
    prompt: str = Field(..., min_length=3)
    character_id: Optional[str] = None
    audio_type: str = "voice"          # voice | music | sfx | ambient
    emotion: str = "neutral"           # neutral | happy | sad | angry | fearful
    duration_seconds: float = Field(default=5.0, ge=0.5, le=300.0)
    output_format: str = "wav"         # wav | mp3 | flac
    # --- Phase 6: Voice identity (optional — advisory, not enforced yet) ---
    identity_name: Optional[str] = None
    identity_strength: float = Field(
        default=0.8, ge=0.0, le=1.5,
        description="Voice identity conditioning strength (Phase 6 — reserved)."
    )


class DocumentGenerationRequest(BaseModel):
    """Validated request for Document/Presentation engines (Phase 7 stub)."""
    prompt: str = Field(..., min_length=3)
    doc_type: str = "report"           # report | article | summary | presentation
    style_id: Optional[str] = None
    topic_keywords: List[str] = Field(default_factory=list)
    target_pages: int = Field(default=5, ge=1, le=100)
    include_images: bool = True
    output_format: str = "docx"        # docx | pdf | pptx


# ---------------------------------------------------------------------------
# Enhanced prompt output from PromptEngine
# ---------------------------------------------------------------------------

class EnhancedPrompt(BaseModel):
    """The final, validated, style-injected prompt ready for an engine."""
    original: str
    enhanced: str
    negative: str
    style_fragment: str = ""
    tokens_estimated: int = 0
    metadata: dict = Field(default_factory=dict)
