"""
Pydantic v2 schemas for all generation request types.

All engines receive one of these validated request objects.
The schema enforces required/optional fields and provides
sane defaults so callers don't need to specify every field.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


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

    # Creative Controls
    style: Optional[str] = "cinematic"
    camera: Optional[str] = "medium"
    environment_detail_level: float = Field(default=0.8, ge=0.0, le=1.0)

    # Model Controls
    model_name: str = "sdxl-base"
    use_refiner: bool = True

    # Technical Controls
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    seed: Optional[int] = 42
    num_inference_steps: int = Field(default=30, ge=10, le=100,
                                      description="Denoising steps for base (and refiner) pass.")

    # --- Phase 4 / 6 backward compatibility ---
    character_id: Optional[str] = None
    scene_id: Optional[str] = None
    identity_name: Optional[str] = None
    identity_strength: float = Field(default=0.8, ge=0.0, le=1.5)

    @field_validator("width", "height")
    @classmethod
    def validate_resolution(cls, v: int) -> int:
        if v % 64 != 0:
            raise ValueError(f"Resolution must be divisible by 64")
        return v

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be blank")
        return v.strip()



class VideoGenerationRequest(BaseModel):
    """Validated request for the VideoEngine (Phase 6 SVD-XT)."""
    prompt: str = Field(..., min_length=3)
    negative_prompt: str = ""
    character_id: Optional[str] = None
    scene_id: Optional[str] = None
    style_id: Optional[str] = None
    num_frames: int = Field(default=16, ge=4, le=60)
    frame_duration: float = Field(
        default=0.5, ge=0.1,
        description="Per-frame duration hint (seconds). Not used by SVD-XT directly; "
                    "output timing is controlled by fps. Reserved for Phase 10 audio sync."
    )
    fps: int = Field(default=8, ge=4, le=60)
    width: int = Field(default=1024, ge=256, le=1920)
    height: int = Field(default=576, ge=256, le=1080)
    seed: Optional[int] = None
    identity_threshold: float = Field(
        default=0.55, ge=0.0, le=1.0,
        description="Cosine similarity threshold for identity drift enforcement (Phase 5). 0.55 = enforced; was 0.85 advisory in Phase 4."
    )
    # --- Identity: stored + drift-tracked, latent conditioning planned Phase 9 ---
    identity_name: Optional[str] = None
    identity_strength: float = Field(
        default=0.8, ge=0.0, le=1.5,
        description="Reserved for Phase 9 latent identity conditioning. Currently advisory only."
    )
    # --- Phase 5: Temporal engine (now mapped to SVD-XT motion_bucket_id) ---
    temporal_strength: float = Field(
        default=0.25, ge=0.0, le=1.0,
        description="Motion strength. 0.0=static, 1.0=high motion. Mapped to SVD motion_bucket_id."
    )
    motion_hint: str = Field(
        default="",
        description="Optional subtle motion suffix appended to prompt (e.g. 'subtle walking motion')."
    )
    num_inference_steps: int = Field(
        default=25, ge=10, le=100,
        description="Denoising steps per frame. 25 is the native SVD-XT default."
    )

    @model_validator(mode='after')
    def validate_resolution(self):
        if self.width % 64 != 0 or self.height % 64 != 0:
            raise ValueError(f"SVD requires dimensions divisible by 64. Got {self.width}x{self.height}")
        return self


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
