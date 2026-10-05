"""
Pydantic v2 schemas for all generation request types.

All engines receive one of these validated request objects.
The schema enforces required/optional fields and prmultigenaides
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
    use_refiner: bool = Field(
        default=False,
        description="SDXL refiner adds a second model load/VRAM cycle. Keep it "
                    "off until the primary image -> Ovi pipeline is stable.",
    )

    # Technical Controls
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    seed: Optional[int] = 42
    num_inference_steps: int = Field(default=30, ge=10, le=100,
                                      description="Denoising steps for base pass.")
    guidance_scale: float = Field(default=6.5, ge=1.0, le=20.0)

    # Refiner Specifics (Phase 13)
    refiner_num_inference_steps: int = Field(default=20, ge=5, le=100)
    refiner_strength: float = Field(default=0.15, ge=0.0, le=1.0)

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
    """Validated request for the Ovi joint audio-video backend.

    The Ovi-era contract is variant-driven: fps, frame count, guidance
    defaults and duration come from the selected Ovi variant (see
    ``multigenai.ovi.contracts``), not from free user parameters. Legacy
    SVD-era fields are kept for backward compatibility but are ignored by
    the Ovi path (RIFE interpolation in particular is retired from the
    cinematic pipeline — Ovi is natively 24 FPS).
    """
    prompt: str = Field(..., min_length=3)
    negative_prompt: str = ""
    character_id: Optional[str] = None
    scene_id: Optional[str] = None
    style_id: Optional[str] = None

    # --- Ovi variant contract (authoritative for the fusion backend) ---
    variant: str = Field(
        default="720x720_5s",
        description="Ovi checkpoint variant: 720x720_5s | 960x960_5s | 960x960_10s.",
    )
    solver: str = Field(default="unipc", description="unipc | dpm++ | euler.")
    shift: float = Field(default=5.0, ge=1.0, le=10.0)
    sample_steps: int = Field(
        default=50, ge=10, le=100,
        description="Diffusion steps. Upstream Ovi default is 50.",
    )
    video_guidance_scale: float = Field(default=4.0, ge=1.0, le=20.0)
    audio_guidance_scale: float = Field(default=3.0, ge=1.0, le=20.0)
    slg_layer: int = Field(
        default=11, ge=-1, le=50,
        description="Skip-layer guidance index (upstream recommendation: 11).",
    )
    video_negative_prompt: str = "jitter, bad hands, blur, distortion"
    audio_negative_prompt: str = "robotic, muffled, echo, distorted"
    audio_description: Optional[str] = Field(
        default=None,
        description="Sound-design line rendered as 'Audio: ...' in the Ovi prompt.",
    )

    # --- Resolution (must be /32; engine snaps area to the variant target) ---
    # NOTE: 704x704 is what the /32 snap of the 720x720_5s target resolves to
    # (round(22.5) == 22 in snap_hw_to_multiple_of_32), i.e. the real native
    # T2V frame size; I2V resizes the keyframe to the variant area with /32
    # granularity at runtime.
    num_frames: int = Field(
        default=121, ge=4, le=250,
        description="Advisory frame count. The Ovi variant decides the real "
                    "temporal length (121 for 5s, 241 for 10s @ 24 FPS).",
    )
    frame_duration: float = Field(
        default=0.5, ge=0.1,
        description="Legacy per-frame duration hint. Unused by the Ovi path."
    )
    fps: int = Field(
        default=24, ge=4, le=60,
        description="Advisory fps. Ovi output is native 24 FPS.",
    )
    width: int = Field(default=704, ge=256, le=1920)
    height: int = Field(default=704, ge=256, le=1080)
    seed: Optional[int] = None

    # --- Keyframe orchestration (image -> Ovi I2V handoff) ---
    generate_keyframes: bool = Field(
        default=True,
        description="Let the orchestrator generate an SDXL keyframe for the "
                    "first scene and feed it to Ovi as the I2V reference.",
    )

    # --- Identity (advisory only — Ovi conditions via the image channel) ---
    identity_threshold: float = Field(
        default=0.55, ge=0.0, le=1.0,
        description="Cosine similarity threshold for identity drift enforcement."
    )
    identity_name: Optional[str] = None
    identity_strength: float = Field(
        default=0.8, ge=0.0, le=1.5,
        description="Advisory identity strength (no latent conditioning in Ovi)."
    )

    # --- Legacy SVD-era fields (unused by the Ovi backend) ---
    temporal_strength: float = Field(
        default=0.25, ge=0.0, le=1.0,
        description="Legacy motion strength. Unused by the Ovi path."
    )
    motion_hint: str = Field(
        default="",
        description="Optional subtle motion suffix appended to prompt."
    )
    sampling_steps: int = Field(
        default=50, ge=10, le=100,
        description="Legacy alias; sample_steps takes precedence.",
    )
    guidance_scale: float = Field(default=5.0, ge=1.0, le=20.0)
    interpolate: bool = Field(
        default=False,
        description="RIFE interpolation is retired from the Ovi path "
                    "(native 24 FPS output). Legacy backends may opt in.",
    )
    interpolation_factor: int = Field(
        default=1, ge=1, le=4,
        description=(
            "Frame multiplication factor for legacy backends only. "
            "Output frames = n + (n-1)*(factor-1)."
        )
    )

    @field_validator("variant")
    @classmethod
    def validate_variant(cls, v: str) -> str:
        from multigenai.ovi.contracts import KNOWN_VARIANTS
        if v not in KNOWN_VARIANTS:
            raise ValueError(
                f"Unknown Ovi variant '{v}'. Known: {sorted(KNOWN_VARIANTS)}"
            )
        return v

    @model_validator(mode='after')
    def validate_request(self):
        # Ovi snaps dimensions to multiples of 32 (Wan VAE 16x + patch 2).
        # 32-divisibility is what upstream's snap_hw_to_multiple_of_32
        # guarantees, and it keeps the latent H/W even for the (1,2,2) patch.
        if self.width % 32 != 0 or self.height % 32 != 0:
            raise ValueError(
                f"Ovi requires dimensions divisible by 32. Got {self.width}x{self.height}"
            )
        # interpolation_factor must be 1 when interpolation is disabled
        if not self.interpolate and self.interpolation_factor != 1:
            raise ValueError("interpolation_factor must be 1 when interpolate=False")
        return self


class AudioGenerationRequest(BaseModel):
    """Validated request for the Audio Engine (Phase 6 stub)."""
    prompt: str = Field(..., min_length=3)
    character_id: Optional[str] = None
    audio_type: str = "voice"          # voice | music | sfx | ambient
    emotion: str = "neutral"           # neutral | happy | sad | angry | fearful
    duration_seconds: float = Field(default=5.0, ge=0.5, le=300.0)
    output_format: str = "wav"         # wav | mp3 | flac
    sampling_steps: int = Field(default=25, ge=10, le=100)
    guidance_scale: float = Field(default=4.5, ge=1.0, le=20.0)
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


class CodeGenerationRequest(BaseModel):
    """Validated request for the Code Engine (Phase 1 stub)."""
    prompt: str = Field(..., min_length=3)
    language: str = "python"           # python | javascript | ...
    file_name: Optional[str] = None


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
