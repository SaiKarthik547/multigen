"""
StyleRegistry — Cinematic style profile storage.

StyleProfiles are reusable across images, video, documents, and presentations
to maintain a consistent visual world across all modalities.

Phase 2 will hook this into the PromptEngine for automatic style injection.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)


@dataclass
class StyleProfile:
    """
    Defines a complete cinematic style for a project or scene.

    Attributes:
        style_id:        Unique identifier (e.g. "noir-thriller", "pastel-dream").
        name:            Human-readable display name.
        color_palette:   List of hex color codes that define the palette.
        contrast_level:  "low" | "medium" | "high" | "extreme"
        film_grain:      "none" | "subtle" | "medium" | "heavy"
        lens_type:       "wide" | "standard" | "telephoto" | "macro" | "fisheye"
        atmosphere_tags: Descriptive words injected into prompts (e.g. ["moody","cinematic"]).
        negative_tags:   Tags to exclude (e.g. ["cartoon","anime","sketch"]).
        document_theme:  Optional theme name for document/PPT styling.
    """
    style_id: str
    name: str
    description: str = ""
    color_palette: List[str] = field(default_factory=list)
    contrast_level: str = "medium"
    film_grain: str = "subtle"
    lens_type: str = "standard"
    atmosphere_tags: List[str] = field(default_factory=list)
    negative_tags: List[str] = field(default_factory=list)
    document_theme: Optional[str] = None

    def to_prompt_fragment(self) -> str:
        """Return a comma-joined prompt fragment for injection into generation prompts."""
        parts = list(self.atmosphere_tags)
        if self.lens_type != "standard":
            parts.append(f"{self.lens_type} lens")
        if self.film_grain not in ("none", ""):
            parts.append(f"{self.film_grain} film grain")
        return ", ".join(parts)

    def to_negative_fragment(self) -> str:
        """Return a comma-joined negative prompt fragment."""
        return ", ".join(self.negative_tags)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StyleProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Built-in style presets loaded on first access
_BUILTIN_STYLES: List[dict] = [
    {
        "style_id": "cinematic-dark",
        "name": "Cinematic Dark",
        "description": "Hollywood-style dark, moody, high-contrast visuals.",
        "color_palette": ["#0a0a0a", "#1a1a2e", "#16213e", "#e94560"],
        "contrast_level": "high",
        "film_grain": "medium",
        "lens_type": "standard",
        "atmosphere_tags": ["cinematic", "dramatic lighting", "moody", "photorealistic", "8k"],
        "negative_tags": ["cartoon", "anime", "sketch", "watercolor", "flat", "bright"],
    },
    {
        "style_id": "pastel-dream",
        "name": "Pastel Dream",
        "description": "Soft, dreamy pastel aesthetic — ideal for fantasy or lifestyle content.",
        "color_palette": ["#ffd6e0", "#ffefef", "#c3f6f3", "#e8d5f5"],
        "contrast_level": "low",
        "film_grain": "none",
        "lens_type": "standard",
        "atmosphere_tags": ["soft lighting", "dreamy", "pastel", "ethereal", "bokeh"],
        "negative_tags": ["dark", "gloomy", "harsh shadows", "horror"],
    },
    {
        "style_id": "documentary-raw",
        "name": "Documentary Raw",
        "description": "Naturalistic, handheld, real-world documentary look.",
        "color_palette": ["#d4c5a9", "#b8a99a", "#6d7463", "#3e4035"],
        "contrast_level": "medium",
        "film_grain": "heavy",
        "lens_type": "wide",
        "atmosphere_tags": ["documentary", "natural light", "realistic", "raw", "handheld"],
        "negative_tags": ["studio lighting", "perfect", "glossy", "airbrushed"],
    },
]


class StyleRegistry:
    """
    JSON-backed cinematic style profile registry.

    Pre-loaded with built-in styles. Custom styles are saved to
    `<store_dir>/styles/<style_id>.json`.

    Usage:
        sr = StyleRegistry()
        sr.register(StyleProfile(style_id="custom", name="My Style", ...))
        profile = sr.get("cinematic-dark")   # built-in
    """

    def __init__(self, store_dir: str = "multigen_outputs/.memory") -> None:
        self._root = pathlib.Path(store_dir) / "styles"
        self._root.mkdir(parents=True, exist_ok=True)
        self._builtins: Dict[str, StyleProfile] = {
            d["style_id"]: StyleProfile.from_dict(d) for d in _BUILTIN_STYLES
        }

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, profile: StyleProfile, overwrite: bool = True) -> None:
        """Persist a custom StyleProfile to disk."""
        path = self._path(profile.style_id)
        if path.exists() and not overwrite:
            raise ValueError(f"Style '{profile.style_id}' already exists.")
        path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
        LOG.debug(f"Style saved: {profile.style_id}")

    def get(self, style_id: str) -> Optional[StyleProfile]:
        """Return a StyleProfile (custom first, then built-in). None if not found."""
        disk_path = self._path(style_id)
        if disk_path.exists():
            try:
                return StyleProfile.from_dict(
                    json.loads(disk_path.read_text(encoding="utf-8"))
                )
            except Exception as exc:
                LOG.warning(f"Could not parse style '{style_id}': {exc}")
        return self._builtins.get(style_id)

    def delete(self, style_id: str) -> bool:
        """Delete a custom style (cannot delete built-ins)."""
        path = self._path(style_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_all(self) -> List[str]:
        """Return all style IDs (built-in + custom, sorted)."""
        custom = {p.stem for p in self._root.glob("*.json")}
        return sorted(set(self._builtins.keys()) | custom)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _path(self, style_id: str) -> pathlib.Path:
        safe = style_id.replace("/", "_").replace("\\", "_")
        return self._root / f"{safe}.json"
