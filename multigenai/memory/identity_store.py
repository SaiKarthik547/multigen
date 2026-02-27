"""
IdentityStore — Persistent character profile storage.

Architecture: the memory layer is modality-agnostic.
It knows nothing about image engines, IP-Adapters, or audio models.
Engines retrieve embeddings through IdentityResolver, not directly.

Schema history:
  v1: original (no face_embedding)
  v2: added face_embedding
  v3: style_embedding, voice_embedding promoted to first-class;
      image-specific fields (wardrobe, lighting_bias, personality_profile)
      moved into metadata{}; generic for all modalities.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from multigenai.core.exceptions import MemoryError as MGOSMemoryError
from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)

# Increment whenever CharacterProfile schema changes in a breaking way.
SCHEMA_VERSION = 3

# Expected embedding dimensions per modality — used for validation.
_EMBEDDING_DIMS: Dict[str, int] = {
    "face": 512,    # ArcFace / InsightFace buffalo_l
    "voice": 256,   # Speaker embedding (Phase 6 — reserved)
    "style": 768,   # Visual style vector (Phase 7 — reserved)
}


@dataclass
class CharacterProfile:
    """
    Modality-agnostic identity record for a character.

    Embeddings:
      face_embedding  — ArcFace 512-d (image/video identity)
      voice_embedding — Speaker 256-d (audio Phase 6)
      style_embedding — Visual style 768-d (Phase 7)

    Metadata dict holds any modality-specific or engine-specific data
    that would otherwise bloat the top-level schema. Examples:
      metadata["wardrobe"]           — {scene_id: outfit}
      metadata["lighting_bias"]      — "neutral"
      metadata["personality_profile"] — {trait: value}

    This design keeps the memory layer clean of engine concerns.
    """
    character_id: str
    name: str
    description: str = ""
    # --- Seed ---
    persistent_seed: Optional[int] = None
    # --- Modality embeddings ---
    face_embedding: Optional[List[float]] = None    # ArcFace 512-d
    voice_embedding: Optional[List[float]] = None   # Speaker embedding (Phase 6)
    style_embedding: Optional[List[float]] = None   # Style vector (Phase 7)
    # --- Phase 5: LoRA ---
    lora_reference: Optional[str] = None
    # --- Open dict for future / engine-specific data ---
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Embedding presence properties
    # ------------------------------------------------------------------

    @property
    def has_embedding(self) -> bool:
        """True if face_embedding is set and non-empty (backward-compat alias)."""
        return bool(self.face_embedding)

    @property
    def has_face_embedding(self) -> bool:
        """True if face_embedding is set and non-empty."""
        return bool(self.face_embedding)

    @property
    def has_voice_embedding(self) -> bool:
        """True if voice_embedding is set and non-empty."""
        return bool(self.voice_embedding)

    @property
    def has_style_embedding(self) -> bool:
        """True if style_embedding is set and non-empty."""
        return bool(self.style_embedding)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CharacterProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class IdentityStore:
    """
    JSON-backed character identity store.

    Each character is saved as ``<store_dir>/identities/<character_id>.json``.

    Engines must access characters through ``IdentityResolver``, not directly
    through this class, to maintain modality isolation.

    Usage:
        store = IdentityStore(store_dir="multigen_outputs/.memory")
        store.add(CharacterProfile(character_id="hero", name="Alice"))
        profile = store.get("hero")

        # Embedding helpers
        store.set_embedding("hero", modality="face", vector=[...])
        vec = store.get_embedding("hero", modality="face")
    """

    def __init__(self, store_dir: str = "multigen_outputs/.memory") -> None:
        self._root = pathlib.Path(store_dir) / "identities"
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, profile: CharacterProfile, overwrite: bool = True) -> None:
        """Persist a CharacterProfile to disk (includes schema_version)."""
        path = self._path(profile.character_id)
        if path.exists() and not overwrite:
            raise MGOSMemoryError(f"Character '{profile.character_id}' already exists.")
        payload = {"schema_version": SCHEMA_VERSION, **profile.to_dict()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOG.debug(f"Identity saved: {profile.character_id}")

    def get(self, character_id: str) -> Optional[CharacterProfile]:
        """Load a CharacterProfile by ID. Returns None if not found."""
        path = self._path(character_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            stored_version = data.pop("schema_version", None)
            data = self._migrate(character_id, data, stored_version)
            return CharacterProfile.from_dict(data)
        except Exception as exc:
            raise MGOSMemoryError(f"Failed to read identity '{character_id}': {exc}") from exc

    # Convenience alias — engines use this for clarity
    def get_profile(self, character_id: str) -> Optional[CharacterProfile]:
        """
        Load a CharacterProfile by ID. Returns None if not found.

        Alias for get() — preferred by engines that need lora_reference,
        persistent_seed, or metadata in addition to embeddings.
        """
        return self.get(character_id)

    def delete(self, character_id: str) -> bool:
        """Delete a character profile. Returns True if deleted, False if not found."""
        path = self._path(character_id)
        if path.exists():
            path.unlink()
            LOG.debug(f"Identity deleted: {character_id}")
            return True
        return False

    def list_all(self) -> List[str]:
        """Return sorted list of all registered character IDs."""
        return sorted(p.stem for p in self._root.glob("*.json"))

    def get_all(self) -> List[CharacterProfile]:
        """Return all stored CharacterProfile objects."""
        profiles = []
        for cid in self.list_all():
            p = self.get(cid)
            if p:
                profiles.append(p)
        return profiles

    # ------------------------------------------------------------------
    # Embedding helpers — modality-aware
    # ------------------------------------------------------------------

    def get_embedding(
        self, character_id: str, modality: str = "face"
    ) -> Optional[List[float]]:
        """
        Return the stored embedding for *character_id* and *modality*.

        Args:
            character_id: Character ID to look up.
            modality:     One of "face", "voice", "style". Unknown keys return None.

        Returns:
            List[float] or None if not found / modality unknown.
        """
        profile = self.get(character_id)
        if profile is None:
            return None
        embedding_field = {
            "face": profile.face_embedding,
            "voice": profile.voice_embedding,
            "style": profile.style_embedding,
        }.get(modality)
        return embedding_field  # None for unknown modality keys

    def set_embedding(
        self, character_id: str, modality: str, vector: List[float]
    ) -> None:
        """
        Persist an embedding for *character_id* under *modality*.

        Validates:
          - vector must be a non-empty list of floats
          - length must match expected dim for known modalities
          - numpy arrays are coerced to list automatically

        Args:
            character_id: Target character.
            modality:     One of "face", "voice", "style".
            vector:       Embedding vector.

        Raises:
            MGOSMemoryError: character not found, or embedding fails validation.
            ValueError:      Unknown modality.
        """
        # Coerce numpy arrays silently
        try:
            import numpy as np  # type: ignore
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()
        except ImportError:
            pass

        if not isinstance(vector, list) or len(vector) == 0:
            raise MGOSMemoryError(
                f"set_embedding: vector for '{character_id}/{modality}' must be "
                "a non-empty list of floats."
            )

        # Coerce to float (guard against int lists)
        try:
            vector = [float(x) for x in vector]
        except (TypeError, ValueError) as exc:
            raise MGOSMemoryError(
                f"set_embedding: vector elements must be numeric: {exc}"
            ) from exc

        # Dimension check for known modalities
        expected_dim = _EMBEDDING_DIMS.get(modality)
        if expected_dim is not None and len(vector) != expected_dim:
            raise MGOSMemoryError(
                f"set_embedding: '{modality}' embedding must have {expected_dim} dimensions, "
                f"got {len(vector)}."
            )

        profile = self.get(character_id)
        if profile is None:
            raise MGOSMemoryError(
                f"set_embedding: character '{character_id}' not found. "
                "Call store.add() first."
            )

        field_map = {"face": "face_embedding", "voice": "voice_embedding", "style": "style_embedding"}
        if modality not in field_map:
            raise ValueError(
                f"set_embedding: unknown modality '{modality}'. "
                f"Supported: {list(field_map)}"
            )

        setattr(profile, field_map[modality], vector)
        self.add(profile, overwrite=True)
        LOG.debug(
            f"IdentityStore: set '{modality}' embedding for '{character_id}' "
            f"(dim={len(vector)})."
        )

    # ------------------------------------------------------------------
    # Schema migration
    # ------------------------------------------------------------------

    @staticmethod
    def _migrate(character_id: str, data: dict, stored_version) -> dict:
        """
        Upgrade a stored profile dict from any older schema to SCHEMA_VERSION.

        Migration is non-sequential: checks `< 2` and `< 3` independently
        so a v1 profile jumping directly to v3 is handled correctly.

        Rules:
          - Never overwrite existing metadata keys (uses setdefault).
          - Fields that no longer exist at the top level are moved into
            metadata{} and removed from the top-level dict.
        """
        if stored_version is None or stored_version == SCHEMA_VERSION:
            return data

        version = stored_version if isinstance(stored_version, int) else 0
        LOG.warning(
            f"IdentityStore: schema_version mismatch for '{character_id}' "
            f"(stored={stored_version}, current={SCHEMA_VERSION}) — running migration."
        )

        if version < 2:
            # v1 had no face_embedding
            data.setdefault("face_embedding", None)
            LOG.info(f"IdentityStore: migrated '{character_id}' v1 → (face_embedding injected).")

        if version < 3:
            # v2/v1 had wardrobe, lighting_bias, personality_profile at top level.
            # Move them into metadata{} — never overwrite what's already there.
            meta = data.setdefault("metadata", {})
            for legacy_key in ("wardrobe", "lighting_bias", "personality_profile"):
                if legacy_key in data:
                    meta.setdefault(legacy_key, data.pop(legacy_key))
            # Also inject new embedding fields that didn't exist
            data.setdefault("voice_embedding", None)
            data.setdefault("style_embedding", None)
            LOG.info(
                f"IdentityStore: migrated '{character_id}' v{version} → v3 "
                "(legacy image fields moved to metadata)."
            )

        return data

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _path(self, character_id: str) -> pathlib.Path:
        safe = character_id.replace("/", "_").replace("\\", "_")
        return self._root / f"{safe}.json"
