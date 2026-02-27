"""
IdentityResolver — Central cross-engine helper for embedding retrieval.

Engines call IdentityResolver, never CharacterProfile directly.
This enforces modality isolation and centralises structured logging.

Usage:
    embedding = IdentityResolver.get_face_embedding("hero_01", store)
    embedding = IdentityResolver.resolve("hero_01", store, modality="voice")
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, List, Optional

from multigenai.core.logging.logger import get_logger

if TYPE_CHECKING:
    from multigenai.memory.identity_store import IdentityStore

LOG = get_logger(__name__)


class IdentityResolver:
    """
    Stateless helper: retrieves embeddings (and full profiles) from an
    IdentityStore on behalf of any engine.

    Design rules:
      - All methods are static — no engine should hold a resolver instance.
      - Logs structured JSON-compatible messages (no free-text warnings).
      - Returns None on any missing-data path; never raises.
    """

    # ------------------------------------------------------------------
    # Generic resolve
    # ------------------------------------------------------------------

    @staticmethod
    def resolve(
        identity_name: Optional[str],
        store: "IdentityStore",
        modality: str = "face",
    ) -> Optional[List[float]]:
        """
        Return the embedding for *identity_name* and *modality*.

        Logs a structured warning if the character or embedding is absent.

        Args:
            identity_name: Character ID to look up. Returns None if falsy.
            store:          IdentityStore instance.
            modality:       One of "face", "voice", "style".

        Returns:
            List[float] or None.
        """
        if not identity_name:
            return None

        profile = store.get(identity_name)
        if profile is None:
            LOG.warning(
                json.dumps({
                    "event": "identity_missing",
                    "character_id": identity_name,
                    "modality": modality,
                })
            )
            return None

        embedding_map = {
            "face":  getattr(profile, "face_embedding", None),
            "voice": getattr(profile, "voice_embedding", None),
            "style": getattr(profile, "style_embedding", None),
        }

        if modality not in embedding_map:
            # Unknown modality — return None cleanly, no crash
            LOG.warning(
                json.dumps({
                    "event": "identity_unknown_modality",
                    "character_id": identity_name,
                    "modality": modality,
                })
            )
            return None

        embedding = embedding_map[modality]
        if not embedding:
            LOG.warning(
                json.dumps({
                    "event": "embedding_missing",
                    "character_id": identity_name,
                    "modality": modality,
                })
            )
            return None

        return embedding

    # ------------------------------------------------------------------
    # Modality shorthands
    # ------------------------------------------------------------------

    @staticmethod
    def get_face_embedding(
        identity_name: Optional[str],
        store: "IdentityStore",
    ) -> Optional[List[float]]:
        """Return the 512-d ArcFace embedding for *identity_name*, or None."""
        return IdentityResolver.resolve(identity_name, store, modality="face")

    @staticmethod
    def get_voice_embedding(
        identity_name: Optional[str],
        store: "IdentityStore",
    ) -> Optional[List[float]]:
        """Return the speaker embedding for *identity_name* (Phase 6), or None."""
        return IdentityResolver.resolve(identity_name, store, modality="voice")

    @staticmethod
    def get_style_embedding(
        identity_name: Optional[str],
        store: "IdentityStore",
    ) -> Optional[List[float]]:
        """Return the visual style vector for *identity_name* (Phase 7), or None."""
        return IdentityResolver.resolve(identity_name, store, modality="style")
