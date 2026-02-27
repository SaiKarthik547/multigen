"""
ConsistencyEnforcer — Cross-modal and identity consistency utilities.

Phase 4: Real implementations for identity seed enforcement and
         face embedding drift detection (advisory only — no hard enforcement).

Phase 8: CLIP-based visual/text alignment (still stub).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

from multigenai.core.logging.logger import get_logger

if TYPE_CHECKING:
    from multigenai.llm.schema_validator import ImageGenerationRequest
    from multigenai.memory.identity_store import IdentityStore

LOG = get_logger(__name__)


class ConsistencyEnforcer:
    """
    Cross-modal consistency validator and enforcer.

    Phase 4:
      - inject_identity(): seeds a request from a character's persistent_seed.
      - enforce_seed(): returns the correct seed to use.
      - check_identity_drift(): cosine similarity between two 512-d embeddings.
      - enforce(): advisory pass-through (always True in Phase 4).

    Phase 8:
      - check_visual_text_alignment(): CLIP similarity (still NotImplementedError).
    """

    # ------------------------------------------------------------------
    # Phase 4 — Identity tools
    # ------------------------------------------------------------------

    def inject_identity(
        self,
        request: "ImageGenerationRequest",
        store: "IdentityStore",
    ) -> "ImageGenerationRequest":
        """
        Copy the character's persistent_seed into the request if the request
        has no seed set yet and the profile has a persistent_seed.

        Args:
            request: The generation request (may have identity_name set).
            store:   IdentityStore to look up the profile.

        Returns:
            The same request object (mutated in-place) or a copy if Pydantic
            model_copy is available.
        """
        if not getattr(request, "identity_name", None):
            return request

        profile = store.get(request.identity_name)
        if profile is None:
            LOG.debug(
                f"ConsistencyEnforcer: no profile found for '{request.identity_name}' — "
                "skipping seed injection."
            )
            return request

        if request.seed is None and profile.persistent_seed is not None:
            try:
                # Pydantic v2
                patched = request.model_copy(update={"seed": profile.persistent_seed})
            except AttributeError:
                # Pydantic v1 fallback
                patched = request.copy(update={"seed": profile.persistent_seed})
            LOG.debug(
                f"ConsistencyEnforcer: injected persistent_seed={profile.persistent_seed} "
                f"for identity '{request.identity_name}'."
            )
            return patched

        return request

    def enforce_seed(self, request: "ImageGenerationRequest", profile) -> Optional[int]:
        """
        Return the seed to use for this generation.

        Priority:
          1. request.seed  (explicit caller preference)
          2. profile.persistent_seed  (character consistency)
          3. None  (random each time)

        Args:
            request: Validated generation request.
            profile: CharacterProfile (may be None).

        Returns:
            int seed or None.
        """
        if request.seed is not None:
            return request.seed
        if profile is not None and getattr(profile, "persistent_seed", None) is not None:
            return profile.persistent_seed
        return None

    def check_embedding_drift(
        self,
        frame_embedding: List[float],
        reference_embedding: List[float],
    ) -> float:
        """
        Return the cosine similarity between two embeddings of any modality.

        Range: -1.0 (opposite) to 1.0 (identical).
        Typical threshold for same-identity: ≥ 0.6.

        NOTE: Advisory only in Phase 4. No hard enforcement — callers
        can log the score or store it for later analysis.
        Drift enforcement activates in Phase 5 temporal system.

        Args:
            frame_embedding:     Embedding vector for the current frame/sample.
            reference_embedding: Reference embedding vector.

        Returns:
            cosine similarity as float.
        """
        if not frame_embedding or not reference_embedding:
            LOG.warning("ConsistencyEnforcer: one or both embeddings empty — returning 0.0")
            return 0.0

        # Pure Python / no torch dependency
        dot = sum(a * b for a, b in zip(frame_embedding, reference_embedding))
        mag_a = math.sqrt(sum(x * x for x in frame_embedding))
        mag_b = math.sqrt(sum(x * x for x in reference_embedding))

        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0

        similarity = dot / (mag_a * mag_b)
        LOG.debug(f"ConsistencyEnforcer: embedding drift score = {similarity:.4f}")
        return float(similarity)

    def check_identity_drift(
        self,
        frame_embedding: List[float],
        reference_embedding: List[float],
    ) -> float:
        """
        Deprecated alias for check_embedding_drift().

        Kept for backward compatibility with existing callers and tests.
        New code should call check_embedding_drift() directly.
        """
        LOG.debug(
            "ConsistencyEnforcer.check_identity_drift: "
            "deprecated — use check_embedding_drift() instead."
        )
        return self.check_embedding_drift(frame_embedding, reference_embedding)

    def enforce(self, result, request) -> bool:
        """
        Phase 4: Advisory pass-through — always returns True.

        Drift score is logged but not enforced. Hard enforcement activates in Phase 5.

        Args:
            result:  Generation result (ImageResult or similar).
            request: The original generation request.

        Returns:
            True (always — Phase 4 is advisory only).
        """
        LOG.debug(
            "ConsistencyEnforcer.enforce: advisory mode (Phase 4) — returning True."
        )
        return True

    # ------------------------------------------------------------------
    # Phase 8 — Visual / text alignment (stub)
    # ------------------------------------------------------------------

    def check_visual_text_alignment(self, image_path: str, description: str) -> float:
        """[Phase 8] Return CLIP similarity score between image and text."""
        raise NotImplementedError("CLIP alignment check activates in Phase 8.")
