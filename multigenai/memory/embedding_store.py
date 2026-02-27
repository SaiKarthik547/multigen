"""
EmbeddingStore — In-memory vector embedding store.

Phase 3 stub: provides the interface used by all engines.
Phase 3 will replace the in-memory dict with a real vector database
(ChromaDB, FAISS, or Pinecone) while keeping this interface identical.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from multigenai.core.exceptions import EmbeddingStoreError
from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)


class EmbeddingStore:
    """
    In-memory key→vector store with cosine similarity search.

    Interface is intentionally vector-DB-compatible so Phase 3 can
    drop in ChromaDB/FAISS without changing callers.

    Usage:
        store = EmbeddingStore()
        store.store("alice_face", [0.1, 0.9, 0.3, ...])
        results = store.similarity_search([0.1, 0.8, 0.3, ...], top_k=3)
    """

    def __init__(self) -> None:
        self._store: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(self, key: str, vector: List[float], metadata: Optional[Any] = None) -> None:
        """
        Store an embedding vector under a key.

        Args:
            key:      Unique identifier (e.g. "alice_face_v1").
            vector:   Embedding vector as a list of floats.
            metadata: Optional metadata dict attached to this entry.
        """
        if not vector:
            raise EmbeddingStoreError(f"Cannot store empty vector for key '{key}'.")
        self._store[key] = vector
        self._metadata[key] = metadata
        LOG.debug(f"Embedding stored: key={key}, dim={len(vector)}")

    def retrieve(self, key: str) -> Optional[List[float]]:
        """Return the vector for a given key, or None if not found."""
        return self._store.get(key)

    def delete(self, key: str) -> bool:
        """Delete an entry. Returns True if deleted, False if not found."""
        if key in self._store:
            del self._store[key]
            self._metadata.pop(key, None)
            return True
        return False

    def list_keys(self) -> List[str]:
        """Return all stored keys."""
        return sorted(self._store.keys())

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def similarity_search(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Return the top-k most similar keys by cosine similarity.

        Args:
            query_vector: The query embedding.
            top_k:        Number of results to return.

        Returns:
            List of (key, cosine_similarity) tuples, sorted descending.
        """
        if not self._store:
            return []
        scores = [
            (key, self._cosine(query_vector, vec))
            for key, vec in self._store.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)
