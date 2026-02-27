"""
FaceEncoder — CPU-side face embedding extraction via InsightFace.

Design:
  - Runs entirely on CPU via ONNX runtime (buffalo_l model).
  - No GPU / VRAM dependency — VRAM guard lives in ImageEngine._inject_identity().
  - Lazy load: InsightFace app is initialised on the first extract() call.
  - Returns a 512-dimensional ArcFace embedding vector (List[float]).
  - Raises IdentityEncoderError if insightface is not installed or
    if no face is detected in the reference image.

Usage:
    encoder = FaceEncoder()
    embedding = encoder.extract("path/to/face.png")  # List[float], len=512
"""

from __future__ import annotations

import pathlib
from typing import List, Optional

from multigenai.core.exceptions import IdentityEncoderError
from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)

# Buffalo_l name passed to insightface — embeddings are produced by ArcFace R100
_INSIGHTFACE_MODEL_NAME = "buffalo_l"
# ctx_id=-1 → CPU inference via ONNX runtime
_INSIGHTFACE_CTX_ID = -1


class FaceEncoder:
    """
    Extracts 512-d ArcFace face embeddings from reference images.

    CPU-only — uses ONNX runtime; no GPU memory consumed.

    Usage:
        enc = FaceEncoder()
        emb = enc.extract("reference.png")   # returns List[float] of length 512
        enc.reset()                           # optional — unloads the app
    """

    def __init__(self) -> None:
        self._app = None  # lazy — loaded on first extract() call

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def extract(self, image_path: str) -> List[float]:
        """
        Extract a 512-d face embedding from a reference image.

        Args:
            image_path: Path to a face image (PNG, JPEG, BMP …).

        Returns:
            List[float] of length 512.

        Raises:
            IdentityEncoderError: insightface not installed, no face detected,
                                  or embedding extraction failed.
        """
        path = pathlib.Path(image_path)
        if not path.exists():
            raise IdentityEncoderError(f"Image not found: {image_path}")

        app = self._get_app()

        try:
            import cv2
            import numpy as np

            img = cv2.imread(str(path))
            if img is None:
                raise IdentityEncoderError(f"Could not decode image: {image_path}")

        except ImportError as exc:
            raise IdentityEncoderError(
                f"opencv-python not installed — required for FaceEncoder: {exc}"
            ) from exc

        try:
            faces = app.get(img)
        except Exception as exc:
            raise IdentityEncoderError(
                f"InsightFace face detection failed on '{image_path}': {exc}"
            ) from exc

        if not faces:
            raise IdentityEncoderError(
                f"No face detected in image: {image_path}. "
                "Provide a clear frontal face photograph."
            )

        # Select the face with the largest bounding box area (primary subject)
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        embedding = best.embedding
        if embedding is None or len(embedding) == 0:
            raise IdentityEncoderError(
                f"ArcFace embedding is empty for '{image_path}'. "
                "Model may not be downloaded yet."
            )

        result: List[float] = embedding.tolist()
        LOG.info(
            f"FaceEncoder: extracted {len(result)}-d embedding "
            f"from '{path.name}' (faces found: {len(faces)})"
        )
        return result

    def reset(self) -> None:
        """Unload the InsightFace app to free CPU memory."""
        self._app = None
        LOG.debug("FaceEncoder: InsightFace app unloaded.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_app(self):
        """Lazy-load and return the InsightFace FaceAnalysis app."""
        if self._app is not None:
            return self._app

        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise IdentityEncoderError(
                f"insightface not installed — run: pip install insightface==0.7.3 onnxruntime==1.17.3 | error: {exc}"
            ) from exc

        LOG.info("FaceEncoder: loading InsightFace buffalo_l model (CPU/ONNX)…")
        try:
            app = FaceAnalysis(
                name=_INSIGHTFACE_MODEL_NAME,
                providers=["CPUExecutionProvider"],
            )
            # det_size=(640, 640) → standard detection resolution for buffalo_l
            app.prepare(ctx_id=_INSIGHTFACE_CTX_ID, det_size=(640, 640))
            self._app = app
            LOG.info("FaceEncoder: buffalo_l loaded successfully.")
        except Exception as exc:
            raise IdentityEncoderError(
                f"Failed to initialise InsightFace buffalo_l: {exc}"
            ) from exc

        return self._app
