"""
Temporal Stability Benchmark — Phase 16

Quantifies the visual stability of a generated video by measuring
frame-to-frame pixel differences.

Interpretation:
- Low std deviation → Stable motion
- High std deviation → Distortion spikes or flickers

Target range for AnimateDiff:
- Mean diff ≈ 6–12
- Std dev < 4
"""

import cv2
import numpy as np
import sys


def frame_difference(video_path: str) -> tuple[float, float]:
    """
    Measures frame-to-frame mean pixel difference.
    Returns (mean_diff, std_diff).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at {video_path}")

    prev = None
    diffs = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev is not None:
            # Absolute difference between consecutive frames
            diff = np.mean(np.abs(gray.astype(np.float32) - prev.astype(np.float32)))
            diffs.append(diff)

        prev = gray

    cap.release()

    if not diffs:
        return 0.0, 0.0

    return float(np.mean(diffs)), float(np.std(diffs))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python temporal_stability.py <video_path>")
        sys.exit(1)

    path = sys.argv[1]
    try:
        mean_d, std_d = frame_difference(path)
        print(f"[{path}] Temporal Stability:")
        print(f"  Mean Diff: {mean_d:.2f} (Target: 6-12)")
        print(f"  Std Dev:   {std_d:.2f} (Target: < 4)")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
