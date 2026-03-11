"""
Motion Quality Diagnostics — Phase 16

Analyzes a generated video using lightweight OpenCV optical flow to
determine the magnitude of motion. 

Helps distinguish between:
- Rigid motion (good, high magnitude)
- Morphing diffusion / Slideshow (bad, near-zero magnitude)

Expected AnimateDiff motion magnitude: 0.3 – 2.5
Near zero indicates slideshow effect.
"""

import cv2
import numpy as np
import sys


def optical_flow(video_path: str) -> float:
    """
    Measures the mean optical flow magnitude across the video.
    Returns the mean magnitude.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at {video_path}")

    ret, frame1 = cap.read()
    if not ret:
        cap.release()
        return 0.0

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    flows = []

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prvs, next_frame, None,
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        magnitude = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
        flows.append(magnitude)

        prvs = next_frame

    cap.release()

    if not flows:
        return 0.0

    return float(np.mean(flows))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python motion_flow_check.py <video_path>")
        sys.exit(1)

    path = sys.argv[1]
    try:
        mag = optical_flow(path)
        print(f"[{path}] Motion Quality:")
        print(f"  Mean Flow Magnitude: {mag:.3f} (Target: 0.3 - 2.5)")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
