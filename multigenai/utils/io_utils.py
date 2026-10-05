import subprocess
import tempfile
import pathlib
from typing import Optional
import numpy as np
from scipy.io import wavfile
from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)

def save_video(
    output_path: str,
    video_numpy: np.ndarray,
    audio_numpy: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    fps: int = 24,
) -> str:
    """
    Combine a sequence of video frames with an optional audio track and save as an MP4 via FFmpeg.
    Replaces legacy moviepy implementation for MGOS Phase 6.
    """
    # 1. Validate and Preprocess
    assert video_numpy.ndim == 4, "video_numpy must have shape (C, F, H, W)"
    video_numpy = video_numpy.transpose(1, 2, 3, 0) # (F, H, W, C)
    
    if video_numpy.max() <= 1.0:
        video_numpy = ((np.clip(video_numpy, -1, 1) + 1) / 2 * 255).astype(np.uint8)
    else:
        video_numpy = video_numpy.astype(np.uint8)

    num_frames, height, width, channels = video_numpy.shape
    out_path = pathlib.Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. FFmpeg Command Construction
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24" if channels == 3 else "gray",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-", # Video from stdin
    ]

    temp_audio = None
    if audio_numpy is not None:
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wavfile.write(temp_audio.name, sample_rate, (audio_numpy * 32767).astype(np.int16))
        cmd.extend(["-i", temp_audio.name])

    cmd.extend([
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
    ])

    if audio_numpy is not None:
        cmd.extend(["-acodec", "aac", "-shortest"])
    else:
        cmd.append("-an")

    cmd.append(str(out_path))

    # 3. Execution
    try:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for frame in video_numpy:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        process.wait()

        if process.returncode != 0:
            err = process.stderr.read().decode()
            LOG.error(f"FFmpeg Error: {err}")
            raise RuntimeError(f"FFmpeg failed: {err}")

    finally:
        if temp_audio:
            pathlib.Path(temp_audio.name).unlink(missing_ok=True)

    return str(out_path)
