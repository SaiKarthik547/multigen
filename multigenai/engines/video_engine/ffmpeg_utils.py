import numpy as np
import subprocess
import pathlib
import shutil
from typing import List, Union
from PIL import Image
from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)

def encode_video(
    frames: List[Union[Image.Image, np.ndarray]],
    out_path: Union[str, pathlib.Path],
    fps: int,
    seed: int
) -> "VideoResult":
    """
    Encode frames to mp4 via ffmpeg. Standard MGOS utility for all video engines.
    """
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class VideoResult:
        path: str
        frame_count: int
        fps: int
        seed: int
        success: bool = True
        error: Optional[str] = None

    frame_count = len(frames)
    if frame_count == 0:
        return VideoResult("", 0, fps, seed, False, "No frames to encode.")

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Get dimensions from first frame
    if isinstance(frames[0], Image.Image):
        width, height = frames[0].size
    else:
        height, width, _ = frames[0].shape

    LOG.info(f"FFmpeg: Encoding {frame_count} frames to {out_path} ({width}x{height} @ {fps}fps)")

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]

    try:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        
        for frame in frames:
            if process.poll() is not None:
                break
            
            if isinstance(frame, Image.Image):
                frame_data = np.asarray(frame.convert("RGB"), dtype=np.uint8)
            else:
                frame_data = frame.astype(np.uint8)
            
            process.stdin.write(frame_data.tobytes())
            del frame_data

        process.stdin.close()
        process.wait()
        
        if process.returncode != 0:
            stderr = process.stderr.read().decode()
            return VideoResult("", 0, fps, seed, False, f"FFmpeg Error: {stderr}")

        return VideoResult(str(out_path), frame_count, fps, seed, True)
        
    except Exception as e:
        LOG.error(f"FFmpeg encoding failed: {e}")
        return VideoResult("", 0, fps, seed, False, str(e))
