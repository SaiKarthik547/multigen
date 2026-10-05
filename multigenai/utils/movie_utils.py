import subprocess
import pathlib
import os
from typing import List
from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)

def stitch_cinematic_movie(
    video_segments: List[str],
    audio_segments: List[str],
    output_path: str
) -> bool:
    """
    Stitches multiple video and audio segments into a single cinematic MP4.
    Uses FFmpeg filter_complex for seamless transitions and audio-video alignment.
    """
    if not video_segments:
        LOG.error("No video segments to stitch.")
        return False
        
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    # Create FFmpeg file list for concatenation
    # Alternatively, use filter_complex for more control
    
    inputs = []
    for v in video_segments:
        inputs.extend(["-i", v])
    for a in audio_segments:
        inputs.extend(["-i", a])
        
    # Build filter_complex string
    # [0:v][1:v]... concat=n=N:v=1:a=0 [v]
    # [N:a][N+1:a]... concat=n=N:v=0:a=1 [a]
    
    num_scenes = len(video_segments)
    filter_str = ""
    
    # Video concat
    for i in range(num_scenes):
        filter_str += f"[{i}:v]"
    filter_str += f"concat=n={num_scenes}:v=1:a=0[v];"
    
    # Audio concat
    for i in range(num_scenes):
        filter_str += f"[{num_scenes + i}:a]"
    filter_str += f"concat=n={num_scenes}:v=0:a=1[a]"
    
    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_str,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        str(out)
    ]
    
    try:
        LOG.info(f"Stitching {num_scenes} scenes into final movie: {output_path}")
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        LOG.error(f"FFmpeg Stitching Error: {e.stderr.decode()}")
        return False
    except Exception as e:
        LOG.error(f"MGOS movie stitching failed: {e}")
        return False
