"""Cinematic movie assembly — sequential AV segments -> one movie.

Ovi emits independent per-scene segments (each with synchronized audio).
Assembly joins them with short cross-fade transitions (ffmpeg ``xfade`` for
video, ``acrossfade`` for audio) so scene switches read as smooth dissolves
rather than hard cuts, then muxes the result into a single MP4.

Fallback ladder:
    1. xfade + acrossfade filter graph (needs ffprobe for durations)
    2. plain concat (previous behaviour) if ffprobe/xfade is unavailable
The function never raises: it returns False with a logged error instead.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
from typing import List, Optional, Sequence, Tuple

from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)

DEFAULT_TRANSITION_SECONDS = 0.5


def probe_duration(path: str) -> Optional[float]:
    """Media duration in seconds via ffprobe, or None when unavailable."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-print_format", "json",
                "-show_entries", "format=duration",
                str(path),
            ],
            check=True, capture_output=True, text=True, timeout=60,
        ).stdout
        return float(json.loads(out)["format"]["duration"])
    except FileNotFoundError:
        LOG.warning("ffprobe not found — crossfade transitions unavailable.")
        return None
    except Exception as exc:
        LOG.warning("ffprobe failed for %s: %s", path, exc)
        return None


def build_transition_filter_graph(
    num_segments: int,
    durations: Sequence[float],
    fade: float,
) -> Tuple[str, str]:
    """Build the xfade (video) + acrossfade (audio) filter_complex strings.

    Video chain: successive xfade joins, each offset at
    ``sum(durations[:k]) - k * fade`` — the standard crossfade accounting.

    Audio chain: successive acrossfade joins with the same fade length so the
    audio and video timelines stay in sync (both shrink by exactly ``fade``
    per join).

    Raises:
        ValueError: on degenerate inputs (fade <= 0, too-short segments).
    """
    if num_segments < 2:
        raise ValueError("transition graph needs at least 2 segments")
    if fade <= 0:
        raise ValueError("fade duration must be positive")
    if len(durations) < num_segments or any(d <= fade for d in durations[:num_segments]):
        raise ValueError(
            "every segment must be longer than the fade duration; "
            f"durations={list(durations)}, fade={fade}"
        )

    video_chain = ""
    offset = 0.0
    prev = "[0:v]"
    for k in range(1, num_segments):
        offset += durations[k - 1] - fade
        out = f"[vx{k}]"
        video_chain += (
            f"{prev}[{k}:v]xfade=transition=fade:"
            f"duration={fade:.3f}:offset={offset:.3f}{out};"
        )
        prev = out

    audio_chain = ""
    prev_a = f"[{num_segments}:a]"
    for k in range(1, num_segments):
        out = f"[ax{k}]"
        audio_chain += f"{prev_a}[{num_segments + k}:a]acrossfade=d={fade:.3f}{out};"
        prev_a = out

    return video_chain, audio_chain


def stitch_cinematic_movie(
    video_segments: List[str],
    audio_segments: List[str],
    output_path: str,
    transition_seconds: float = DEFAULT_TRANSITION_SECONDS,
) -> bool:
    """Stitch per-scene AV segments into one cinematic MP4.

    Args:
        video_segments: per-scene MP4 paths (video-only streams).
        audio_segments: per-scene WAV paths (16 kHz mono).
        output_path:    final MP4 path.
        transition_seconds: cross-fade length between consecutive scenes.

    Returns:
        True on success, False (with a logged error) otherwise.
    """
    if not video_segments:
        LOG.error("No video segments to stitch.")
        return False
    if len(audio_segments) != len(video_segments):
        LOG.error(
            "AV segment count mismatch: %d video vs %d audio.",
            len(video_segments), len(audio_segments),
        )
        return False
    missing = [p for p in (*video_segments, *audio_segments) if not pathlib.Path(p).exists()]
    if missing:
        LOG.error("Missing segment files before assembly: %s", missing)
        return False

    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    num = len(video_segments)
    fade = float(transition_seconds)

    durations: List[Optional[float]] = []
    use_crossfade = num > 1 and fade > 0
    if use_crossfade:
        durations = [probe_duration(v) for v in video_segments]
        use_crossfade = all(d is not None for d in durations) and all(
            d > fade + 0.5 for d in durations
        )
        if not use_crossfade:
            LOG.warning(
                "Crossfade skipped (ffprobe unavailable or segments too short); "
                "falling back to plain concat."
            )

    # ---- 1. Crossfade assembly -------------------------------------------
    if use_crossfade:
        try:
            video_chain, audio_chain = build_transition_filter_graph(
                num, [d for d in durations if d is not None], fade
            )
        except ValueError as exc:
            LOG.warning("Crossfade skipped: %s", exc)
            use_crossfade = False

    if use_crossfade:
        inputs: List[str] = []
        for v in video_segments:
            inputs += ["-i", str(v)]
        for a in audio_segments:
            inputs += ["-i", str(a)]

        filter_str = (
            f"{video_chain}{audio_chain}"
            f"[vx{num - 1}][ax{num - 1}]"
        )
        expected = sum(d for d in durations if d is not None) - (num - 1) * fade
        cmd = [
            "ffmpeg", "-y", *inputs,
            "-filter_complex", filter_str,
            "-map", f"[vx{num - 1}]", "-map", f"[ax{num - 1}]",
            "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            str(out),
        ]
        if _run_ffmpeg(cmd, f"crossfade stitching of {num} scenes"):
            _validate_output(str(out), expected)
            return True
        LOG.warning("Crossfade assembly failed — falling back to concat.")

    # ---- 2. Plain concat (fallback / single segment) ----------------------
    inputs: List[str] = []
    for v in video_segments:
        inputs += ["-i", str(v)]
    for a in audio_segments:
        inputs += ["-i", str(a)]

    filter_str = ""
    for i in range(num):
        filter_str += f"[{i}:v]"
    filter_str += f"concat=n={num}:v=1:a=0[v];"
    for i in range(num):
        filter_str += f"[{num + i}:a]"
    filter_str += f"concat=n={num}:v=0:a=1[a]"

    expected = None
    if num == 1:
        # Single segment: no concat filters needed, just remux cleanly.
        filter_str = None
        try:
            expected = probe_duration(video_segments[0])
        except Exception:
            expected = None

    cmd = [
        "ffmpeg", "-y", *inputs,
        *([] if filter_str is None else ["-filter_complex", filter_str]),
        *([] if filter_str is None else ["-map", "[v]", "-map", "[a]"]),
        "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        str(out),
    ]
    if _run_ffmpeg(cmd, f"concat stitching of {num} scenes"):
        _validate_output(str(out), expected)
        return True
    return False


def _run_ffmpeg(cmd: List[str], context: str) -> bool:
    try:
        LOG.info("FFmpeg: %s", context)
        subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
        return True
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else "<no stderr>"
        LOG.error("FFmpeg error [%s]: %s", context, stderr[-2000:])
        return False
    except Exception as e:
        LOG.error("FFmpeg assembly failed [%s]: %s", context, e)
        return False


def _validate_output(path: str, expected_duration: Optional[float]) -> None:
    """Post-assembly gate: the output must be a playable file whose video
    duration is in the right ballpark. Logs problems; never raises."""
    duration = probe_duration(path)
    if duration is None:
        LOG.warning("Movie validation: could not probe %s", path)
        return
    if duration <= 0:
        LOG.error("Movie validation: %s has non-positive duration %.3f", path, duration)
        return
    if expected_duration and abs(duration - expected_duration) > 1.0:
        LOG.warning(
            "Movie validation: duration %.2fs differs from expected %.2fs "
            "(audio/video drift?)", duration, expected_duration,
        )
    LOG.info("Movie assembled: %s (%.2fs)", path, duration)
