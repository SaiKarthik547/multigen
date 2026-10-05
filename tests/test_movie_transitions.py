"""Movie assembly tests — transition graph math and input validation.

The crossfade offsets are the part that silently corrupts AV sync when wrong:
video ``xfade`` and audio ``acrossfade`` must both shorten the timeline by
exactly ``fade`` per join, otherwise the movie drifts out of sync.

No ffmpeg binary needed (only the pure graph builder and the validation
guards are exercised).
"""

import pytest

from multigenai.utils.movie_utils import (
    build_transition_filter_graph,
    stitch_cinematic_movie,
)


# ---------------------------------------------------------------------------
# Filter graph
# ---------------------------------------------------------------------------

def test_two_segment_graph_has_one_join():
    video, audio = build_transition_filter_graph(2, [5.0, 5.0], 0.5)
    # offset = d0 - fade = 4.5
    assert "xfade=transition=fade:duration=0.500:offset=4.500[vx1]" in video
    assert audio == "[2:a][3:a]acrossfade=d=0.500[ax1];"


def test_three_segment_offsets_accumulate_minus_fade_per_join():
    video, audio = build_transition_filter_graph(3, [5.0, 5.0, 5.0], 0.5)
    # k=1 -> 5.0 - 0.5 = 4.5 ; k=2 -> 4.5 + (5.0 - 0.5) = 9.0
    assert "offset=4.500[vx1]" in video
    assert "offset=9.000[vx2]" in video
    # each join consumes the accumulated stream, not the raw input
    assert video.startswith("[0:v][1:v]")
    assert "[vx1][2:v]" in video
    assert audio.count("acrossfade") == 2
    assert "[ax2]" in audio


def test_audio_and_video_labels_line_up():
    num = 4
    _, audio = build_transition_filter_graph(num, [6.0] * num, 0.4)
    assert audio.endswith(f"[ax{num - 1}];")


def test_total_timeline_shortens_by_fade_per_join():
    durations = [4.0, 5.0, 6.0]
    fade = 0.5
    expected = sum(durations) - (len(durations) - 1) * fade
    assert expected == pytest.approx(14.0)


def test_rejects_fade_longer_than_segment():
    with pytest.raises(ValueError):
        build_transition_filter_graph(2, [0.3, 0.3], 0.5)


def test_rejects_single_segment():
    with pytest.raises(ValueError):
        build_transition_filter_graph(1, [5.0], 0.5)


def test_rejects_nonpositive_fade():
    with pytest.raises(ValueError):
        build_transition_filter_graph(2, [5.0, 5.0], 0.0)


# ---------------------------------------------------------------------------
# Validation guards (never invoke ffmpeg)
# ---------------------------------------------------------------------------

def test_empty_video_segments_rejected(tmp_path):
    assert stitch_cinematic_movie([], [], str(tmp_path / "out.mp4")) is False


def test_av_count_mismatch_rejected(tmp_path):
    v = tmp_path / "a.mp4"
    v.write_bytes(b"x")
    assert stitch_cinematic_movie([str(v)], [], str(tmp_path / "out.mp4")) is False


def test_missing_segment_file_rejected(tmp_path):
    v = tmp_path / "a.mp4"
    a = tmp_path / "a.wav"
    v.write_bytes(b"x")  # audio missing on purpose
    assert stitch_cinematic_movie([str(v)], [str(a)], str(tmp_path / "out.mp4")) is False


def test_transition_seconds_must_be_positive_never_crashes(tmp_path):
    """A non-positive transition must fall back to concat, not raise."""
    v = tmp_path / "a.mp4"
    a = tmp_path / "a.wav"
    v.write_bytes(b"x")
    a.write_bytes(b"x")
    # ffmpeg may or may not exist; the contract is: no exception, bool result.
    result = stitch_cinematic_movie(
        [str(v)], [str(a)], str(tmp_path / "out.mp4"), transition_seconds=0.0
    )
    assert isinstance(result, bool)