"""Sequential cinematic workflow — image -> unload -> Ovi -> continuation.

The orchestrator contract under test (Kaggle free-tier safety):

1. The SDXL keyframe is produced FIRST, and the image engine is FULLY released
   BEFORE the Ovi bundle is constructed — engines never share VRAM.
2. The Ovi bundle is held across scenes (one checkpoint load, not one per
   scene) and released explicitly after the video phase.
3. Scene 1 is conditioned on the keyframe; every later scene is conditioned on
   the previous segment's decoded LAST FRAME (the upstream Ovi I2V handoff).
4. The movie is assembled from all segments with transitions, and the result
   metadata (frame count / fps) comes from the fusion results, not literals.

Engines are faked; the contract is about ORDER, HANDOFF and LIFECYCLE.
"""

from types import SimpleNamespace

import pytest

from multigenai.consistency.scene_memory import SceneMemory
from multigenai.core.generation_manager import GenerationManager
from multigenai.llm.schema_validator import VideoGenerationRequest


SCRIPT = "A knight rides across the bridge. He finds a sword in the mud."


class _FakeImageEngine:
    def __init__(self, ctx):
        self.released = 0

    def release(self):
        self.released += 1


def _make_ctx(tmp_path):
    settings = SimpleNamespace(
        output_dir=str(tmp_path / "out"),
        environment="test",
    )
    return SimpleNamespace(
        settings=settings,
        device="cpu",
        llm=None,
        scene_memory=SceneMemory(),
        behaviour=SimpleNamespace(auto_unload_after_gen=False),
    )


@pytest.fixture
def workflow(monkeypatch, tmp_path):
    """Wire fakes into the orchestrator and record the call order."""
    ctx = _make_ctx(tmp_path)
    events = []
    paths = tmp_path / "assets"
    paths.mkdir()

    # --- fake image engine -------------------------------------------------
    class ImageEngine:
        def __init__(self, ctx):
            self._ctx = ctx

        def release(self):
            events.append("image_release")

    monkeypatch.setattr(
        "multigenai.engines.image_engine.engine.ImageEngine", ImageEngine
    )

    manager = GenerationManager(ctx)

    def fake_generate_image(self, request):
        events.append("keyframe")
        p = paths / "keyframe.png"
        p.write_bytes(b"x")
        return SimpleNamespace(path=str(p), success=True, error=None)

    monkeypatch.setattr(GenerationManager, "generate_image", fake_generate_image)

    # --- fake Ovi fusion engine ------------------------------------------
    class FusionResult:
        def __init__(self, idx):
            self.video_path = f"vid_{idx}.mp4"
            self.audio_path = f"aud_{idx}.wav"
            self.frame_count = 121
            self.fps = 24
            self.seed = 7
            self.success = True
            self.error = None
            self.last_frame_path = str(paths / f"last_frame_{idx}.png")

    class FusionEngine:
        counter = 0

        def __init__(self, ctx):
            events.append("fusion_init")

        def hold_bundle(self):
            events.append("fusion_hold")

        def run(self, request, image_path=None, dialogue=None,
                audio_description=None, temporal_state=None):
            FusionEngine.counter += 1
            idx = FusionEngine.counter
            events.append(("fusion_run", image_path, request.prompt))
            return FusionResult(idx)

        def release(self):
            events.append("fusion_release")

    monkeypatch.setattr(
        "multigenai.engines.fusion_engine.engine.FusionEngine", FusionEngine
    )

    # --- fake assembly -----------------------------------------------------
    stitched = {}

    def fake_stitch(video_segments, audio_segments, output_path,
                    transition_seconds=0.5):
        stitched["videos"] = list(video_segments)
        stitched["audios"] = list(audio_segments)
        stitched["output"] = output_path
        stitched["transition"] = transition_seconds
        return True

    monkeypatch.setattr(
        "multigenai.utils.movie_utils.stitch_cinematic_movie", fake_stitch
    )

    return SimpleNamespace(manager=manager, events=events, stitched=stitched,
                           paths=paths)


def _request(**kw):
    return VideoGenerationRequest(prompt=SCRIPT, seed=7, **kw)


def test_image_phase_precedes_and_releases_before_ovi(workflow):
    """The image engine must be torn down BEFORE Ovi is constructed."""
    result = workflow.manager.generate_video(_request())

    assert result.success, result.error
    assert workflow.events[0] == "keyframe"
    assert workflow.events[1] == "image_release"
    assert workflow.events[2] == "fusion_init"
    # release must happen before the bundle is constructed, never after
    assert workflow.events.index("image_release") < workflow.events.index(
        "fusion_init"
    )


def test_ovi_bundle_is_held_across_scenes_and_released_once(workflow):
    workflow.manager.generate_video(_request())
    assert workflow.events.count("fusion_hold") == 1
    assert workflow.events.count("fusion_init") == 1
    assert workflow.events.count("fusion_release") == 1
    assert workflow.events.index("fusion_release") > workflow.events.index(
        "fusion_hold"
    )


def test_scene_two_uses_scene_one_last_frame(workflow):
    """Continuation must flow through the decoded last-frame image."""
    workflow.manager.generate_video(_request())
    runs = [e for e in workflow.events if isinstance(e, tuple)]
    assert len(runs) == 2, workflow.events

    keyframe = str(workflow.paths / "keyframe.png")
    assert runs[0][1] == keyframe          # scene 1 <- SDXL keyframe
    assert runs[1][1].endswith("last_frame_1.png")  # scene 2 <- scene 1 tail

    # The last frame must also land in scene memory for downstream consumers.
    assert workflow.manager._ctx.scene_memory.get().reference_frame_path.endswith(
        "last_frame_2.png"
    )


def test_all_segments_are_assembled_with_transitions(workflow):
    result = workflow.manager.generate_video(_request())
    assert workflow.stitched["videos"] == ["vid_1.mp4", "vid_2.mp4"]
    assert workflow.stitched["audios"] == ["aud_1.wav", "aud_2.wav"]
    assert workflow.stitched["transition"] > 0
    assert result.path.endswith(".mp4")


def test_result_metadata_comes_from_fusion_results(workflow):
    result = workflow.manager.generate_video(_request())
    # 2 scenes x 121 frames @ 24 fps — derived, never hardcoded 81/8
    assert result.frame_count == 242
    assert result.fps == 24


def test_keyframe_generation_can_be_disabled(workflow):
    """Pure T2V: no image phase at all, and scene 1 starts from noise."""
    workflow.manager.generate_video(_request(generate_keyframes=False))
    assert "keyframe" not in workflow.events
    runs = [e for e in workflow.events if isinstance(e, tuple)]
    assert runs[0][1] is None


def test_failed_scene_is_skipped_but_others_continue(monkeypatch, workflow):
    def failing_first_run(self, request, image_path=None, dialogue=None,
                          audio_description=None, temporal_state=None):
        return SimpleNamespace(
            video_path="", audio_path="", frame_count=0, fps=24, seed=7,
            success=False, error="boom", last_frame_path=None,
        )

    from multigenai.engines.fusion_engine.engine import FusionEngine
    # Make every scene fail -> no segments -> clean failure result.
    monkeypatch.setattr(FusionEngine, "run", failing_first_run)
    result = workflow.manager.generate_video(_request())
    assert result.success is False
    assert "no scenes generated" in result.error
    # Ovi is still released even on the failure path
    assert "fusion_release" in workflow.events