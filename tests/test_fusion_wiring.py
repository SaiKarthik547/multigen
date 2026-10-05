"""FusionEngine wiring tests — the REAL run() path with stubbed weights.

Everything except the model math is exercised for real here: prompt
assembly, T5 slot ordering, I2V encode, geometry resolution, the guarded
sampling loop, dual schedulers, decode, output files and the last-frame
handoff.

Guards the specific regressions that silent ``strict=False`` loading and
shared-scheduler bugs produced:
  * two INDEPENDENT schedulers are created and stepped
  * ``first_frame_is_clean`` is forwarded on BOTH guided and unguided passes
  * SLG is applied on the negative pass only
  * ``clip_fea`` is never passed (FusionModel hard-asserts it is None)
  * the first-frame latent is re-imposed every step
  * outputs are unique per call even when the seed repeats
  * the last frame is a real PNG derived from the decoded tensor
"""

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from multigenai.engines.fusion_engine.engine import FusionEngine, FusionResult
from multigenai.ovi.loader import build_runtime_spec


FRAME_COUNT = 5          # decoded video frames returned by the stub VAE
LATENT_F = 31            # upstream contract for 5s variants
LATENT_HW = 22


class _FakeModel:
    """Records every forward call; returns zero velocity."""

    def __init__(self):
        self.calls = []

    def __call__(self, vid=None, audio=None, t=None, vid_context=None,
                 audio_context=None, vid_seq_len=None, audio_seq_len=None,
                 clip_fea=None, clip_fea_audio=None, y=None,
                 first_frame_is_clean=False, slg_layer=False):
        self.calls.append({
            "first_frame_is_clean": first_frame_is_clean,
            "slg_layer": slg_layer,
            "clip_fea": clip_fea,
            "clip_fea_audio": clip_fea_audio,
            "vid_seq_len": vid_seq_len,
            "audio_seq_len": audio_seq_len,
            "t": t,
            "first_slice_is_image": None,  # filled by the test via vid
            "vid": vid[0].clone(),
        })
        return [torch.zeros_like(vid[0])], [torch.zeros_like(audio[0])]


class _FakeVideoVAE:
    model = None

    def wrapped_encode(self, video):
        """Returns a LATENT (b, c, 1, h, w) — non-zero so the test can tell
        the image latent apart from pure noise."""
        b = video.shape[0]
        return torch.ones((b, 48, 1, LATENT_HW, LATENT_HW))

    def wrapped_decode(self, latents):
        return torch.zeros((1, 3, FRAME_COUNT, 64, 64))


class _FakeAudioVAE:
    def wrapped_decode(self, latents):
        return torch.linspace(-0.5, 0.5, 16000)


class _FakeT5:
    def __init__(self):
        self.device = "cpu"
        self.model = None
        self.texts = None

    def __call__(self, texts, device):
        self.texts = list(texts)
        return [torch.zeros((8, 64), dtype=torch.bfloat16) for _ in texts]


class _FakeScheduler:
    def __init__(self, name):
        self.name = name
        self.steps = []

    def step(self, model_output, t, sample, return_dict=False):
        self.steps.append((self.name, float(t)))
        return [sample * 0.5]


def _make_engine(tmp_path, image_path, monkeypatch, i2v=True):
    ctx = SimpleNamespace(
        settings=SimpleNamespace(output_dir=str(tmp_path / "out")),
        device="cpu",
        behaviour=SimpleNamespace(auto_unload_after_gen=False),
        environment=None,
    )
    engine = FusionEngine(ctx)

    spec = build_runtime_spec("720x720_5s")
    bundle = SimpleNamespace(
        model=_FakeModel(),
        v_vae=_FakeVideoVAE(),
        a_vae=_FakeAudioVAE(),
        t5=_FakeT5(),
        spec=spec,
        device="cpu",
        cpu_offload=False,
        fp8=False,
        model_to_device=lambda: None,
        model_offload=lambda: None,
        t5_to_device=lambda: None,
        t5_offload=lambda: None,
        vae_video_to_device=lambda: None,
        vae_video_offload=lambda: None,
        vae_audio_to_device=lambda: None,
        vae_audio_offload=lambda: None,
    )
    monkeypatch.setattr(
        FusionEngine, "_load_bundle", lambda self: setattr(self, "bundle", bundle) or bundle.spec
    )

    schedulers = {}

    def fake_sched(self, sampling_steps, solver_name="unipc", device=0, shift=5.0):
        idx = len(schedulers)
        name = "video" if idx == 0 else "audio"
        sch = _FakeScheduler(name)
        schedulers[name] = sch
        return sch, [torch.tensor(1000.0), torch.tensor(500.0)]

    monkeypatch.setattr(FusionEngine, "get_scheduler_time_steps", fake_sched)

    def fake_encode_video(frames, out_path, fps, seed):
        """Mirrors ffmpeg_utils.encode_video's uint8 conversion without ffmpeg."""
        from PIL import Image
        out = str(out_path)
        Image.fromarray(np.asarray(frames[0]).astype(np.uint8)).save(out + ".png")
        return SimpleNamespace(path=out, success=True, error=None)

    monkeypatch.setattr(
        "multigenai.engines.video_engine.ffmpeg_utils.encode_video", fake_encode_video
    )
    return engine, bundle, schedulers


def _request(**kw):
    from multigenai.llm.schema_validator import VideoGenerationRequest
    return VideoGenerationRequest(prompt="A knight crosses the bridge", seed=42, **kw)


@pytest.fixture
def keyframe(tmp_path):
    from PIL import Image
    p = tmp_path / "keyframe.png"
    Image.fromarray(
        (np.random.default_rng(0).random((512, 512, 3)) * 255).astype(np.uint8)
    ).save(p)
    return str(p)


def test_sampling_uses_two_independent_schedulers(tmp_path, keyframe, monkeypatch):
    engine, bundle, schedulers = _make_engine(tmp_path, keyframe, monkeypatch)
    res = engine.run(_request(), image_path=keyframe, dialogue=None)
    assert res.success, res.error

    assert set(schedulers) == {"video", "audio"}
    assert schedulers["video"].steps and schedulers["audio"].steps
    assert [s[0] for s in schedulers["video"].steps] == ["video"] * 2
    assert [s[0] for s in schedulers["audio"].steps] == ["audio"] * 2
    # Same timestep drive, but through two distinct stateful objects.
    assert schedulers["video"].steps[0][1] == schedulers["audio"].steps[0][1]


def test_first_frame_is_clean_forwarded_on_both_passes(tmp_path, keyframe, monkeypatch):
    engine, bundle, schedulers = _make_engine(tmp_path, keyframe, monkeypatch)
    res = engine.run(_request(), image_path=keyframe)
    assert res.success, res.error

    # 2 timesteps x (positive + negative) = 4 calls
    assert len(bundle.model.calls) == 4
    assert all(c["first_frame_is_clean"] is True for c in bundle.model.calls)


def test_slg_applied_on_negative_pass_only(tmp_path, keyframe, monkeypatch):
    engine, bundle, schedulers = _make_engine(tmp_path, keyframe, monkeypatch)
    res = engine.run(_request(), image_path=keyframe)
    assert res.success, res.error

    slg = [c["slg_layer"] for c in bundle.model.calls]
    # order per step: positive, negative
    assert slg == [False, 11, False, 11]


def test_clip_fea_is_never_passed(tmp_path, keyframe, monkeypatch):
    """FusionModel.forward hard-asserts clip_fea is None — ArcFace injection
    would have crashed here (and was conceptually wrong anyway)."""
    engine, bundle, schedulers = _make_engine(tmp_path, keyframe, monkeypatch)
    res = engine.run(_request(), image_path=keyframe, )
    assert res.success, res.error
    assert all(c["clip_fea"] is None for c in bundle.model.calls)
    assert all(c["clip_fea_audio"] is None for c in bundle.model.calls)


def test_sequence_lengths_come_from_variant_contract(tmp_path, keyframe, monkeypatch):
    from multigenai.ovi.contracts import derive_seq_len

    engine, bundle, schedulers = _make_engine(tmp_path, keyframe, monkeypatch)
    res = engine.run(_request(), image_path=keyframe)
    assert res.success, res.error

    call = bundle.model.calls[0]
    assert call["audio_seq_len"] == 157
    # video: 31 latent frames x measured 22x22 spatial -> per-dim conv output
    assert call["vid_seq_len"] == derive_seq_len(
        (LATENT_F, LATENT_HW, LATENT_HW), (1, 2, 2)
    )


def test_t5_receives_prompt_video_negative_audio_negative(tmp_path, keyframe, monkeypatch):
    engine, bundle, schedulers = _make_engine(tmp_path, keyframe, monkeypatch)
    res = engine.run(
        _request(), image_path=keyframe,
        dialogue="We must cross now.",
        audio_description="rain on stone",
    )
    assert res.success, res.error

    prompt, video_neg, audio_neg = bundle.t5.texts
    # 720x720_5s uses the internal AUDCAP tag form
    assert "<S>We must cross now.<E>" in prompt
    assert "<AUDCAP>rain on stone<ENDAUDCAP>" in prompt
    assert video_neg == "jitter, bad hands, blur, distortion"
    assert audio_neg == "robotic, muffled, echo, distorted"


def test_audio_description_not_duplicated_when_planner_supplies_one(
    tmp_path, keyframe, monkeypatch
):
    engine, bundle, schedulers = _make_engine(tmp_path, keyframe, monkeypatch)
    res = engine.run(
        _request(), image_path=keyframe,
        dialogue="Look!", audio_description="distant bells",
    )
    assert res.success, res.error
    prompt = bundle.t5.texts[0]
    assert "clear natural speech" not in prompt
    assert "<AUDCAP>distant bells<ENDAUDCAP>" in prompt


def test_first_frame_latent_reimposed_every_step(tmp_path, keyframe, monkeypatch):
    engine, bundle, schedulers = _make_engine(tmp_path, keyframe, monkeypatch)
    res = engine.run(_request(), image_path=keyframe)
    assert res.success, res.error

    # The stub velocity output forces the scheduler to scale the sample; after
    # each step the engine must restore slot 0 to the image latent.
    for call in bundle.model.calls:
        assert torch.count_nonzero(call["vid"][:, :1]) > 0


def test_metadata_derived_from_decoded_tensor(tmp_path, keyframe, monkeypatch):
    engine, bundle, schedulers = _make_engine(tmp_path, keyframe, monkeypatch)
    res = engine.run(_request(), image_path=keyframe)
    assert res.success, res.error
    assert isinstance(res, FusionResult)
    assert res.frame_count == FRAME_COUNT
    assert res.fps == 24
    assert res.duration_seconds == pytest.approx(FRAME_COUNT / 24)


def test_outputs_unique_across_calls_with_same_seed(tmp_path, keyframe, monkeypatch):
    """Same seed on two calls must not overwrite the first scene's files."""
    engine, bundle, schedulers = _make_engine(tmp_path, keyframe, monkeypatch)
    first = engine.run(_request(), image_path=keyframe)
    second = engine.run(_request(), image_path=keyframe)
    assert first.success and second.success, (first.error, second.error)
    assert first.video_path != second.video_path
    assert first.audio_path != second.audio_path
    assert first.last_frame_path != second.last_frame_path


def test_last_frame_is_valid_png_of_the_decoded_tail(tmp_path, keyframe, monkeypatch):
    from PIL import Image

    engine, bundle, schedulers = _make_engine(tmp_path, keyframe, monkeypatch)
    res = engine.run(_request(), image_path=keyframe)
    assert res.success, res.error
    assert res.last_frame_path and res.last_frame_path.endswith(".png")
    with Image.open(res.last_frame_path) as img:
        assert img.size == (64, 64)      # stub decode shape (w, h)
        assert img.mode == "RGB"


def test_t2v_mode_runs_without_reference_image(tmp_path, keyframe, monkeypatch):
    engine, bundle, schedulers = _make_engine(tmp_path, keyframe, monkeypatch)
    res = engine.run(_request(), image_path=None)
    assert res.success, res.error
    assert all(c["first_frame_is_clean"] is False for c in bundle.model.calls)


def test_engine_failure_returns_failed_result_not_exception(tmp_path, monkeypatch):
    ctx = SimpleNamespace(
        settings=SimpleNamespace(output_dir=str(tmp_path / "out")),
        device="cpu",
        behaviour=SimpleNamespace(auto_unload_after_gen=False),
        environment=None,
    )
    engine = FusionEngine(ctx)
    monkeypatch.setattr(
        FusionEngine, "_load_bundle",
        lambda self: (_ for _ in ()).throw(RuntimeError("checkpoint missing")),
    )
    res = engine.run(_request(), image_path=None)
    assert res.success is False
    assert "checkpoint missing" in res.error