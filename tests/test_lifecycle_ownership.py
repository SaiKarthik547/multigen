"""P0-C — lifecycle ownership / VRAM reclamation tests.

Before this fix, ``ModelRegistry.unload()`` only dropped its own reference and
logged.  The real teardown (``del`` -> ``gc`` -> ``empty_cache`` ->
``ipc_collect``) lived in ``ModelLifecycle`` and was never invoked from that
path, and no engine ever nulled its own ``self.bundle``.  Net effect: unloading
a model did **not** return VRAM to the allocator.

These tests assert the ownership contract, using a fake teardown spy so they
run without torch and without a GPU.

Run:  python -m pytest tests/test_lifecycle_ownership.py -v
"""

import gc
import pathlib
import re

import pytest

from multigenai.core.model_lifecycle import ModelLifecycle
from multigenai.core.model_registry import ModelRegistry
from multigenai.core.exceptions import ModelNotFoundError


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def fresh_registry():
    """An isolated ModelRegistry (the class is a singleton in production)."""
    return ModelRegistry()


@pytest.fixture
def teardown_spy(monkeypatch):
    """Record every enforce_cleanup() call without touching CUDA."""
    calls = []
    monkeypatch.setattr(
        ModelLifecycle,
        "enforce_cleanup",
        staticmethod(lambda context="system": calls.append(context)),
    )
    return calls


# ---------------------------------------------------------------------------
# Registry teardown
# ---------------------------------------------------------------------------

def test_unload_runs_teardown(fresh_registry, teardown_spy):
    """unload() must invoke the gc/CUDA sweep, not just drop a reference."""
    sentinel = object()
    fresh_registry.register("m1", loader=lambda: sentinel, min_vram_gb=0.0)
    assert fresh_registry.get("m1") is sentinel
    assert fresh_registry.is_loaded("m1")

    fresh_registry.unload("m1")

    assert not fresh_registry.is_loaded("m1")
    assert teardown_spy, "unload() did not run ModelLifecycle.enforce_cleanup()"
    assert any("m1" in ctx for ctx in teardown_spy)


def test_unload_all_runs_teardown_once(fresh_registry, teardown_spy):
    a, b = object(), object()
    fresh_registry.register("a", loader=lambda: a, min_vram_gb=0.0)
    fresh_registry.register("b", loader=lambda: b, min_vram_gb=0.0)
    fresh_registry.get("a")
    fresh_registry.get("b")

    fresh_registry.unload_all()

    assert not fresh_registry.is_loaded("a")
    assert not fresh_registry.is_loaded("b")
    assert teardown_spy == ["registry.unload_all"]


def test_unload_all_with_nothing_loaded_is_a_noop(fresh_registry, teardown_spy):
    fresh_registry.register("a", loader=lambda: object(), min_vram_gb=0.0)
    fresh_registry.unload_all()
    assert teardown_spy == []


def test_unload_unknown_model_raises(fresh_registry):
    with pytest.raises(ModelNotFoundError):
        fresh_registry.unload("never_registered")


def test_unload_not_loaded_model_does_not_teardown(fresh_registry, teardown_spy):
    fresh_registry.register("x", loader=lambda: object(), min_vram_gb=0.0)
    fresh_registry.unload("x")  # never loaded
    assert teardown_spy == []


# ---------------------------------------------------------------------------
# Engine ownership contract (static — engines need torch to import)
# ---------------------------------------------------------------------------

ENGINE_DIR = REPO_ROOT / "multigenai" / "engines"


@pytest.mark.parametrize(
    "engine_file",
    [
        "video_engine/engine.py",
        "fusion_engine/engine.py",
        "audio_engine/engine.py",
    ],
)
def test_engine_nulls_bundle_before_registry_unload(engine_file):
    """Each engine must drop `self.bundle` before the registry teardown runs.

    Otherwise the engine keeps the models reachable and VRAM is never freed.
    Checked statically because these modules import torch at module level.
    """
    src = (ENGINE_DIR / engine_file).read_text(encoding="utf-8")

    # Locate the unload method.
    match = re.search(
        r"def _unload(?:_model)?\(self\).*?(?=\n    def |\Z)", src, re.DOTALL
    )
    assert match, f"{engine_file}: no _unload/_unload_model method found"

    body = match.group(0)
    bundle_null = body.find("self.bundle = None")
    registry_unload = body.find("registry.unload")

    assert bundle_null != -1, f"{engine_file}: unload does not null self.bundle"
    assert registry_unload != -1, f"{engine_file}: unload does not call registry.unload"
    assert bundle_null < registry_unload, (
        f"{engine_file}: must null self.bundle BEFORE registry.unload(), "
        "otherwise the models stay reachable during the gc sweep"
    )


@pytest.mark.parametrize(
    "engine_file",
    [
        "video_engine/engine.py",
        "fusion_engine/engine.py",
        "audio_engine/engine.py",
        "image_engine/engine.py",
        "interpolation_engine/engine.py",
    ],
)
def test_engine_does_not_read_nonexistent_auto_unload_attribute(engine_file):
    """`settings.auto_unload` / `settings.video.auto_unload` never existed.

    Reading them raised AttributeError inside a finally block, which masked the
    real generation error. Engines must use behaviour.auto_unload_after_gen.
    """
    src = (ENGINE_DIR / engine_file).read_text(encoding="utf-8")
    assert "settings.auto_unload" not in src
    assert "settings.video.auto_unload" not in src
    assert "settings.audio.auto_unload" not in src


def test_engines_use_package_relative_config_paths():
    """P0-N: no CWD-relative config access in executable code.

    Comments are stripped first — the fusion engine's comment block mentions
    the old CWD-relative form when explaining why it was replaced, and that
    must not trip the check.
    """
    import io
    import tokenize

    for engine_file in ("video_engine/engine.py", "fusion_engine/engine.py"):
        path = ENGINE_DIR / engine_file
        raw = path.read_text(encoding="utf-8")

        # Rebuild source with comments and docstrings removed.
        code_only = []
        prev_type = tokenize.INDENT
        for tok in tokenize.generate_tokens(io.StringIO(raw).readline):
            if tok.type == tokenize.COMMENT:
                continue
            if tok.type == tokenize.STRING and prev_type in (
                tokenize.INDENT, tokenize.NEWLINE, tokenize.NL, tokenize.DEDENT,
            ):
                continue  # docstring
            if tok.type not in (tokenize.NL, tokenize.NEWLINE):
                code_only.append(tok.string)
            prev_type = tok.type
        src = " ".join(code_only)

        assert 'Path("multigenai/' not in src, (
            f"{engine_file}: CWD-relative path breaks when run outside the repo root"
        )


# ---------------------------------------------------------------------------
# ModelLifecycle behaviour that is safe to exercise without CUDA
# ---------------------------------------------------------------------------

def test_safe_unload_accepts_none():
    ModelLifecycle.safe_unload(None)  # must not raise


def test_enforce_cleanup_is_safe_without_cuda(monkeypatch):
    """Should be a no-op that still runs gc, even with torch absent."""
    ModelLifecycle.enforce_cleanup("test")  # must not raise


def test_assert_vram_clean_is_noop_without_cuda():
    ModelLifecycle.assert_vram_clean(threshold_gb=0.0, context="test")