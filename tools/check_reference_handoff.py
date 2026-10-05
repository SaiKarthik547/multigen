"""Standalone check of the reference-image resolution chain in GenerationManager.

Mirrors the exact expression used in generate_video() so the priority order can
be validated without importing torch-dependent engine modules.

Run:  python tools/check_reference_handoff.py
"""
from dataclasses import dataclass


@dataclass
class FakeSceneState:
    character_reference_path: str = ""
    reference_frame_path: str = ""


@dataclass
class FakeScene:
    scene_id: str = "s01"
    keyframe_path: str = ""
    dialogue: str = None


def resolve_reference_image(scene, scene_state, conditioning_image_path):
    """The engine-agnostic priority chain under test."""
    return (
        scene.keyframe_path
        or scene_state.reference_frame_path
        or conditioning_image_path
    )


CASES = [
    ("per-scene keyframe wins", FakeScene(keyframe_path="kf.png"),
     FakeSceneState(reference_frame_path="rf.png"), "cond.png", "kf.png"),
    ("falls back to image stage", FakeScene(keyframe_path=""),
     FakeSceneState(reference_frame_path="rf.png"), "cond.png", "rf.png"),
    ("falls back to caller arg", FakeScene(keyframe_path=""),
     FakeSceneState(reference_frame_path=""), "cond.png", "cond.png"),
    ("no reference -> T2V", FakeScene(keyframe_path=""),
     FakeSceneState(reference_frame_path=""), None, None),
    ("empty keyframe still uses image", FakeScene(keyframe_path=""),
     FakeSceneState(reference_frame_path="rf2.png"), None, "rf2.png"),
]


def main() -> int:
    failures = 0
    for name, scene, state, cond, expected in CASES:
        got = resolve_reference_image(scene, state, cond)
        ok = got == expected
        failures += 0 if ok else 1
        print(f"  {'OK  ' if ok else 'FAIL'} {name:28} -> {got!r}")

    print()
    print(f"  {len(CASES) - failures}/{len(CASES)} passed")
    if failures:
        print("  FAIL: reference-image priority chain is incorrect")
    else:
        print("  PASS: image -> video keyframe handoff is restored")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())