"""Ovi prompt-contract tests — speech / audio tags per variant.

Upstream contract (character-ai/Ovi README + NAME_TO_MODEL_SPECS_MAP):
  * speech is rendered inside ``<S>...</E>``
  * sound design is rendered as ``Audio: ...``
  * the 720x720_5s checkpoint expects the internal ``<AUDCAP>...<ENDAUDCAP>``
    form; the 960x960_* checkpoints consume the plain ``Audio:`` form.

No torch required.
"""

from multigenai.ovi.prompt import (
    DEFAULT_AUDIO_NEGATIVE_PROMPT,
    DEFAULT_VIDEO_NEGATIVE_PROMPT,
    apply_variant_audio_tags,
    build_ovi_prompt,
)


def test_speech_is_wrapped_in_upstream_tags():
    prompt = build_ovi_prompt(
        scene_description="A lighthouse at dusk",
        dialogue="The storm is coming.",
        audio_tag_style="audio",
    )
    text = prompt.render()
    assert "<S>The storm is coming.<E>" in text
    assert text.startswith("A lighthouse at dusk")


def test_audio_description_rendered_as_audio_line():
    prompt = build_ovi_prompt(
        scene_description="A forest path",
        audio_description="soft rain, distant thunder",
        audio_tag_style="audio",
    )
    assert "Audio: soft rain, distant thunder" in prompt.render()


def test_audcap_style_rewrites_audio_line():
    """720x720_5s checkpoints expect <AUDCAP>...<ENDAUDCAP>."""
    prompt = build_ovi_prompt(
        scene_description="A city street",
        audio_description="traffic hum",
        audio_tag_style="audcap",
    )
    text = prompt.render()
    assert "<AUDCAP>traffic hum<ENDAUDCAP>" in text
    assert "Audio: traffic hum" not in text


def test_dialogue_only_scene_gets_neutral_audio_descriptor():
    """A speech scene with no explicit sound design must not leave the audio
    branch empty — a neutral speech descriptor is injected."""
    prompt = build_ovi_prompt(
        scene_description="Close-up of the captain",
        dialogue="All hands on deck!",
    )
    text = prompt.render()
    assert "<S>All hands on deck!<E>" in text
    assert "Audio:" in text


def test_no_speech_no_audio_leaves_visual_only():
    prompt = build_ovi_prompt(scene_description="Waves on a rocky shore")
    text = prompt.render()
    assert "<S>" not in text
    assert "Audio:" not in text
    assert text == "Waves on a rocky shore"


def test_apply_variant_audio_tags_roundtrip():
    tagged = apply_variant_audio_tags("scene. Audio: wind", "audcap")
    assert tagged == "scene. <AUDCAP>wind<ENDAUDCAP>"
    untagged = apply_variant_audio_tags(tagged, "audio")
    assert untagged == "scene. Audio: wind"


def test_apply_variant_audio_tags_rejects_unknown_style():
    import pytest

    with pytest.raises(ValueError):
        apply_variant_audio_tags("x", "nope")


def test_negative_prompt_defaults_match_upstream_recommendations():
    assert DEFAULT_VIDEO_NEGATIVE_PROMPT == "jitter, bad hands, blur, distortion"
    assert DEFAULT_AUDIO_NEGATIVE_PROMPT == "robotic, muffled, echo, distorted"
    prompt = build_ovi_prompt(scene_description="x")
    assert prompt.video_negative_prompt == DEFAULT_VIDEO_NEGATIVE_PROMPT
    assert prompt.audio_negative_prompt == DEFAULT_AUDIO_NEGATIVE_PROMPT


def test_whitespace_in_dialogue_is_normalized():
    prompt = build_ovi_prompt(
        scene_description="A quiet room",
        dialogue="  hello \n world  ",
    )
    assert "<S>hello world<E>" in prompt.render()
