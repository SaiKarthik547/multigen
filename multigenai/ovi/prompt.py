"""Ovi prompt contract — speech and audio conditioning text.

Upstream Ovi (character-ai/Ovi README) trains its joint audio-video model on
prompts that encode spoken dialogue and sound design *in the text channel*:

    <S>spoken dialogue<E>            — speech is synthesised verbatim
    Audio: sound description         — ambient / SFX description
    <AUDCAP>...<ENDAUDCAP>           — internal form for the 720x720_5s ckpt

The released ``720x720_5s`` checkpoint expects the ``<AUDCAP>...<ENDAUDCAP>``
form (the upstream engine converts ``Audio: ...`` to tags via its per-variant
``formatter``), while the ``960x960_*`` checkpoints consume plain
``Audio: ...``.  This module is the single place where that conversion lives.

Everything here is pure text manipulation: no torch, no I/O — safe to unit
test on a CPU-only box.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "OviPrompt",
    "build_ovi_prompt",
    "apply_variant_audio_tags",
    "DEFAULT_VIDEO_NEGATIVE_PROMPT",
    "DEFAULT_AUDIO_NEGATIVE_PROMPT",
]

#: Upstream-recommended video negative prompt (Ovi README / demo defaults).
DEFAULT_VIDEO_NEGATIVE_PROMPT = "jitter, bad hands, blur, distortion"

#: Upstream-recommended audio negative prompt (Ovi README / demo defaults).
DEFAULT_AUDIO_NEGATIVE_PROMPT = "robotic, muffled, echo, distorted"

_SPEECH_OPEN = "<S>"
_SPEECH_CLOSE = "<E>"


@dataclass
class OviPrompt:
    """Fully decomposed Ovi conditioning text for one generation."""

    #: Visual scene description (doubles as the joint video+audio positive
    #: context, matching upstream, which feeds text_embeddings[0] to BOTH
    #: modalities).
    visual_text: str
    #: Verbatim speech to synthesise, if any.
    speech: Optional[str] = None
    #: Ambient/SFX description, if any.
    audio_description: Optional[str] = None
    video_negative_prompt: str = DEFAULT_VIDEO_NEGATIVE_PROMPT
    audio_negative_prompt: str = DEFAULT_AUDIO_NEGATIVE_PROMPT
    #: Which audio-tag convention the target checkpoint expects.
    audio_tag_style: str = "audio"

    def render(self) -> str:
        """Compose the final prompt string in upstream order and tag style.

        Layout (upstream README examples)::

            {scene description}. <S>{dialogue}<E> Audio: {sound design}

        then, for checkpoints trained on the internal tag form
        (``audio_tag_style == "audcap"``), ``Audio: ...`` is rewritten to
        ``<AUDCAP>...<ENDAUDCAP>``.
        """
        parts = [self.visual_text.strip().rstrip(".")]

        if self.speech and self.speech.strip():
            dialogue = " ".join(self.speech.strip().split())
            parts.append(f"{_SPEECH_OPEN}{dialogue}{_SPEECH_CLOSE}")

        if self.audio_description and self.audio_description.strip():
            parts.append(f"Audio: {self.audio_description.strip()}")

        text = ". ".join(p for p in parts if p)
        return apply_variant_audio_tags(text, self.audio_tag_style)


def build_ovi_prompt(
    scene_description: str,
    dialogue: Optional[str] = None,
    audio_description: Optional[str] = None,
    video_negative_prompt: str = DEFAULT_VIDEO_NEGATIVE_PROMPT,
    audio_negative_prompt: str = DEFAULT_AUDIO_NEGATIVE_PROMPT,
    audio_tag_style: str = "audio",
) -> OviPrompt:
    """Build an :class:`OviPrompt` from scene planning outputs.

    Args:
        scene_description: the visual description of the shot.
        dialogue: character speech, if the scene has any. Rendered inside
            ``<S>...</E>`` so the model speaks it verbatim.
        audio_description: sound-design line. When the caller did not supply
            one but the scene has dialogue, a neutral speech-only descriptor is
            used so the audio branch is never silently empty.
        audio_tag_style: ``"audcap"`` for the 720x720_5s checkpoint, else
            ``"audio"``.
    """
    if not audio_description and dialogue and dialogue.strip():
        audio_description = "clear natural speech"

    return OviPrompt(
        visual_text=scene_description,
        speech=dialogue,
        audio_description=audio_description,
        video_negative_prompt=video_negative_prompt,
        audio_negative_prompt=audio_negative_prompt,
        audio_tag_style=audio_tag_style,
    )


def apply_variant_audio_tags(text: str, audio_tag_style: str) -> str:
    """Convert ``Audio: ...`` between the two upstream checkpoint conventions.

    - ``"audcap"``: ``Audio: X``  ->  ``<AUDCAP>X<ENDAUDCAP>``  (720x720_5s)
    - ``"audio"``:  text unchanged (960x960_* checkpoints consume ``Audio:``)

    This mirrors the per-variant ``formatter`` lambdas in the upstream
    ``NAME_TO_MODEL_SPECS_MAP``.
    """
    if audio_tag_style == "audcap":
        return re.sub(
            r"Audio:\s*(.*)", r"<AUDCAP>\1<ENDAUDCAP>", text, flags=re.S
        )
    if audio_tag_style == "audio":
        # Normalize any leaked internal tags back to the plain form.
        return re.sub(
            r"<AUDCAP>(.*?)<ENDAUDCAP>", r"Audio: \1", text, flags=re.S
        )
    raise ValueError(
        f"unknown audio_tag_style {audio_tag_style!r}; expected 'audcap' or 'audio'"
    )
