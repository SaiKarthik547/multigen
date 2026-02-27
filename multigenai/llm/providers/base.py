"""
LLMProvider — Abstract base class for all LLM backends.

Design rules:
  - No SDK imports here (no openai, no google.generativeai)
  - No network calls here
  - All providers receive their config via constructor
  - structured_generate() applies retry + JSON extraction + schema validation
  - All failures surface as typed exceptions (never raw Exception)

Exception hierarchy (defined in core.exceptions):
  ProviderUnavailableError   — backend unreachable (general)
    ProviderTimeoutError     — connect/read timeout
    ProviderAuthError        — 401/403 or missing credentials
    ProviderResponseError    — HTTP 4xx/5xx other than auth
    ProviderResponseFormatError — malformed JSON from model
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Type

from multigenai.core.logging.logger import get_logger

if TYPE_CHECKING:
    from pydantic import BaseModel

LOG = get_logger(__name__)

# System prompt used for structured JSON generation
_JSON_SYSTEM_PROMPT = (
    "You are a precise JSON generator. "
    "Respond ONLY with valid JSON that matches the requested schema. "
    "Do not include markdown fences, explanations, or extra text."
)

# Prompt fed to the model when its first response was invalid JSON
_JSON_FIX_PROMPT = (
    "Your previous response was not valid JSON. "
    "Return ONLY the corrected JSON object with no other text."
)


def extract_json_candidates(text: str) -> list[str]:
    """
    Extract ALL valid JSON objects/arrays from raw LLM output.

    Returns a list of JSON strings ordered by appearance, de-duplicated.
    Handles markdown fences, leading/trailing prose, multiple blocks.
    """
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)

    candidates: list[str] = []
    for start_char, end_char in ("{", "}"), ("[", "]"):
        pos = 0
        while pos < len(text):
            start = text.find(start_char, pos)
            if start == -1:
                break
            depth = 0
            for i, ch in enumerate(text[start:], start=start):
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            json.loads(candidate)
                            if candidate not in candidates:
                                candidates.append(candidate)
                        except json.JSONDecodeError:
                            pass
                        pos = i + 1
                        break
            else:
                break
    return candidates


def extract_json(text: str) -> str:
    """
    Extract the first valid JSON object or array from raw LLM output.

    Handles common LLM wrapping patterns:
      - ```json ... ``` fences
      - Bare JSON with leading/trailing noise
      - Single object {} or array []

    Returns:
        Extracted JSON string (the first valid candidate).

    Raises:
        ValueError: if no valid JSON block could be found.
    """
    candidates = extract_json_candidates(text)
    if candidates:
        return candidates[0]
    raise ValueError(f"No valid JSON found in LLM response: {text[:200]!r}")




class LLMProvider(ABC):
    """
    Abstract base for all LLM provider backends.

    Subclasses implement:
      generate()            — raw text generation
      structured_generate() — JSON-schema-validated structured output
                              (with retry and JSON extraction built in)

    The base class provides the shared structured_generate() implementation.
    Subclasses only need to implement generate().
    """

    MAX_RETRIES: int = 2  # attempts for structured output

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a text response for the given prompt.

        Args:
            prompt:        User prompt.
            system_prompt: Optional system/instruction prefix.

        Returns:
            Model response as a plain string.

        Raises:
            ProviderTimeoutError: on connect/read timeout.
            ProviderAuthError:    on authentication failure.
            ProviderResponseError: on other HTTP errors.
            ProviderUnavailableError: on network-level failure.
        """

    def structured_generate(self, prompt: str, schema: "Type[BaseModel]") -> "BaseModel":
        """
        Generate a response and parse it into a Pydantic model.

        Flow:
          1. Prepend JSON system prompt
          2. Call generate()
          3. Extract ALL JSON candidates from response
          4. Try each candidate against schema until one validates
          5. If none validate, retry with fix prompt (max MAX_RETRIES times)
          6. Wrap all failures as ProviderResponseFormatError

        Args:
            prompt: Instruction describing what to generate.
            schema: Pydantic BaseModel class to validate against.

        Returns:
            Validated Pydantic model instance.

        Raises:
            ProviderResponseFormatError: if no candidate validates
                                        after all retries.
            Any ProviderUnavailableError subclass: from generate() itself.
        """
        from multigenai.core.exceptions import ProviderResponseFormatError

        last_error: Exception = Exception("Unknown error")
        current_prompt = prompt

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                raw = self.generate(current_prompt, system_prompt=_JSON_SYSTEM_PROMPT)
                candidates = extract_json_candidates(raw)
                if not candidates:
                    raise ValueError(f"No JSON found in response: {raw[:100]!r}")
                for candidate in candidates:
                    try:
                        return schema.model_validate_json(candidate)
                    except Exception:
                        continue
                raise ValueError(
                    f"None of {len(candidates)} JSON candidates matched schema '{schema.__name__}'"
                )
            except Exception as exc:
                last_error = exc
                LOG.warning(
                    f"structured_generate attempt {attempt}/{self.MAX_RETRIES} failed: {exc}"
                )
                current_prompt = f"{prompt}\n\n{_JSON_FIX_PROMPT}"

        raise ProviderResponseFormatError(
            f"Structured generation failed after {self.MAX_RETRIES} attempts",
            details={"schema": schema.__name__, "cause": str(last_error)},
        ) from last_error
