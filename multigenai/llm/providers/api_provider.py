"""
APILLMProvider — Cloud LLM API backend (Gemini / OpenAI).

Uses plain requests.post — no SDK, no vendor lock-in.

API key is read from an env var whose NAME is stored in config
(e.g. api_key_env: MGOS_LLM_API_KEY). The key itself is never
hardcoded or stored in config files.

Architecture:
  APILLMProvider.generate()
    → _call_openai()   if api_mode == "openai"
    → _call_gemini()   if api_mode == "gemini"

Switching vendors: change config.yaml `llm.api_mode` only.
"""

from __future__ import annotations

import os
from typing import Optional

from multigenai.core.logging.logger import get_logger
from multigenai.llm.providers.base import LLMProvider

LOG = get_logger(__name__)

_DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_DEFAULT_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
_DEFAULT_MODEL_OPENAI = "gpt-4o-mini"
_DEFAULT_MODEL_GEMINI = "gemini-1.5-flash"
_DEFAULT_TIMEOUT = 60.0


class APILLMProvider(LLMProvider):
    """
    LLM provider that calls a cloud API (OpenAI or Gemini).

    Config-driven — no vendor logic mixed inline.

    Usage (via config):
        llm:
          provider: api
          api_mode: gemini               # openai | gemini
          model: gemini-1.5-flash
          api_key_env: MGOS_LLM_API_KEY  # env var holding the key
          timeout_seconds: 60

    Usage (direct):
        provider = APILLMProvider(api_mode="openai", api_key_env="OPENAI_API_KEY")
        text = provider.generate("Enhance this prompt: a stormy sea")
    """

    def __init__(
        self,
        api_mode: str = "gemini",
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key_env: str = "MGOS_LLM_API_KEY",
        timeout_seconds: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._api_mode = api_mode.lower()
        self._timeout = timeout_seconds
        self._api_key_env = api_key_env
        self._api_key: Optional[str] = os.environ.get(api_key_env)

        if self._api_mode == "openai":
            self._model = model or _DEFAULT_MODEL_OPENAI
            self._endpoint = endpoint or _DEFAULT_OPENAI_ENDPOINT
        elif self._api_mode == "gemini":
            self._model = model or _DEFAULT_MODEL_GEMINI
            self._endpoint = endpoint or _DEFAULT_GEMINI_ENDPOINT.format(
                model=self._model
            )
        else:
            raise ValueError(
                f"Unknown api_mode '{api_mode}'. Choose 'openai' or 'gemini'."
            )

        LOG.debug(
            f"APILLMProvider configured: api_mode={self._api_mode} "
            f"model={self._model} key_env={api_key_env} "
            f"key_present={'yes' if self._api_key else 'NO'}"
        )

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Dispatch to the configured API backend.

        Raises:
            ProviderAuthError:       on missing key or 401/403.
            ProviderTimeoutError:    on request timeout.
            ProviderResponseError:   on other HTTP errors.
            ProviderUnavailableError: on network failure.
        """
        from multigenai.core.exceptions import ProviderAuthError

        if not self._api_key:
            raise ProviderAuthError(self._endpoint)

        if self._api_mode == "openai":
            return self._call_openai(prompt, system_prompt)
        else:
            return self._call_gemini(prompt, system_prompt)

    # ------------------------------------------------------------------
    # Vendor-specific call methods (isolated, no inline branching)
    # ------------------------------------------------------------------

    def _call_openai(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Send request to OpenAI-compatible chat completions endpoint."""
        try:
            import requests
        except ImportError:
            from multigenai.core.exceptions import ProviderUnavailableError
            raise ProviderUnavailableError("'requests' not installed.")

        from multigenai.core.exceptions import (
            ProviderAuthError,
            ProviderResponseError,
            ProviderTimeoutError,
            ProviderUnavailableError,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.7,
        }

        try:
            resp = requests.post(
                self._endpoint,
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
        except requests.Timeout:
            raise ProviderTimeoutError(self._endpoint, self._timeout)
        except requests.ConnectionError as exc:
            raise ProviderUnavailableError(
                f"Network error reaching OpenAI: {exc}",
                details={"endpoint": self._endpoint},
            ) from exc

        if resp.status_code in (401, 403):
            raise ProviderAuthError(self._endpoint)
        if not resp.ok:
            raise ProviderResponseError(self._endpoint, resp.status_code, resp.text)

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            from multigenai.core.exceptions import ProviderResponseFormatError
            raise ProviderResponseFormatError(
                "Unexpected OpenAI response structure",
                details={"keys": list(data.keys())},
            ) from exc

    def _call_gemini(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Send request to the Gemini generateContent endpoint."""
        try:
            import requests
        except ImportError:
            from multigenai.core.exceptions import ProviderUnavailableError
            raise ProviderUnavailableError("'requests' not installed.")

        from multigenai.core.exceptions import (
            ProviderAuthError,
            ProviderResponseError,
            ProviderTimeoutError,
            ProviderUnavailableError,
        )

        # Gemini uses ?key= query param
        url = f"{self._endpoint}?key={self._api_key}"
        parts = [{"text": prompt}]
        if system_prompt:
            parts.insert(0, {"text": f"[System] {system_prompt}\n\n"})
        payload = {"contents": [{"parts": parts}]}

        try:
            resp = requests.post(url, json=payload, timeout=self._timeout)
        except requests.Timeout:
            raise ProviderTimeoutError(self._endpoint, self._timeout)
        except requests.ConnectionError as exc:
            raise ProviderUnavailableError(
                f"Network error reaching Gemini: {exc}",
                details={"endpoint": self._endpoint},
            ) from exc

        if resp.status_code in (401, 403):
            raise ProviderAuthError(self._endpoint)
        if not resp.ok:
            raise ProviderResponseError(self._endpoint, resp.status_code, resp.text)

        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError) as exc:
            from multigenai.core.exceptions import ProviderResponseFormatError
            raise ProviderResponseFormatError(
                "Unexpected Gemini response structure",
                details={"keys": list(data.keys())},
            ) from exc
