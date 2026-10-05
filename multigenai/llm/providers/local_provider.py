"""
LocalLLMProvider — Ollama-compatible local LLM backend.

Communicates with Ollama's HTTP API (default: http://localhost:11434).
Also works with any Ollama-compatible local server (LM Studio, etc.).

Kaggle safety:
  - `requests` imported lazily inside methods — never at module level
  - Falls back to ProviderUnavailableError on any failure — never crashes caller

Retry policy:
  - max_retries: 2 attempts on transient network errors
  - timeout respected per request (from settings)
"""

from __future__ import annotations

from typing import Optional

from multigenai.core.logging.logger import get_logger
from multigenai.llm.providers.base import LLMProvider

LOG = get_logger(__name__)

_DEFAULT_ENDPOINT = "http://localhost:11434/api/generate"
_DEFAULT_MODEL = "mistral"
_DEFAULT_TIMEOUT = 30.0
_MAX_NETWORK_RETRIES = 2


class LocalLLMProvider(LLMProvider):
    """
    LLM provider that calls a locally-running Ollama-compatible server.

    Usage (via config):
        llm:
          provider: local
          model: mistral
          endpoint: http://localhost:11434/api/generate
          timeout_seconds: 30

    Usage (direct):
        provider = LocalLLMProvider(model="mistral")
        text = provider.generate("Enhance this prompt: a stormy sea")
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        endpoint: str = _DEFAULT_ENDPOINT,
        timeout_seconds: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._model = model
        self._endpoint = endpoint
        self._timeout = timeout_seconds
        LOG.debug(f"LocalLLMProvider configured: model={model} endpoint={endpoint}")

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a generation request to the local Ollama server.

        Args:
            prompt:        User prompt text.
            system_prompt: Optional system instruction.

        Returns:
            Generated text string.

        Raises:
            ProviderTimeoutError:    on timeout.
            ProviderResponseError:   on HTTP 4xx/5xx (non-auth).
            ProviderUnavailableError: on connection error.
        """
        try:
            import requests
        except ImportError:
            from multigenai.core.exceptions import ProviderUnavailableError
            raise ProviderUnavailableError(
                "The 'requests' library is not installed. "
                "Run: pip install requests"
            )

        from multigenai.core.exceptions import (
            ProviderResponseError,
            ProviderTimeoutError,
            ProviderUnavailableError,
        )

        payload: dict = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        last_error: Exception = Exception("Unknown")

        for attempt in range(1, _MAX_NETWORK_RETRIES + 1):
            try:
                LOG.debug(
                    f"LocalLLMProvider attempt {attempt}/{_MAX_NETWORK_RETRIES} "
                    f"model={self._model}"
                )
                resp = requests.post(
                    self._endpoint,
                    json=payload,
                    timeout=self._timeout,
                )

                if resp.status_code in (401, 403):
                    # Ollama doesn't normally auth, but guard anyway
                    from multigenai.core.exceptions import ProviderAuthError
                    raise ProviderAuthError(self._endpoint)

                if not resp.ok:
                    raise ProviderResponseError(
                        self._endpoint, resp.status_code, resp.text
                    )

                data = resp.json()
                # Ollama format: {"response": "...", "done": true}
                if "response" not in data:
                    from multigenai.core.exceptions import ProviderResponseFormatError
                    raise ProviderResponseFormatError(
                        "Ollama response missing 'response' key",
                        details={"keys": list(data.keys())},
                    )

                text = data["response"].strip()
                LOG.debug(f"LocalLLMProvider got {len(text)} chars")
                return text

            except requests.Timeout:
                last_error = ProviderTimeoutError(self._endpoint, self._timeout)
                LOG.warning(f"LocalLLMProvider timeout attempt {attempt}")
                continue

            except requests.ConnectionError as exc:
                last_error = ProviderUnavailableError(
                    f"Cannot connect to Ollama at {self._endpoint}: {exc}",
                    details={"endpoint": self._endpoint},
                )
                LOG.warning(f"LocalLLMProvider connection error attempt {attempt}: {exc}")
                continue

            except (ProviderResponseError, ProviderUnavailableError):
                raise  # re-raise typed errors immediately (no retry benefit)

        # All retries exhausted
        raise last_error
