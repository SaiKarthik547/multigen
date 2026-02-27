"""
Structured logger for MultiGenAI OS.

Supports:
  - Pretty console mode (human-readable, colored via rich)
  - JSON mode (machine-readable, for log aggregators)
  - Optional file sink
  - Correlation IDs attached via contextvars
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import pathlib
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Correlation ID context variable (set once per request/job)
# ---------------------------------------------------------------------------
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def new_correlation_id() -> str:
    """Generate and set a new correlation ID for the current execution context."""
    cid = uuid.uuid4().hex[:12]
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Return the current correlation ID (or empty string if unset)."""
    return _correlation_id.get()


# ---------------------------------------------------------------------------
# Custom formatters
# ---------------------------------------------------------------------------

class _PrettyFormatter(logging.Formatter):
    """Human-readable colored formatter with optional rich fallback."""

    _LEVEL_COLORS = {
        "DEBUG":    "\033[36m",   # Cyan
        "INFO":     "\033[32m",   # Green
        "WARNING":  "\033[33m",   # Yellow
        "ERROR":    "\033[31m",   # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._LEVEL_COLORS.get(record.levelname, "")
        reset = self._RESET
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        cid = _correlation_id.get()
        cid_part = f" [{cid}]" if cid else ""
        msg = super().format(record)
        return f"{color}[{record.levelname:<8}]{reset} {ts}{cid_part} {msg}"


class _JsonFormatter(logging.Formatter):
    """Machine-readable JSON formatter for log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "correlation_id": _correlation_id.get() or None,
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure_logging(
    level: str = "INFO",
    mode: str = "pretty",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure the root MultiGenAI logger.

    Call once during application startup (LifecycleManager handles this).

    Args:
        level:    Log level string ("DEBUG", "INFO", "WARNING", "ERROR").
        mode:     "pretty" for console color output; "json" for JSON lines.
        log_file: Optional file path. If given, logs are also written there.
    """
    root = logging.getLogger("multigenai")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        _PrettyFormatter() if mode != "json" else _JsonFormatter()
    )
    root.addHandler(console_handler)

    # Optional file handler (rotating, 10 MB × 3 backups)
    if log_file:
        file_path = pathlib.Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            file_path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(_JsonFormatter())
        root.addHandler(file_handler)

    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger namespaced under 'multigenai'.

    Usage:
        LOG = get_logger(__name__)
        LOG.info("Hello world")
    """
    return logging.getLogger(f"multigenai.{name}" if not name.startswith("multigenai") else name)
