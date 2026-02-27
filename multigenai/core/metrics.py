"""
GenerationMetrics — Per-generation compute metrics tracking.

Captures:
  - Model ID and resolution used
  - Wall-clock duration
  - VRAM before / peak / after (0 on CPU)
  - Whether a resolution downgrade occurred
  - Success flag and optional error string

MetricsCollector is a process-level singleton that accumulates all
GenerationMetrics objects and can print a Rich summary table.

GenerationTimer is a context manager that auto-captures:
  - VRAM watermarks (via torch if available)
  - Wall-clock duration
  - Attaches result to a GenerationMetrics object in-place.

Usage:
    metrics = GenerationMetrics(model_id="sdxl", width=1024, height=1024)
    with GenerationTimer(metrics):
        ... run generation ...
    MetricsCollector.instance().record(metrics)
    MetricsCollector.instance().log_summary()
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# GenerationMetrics — immutable record of one generation run
# ---------------------------------------------------------------------------

@dataclass
class GenerationMetrics:
    """
    Record of a single image/video generation attempt.

    All VRAM values in MB. 0 means CPU-only or undetectable.
    """
    model_id: str
    width: int
    height: int
    duration_seconds: float = 0.0
    vram_before_mb: int = 0
    vram_after_mb: int = 0
    peak_vram_mb: int = 0
    downgraded: bool = False          # True if resolution was reduced due to OOM
    success: bool = True
    error: Optional[str] = None
    # --- Phase 4: Identity conditioning ---
    identity_used: bool = False       # True if IP-Adapter identity was applied
    identity_name: Optional[str] = None  # Name of identity profile used

    @property
    def resolution_label(self) -> str:
        return f"{self.width}×{self.height}"

    @property
    def vram_delta_mb(self) -> int:
        """VRAM consumed after vs before (can be negative if freed)."""
        return self.vram_after_mb - self.vram_before_mb


# ---------------------------------------------------------------------------
# GenerationTimer — context manager for auto-capturing metrics
# ---------------------------------------------------------------------------

def _read_vram_mb() -> int:
    """Return current allocated VRAM in MB. Returns 0 if torch not available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() // (1024 * 1024)
    except ImportError:
        pass
    return 0


def _read_peak_vram_mb() -> int:
    """Return peak allocated VRAM since last reset_peak_memory_stats call, in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() // (1024 * 1024)
    except ImportError:
        pass
    return 0


def _reset_peak_vram() -> None:
    """Reset the CUDA peak memory counter."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


class GenerationTimer:
    """
    Context manager that captures VRAM watermarks and wall-clock duration
    and writes them directly into a GenerationMetrics object.

    Usage:
        m = GenerationMetrics(model_id="sdxl", width=1024, height=1024)
        with GenerationTimer(m):
            ... run generation ...
        # m.duration_seconds, m.vram_before_mb, m.peak_vram_mb, m.vram_after_mb now set
    """

    def __init__(self, metrics: GenerationMetrics) -> None:
        self._metrics = metrics
        self._t0 = 0.0

    def __enter__(self) -> "GenerationTimer":
        _reset_peak_vram()
        self._metrics.vram_before_mb = _read_vram_mb()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._metrics.duration_seconds = round(time.perf_counter() - self._t0, 3)
        self._metrics.peak_vram_mb = _read_peak_vram_mb()
        self._metrics.vram_after_mb = _read_vram_mb()
        # Don't suppress exceptions
        return False


# ---------------------------------------------------------------------------
# MetricsCollector — singleton that accumulates and reports
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Process-level singleton that accumulates GenerationMetrics records.

    Thread-safe via internal lock.

    Usage:
        MetricsCollector.instance().record(metrics)
        MetricsCollector.instance().log_summary()
        MetricsCollector.instance().reset()
    """

    _singleton: Optional["MetricsCollector"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._records: List[GenerationMetrics] = []
        self._record_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "MetricsCollector":
        """Return the global singleton MetricsCollector."""
        if cls._singleton is None:
            with cls._lock:
                if cls._singleton is None:
                    cls._singleton = cls()
        return cls._singleton

    def record(self, metrics: GenerationMetrics) -> None:
        """Append a GenerationMetrics record and log a one-liner summary."""
        with self._record_lock:
            self._records.append(metrics)
        status = "✅" if metrics.success else "❌"
        downgrade = " [downgraded]" if metrics.downgraded else ""
        identity_info = (
            {"identity_used": True, "identity_name": metrics.identity_name}
            if metrics.identity_used
            else {"identity_used": False}
        )
        LOG.info(
            f"METRICS {status} {metrics.model_id} | {metrics.resolution_label}{downgrade} | "
            f"{metrics.duration_seconds:.1f}s | VRAM {metrics.vram_before_mb}→"
            f"{metrics.vram_after_mb}MB (peak {metrics.peak_vram_mb}MB) | "
            f"identity={identity_info}"
        )

    def summary(self) -> dict:
        """
        Return a summary dict of all recorded metrics.

        Keys: total, successes, failures, downgrades,
              avg_duration_s, avg_peak_vram_mb, records (list)
        """
        with self._record_lock:
            records = list(self._records)

        total = len(records)
        if total == 0:
            return {"total": 0, "successes": 0, "failures": 0,
                    "downgrades": 0, "avg_duration_s": 0.0,
                    "avg_peak_vram_mb": 0, "records": []}

        successes = sum(1 for r in records if r.success)
        downgrades = sum(1 for r in records if r.downgraded)
        avg_dur = round(sum(r.duration_seconds for r in records) / total, 2)
        avg_peak = int(sum(r.peak_vram_mb for r in records) / total)

        return {
            "total": total,
            "successes": successes,
            "failures": total - successes,
            "downgrades": downgrades,
            "avg_duration_s": avg_dur,
            "avg_peak_vram_mb": avg_peak,
            "records": records,
        }

    def log_summary(self) -> None:
        """Print a formatted summary to the log."""
        s = self.summary()
        LOG.info(
            f"MetricsCollector summary | total={s['total']} "
            f"success={s['successes']} fail={s['failures']} "
            f"downgrades={s['downgrades']} "
            f"avg_dur={s['avg_duration_s']}s "
            f"avg_peak_vram={s['avg_peak_vram_mb']}MB"
        )

    def reset(self) -> None:
        """Clear all recorded metrics (useful between test runs)."""
        with self._record_lock:
            self._records.clear()
        LOG.debug("MetricsCollector reset.")
