"""
ModelLifecycle — Centralized GPU/RAM teardown helper.

All engines call ModelLifecycle.safe_unload() in their finally blocks.
This prevents VRAM leaks, memory fragmentation, and IPC handle exhaustion
across sequential generations on resource-constrained hardware (Kaggle T4, etc).

Teardown sequence (in order):
  1. del obj                  — remove Python reference → cpython refcount hits 0
  2. gc.collect()             — force sweep of any circular-ref survivors
  3. cuda.empty_cache()       — return freed VRAM blocks to the CUDA allocator pool
  4. cuda.ipc_collect()       — reclaim IPC memory handles (prevents handle leak
                                when engines run in separate processes/subproceses)
"""

from __future__ import annotations

import gc

from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)


class ModelLifecycle:
    """Centralized model lifecycle handler for safe, logged unloading."""

    @staticmethod
    def safe_unload(model: object) -> None:
        """
        Safely destroy a model object and aggressively reclaim GPU/host memory.

        IMPORTANT: This function removes the local reference but the caller MUST 
        nullify their own reference (e.g., `self.model = None`) after calling this 
        to ensure Python's GC can reclaim the memory.
        
        Example:
            ModelLifecycle.safe_unload(pipe)
            pipe = None

        Safe to call with None (no-op).
        Safe to call multiple times on the same object (idempotent via None check).
        Logs CUDA stats when DEBUG level is active.

        Args:
            model: Any object (diffusers pipeline, torch module, etc.) or None.
        """
        if model is None:
            return

        model_name = type(model).__name__
        LOG.debug(f"ModelLifecycle: unloading {model_name}...")

        try:
            del model
        finally:
            ModelLifecycle.enforce_cleanup(model_name)

    @staticmethod
    def enforce_cleanup(context: str = "system") -> None:
        """
        Aggressively reclaim GPU/host memory.
        Safe for both CUDA and CPU environments.
        """
        import gc
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                before_mb = torch.cuda.memory_reserved() / 1024 / 1024
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
                after_mb = torch.cuda.memory_reserved() / 1024 / 1024
                LOG.debug(
                    f"ModelLifecycle: VRAM reclaimed [{context}] — "
                    f"{before_mb:.0f}MB → {after_mb:.0f}MB reserved"
                )
        except ImportError:
            pass
        except Exception as exc:
            LOG.warning(f"ModelLifecycle: CUDA flush error [{context}]: {exc}")

    @staticmethod
    def assert_vram_clean(threshold_gb: float = 1.5, context: str = "system") -> None:
        """
        Phase 15 Health Guard: Raise RuntimeError if reserved VRAM exceeds threshold.

        Call after every hard unload to immediately surface memory leaks rather
        than allowing them to accumulate silently and cause OOM mid-generation.

        Args:
            threshold_gb: Maximum acceptable reserved VRAM in GB (default 1.5).
            context:      Label for the error/log message.

        Raises:
            RuntimeError: if reserved VRAM > threshold_gb.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return
            reserved_gb = torch.cuda.memory_reserved() / 1024 ** 3
            if reserved_gb > threshold_gb:
                raise RuntimeError(
                    f"VRAM leak detected [{context}]: "
                    f"{reserved_gb:.2f} GB reserved > {threshold_gb:.1f} GB threshold. "
                    "Abort to prevent OOM."
                )
            LOG.debug(f"ModelLifecycle: VRAM clean [{context}] — {reserved_gb:.2f} GB reserved.")
        except ImportError:
            pass
