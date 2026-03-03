"""
ModelLifecycle Helper

Centralizes safe memory unloading logic for all generative engines,
ensuring consistent VRAM reclamation (e.g., handling garbage collection,
CUDA cache clearing, and IPC handle cleanup) to prevent memory fragmentation
across the application.
"""

from __future__ import annotations

import gc

class ModelLifecycle:
    """
    Centralized model lifecycle handler for safe unloading.
    """

    @staticmethod
    def safe_unload(obj: object) -> None:
        """
        Safely deletes an object and forcefully clears VRAM/IPC handles.
        All engines must use this helper when tearing down models.
        """
        if obj is None:
            return

        try:
            del obj
        except Exception:
            pass

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
