"""
CapabilityReport — System capability detection and reporting.

Detects and reports:
  - OS, Python, platform
  - CUDA availability and VRAM
  - All optional library statuses (torch, diffusers, PIL, moviepy, etc.)

Outputs a formatted table to stdout and returns a structured dict.
"""

from __future__ import annotations

import importlib
import platform
import sys
from typing import Any, Dict

from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)

# Libraries to probe  (display_name, import_name)
_OPTIONAL_LIBS = [
    ("PyTorch",       "torch"),
    ("Diffusers",     "diffusers"),
    ("Transformers",  "transformers"),
    ("Accelerate",    "accelerate"),
    ("Pillow",        "PIL"),
    ("MoviePy",       "moviepy"),
    ("NLTK",          "nltk"),
    ("Matplotlib",    "matplotlib"),
    ("python-pptx",   "pptx"),
    ("python-docx",   "docx"),
    ("Wikipedia-API", "wikipediaapi"),
    ("PyYAML",        "yaml"),
    ("Pydantic",      "pydantic"),
    ("Typer",         "typer"),
    ("Rich",          "rich"),
    ("torch-directml","torch_directml"),
]


def _check_lib(import_name: str) -> tuple[bool, str]:
    """Try to import a library; return (available, version_str)."""
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "?")
        return True, str(version)
    except ImportError:
        return False, "not installed"
    except Exception as exc:
        return False, f"error: {exc}"


class CapabilityReport:
    """
    Gathers system capabilities and produces a human-readable report.

    Usage:
        cr = CapabilityReport()
        result = cr.report()   # prints + returns dict
        data   = cr.to_dict()  # dict only (no print)
    """

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Collect and return capability data as a structured dict."""
        if not self._data:
            self._collect()
        return self._data

    def report(self) -> Dict[str, Any]:
        """Print a formatted capability table and return the data dict."""
        data = self.to_dict()
        self._print(data)
        return data

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect(self) -> None:
        libs: Dict[str, Dict[str, Any]] = {}
        for display, import_name in _OPTIONAL_LIBS:
            ok, ver = _check_lib(import_name)
            libs[display] = {"available": ok, "version": ver}

        # CUDA details (lazy torch access)
        cuda_info: Dict[str, Any] = {"available": False}
        if libs["PyTorch"]["available"]:
            try:
                import torch
                if torch.cuda.is_available():
                    cuda_info = {
                        "available": True,
                        "device_count": torch.cuda.device_count(),
                        "device_name": torch.cuda.get_device_name(0),
                    }
                    try:
                        free_b, total_b = torch.cuda.mem_get_info()
                        cuda_info["vram_free_gb"]  = round(free_b  / 1024 ** 3, 2)
                        cuda_info["vram_total_gb"] = round(total_b / 1024 ** 3, 2)
                    except Exception:
                        pass
            except Exception:
                pass

        self._data = {
            "system": {
                "os":       platform.system(),
                "platform": platform.platform(),
                "python":   sys.version.split()[0],
                "arch":     platform.machine(),
            },
            "cuda": cuda_info,
            "libraries": libs,
        }

    def _print(self, data: Dict[str, Any]) -> None:
        sep = "─" * 58
        print(f"\n{'MultiGenAI OS — Capability Report':^58}")
        print(sep)

        sys_info = data["system"]
        print(f"  OS       : {sys_info['os']} ({sys_info['arch']})")
        print(f"  Platform : {sys_info['platform'][:50]}")
        print(f"  Python   : {sys_info['python']}")

        cuda = data["cuda"]
        if cuda["available"]:
            gpu = cuda.get("device_name", "unknown")
            free = cuda.get("vram_free_gb", "?")
            total = cuda.get("vram_total_gb", "?")
            print(f"  GPU      : ✔ {gpu}")
            print(f"  VRAM     : {free} GB free / {total} GB total")
        else:
            print("  GPU      : ✘ CUDA not available (CPU mode)")

        print(sep)
        print(f"  {'Library':<18} {'Status':<12} {'Version'}")
        print(f"  {'───────':<18} {'──────':<12} {'───────'}")
        for name, info in data["libraries"].items():
            mark = "✔" if info["available"] else "✘"
            print(f"  {mark} {name:<17} {'OK' if info['available'] else 'missing':<12} {info['version']}")

        print(sep + "\n")
