"""
CodeEngine — Code file generation.

Phase 1: Writes structured code templates from prompt context.
Future phases: Will hook into a real LLM for code generation.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from multigenai.core.logging.logger import get_logger

if TYPE_CHECKING:
    from multigenai.core.execution_context import ExecutionContext

LOG = get_logger(__name__)

_LANG_MAP = {
    "python": "py", "javascript": "js", "typescript": "ts",
    "html": "html", "css": "css", "java": "java",
    "c++": "cpp", "c#": "cs", "go": "go",
    "rust": "rs", "ruby": "rb", "sql": "sql", "shell": "sh",
}


@dataclass
class CodeResult:
    """Output from the CodeEngine."""
    path: str
    language: str
    success: bool = True
    error: Optional[str] = None


class CodeEngine:
    """
    Generates code files from natural language prompts.

    Usage:
        engine = CodeEngine(ctx)
        result = engine.run("write a python fibonacci function")
    """

    def __init__(self, ctx: "ExecutionContext") -> None:
        self._ctx = ctx
        self._out_dir = pathlib.Path(ctx.settings.output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def run(self, prompt: str) -> CodeResult:
        """Generate a code file from a natural language prompt."""
        lang, ext = self._detect_language(prompt)
        slug = re.sub(r"[^A-Za-z0-9\-_]+", "_", prompt)[:40]
        slug += f"_{hashlib.sha1(prompt.encode()).hexdigest()[:8]}"
        out_path = self._out_dir / f"{slug}.{ext}"

        content = (
            f"# Language: {lang}\n"
            f"# Prompt: {prompt}\n\n"
            f"# NOTE: Phase 1 placeholder — LLM code generation activates in a future phase.\n"
            f"# Replace this with actual generated code.\n\n"
            f"def placeholder():\n"
            f"    \"\"\"Generated stub for: {prompt}\"\"\"\n"
            f"    raise NotImplementedError('LLM code generation not yet active')\n"
        ) if lang == "python" else (
            f"// Language: {lang}\n"
            f"// Prompt: {prompt}\n\n"
            f"// NOTE: Phase 1 placeholder — LLM code generation activates in a future phase.\n"
        )

        out_path.write_text(content, encoding="utf-8")
        LOG.info(f"Code file saved: {out_path} ({lang})")
        return CodeResult(path=str(out_path), language=lang)

    def _detect_language(self, prompt: str) -> tuple[str, str]:
        lower = prompt.lower()
        for lang, ext in _LANG_MAP.items():
            if lang in lower:
                return lang, ext
        return "python", "py"
