"""
Prompting Subsystem — Phase 9 Advanced Prompt Processing Engine

Provides token-safe segmentation, semantic analysis, segment expansion,
and negative prompt management for arbitrarily long user prompts and scripts.

Public API:
    from multigenai.prompting import PromptProcessor
    plan = PromptProcessor(settings).process(prompt, negative_prompt)
"""

from multigenai.prompting.prompt_processor import PromptProcessor

__all__ = ["PromptProcessor"]
