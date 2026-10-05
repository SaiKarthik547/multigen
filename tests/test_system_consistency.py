import os
import pathlib
import re
import pytest

# MGOS Static Consistency Audit (Dependency-Free)
# This test validates rebranding and registry keys without importing heavy models.

def get_project_files():
    # Root of the package
    root = pathlib.Path(__file__).parent.parent / "multigenai"
    return list(root.rglob("*.py"))

def test_static_rebranding_audit():
    """Ensure no references to *removed* upstream module paths survive.

    HISTORY
    -------
    This test used to forbid the bare word ``ovi``. That was correct while the
    vendored upstream Ovi code was considered legacy scaffolding to be
    rebranded away. It is now inverted: Ovi is the intended cinematic backend,
    so the literal word is legitimate.

    What is still genuinely stale -- and what this test now guards -- are
    references to module paths that **no longer exist**:

      * ``multigenai.modules.*``  (pre-merge package layout, renamed to
        ``multigenai.models.*``)
      * ``multigenai.llm.providers`` under its corrupted spelling
      * ``ovi_fusion_engine`` / other one-off script names

    A stale import is a hard crash at module import time, which is exactly the
    class of defect this audit exists to catch.
    """
    files = get_project_files()

    stale_patterns = [
        (re.compile(r"multigenai\.modules\."), "removed package 'multigenai.modules'"),
        (re.compile(r"\bprmultigenaider", re.IGNORECASE), "corrupted 'provider' rename"),
        (re.compile(r"\bovi_fusion_engine\b"), "removed script 'ovi_fusion_engine'"),
    ]

    failures = []
    for f in files:
        # Skip the audit tools themselves and vendored third-party extensions.
        if "tests" in str(f) or "system_check" in str(f) or "mmaudio_core" in str(f):
            continue

        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                for i, line in enumerate(fh):
                    # Only executable import statements count. Docstrings and
                    # comments may legitimately *mention* the removed paths to
                    # explain why they were replaced.
                    stripped = line.strip()
                    if not (stripped.startswith(("import ", "from "))):
                        continue
                    for pattern, reason in stale_patterns:
                        if pattern.search(stripped):
                            rel = f.relative_to(f.parent.parent.parent)
                            failures.append(f"{rel}:L{i+1} -> {reason}: {stripped}")
                            break
        except Exception:
            continue

    assert not failures, "Found stale upstream references:\n" + "\n".join(failures)

def test_registry_key_consistency():
    """Verify Engines use the correct MGOS/Kaggle registry keys."""
    engines_dir = pathlib.Path(__file__).parent.parent / "multigenai" / "engines"
    required_keys = ["wan_model_path", "mmaudio_model_path", "shared_t5_path", "t5_tokenizer_path", "fusion_adapter_path"]
    
    engine_files = list(engines_dir.rglob("engine.py"))
    
    found_keys = set()
    for f in engine_files:
        with open(f, "r", encoding="utf-8") as fh:
            content = fh.read()
            for key in required_keys:
                if key in content:
                    found_keys.add(key)
    
    missing = [k for k in required_keys if k not in found_keys]
    assert not missing, f"Registry keys missing from engines: {missing}"

def test_config_kaggle_paths():
    """Ensure model_config.yaml is correctly set for Kaggle environments."""
    root = pathlib.Path(__file__).parent.parent
    config_path = root / "multigenai" / "core" / "config" / "model_config.yaml"
    assert config_path.exists(), "Missing model_config.yaml"
    
    with open(config_path, "r") as f:
        content = f.read()
    
    assert "/kaggle/input/" in content, "Model paths are not Kaggle-formatted in model_config.yaml"
