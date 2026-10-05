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
    """Ensure no legacy 'ovi' references exist in the core codebase."""
    files = get_project_files()
    legacy_pattern = re.compile(r'\bovi\b', re.IGNORECASE)
    
    failures = []
    for f in files:
        # Skip this check for the audit tools themselves and 3rd party ext
        if "tests" in str(f) or "system_check" in str(f) or "mmaudio_core\\ext" in str(f):
            continue
            
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                for i, line in enumerate(fh):
                    if legacy_pattern.search(line):
                        # Filter out known false positives (system-level strings)
                        if "multigenai" in line.lower():
                            continue
                        failures.append(f"{f.relative_to(f.parent.parent.parent)}:L{i+1} -> {line.strip()}")
        except Exception:
            continue
    
    assert not failures, f"Found legacy Ovi references:\n" + "\n".join(failures)

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
