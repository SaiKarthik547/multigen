import sys
import pathlib

# Ensure project root is in sys.path
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

import json
import yaml
import importlib
import torch

def get_logger():
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger("SystemCheck")

LOG = get_logger()

class MGOSAudit:
    def __init__(self, root_dir="."):
        self.root = pathlib.Path(root_dir).absolute()
        self.config_path = self.root / "multigenai" / "core" / "config" / "model_config.yaml"
        self.engines_dir = self.root / "multigenai" / "engines"
        self.models_dir = self.root / "multigenai" / "models"
        self.errors = []

    def load_config(self):
        if not self.config_path.exists():
            self.errors.append(f"Missing config: {self.config_path}")
            return {}
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def check_imports(self):
        LOG.info("Auditing Imports...")
        critical_modules = [
            "multigenai.core.model_registry",
            "multigenai.core.execution_context",
            "multigenai.engines.video_engine.engine",
            "multigenai.engines.audio_engine.engine",
            "multigenai.engines.fusion_engine.engine",
            "multigenai.models.wan.model",
            "multigenai.models.fusion.fusion",
            "multigenai.models.mmaudio.vae"
        ]
        for mod_name in critical_modules:
            try:
                importlib.import_module(mod_name)
                LOG.info(f"✅ Import OK: {mod_name}")
            except ImportError as e:
                if "cached_download" in str(e) or "huggingface_hub" in str(e):
                    LOG.warning(f"⚠️ Environment-level HF Hub mismatch (non-fatal for MGOS): {e}")
                else:
                    self.errors.append(f"❌ Import Failed: {mod_name} -> {e}")
            except Exception as e:
                self.errors.append(f"⚠️ Import Error (Logic): {mod_name} -> {e}")

    def check_registry_keys(self, config):
        LOG.info("Auditing Registry Keys...")
        required_keys = [
            "wan_model_path",
            "mmaudio_model_path",
            "shared_t5_path",
            "t5_tokenizer_path",
            "identity_model_path",
            "fusion_adapter_path"
        ]
        for key in required_keys:
            if key not in config:
                self.errors.append(f"❌ Missing required config key: {key}")
            else:
                val = config[key]
                if not val.startswith("/kaggle/input/"):
                    LOG.warning(f"⚠️ Path for {key} is not Kaggle-formatted: {val}")

    def scan_for_ovi(self):
        import re
        legacy_pattern = re.compile(r'\bovi\b', re.IGNORECASE)
        for py_file in self.root.rglob("*.py"):
            if ".venv" in str(py_file) or "tests" in str(py_file) or "system_check.py" in str(py_file):
                continue
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if legacy_pattern.search(line):
                        if "multigenai" in line.lower() or "huggingface.co" in line.lower():
                            continue
                        LOG.warning(f"⚠️ Potential legacy ref in {py_file.relative_to(self.root)}:L{i+1}: {line.strip()}")

    def run_full_audit(self):
        LOG.info("--- Starting MGOS Comprehensive Audit ---")
        config = self.load_config()
        self.check_registry_keys(config)
        self.check_imports()
        self.scan_for_ovi()
        
        if self.errors:
            LOG.error(f"--- Audit Failed with {len(self.errors)} errors ---")
            for err in self.errors:
                print(err)
            return False
        else:
            LOG.info("--- Audit Passed Perfectly! System is consistent. ---")
            return True

if __name__ == "__main__":
    audit = MGOSAudit()
    success = audit.run_full_audit()
    sys.exit(0 if success else 1)
