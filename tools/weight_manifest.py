import os
import json
import pathlib
from dataclasses import dataclass, asdict

@dataclass
class WeightEntry:
    name: str
    target_path: str
    source_url: str
    description: str
    estimated_size_gb: float

WEIGHTS = [
    WeightEntry(
        "Wan 2.2 DiT (5B)",
        "Wan2.2-TI2V-5B/model.safetensors",
        "https://huggingface.co/Wan-AI/Wan2.2-T2V-2.1B-Character/resolve/main/model.safetensors", # Placeholder Example
        "Core Video Generation Transformer Backbone",
        20.0
    ),
    WeightEntry(
        "Wan 2.2 VAE",
        "Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
        "https://huggingface.co/Wan-AI/Wan2.2-T2V-2.1B-Character/resolve/main/Wan2.2_VAE.pth",
        "Temporal VAE for video encoding/decoding",
        5.0
    ),
    WeightEntry(
        "MMAudio Backbone",
        "MMAudio/model.safetensors",
        "https://huggingface.co/hkchengrex/MMAudio/resolve/main/model.safetensors",
        "Core Audio Generation Transformer Backbone",
        8.0
    ),
    WeightEntry(
        "BigVGAN Vocoder",
        "MMAudio/ext_weights/best_netG.pt",
        "https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x/resolve/main/bigvgan_generator.pt",
        "High-fidelity neural vocoder for MMAudio",
        1.5
    ),
    WeightEntry(
        "Shared T5 Encoder",
        "Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
        "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth",
        "Unified Text Encoder for Video and Audio",
        22.0
    ),
    WeightEntry(
        "Fusion Adapter weights",
        "fusion_adapter/model.safetensors",
        "https://huggingface.co/chetwinlow1/Ovi/resolve/main/model.safetensors",
        "MGOS Cross-Attention Fusion Adapter",
        2.5
    )
]

def generate_manifest():
    root = pathlib.Path("./weights_prep")
    root.mkdir(exist_ok=True)
    
    manifest_path = root / "mgos_weights_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump([asdict(w) for w in WEIGHTS], f, indent=4)
    
    print(f"✅ Manifest generated at {manifest_path}")
    print("\n--- Required Weights Directory Structure for Kaggle ---")
    print("mgos-weights/")
    for w in WEIGHTS:
        print(f"  └── {w.target_path} ({w.estimated_size_gb} GB)")
    
    print("\n💡 Action: Download these files into a folder named 'mgos-weights', zip it, and upload to Kaggle as a dataset.")

if __name__ == "__main__":
    generate_manifest()
