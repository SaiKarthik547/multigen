# Legacy — Archived Experimental Code

This directory contains experimental or superseded implementations that were moved out of the main codebase to keep it clean and production-focused.

Files here are **preserved for reference** and potential reuse in future phases. They are **not imported** anywhere in the active codebase.

---

## `models/ip_adapter/` — Phase 15 Retired: IP-Adapter

IP-Adapter was retired in **Phase 15** to stay within the Kaggle T4 15 GB VRAM budget.
Character identity is now handled entirely via `IdentityLatentEncoder` (VAE latent injection).

| File | Original Path | Reason Archived |
|---|---|---|
| `ip_adapter_manager.py` | `multigenai/consistency/ip_adapter_manager.py` | Exceeds VRAM budget on T4; replaced by latent-space identity conditioning |

### Reuse Conditions
- Requires ≥ 20 GB VRAM (A100/H100)
- Re-enable via `enable_ip_adapter: true` in `config.yaml` and restore real file to `multigenai/consistency/`

---

## `models/controlnet/` — Phase 15 Retired: ControlNet + DepthAnything

ControlNet + DepthAnythingSmall was retired in **Phase 15** to reclaim ~2–3 GB VRAM on the T4.
Structural conditioning is now achieved via keyframe latent anchoring in `IdentityLatentEncoder`.

| File | Original Path | Reason Archived |
|---|---|---|
| `controlnet_manager_sdxl.py` | `multigenai/consistency/controlnet_manager.py` | Full SDXL ControlNet + DepthAnything — too large for T4 |
| `controlnet_manager_stub.py` | `multigenai/control/controlnet_manager.py` | Phase 4 stub, never activated |

### Reuse Conditions
- Requires ≥ 20 GB VRAM
- Restore real file to `multigenai/consistency/controlnet_manager.py`

---

## `temporal/` — Phase 5 Temporal Stubs

These files were created as Phase 5 placeholders for optical-flow-based temporal smoothing.

They were superseded in **Phase 6** by the SVD-XT (`StableVideoDiffusionPipeline`) approach, which provides native temporal coherence without manual optical flow computation.

| File | Original Path | Reason Archived |
|---|---|---|
| `optical_flow.py` | `multigenai/temporal/optical_flow.py` | Phase 5 stub, `raise NotImplementedError`, not imported |
| `motion_engine.py` | `multigenai/temporal/motion_engine.py` | Phase 5 stub, `raise NotImplementedError`, not imported |

### Phase 9 Reuse Potential

- `optical_flow.py`: Could be revisited for **depth-guided motion** or **ControlNet temporal conditioning** in Phase 9.
- `motion_engine.py`: Could support **AnimateDiff-style motion module injection** if Phase 9 moves beyond SVD-XT.

---

## Policy

> **Do not import from `legacy/` in production code.**  
> If a legacy file is being reused, move it back to the appropriate module and document the decision.
