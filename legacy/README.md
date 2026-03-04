# Legacy — Archived Experimental Code

This directory contains experimental or superseded implementations that were moved out of the main codebase to keep it clean and production-focused.

Files here are **preserved for reference** and potential reuse in future phases. They are **not imported** anywhere in the active codebase.

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
