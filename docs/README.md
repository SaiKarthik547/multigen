# docs/

Review and baseline documentation for the 50-phase Ovi integration plan.

**No source code has been changed.** These documents are verification artefacts only.

## Files

| File | Purpose |
|---|---|
| [`PLAN_REVIEW.md`](./PLAN_REVIEW.md) | Full review of the 50-phase plan. Verdict, verified claims, blockers the plan misses, ordering problems, per-phase table, amended execution order. |
| [`BASELINE_SNAPSHOT.md`](./BASELINE_SNAPSHOT.md) | Measured pre-repair state: environment, test baseline, zero-byte files, rename corruption, missing attributes, verified crashes, VRAM declarations vs hardware, what is genuinely strong. |
| [`ROADMAP_REVIEW.md`](./ROADMAP_REVIEW.md) | Review of the *revised* roadmap. Four fixes required before starting: a wrong `vid_seq_len` formula, an unverified `157` constant, a missed audio-channel bug, and a keyframe step that isn't actually restorable. |
| [`_check_md.py`](./_check_md.py) | Markdown QA helper — verifies GFM table column counts, relative links, and heading anchors in these docs. Run: `python docs/_check_md.py` |

## Headline

The plan's **architecture is correct** and its diagnosis is largely right — verified by direct execution, not by reading prose. Three areas are worse than the plan states, and four hard blockers are missing entirely.

**Verified as correct:** Phases 2, 5, 6, 8/15, 11, 14, 16, 20/37, 26, 28, 29, 31/32/50, 43, 38.
Every one was reproduced by running code.

### Worse than the plan states

1. **Phase 6** — `SegmentExpander` injects a *single global* environment string into *every* segment. Reproduced live: a forest scene receives `mountain, temple`. This is a P0 correctness bug, not a refinement, and it sits inside the layer the plan calls "strongly implemented."
2. **Phase 16** — the `additional_emb_dim=512` injection is a *provable* constructor crash, not a latent risk. I2V has never once executed.
3. **Phase 11/12** — `vid_seq_len=81` is not just a magic number, it is a unit error: a frame count is being passed where a patched token count is required. The model asserts on it. (The exact correct number depends on an unresolved VAE temporal question — see *Status* below.)

### Missing blockers

| # | Issue |
|---|---|
| 1 | `settings.output_dir` is `str` but used with `/` → `TypeError` on the cinematic success path |
| 2 | `scene.keyframe_path` is **never assigned**, and the `reference_frame_path` fallback was lost → I2V can never trigger (second independent lock on the same door) |
| 3 | `ModelRegistry.unload()` never calls the teardown contract, and no engine nulls `self.bundle` → VRAM is not actually released |
| 4 | The entire Ovi variant system (`model_name`, `fp8`, `sample_steps`) is absent, and `inference_fusion.yaml` is read by nothing |

### Most useful unmined asset

`utils/model_loading_utils.py` is the **upstream Ovi reference loader already vendored in this repo**. It already does `strict=True` and returns `(missing, unexpected)` (Phase 14's goal), and already constructs `FusionModel` without the `additional_emb_dim` injection (Phase 16's goal). Its only defects are 4 bad import paths and 2 CWD-relative config paths. **Repair it and delegate to it — do not reimplement loading.**

### Hardware verdict (measured)

RTX 5060 Laptop, **8,151 MiB VRAM**, **15.2 GB RAM (5.7 GB free)**.

Two DiTs ≈ **14.1 GB in bf16**. Repo guards declare 14 GB (video) / 15 GB (fusion); upstream Ovi states **32 GB minimum, 24 GB with FP8**.

- Video and fusion → **Kaggle/remote only**. Plan is correct.
- The plan missed that **CPU offload is also unviable locally** — 15.2 GB RAM cannot park offloaded weights regardless of GPU.
- Additionally `requirements.txt` pins Jan-2024 libs that predate Blackwell `sm_120`.

## Recommendation

Approve the plan, with these amendments:

1. **Restore, don't rewrite** (Phase 2). `git checkout e1fbbbb -- <files>` recovers a 14,913-byte ImageEngine that already had the *correct* lifecycle and the exact contract 4 test files encode. The same commit also holds an **image→video keyframe handoff the current tree lost** — `e1fbbbb` fell back to `scene_state.reference_frame_path` (which the image path *does* populate) and threaded a `keyframe_latent` into `VideoEngine.generate_frames`. HEAD dropped both, so Phase 7's handoff must be rebuilt too.
2. **Reorder.** Insert a `B0/B1` (repair `model_loading_utils`, make `inference_fusion.yaml` load) before image work, and pull Level-2 model-contract tests ahead of Phase 11/12 — they would have caught the `vid_seq_len` bug immediately.
3. **Coordinate Phases 23, 20, 37, 46.** Removing schema fields before isolating legacy engines breaks `generation_manager.py:275`; removing `AudioEngine` also requires updating `cli.py:145`.
4. **Lower stated confidence from ~0.93 to ~0.80** until Phases G/H/I pass a real forward pass.

See [`PLAN_REVIEW.md` §9](./PLAN_REVIEW.md#9-recommended-amended-order) for the full amended order.

## Status: not yet started

Implementation has **not** begun. The revised roadmap was reviewed first and needs four corrections before it is safe to execute — see [`ROADMAP_REVIEW.md`](./ROADMAP_REVIEW.md).

The most important: **do not freeze `39,204` as the target sequence length.** Re-checking the vendored VAE shows `temperal_downsample=[False, True, True]` gives **4x temporal compression**, so the latent for 81 frames is likely `[48, 21, 45, 45]`, not `[48, 81, 45, 45]` — making the real sequence length **10,164**, not 39,204. Both my earlier review and the revised roadmap inherited `81` from the engine's own hardcoded assumption. The spatial factor (16x → 45) is solid; the temporal one is not. Settle it empirically by encoding a real clip before writing any invariant.

## Limitations

- No inference was run — no weights are mounted (`/kaggle` absent; every `model_config.yaml` path is Kaggle-only).
- `torch` is not installed in the active interpreter; 6 test modules error at collection. Their status is **unknown, not passing**.
- The CLIP tokenizer under-count risk (Phase 5) is **reasoned, not measured** — `transformers` unavailable. Measure once installed.
- Upstream Ovi was compared via its README, not a line-by-line diff against `ovi/`. That comparison is Phase 0's job and has not been done.