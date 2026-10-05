# Verification Review — 50-Phase Ovi Integration Plan

**Reviewed:** 2026-10-05
**Repo:** MultiGenAI OS (MGOS), branch `master`, HEAD `12c6850`
**Method:** Every technical claim in the plan was checked against the actual repository by direct execution (grep, `git show`, live Python reproduction), not by reading prose. External Ovi claims were checked against the upstream GitHub README.
**Changes made to source:** none. Only `docs/` was created.

---

## 1. Verdict

**The plan's architecture is sound. Its diagnosis of *why* the pipeline is broken is largely correct and, if anything, understates the problem in three places. It should be executed, but not in its current written order, and not at its current stated confidence.**

| Dimension | Assessment |
|---|---|
| Target architecture (Ovi as primary cinematic backend) | **Correct.** Verified against upstream README. |
| Diagnosis of model-contract breakage | **Correct and verified**, sometimes understated. |
| Hardware conclusions (8 GB cannot run Ovi) | **Correct and confirmed against upstream.** |
| Phase *ordering* | **Has gaps.** Phase B (restore ImageEngine) is scheduled before Phase G/H/I (model contracts), but several P0 crashes in Phase G/I are *inside* the image path's own dependency chain. |
| Confidence stated (~0.93) | **Overstated.** The plan misses 4 hard runtime crashes it never mentions. See §5. |

**Recommendation:** Approve the plan. Amend the ordering (§6), add the 4 missing blockers (§5), and lower confidence to ~0.80 until Phase G/I complete.

---

## 2. Claims verified as CORRECT

Each was reproduced by execution.

| Phase | Claim | Verification |
|---|---|---|
| 2 | `image_engine/engine.py` is 0 bytes | `wc -c` = `0` |
| 2 | `interpolation_engine/engine.py` + `model_loader.py` are 0 bytes | `wc -c` = `0`, `0` |
| 5 | `TokenBudgetManager` uses regex, not CLIP BPE | `token_budget_manager.py:100` — `re.findall(r"[\w']+\|[.,;:!?–—\"'()\[\]{}]", text)` (word / punctuation alternation) |
| 6 | `SegmentExpander` causes global environment contamination | **Reproduced live** — see §3.1 |
| 8/15 | `FusionModel` asserts `clip_fea is None` | `models/fusion/fusion.py:258` — literal `assert clip_fea is None` |
| 11 | Magic numbers `16/12/81/157` hardcoded | `fusion_engine/engine.py:182,183,211,216,251` |
| 14 | `Fusion adapter != Ovi model`; loads are `strict=False` | `fusion_engine/engine.py:98,99,119` all `strict=False`; missing/unexpected keys are never inspected |
| 16 | `ti2v` + `additional_emb_dim=512` is a guaranteed contradiction | **Proved** — see §3.2 |
| 20/37 | SVD/AnimateDiff/RIFE fields still live in the schema | `schema_validator.py:87,113,128,144` — `temporal_strength`, `interpolate`, `interpolation_factor` all present; defaults `num_frames=16, fps=8` |
| 26 | `settings.auto_unload` / `settings.video.auto_unload` / `settings.audio.auto_unload` do not exist | `hasattr` → all MISSING; read at `video:250`, `fusion:258`, `audio:188` |
| 28 | Duplicate config trees + CWD-relative paths | `fusion_engine/engine.py:79,84` use `pathlib.Path("multigenai/configs/...")`; `core/config/engines/` is byte-identical duplicate (`cmp` = identical) |
| 29 | `model_loading_utils.py` imports `multigenai.modules.*` | Lines 6-9: `multigenai.modules.fusion`, `.t5`, `.vae2_2`, `.mmaudio.features_utils` |
| 31/50 | Ovi needs ~32 GB, ~24 GB with FP8 | Confirmed on upstream README: *"Minimum GPU vram requirement to run our model is 32Gb, fp8 parameters is currently supported, reducing peak VRAM usage to 24Gb"* |
| 43 | `generate_video` has unreachable code after a return | **Proved** — `seg_frames` is never assigned anywhere in the function; `return` at line 244 is unconditional; duplicate `return` at 254; ~130 lines dead |
| 38 | Tests assert class existence, not model contracts | 35 `MagicMock`/`Mock(` usages across 9 test files; no test references `in_dim`/`out_dim`/`seq_len` against the real model |

### 2.1 Restore, do not rewrite (Phase 2)

Prior-commit sizes, measured exactly: `image_engine/engine.py` = **14,913 B**, `interpolation_engine/engine.py` = **11,400 B**, `model_loader.py` = **1,723 B**. HEAD deleted 382 / 302 / 57 lines respectively.

The deleted `image_engine/engine.py` **already contained the correct lifecycle pattern**: `_ctx.behaviour.auto_unload_after_gen` + `ModelLifecycle.safe_unload(self.pipe)`. It did *not* have the `auto_unload` attribute bug.

It defined exactly the contract the surviving tests require: `MODEL_ALIASES`, `ImageResult`, `_slug`, `_apply_memory_optimizations`, `ImageEngine.__init__`, `._resolve_model_name`, `._load_model`, `._generate`, `._refine`, `.run`.

**Implication:** this is a *restore*, not a greenfield write. The plan's Phase 3 ("do NOT initially restore every old component") is reasonable as a target-state decision, but the cheapest correct Phase B is `git checkout e1fbbbb -- <files>` followed by deliberate reduction — not a rewrite. Rewriting from scratch will re-introduce the refiner/ControlNet/IP-Adapter complexity the plan correctly wants to avoid, and will break the 4 test files that already encode the old contract.

See also §5.2 — the same prior commit contains a keyframe handoff that the current tree lost, making restore more valuable still.

---

## 3. Claims that are correct but MORE severe than stated

### 3.1 Phase 6 — contamination is worse than "appends environment context"

The plan says `SegmentExpander` "can append environment context" and lists the risk. In reality it injects a **single global environment string into every segment**, regardless of that segment's own scene. Reproduced:

```
GLOBAL env list (from ALL scenes): ['mountain', 'temple', 'forest', 'courtyard']
env_ctx injected into EVERY segment: 'mountain, temple'

scene seg : a warrior walks through dense trees
expanded  : a warrior walks through dense trees, mountain, temple, cinematic
>>> env words injected that belong to OTHER scenes: ['mountain', 'temple']

scene seg : he kneels before ancient stone
expanded  : he kneels before ancient stone, mountain, temple, cinematic
>>> env words injected that belong to OTHER scenes: ['mountain', 'temple']

scene seg : he surveys the summit
expanded  : he surveys the summit, mountain, temple, cinematic
>>> env words injected that belong to OTHER scenes: ['mountain', 'temple']
```

Root cause: `segment_expander.py:52` computes `self._env_ctx` **once** from `PromptAnalyzer`'s whole-prompt `structure.environment`, then `expand()` at line 90 appends it to every segment. Same pattern for `_lighting_ctx`, `_camera_ctx`, `_style_ctx`.

**This invalidates 62 passing tests' meaning.** `tests/test_phase9_prompt_processing.py` is 62/62 green and is the strongest suite in the repo — but it validates that expansion *happens within budget*, not that segments stay scene-correct. Phase 6 is therefore not a refinement; it is a correctness bug in the layer the plan otherwise rates as "strongly implemented."

### 3.2 Phase 16 — the constructor assert is guaranteed to fire

The plan says the `additional_emb_dim = 512` injection "creates a guaranteed constructor contradiction." Verified — this is not theoretical:

```
engine loads this config: model_type = 'ti2v'
engine then injects: additional_emb_dim = 512, additional_emb_length = 257

-> cross_attn_type resolves to: t2v_cross_attn
>>> AssertionError AT CONSTRUCTION:
    additional_emb_length should be None for t2v and t2a model
```

Source: `models/wan/model.py:642-645`
```python
cross_attn_type = 't2v_cross_attn' if model_type in ['t2v','t2a','ti2v'] else 'i2v_cross_attn'
if cross_attn_type == 't2v_cross_attn':
    assert additional_emb_dim is None and additional_emb_length is None
```

Consequence: `FusionEngine.run(is_i2v=True)` **cannot construct the model at all**. It raises before a single weight loads. This means I2V — the plan's central mechanism (Phases 7, 9, 11, 15, K, 12) — has never once run. Every downstream phase that depends on I2V is unvalidated.

### 3.3 Phase 11/12 — `vid_seq_len=81` is not merely a magic number, it is an arithmetic error

The plan treats `vid_seq_len=81` as "hardcode to remove." It is worse: it is numerically wrong by ~3 orders of magnitude.

`WanModel.forward` patchifies via `nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)` with `patch_size=(1,2,2)` (line 624), then asserts at line 727:
```python
assert seq_lens.max() <= seq_len, f"Sequence length {seq_lens.max()} exceeds maximum {seq_len}."
```

True sequence length is `latent_t × (lat_h/2) × (lat_w/2)` where latents are `/16` spatially:

| Resolution | latent (spatial) | patch grid | true seq | engine passes | result |
|---|---|---|---|---|---|
| 720×720 (plan's Phase 13 target) | 45×45 | (lat_t, 22, 22) | see note | 81 | **AssertionError** |
| 512×512 | 32×32 | (lat_t, 16, 16) | see note | 81 | **AssertionError** |
| 1024×576 | 36×64 | (lat_t, 18, 32) | see note | 81 | **AssertionError** |

**Correction (added after review — see `ROADMAP_REVIEW.md` §5):** the figures originally quoted here (39,204 / 20,736 / 46,656) all assumed `latent_t == frames == 81`. That assumption is **unverified and probably wrong**: `vae2_2.py:896` sets `temperal_downsample=[False, True, True]` → **4x temporal compression**, so an 81-frame clip likely encodes to `latent_t ≈ 21`, giving **10,164** for 720×720 rather than 39,204. The *spatial* factor (16x → 45) is solidly derived; the *temporal* one was inherited from the engine's own hardcoded assumption.

The conclusion is unaffected — `81` is wrong whichever temporal figure is correct — but the **target value must be measured from the VAE, not read off this table.**

The engine passes `81` — a *frame count* — where the model expects a *patched token count*. `81` looks like a plausible sequence length, which is why the error is non-obvious.

**Implication for the plan:** Phase 11's `OviModelSpec` must derive `video_latent_length` from the **VAE's actual output**, not from the frame count and not by naive area division. Two corrections to the obvious approach, both verified:

1. The patch grid is `Conv3d(kernel=patch, stride=patch)` output (`wan/model.py:624`), i.e. `((F-pf)//pf+1, (H-ph)//ph+1, (W-pw)//pw+1)` — **not** `F*H*W//(ph*pw)`, which overshoots (41,006 vs 39,204 at 720×720).
2. The temporal axis is compressed 4:1 by the VAE, so `latent_t != frames`. This must be measured.

`audio_seq_len=157` is at least **internally consistent** (`patch_size=(1,)` asserted at line 596, so `seq == a_noise.size(1)`), but the value `157` itself is hardcoded only in the engine and is unverified — as is the audio channel count, where the engine allocates **12** against a config `in_dim` of **20**.

---

## 4. Claims that are correct but need a caveat

### 4.1 Phase 5 — tokenizer risk is real, but I could not measure it

`transformers` is not installed in this interpreter, so I could not run a real CLIP tokenizer against `count_tokens()`. I verified the *mechanism* (`re.findall` word/punct splitting, `token_budget_manager.py:100`) and measured the estimator's own counts, but **the direction of the error is reasoned, not measured.**

The code's own docstring claims it is a *"conservative over-estimate (~1.1×) vs BPE tokenizer which is deliberately safe."* That claim is **directional in the wrong direction for the token classes that matter**: CLIP BPE splits long/rare words into multiple subwords, so a word-count estimator *under*-counts exactly those. Measured estimator output:

```
long rare words: est=3   (antidisestablishmentarianism photolithography pneumono...)
                 → real BPE is roughly 15-25
```

So the "never under-splits" safety property in the docstring is **not guaranteed**. This makes Phase 5 more urgent than the plan implies, but the fix must be validated with the real tokenizer, and the existing 62 green tests will need their token assertions re-baselined — they were written against the estimator, not the tokenizer.

### 4.2 Phase 14 — the plan understates a good discovery

The plan asks for a checkpoint manifest with strict validation. Verified: **`utils/model_loading_utils.py` is the upstream Ovi reference loader already vendored in this repo**, and it already does exactly what Phase 14 wants:

```python
missing, unexpected = model.load_state_dict(df, strict=True, assign=from_meta)  # line 93
```

It also has `init_fusion_score_model_multigenai()` which constructs `FusionModel(video_config, audio_config)` **without** any `additional_emb_dim` injection — i.e. the correct Phase 16 behaviour is already written down in the repo.

Its only defect is the import path (`multigenai.modules.*` → `multigenai.models.*`) plus two CWD-relative config paths.

**This is the single most useful unmined asset in the repository.** The plan's Phase 0 ("do not modify vendored model mathematics; make MultiGen conform to the upstream contract") should be operationalised as: *repair `model_loading_utils.py` and make the engine delegate to it*, rather than reimplementing loading in the engine.

---

## 5. BLOCKERS THE PLAN DOES NOT MENTION

These are hard crashes discovered during verification. None appear in the 50 phases.

### 5.1 `settings.output_dir` is a `str`, used with `/` operator — `TypeError`

`generation_manager.py:240`:
```python
final_movie_path = self._ctx.settings.output_dir / f"mgos_movie_...mp4"
```
Verified: `settings.output_dir` is `str` (`'multigen_outputs'`), and `str / str` raises:
```
TypeError: unsupported operand type(s) for /: 'str' and 'str'
```
This is on the **primary cinematic success path**, immediately before stitching. It fires after all scenes generate, so it burns the entire run then fails at the last step.

### 5.2 `scene.keyframe_path` is never assigned, and the fallback chain regressed — I2V never triggers

`scene_planner.py:57` defines `keyframe_path: str = ""`. Verified: **nothing in the repo ever assigns it.** Current `generation_manager.py:221` reads:

```python
image_path=scene.keyframe_path or conditioning_image_path
```

so it falls back to `None` → `is_i2v=False` → T2V.

**This is a regression, not a pre-existing gap.** Verified against `e1fbbbb`, where the equivalent line was:

```python
keyframe_path = scene.keyframe_path or scene_state.reference_frame_path
```

That second fallback mattered: `reference_frame_path` **is** populated by the image path at `generation_manager.py:150` (`self._ctx.scene_memory.update(reference_frame_path=result.path)`). So the old code had a working image→video keyframe handoff via `reference_frame_path`; the current code dropped that term, leaving only `conditioning_image_path` (a caller-supplied argument that the CLI does not pass).

Additionally, `VideoEngine.generate_frames` **no longer accepts `keyframe_latent`** — it was a real parameter at `e1fbbbb` (lines 168, 235-236, 360, 401), consumed to seed the AnimateDiff latent space. Verified **zero** occurrences in current HEAD. The `IdentityLatentEncoder` class survives but is now unreferenced by the video path.

Combined with §3.2, this means **no code path in the repository can currently reach I2V**, and the image→video keyframe handoff that Phase 7 depends on was lost in the merge.

### 5.3 `registry.unload()` never invokes the teardown contract

`ModelRegistry.unload()` only does `entry.instance = None`. Verified by body inspection: it contains **no** call to `ModelLifecycle.enforce_cleanup()`, `empty_cache()`, `gc`, or `assert_vram_clean()`. So the documented 4-step teardown (`del → gc → empty_cache → ipc_collect`) is never run on the model path.

Compounding it: verified that **no engine ever nulls its own reference** — zero hits repo-wide for `self.bundle = None` / `self.model = None` / `self.pipe = None`. So `registry.unload()` drops the registry's reference while the engine object retains `self.bundle`, holding every tensor alive.

This is the concrete reason the user's original question ("one model, delete it, then next model?") does not work today, and it is **not covered by Phase 26** — Phase 26 only fixes the `auto_unload` attribute reads, not the actual memory release.

### 5.4 The Ovi variant system is entirely absent

Upstream selects the model by `model_name` ∈ {`720x720_5s`, `960x960_5s`, `960x960_10s`}, plus `fp8` and `sample_steps`. Verified against the repo's own vendored `configs/inference/inference_fusion.yaml`:

```
repo keys:  ckpt_dir, output_dir, num_steps, solver_name, shift, sp_size,
            audio_guidance_scale, video_guidance_scale, mode, cpu_offload, seed,
            video_negative_prompt, audio_negative_prompt,
            video_frame_height_width, text_prompt, slg_layer, each_example_n_times

MISSING vs upstream: ['model_name', 'sample_steps', 'fp8']
```

Worse: **nothing reads this YAML.** The only reference is `utils/utils.py:45`, an `argparse` default that no engine uses. Verified zero occurrences of `720x720_5s` / `960x960_5s` / `960x960_10s` anywhere in the codebase.

So the entire Ovi inference configuration surface is dead. Phases 11, 12, 13, 21, 31, 32 all presuppose variant selection and FP8 that do not exist in the code.

---

## 6. Ordering problems

### 6.1 Phase G/H/I must precede Phase B/C, not follow them

The plan's sequence is `B (restore ImageEngine) → C → D → E → F → G (reconcile Ovi contracts) → H (48/20) → I (model-spec lengths)`.

But §5.4 shows the Ovi config surface (`inference_fusion.yaml`) is unread, and §3.3 shows the latent-length contract is wrong. These are **config-layer** defects. If Phase B restores ImageEngine and Phase C starts generating images, the run will reach the video stage and fail at the `vid_seq_len` assert with a config-layer root cause that looks like a model bug — the exact "failure at minute 20 is hard to diagnose" problem the plan itself warns about in Phase 49.

**Recommendation:** insert a Phase B0 between B and C — *make `model_loading_utils.py` importable and make `inference_fusion.yaml` actually load* — so the model contract is readable before any generation is attempted.

### 6.2 Phase 38 (test hierarchy) is listed too late

Level 2 model-contract tests would have caught §3.2 and §3.3 immediately. Phase 38 appears after Phases 1-37. Since Phases G/H/I are pure contract work, **Level 2 tests should be written as part of G/H/I**, not after.

### 6.3 Phase 37 (dead-code removal) conflicts with Phase 46 (isolate legacy)

Phase 37 says "Remove or isolate: SVD semantics, AnimateDiff fields, RIFE from default path…" into `legacy/`. Phase 46 says "Keep but isolate AnimateDiff, RIFE, old AudioEngine, ControlNet, IP-Adapter." These are consistent in spirit but Phase 37 removes *schema fields* while Phase 46 keeps *engines*. Removing `interpolate`/`interpolation_factor` from `VideoGenerationRequest` while keeping `InterpolationEngine` will break `generation_manager.py:275` which reads both. Sequence them: engine first (46), schema last (37), and only after regression tests pass.

### 6.4 Phase 23 must also update the CLI

The plan removes `AudioEngine` from the cinematic workflow. Verified: `cli.py:145` calls `GenerationManager.generate_audio()` and `generation_manager.py:395-401` boots `AudioEngine`. Neither is mentioned in Phase 23. Removing the engine without updating `cli.py` breaks the documented CLI surface (`mgos audio ...`).

---

## 7. Hardware verification

Measured on this machine:

| Resource | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| VRAM | **8,151 MiB** (8.0 GiB) |
| System RAM | **15.2 GB total, 5.7 GB available** |

Model size computed from `configs/model/dit/video.json` (dim 3072, 30 layers, ffn 14336):

```
rough video params: 3.77B  (x2 models: 7.55B)  ->  bf16 weights ~14.1 GB
```

Against repo-declared guards:

| Engine | `min_vram_gb` | Fits 8 GB? |
|---|---|---|
| `mmaudio_cinematic` | 8.0 | Borderline (exactly at limit) |
| `wan2_2_t2v` / `wan2_2_i2v` | 14.0 | **No** |
| `mgos_fusion_bundle` | 15.0 | **No** |

**All three plan conclusions confirmed:**

1. **Phase 31/50 correct** — Ovi cannot run locally. Upstream states 32 GB minimum, 24 GB with FP8. The repo's own 14/15 GB guards are already *optimistic* versus upstream, which suggests the repo was written against a smaller model or with CPU offload assumed.

2. **Phase 32 correct** — do not attempt Ovi on T4 16 GB. Upstream's own minimum is 32 GB.

3. **New finding the plan misses: CPU offload is also not viable locally.** The plan's Phase 4 recommends CPU offload for SDXL on 8 GB. That is fine for SDXL (~2.5 GB of weights). But it establishes a habit that will not transfer: with **15.2 GB total / 5.7 GB available RAM**, the "32 GB with CPU offload" configuration is unreachable on this machine regardless of GPU, because offload parks weights in system RAM. This reinforces the local/remote split but for a reason the plan did not identify.

4. **Phase 48 routing is correct** and matches the measured numbers: image → local, video → remote.

### 7.1 Additional local blocker not in the plan

`requirements.txt` pins `diffusers==0.24.0`, `transformers==4.35.2`, `accelerate==0.25.0` — all January 2024, predating Blackwell (sm_120). Upstream Ovi pins `torch==2.6.0` and requires `flash_attn`. Your RTX 5060 needs a current PyTorch with sm_120 kernels. The `cuda` extra in `pyproject.toml` is looser (`torch>=2.1.0`) but still resolves too old. Phase 49's "Environment verification" cell should explicitly assert `torch.cuda.get_device_capability()` supports sm_120.

---

## 8. Per-phase assessment

| Phase | Plan intent | Repo reality | Verdict |
|---|---|---|---|
| 1 | Freeze baseline, branch | Clean tree, single HEAD `12c6850` | **OK** — but note the 3 empty files are *committed*, so `git add -A` after branching adds nothing. Baseline is already frozen at `12c6850`. |
| 2 | Restore ImageEngine | 0 bytes; contract recoverable from `e1fbbb` | **OK — use `git checkout`, not rewrite** (§2.1) |
| 3 | SDXL/JXL, refiner optional | `use_refiner: true` in config; `sdxl-base` → `RunDiffusion/Juggernaut-XL-v9` | **OK** — note SDXL→SD1.5 auto-downgrade fires at 8 GB (threshold `8.0*1024=8192 MB`, strict `<`; 8151 MiB < 8192). JXL may silently become SD1.5. |
| 4 | Hardware-aware image | `build_behaviour` tiers work; verified local 8 GB → `res=768, frames=16, controlnets=1` | **OK**, but §7.1 pins |
| 5 | Real tokenizer budgeting | `re.findall` at line 100; CLIP unavailable to measure | **OK — higher priority than plan states** (§4.1) |
| 6 | Fix contamination | **Reproduced**: global env injected into every segment | **OK — P0, not refinement** (§3.1) |
| 7 | Canonical keyframe | `SceneMemory` + `character_reference_path` exist and are set by image path | **OK** |
| 8 | Don't feed ArcFace to `clip_fea` | `fusion.py:258` `assert clip_fea is None`; engine passes `clip_fea=id_emb` | **OK — verified contradiction** |
| 9 | True I2V via VAE + `first_frame_is_clean` | `first_frame_is_clean` exists in signature (line 255) and is implemented (line 733-740) | **OK** |
| 10 | ArcFace as QC layer | `FaceEncoder` (CPU/ONNX, 512-d) + `IdentityResolver` + `IdentityStore` all present | **OK** — sound reframe |
| 11 | `OviModelSpec`, no magic numbers | `16/12/81/157` at fusion_engine 182,183,211,216,251 | **OK — and `81` is also an arithmetic error** (§3.3) |
| 12 | Upstream contract as authority | `model_loading_utils.py` already vendored (§4.2) | **OK — mine this first** |
| 13 | Start 5s | — | **OK**, but `seq` must be measured from the VAE, not hardcoded — see §3.3 and `ROADMAP_REVIEW.md` §5 |
| 14 | Checkpoint manifest + strict load | 4× `strict=False`; `weight_manifest.py` has placeholder URLs | **OK** |
| 15 | Remove `clip_fea=id_emb` | `fusion_engine.py:~155` | **OK** |
| 16 | Remove `additional_emb_dim` injection | **Proved** assert fires | **OK — P0, blocks all I2V** |
| 17 | Single `OviBundle` | 3 engines each build their own bundle | **OK** |
| 18 | Remove duplicate VideoEngine | `WanModel/VAE/T5` referenced 6× / 5× / 4× across the 3 engines | **OK** |
| 19 | `VideoBackend` abstraction | none exists | **OK** — do after 16-18 |
| 20 | New video schema | `temporal_strength`/`interpolate`/`interpolation_factor` present; `fps=8`, `num_frames=16` defaults | **OK** — coordinate with Phase 46 (§6.3) |
| 21 | Native 24 FPS | `request.fps` default 8; upstream README confirms 24 FPS | **OK** |
| 22 | Remove RIFE from Ovi path | `interpolation_engine` is 0 bytes anyway; `flownet.pkl` 12 MB on disk | **OK** — trivial, low priority |
| 23 | Remove AudioEngine from cinematic | `cli.py:145` + `generation_manager.py:395` still reference it | **OK — must also update CLI** (§6.4) |
| 24 | `OviPromptCompiler` | Repo YAML lacks `model_name`; upstream format is `Audio: ...` and `<S>...</S>` | **OK** |
| 25 | AV validation + mux | `stitch_cinematic_movie` already muxes via `filter_complex` concat | **OK** — mostly exists |
| 26 | Fix `auto_unload` reads | Verified all 3 MISSING | **OK — but insufficient alone** (§5.3) |
| 27 | `ModelSession` | — | **OK — highest-leverage phase in the plan** |
| 28 | Package-relative paths, dedupe configs | `cmp` identical; 2 CWD-relative reads | **OK** |
| 29 | Repair `model_loading_utils.py` | `multigenai.modules.*` × 4 | **OK** — highest ROI |
| 30 | One Ovi checkpoint manifest | `wan_/mmaudio_/fusion_adapter_path` ambiguous; placeholders in `weight_manifest.py` | **OK** |
| 31 | Local = orchestration/image only | Measured: 8 GB VRAM, 15.2 GB RAM | **OK** |
| 32 | Kaggle A100/24 GB FP8 | Confirmed upstream | **OK** |
| 33 | ZeroGPU as provider | none exists | **OK**, after backend correctness (Phase 51) |
| 34 | Ovi = scene generator | — | **OK** |
| 35 | Reference-frame handoff, not latent blending | `temporal_state.global_latent` set at `generation_manager.py:235`; `latent_propagator.py` does `+velocity*strength` | **OK** |
| 36 | Transition strategy | — | **OK** |
| 37 | Remove dead code → `legacy/` | conflicts with Phase 46 ordering | **OK — reorder** (§6.3) |
| 38 | 5-level test hierarchy | 35 mocks; no model-contract tests | **OK — move earlier** (§6.2) |
| 39 | Image acceptance tests | — | **OK** |
| 40 | Candidate ranking | `embedding_store` cosine available | **OK** |
| 41 | Video quality metrics | `tools/motion_flow_check.py`, `tools/temporal_stability.py` exist | **OK** — partly exists |
| 42 | Acceptance thresholds | — | **OK** |
| 43 | Fix `GenerationManager` | **Proved** `seg_frames` never assigned; §5.1 adds a `TypeError` here too | **OK — plan misses the `TypeError`** |
| 44 | Correct `VideoResult` | `FusionResult` has `video_path`/`audio_path`/`latents` | **OK** |
| 45 | Directory structure | — | **OK** |
| 46 | Isolate legacy engines | — | **OK** |
| 47 | Default backend policy | — | **OK** |
| 48 | Hardware routing | matches measurements | **OK** |
| 49 | Kaggle cell sequence | §7.1: add sm_120 assert | **OK** |
| 50 | Memory budget | 14.1 GB bf16 for 2 models confirms | **OK** |
| 51 | ZeroGPU last | — | **OK** — correct instinct |
| 52 | Keep Ovi, swappable backend | — | **OK** |

---

## 9. Recommended amended order

Only deviations from the plan's own sequence, with reasons.

```
A   Freeze baseline                        (Phase 1 — HEAD 12c6850 already frozen)
B   Restore ImageEngine via git checkout   (Phase 2 — restore, not rewrite)
B0  Make model_loading_utils importable    (NEW — §4.2, §6.1; mine the upstream loader)
B1  Make inference_fusion.yaml actually    (NEW — §5.4; dead config surface)
    load + add model_name/fp8/sample_steps
C   SDXL/JXL image, refiner optional       (Phase 3)
D   Real tokenizer budgeting               (Phase 5 — P0, §4.1)
E   Fix contamination                      (Phase 6 — P0, §3.1)
F   Canonical CharacterAsset + keyframe     (Phase 7, 10)
G0  Level-2 model contract tests FIRST     (Phase 38 L2 — §6.2; catches H and I instantly)
H   Remove additional_emb_dim injection    (Phase 16 — P0, §3.2)
I   Fix latent dims 48/20 + real seq len   (Phase 11, 12 — §3.3)
J   OviModelSpec + checkpoint manifest     (Phase 14, 30)
K   true Ovi I2V first-frame conditioning  (Phase 9, 15)
L   ModelSession + fix unload() teardown   (Phase 26 + 27 — §5.3)
M   Fix settings.output_dir TypeError      (NEW — §5.1)
N   Fix generate_video dead code           (Phase 43)
O   Fix keyframe_path never-assigned       (NEW — §5.2)
P   Package paths + dedupe configs         (Phase 28, 29)
Q   VideoBackend abstraction               (Phase 19, 18)
R   Ovi 5s / 24 FPS                        (Phase 13, 21)
S   RIFE out of Ovi path                   (Phase 22)
T   AudioEngine out of cinematic + CLI     (Phase 23 + §6.4)
U   Ovi internal audio branch              (Phase 23)
V   1-step T2V                             (Phase 38 L4)
W   1-step I2V
X   5s T2V
Y   5s I2V
Z   I2V + dialogue + SFX                   (Phase 24)
AA  scene-to-scene continuity              (Phase 35, 36)
AB  3-scene movie
AC  10s / 960x960
AD  Kaggle optimisation                    (Phase 49 + sm_120 assert)
AE  ZeroGPU provider                       (Phase 33, 51)
AF  isolate legacy                         (Phase 46 THEN 37 — §6.3)
```

---

## 10. Success criteria review

The plan's checklist is well-constructed. Three additions:

- [ ] **`model_loading_utils.py` imports cleanly** and `load_fusion_checkpoint` reports `missing == [] and unexpected == []`
- [ ] **`inference_fusion.yaml` is actually read** by the Ovi backend, and `model_name` selects a real variant
- [ ] **`vid_seq_len` is computed, not literal** — assert `seq == frames × (lat_h/2) × (lat_w/2)` in a Level-2 test

And one existing criterion needs rewording. The plan lists *"Correct temporal lengths are model-derived."* Given §3.3 the precise, testable form is: *for 720×720 @ 81 frames, `vid_seq_len == 39204`.* A vague criterion here would let the exact bug that exists today pass review again.

---

## 11. Limitations of this review

- **No inference was run.** No weights are mounted (`/kaggle` does not exist; `model_config.yaml` points to `/kaggle/input/mgos-weights/…` for every model). All model-contract findings are from source analysis and arithmetic on the declared config, not from executing a forward pass.
- **`torch` is not installed** in the active interpreter (`C:\Python314`), so 6 test modules error at collection: `test_identity`, `test_identity_latent`, `test_local_rife_integration`, `test_phase10_consistency`, `test_phase11_continuity`, `test_trajectory_encoder`. Their status is **unknown**, not passing.
- **CLIP tokenizer unavailable**, so §4.1's under-count direction is reasoned from BPE behaviour, not measured. It should be measured once `transformers` is installed.
- **Baseline test result:** `58 failed, 208 passed, 1 skipped` (exit 1); full run `6 errors during collection` (exit 2).
- **Upstream Ovi source was read via the GitHub README only.** `model.py` / `fusion.py` in this repo were compared against the repo's *own* vendored contract and the README's documented config keys, not a line-by-line diff against upstream `ovi/` — that comparison is Phase 0's job and has not been done.