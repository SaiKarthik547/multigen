# Repo Baseline Snapshot

Captured 2026-10-05, before any repair. Regenerate with the commands in §7.

## 1. Environment

| Item | Value |
|---|---|
| Repo | MultiGenAI OS (MGOS) |
| Branch | `master` |
| HEAD | `12c6850` — *"feat: initialize multigenai core architecture, engines, and model integrations"* |
| Working tree | clean (`git status --porcelain` → empty) |
| Python (active) | 3.14.7 at `C:\Python314` — **no torch installed** |
| Project `.venv` | **broken** — points at a missing `Python312` interpreter |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| VRAM | 8,151 MiB total (8.0 GiB), 0 MiB used |
| System RAM | 15.2 GB total, 5.7 GB available |
| `nvidia-smi` driver | 617.14, CUDA UMD 13.4 |

## 2. Test baseline

```
$ python -m pytest -q
6 errors during collection          (exit 2)

$ python -m pytest -q --ignore=<the 6 torch-error modules>
58 failed, 208 passed, 1 skipped   (exit 1)
```

### Per-file

| File | Result |
|---|---|
| `test_phase9_prompt_processing.py` | **62 passed** — only fully green suite |
| `test_system_consistency.py` | 3 passed |
| `test_debug_path.py` | 1 passed |
| `test_phase1.py` | 11 failed, 58 passed |
| `test_phase7_hardening.py` | 8 failed, 15 passed, 1 skipped |
| `test_phase7_optimization.py` | 5 failed, 4 passed |
| `test_phase8_interpolation.py` | 10 failed, 12 passed |
| `test_compute_stability.py` | 6 failed, 31 passed |
| `test_environment.py` | 2 failed, 20 passed |
| `test_llm_providers.py` | 16 failed, 2 passed |
| `test_identity.py` | collection error (torch) |
| `test_identity_latent.py` | collection error (torch) |
| `test_local_rife_integration.py` | collection error (torch) |
| `test_phase10_consistency.py` | collection error (torch) |
| `test_phase11_continuity.py` | collection error (torch) |
| `test_trajectory_encoder.py` | collection error (torch) |

### Failure causes (measured, not estimated)

| Count | Cause |
|---|---|
| 28 | `ModuleNotFoundError: No module named 'torch'` |
| 6 | `ModuleNotFoundError: No module named 'multigenai.llm.prmultigenaiders'` |
| 6 | `TypeError: EnhancementEngine.__init__() got an unexpected keyword argument 'provider'` |
| 3 | `AttributeError: 'LLMSettings' object has no attribute 'provider'` |
| 2 | `ImportError: cannot import name 'ProviderUnavailableError' from 'multigenai.core.exceptions'` |
| 2 | `ImportError: cannot import name 'ProviderResponseFormatError'` |
| 2 | `TypeError: ScenePlanner.__init__() got an unexpected keyword argument 'provider'` |
| 1 | `AttributeError: 'VideoGenerationRequest' object has no attribute 'num_inference_steps'` |
| 1 | `FileNotFoundError: .../interpolation_engine/engine.py` |

## 3. Zero-byte files (P0)

| Path | Bytes | Last non-empty commit |
|---|---|---|
| `multigenai/engines/image_engine/engine.py` | **0** | `e1fbbbb` (14,913 B) |
| `multigenai/engines/interpolation_engine/engine.py` | **0** | `e1fbbbb` (11,400 B) |
| `multigenai/engines/interpolation_engine/model_loader.py` | **0** | `e1fbbbb` (1,723 B) |

`git show --stat HEAD` confirms commit `12c6850` **deleted** these (`-382`, `-302`, `-57` lines) and replaced them with empty files.

### Recoverable contract (from `e1fbbb`)

`image_engine/engine.py` defined:
`MODEL_ALIASES`, `ImageResult`, `_slug`, `_apply_memory_optimizations`, `ImageEngine.__init__`, `._resolve_model_name`, `._load_model`, `._generate`, `._refine`, `.run`

It used the **correct** lifecycle — `_ctx.behaviour.auto_unload_after_gen` + `ModelLifecycle.safe_unload(self.pipe)` — and did **not** have the `auto_unload` attribute bug.

`interpolation_engine/engine.py` defined:
`InterpolationResult`, `InterpolationEngine.__init__`, `._load_model`, `._unload_model`, `._interpolate_pair`, `.interpolate`

Surviving tests encode this contract (e.g. `test_phase7_hardening.py:28` asserts `MODEL_ALIASES["sdxl-base"] == "RunDiffusion/Juggernaut-XL-v9"`).

## 4. Corrupted rename: `provider` → `prmultigenaider`

**174 occurrences across 11 files.** A bad find/replace.

Affected: `core/config/config.yaml`, `core/config/settings.py`, `core/exceptions.py`, `core/execution_context.py`, `identity/face_encoder.py`, `llm/enhancement_engine.py`, `llm/providers/api_provider.py`, `llm/providers/base.py`, `llm/providers/local_provider.py`, `llm/providers/__init__.py`, `llm/scene_planner.py`

Renamed symbols:
- `LLMProvider` → `LLMPrmultigenaider`
- `ProviderUnavailableError` → `PrmultigenaiderUnavailableError`
- `ProviderResponseFormatError` → `PrmultigenaiderResponseFormatError`
- `settings.llm.provider` → `settings.llm.prmultigenaider`

### Dead import path

`execution_context.py:164,171` imports:
```python
from multigenai.llm.prmultigenaiders.local_prmultigenaider import LocalLLMPrmultigenaider
from multigenai.llm.prmultigenaiders.api_prmultigenaider import APILLMPrmultigenaider
```
`multigenai/llm/prmultigenaiders/` **does not exist**. The real package is `multigenai/llm/providers/` with `local_provider.py` / `api_provider.py`. **The entire LLM path is dead.**

## 5. Missing attributes causing `AttributeError`

Verified via `hasattr` on a live `Settings`:

| Attribute | Status | Read at |
|---|---|---|
| `settings.auto_unload` | **MISSING** | `fusion_engine/engine.py:258` |
| `settings.audio` | **MISSING** | `audio_engine/engine.py:188` |
| `settings.video.auto_unload` | **MISSING** | `video_engine/engine.py:250` |
| `settings.video.enable_compile` | EXISTS | — |

All three misses are inside `finally:` / post-generation blocks, so they **mask the real generation error** with an `AttributeError`.

## 6. Verified runtime crashes

| # | Site | Error |
|---|---|---|
| 1 | `fusion_engine/engine.py` `is_i2v=True` | `AssertionError: additional_emb_length should be None for t2v and t2a model` (`wan/model.py:645`) |
| 2 | `fusion_engine/engine.py:211` | `AssertionError: Sequence length 39204 exceeds maximum 81` (`wan/model.py:727`) |
| 3 | `generation_manager.py:240` | `TypeError: unsupported operand type(s) for /: 'str' and 'str'` — `settings.output_dir` is `str` |
| 4 | `fusion_engine/engine.py:~155` | `AssertionError` from `fusion.py:258` `assert clip_fea is None` — engine passes `clip_fea=id_emb` |
| 5 | I2V never triggers | `scene.keyframe_path` is `""` and **never assigned anywhere**; the `e1fbbbb` fallback to `reference_frame_path` was also lost (see §9.1) |
| 6 | `llm.enabled=True` | `ModuleNotFoundError: multigenai.llm.prmultigenaiders` |

### Crash 2 arithmetic

`vid_seq_len` must equal `latent_t × (lat_h/2) × (lat_w/2)`. Spatial latents are `/16` (`patchify(2)` at `vae2_2.py:785` × 3 downsamples), `patch_size=(1,2,2)`:

| Resolution | latent (spatial) | patch grid | passed | result |
|---|---|---|---|---|
| 720×720 @81 | 45×45 | (lat_t, 22, 22) | 81 | crash |
| 512×512 @81 | 32×32 | (lat_t, 16, 16) | 81 | crash |
| 1024×576 @81 | 36×64 | (lat_t, 18, 32) | 81 | crash |

**The temporal factor is unresolved.** `vae2_2.py:896` sets `temperal_downsample=[False, True, True]` → 4x temporal compression, so `latent_t ≈ 21` for 81 frames, not 81:

- if `latent_t = 81` → seq = **39,204**
- if `latent_t = 21` → seq = **10,164**

Either way `81` is wrong (a frame count was passed where a token count was expected). The exact target must be measured from the VAE. See `ROADMAP_REVIEW.md` §5.

Also unverified: `audio_seq_len=157` is internally consistent (`patch_size=(1,)` asserted at `wan/model.py:596`) but the value is hardcoded only in the engine; and the engine allocates **12** audio channels against a config `in_dim` of **20**.

## 7. VRAM declarations vs hardware

| Engine | `min_vram_gb` | vs 8.0 GiB |
|---|---|---|
| `mmaudio_cinematic` | 8.0 | borderline |
| `wan2_2_t2v` / `wan2_2_i2v` | 14.0 | **fails** |
| `mgos_fusion_bundle` | 15.0 | **fails** |

Model size from `configs/model/dit/video.json` (dim 3072, 30 layers, ffn 14336):
**3.77 B params per DiT → 7.55 B for video+audio → ~14.1 GB in bf16.**

Upstream Ovi README: *"Minimum GPU vram requirement to run our model is 32Gb, fp8 parameters is currently supported, reducing peak VRAM usage to 24Gb."* The repo's own 14/15 GB guards are **optimistic** vs upstream.

### Simulated 8 GB tier

```
EnvironmentProfile(platform='local', device_type='cuda', vram_mb=7952)
-> mode: dev
-> max_image_resolution=768, max_video_frames=16, max_controlnets=1,
   ip_adapter_allowed=True, auto_unload_after_gen=False
```

`auto_unload_after_gen=False` on local — the unload path is skipped outside Kaggle.

### SDXL auto-downgrade

`_SDXL_VRAM_THRESHOLD_GB = 8.0` → 8192 MB, strict `<`:
`7952 → True`, `8151 → True`, `8192 → False`. **Juggernaut XL silently becomes SD1.5 on this card.**

## 8. What is genuinely strong

| Area | Evidence |
|---|---|
| Prompt processing | `prompting/` ~1,300 lines, 62/62 tests green. *But* see contamination bug below. |
| Core infra | `device_manager.py`, `environment.py`, `capability_report.py`, `model_registry.py`, `model_lifecycle.py`, `lifecycle.py`, `metrics.py`, `exceptions.py` |
| Model code | `models/wan/`, `models/fusion/`, `models/mmaudio/` (BigVGAN v1+v2, autoencoder), `models/shared/` (T5, CLIP), `utils/fm_solvers*.py` (1,677 lines) |
| RIFE weights | `models/rife/flownet.pkl` — 12,186,817 bytes actually on disk |
| Memory/identity/creative | `style_registry`, `world_state`, `embedding_store` (real cosine), `scene_memory`, `face_encoder` (CPU/ONNX 512-d), `scene_designer`, `prompt_compiler` |
| AV muxing | `movie_utils.stitch_cinematic_movie` already muxes via `filter_complex` concat |
| Upstream loader | `utils/model_loading_utils.py` already does `strict=True` + returns `(missing, unexpected)` |

## 9. Known defects (no crash, wrong behaviour)

### Lost image→video keyframe handoff (regression vs `e1fbbbb`)

| | `e1fbbbb` | current HEAD |
|---|---|---|
| fallback chain | `scene.keyframe_path or scene_state.reference_frame_path` | `scene.keyframe_path or conditioning_image_path` |
| `VideoEngine.generate_frames(keyframe_latent=...)` | accepted (lines 168, 235-236, 360, 401) | **removed** — 0 occurrences |
| `reference_frame_path` populated by image gen | yes | yes (`generation_manager.py:150`) |

`reference_frame_path` **is** set by the image path, so the old fallback chain gave a working image→video handoff. The current chain dropped it. `IdentityLatentEncoder` still exists but is no longer referenced by the video path.

Consequence: even setting aside the `additional_emb_dim` crash, **I2V cannot trigger**, and the keyframe handoff that Phase 7 depends on must be rebuilt.

### Prompt contamination — reproduced

```
GLOBAL env list: ['mountain','temple','forest','courtyard']
env_ctx injected into EVERY segment: 'mountain, temple'

  a warrior walks through dense trees
  -> a warrior walks through dense trees, mountain, temple, cinematic
  >>> leaked from other scenes: ['mountain','temple']
```
Root cause `segment_expander.py:52` — `_env_ctx` computed once globally, appended at line 90. Same for `_lighting_ctx`, `_camera_ctx`, `_style_ctx`.

### Unload contract not honoured

- `ModelRegistry.unload()` contains **no** `enforce_cleanup` / `empty_cache` / `gc` / `assert_vram_clean` call.
- **Zero** engines null their reference (`self.bundle = None` → no hits repo-wide).
- `ModelLifecycle.safe_unload(model)` only `del`s its own **local parameter**; callers must null their own ref, and none do.

→ `registry.unload()` drops one reference while the engine retains `self.bundle`. VRAM is not actually released.

### Unreachable code in `generate_video`

`seg_frames` is **never assigned** anywhere in the function. `return` at line 244 is unconditional; duplicate `return` at 254; lines 281–373 (interpolation, blending, encoding) are dead. `_generate_video_segment()` is a `pass` stub.

### Duplicate config trees — byte-identical

| File | `configs/` | `core/config/engines/` |
|---|---|---|
| `video.json` | `9a778eaa` | `9a778eaa` — identical |
| `audio.json` | `82168787` | `82168787` — identical |
| `inference_fusion.yaml` | identical | identical |

### CWD-relative paths

`fusion_engine/engine.py:79,84`; `model_loading_utils.py:36,37`; `utils/utils.py:45`

### Dead Ovi config surface

Repo `inference_fusion.yaml` is missing `model_name`, `sample_steps`, `fp8` vs upstream.
**Nothing reads the YAML** — only an `argparse` default at `utils/utils.py:45`.
Zero occurrences of `720x720_5s` / `960x960_5s` / `960x960_10s` anywhere.

### Stale schema fields

`VideoGenerationRequest` still carries `temporal_strength` ("Mapped to SVD motion_bucket_id"), `interpolate`, `interpolation_factor`, `num_frames` (default 16), `fps` (default 8). Docstrings reference SVD-XT at lines 87, 96, 113, 128, 144, 146.

### Pinched dependency versions

`requirements.txt`: `diffusers==0.24.0`, `transformers==4.35.2`, `accelerate==0.25.0` — Jan 2024, **predate Blackwell sm_120**. Upstream Ovi pins `torch==2.6.0` + `flash_attn`.

### Kaggle-only model paths

Every entry in `core/config/model_config.yaml` is `/kaggle/input/mgos-weights/…`. `/kaggle` does not exist locally. No weights mounted.

## 10. Regenerating this snapshot

```bash
git log --oneline -1
git status --porcelain
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python -m pytest -q -p no:cacheprovider
for f in multigenai/engines/*/engine.py; do echo "$f: $(wc -c < $f)"; done
grep -rni prmultigenaider --include=*.py --include=*.yaml . | grep -v .venv | wc -l   # 174
```