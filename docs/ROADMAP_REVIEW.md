# Review of the Revised Roadmap — errors found before starting

**Reviewed:** 2026-10-05
**Subject:** the second response ("Gate 0–Gate 8", "Final corrected roadmap", "Bottom line")
**Method:** every formula, sequence number, and ordering claim re-checked against the repo by execution.
**Status:** the revision's *architecture* and *confidence framing* are sound and I endorse them. But **three of its technical specifics are wrong**, and one is a formula that would embed a bug into the contract test meant to catch bugs. Do not start implementation from it as written.

**No source files were changed.**

---

## 1. Blockers to fix before starting

### 1.1 The proposed `vid_seq_len` invariant formula is wrong

The revision proposes making this executable:

```python
assert vid_seq_len == (
    video_noise.shape[1] * video_noise.shape[2] * video_noise.shape[3] // (patch_h * patch_w)
)
```

For the document's own worked example (720×720, 81 frames):

| | value |
|---|---|
| video_noise `[48, 81, 45, 45]` | — |
| formula `81*45*45 // 4` | **41,006** |
| actual Conv3d output | **39,204** |
| error | **+1,802** |

The formula divides the **product** `lh*lw` instead of each dimension separately. `wan/model.py:624` uses `nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)`, so each spatial dim is floored independently.

Verified mechanism (per-dim, `patch=2`): `(l-2)//2 + 1 == l//2` for **all** `l >= 1` — so floor-dividing *per dimension* is actually correct. The bug is that the naive form computes `(lh*lw)//4`, which differs from `(lh//2)*(lw//2)` whenever `lh*lw % 4 != 0`:

| latent | `lh*lw` | `%4` | `(lh*lw)//4` | correct | naive? |
|---|---|---|---|---|---|
| (45, 45) — 720×720 | 2025 | 1 | 506 | 484 | **WRONG** |
| (32, 32) — 512×512 | 1024 | 0 | 256 | 256 | ok |
| (36, 64) — 1024×576 | 2304 | 0 | 576 | 576 | ok |
| (60, 60) — 960×960 | 3600 | 0 | 900 | 900 | ok |
| (21, 45) | 945 | 1 | 236 | 220 | **WRONG** |

**Why this is dangerous: the naive form is right 8 times out of 10** across the resolutions that matter. It passes at 512², 960², 1024×576, 1280×704 — and fails only at **720×720, the exact resolution the plan targets.** A gate test written that way would look healthy until the first real 720p run.

**Correct invariant:**

```python
def derive_seq_len(frames, lat_h, lat_w, patch):
    pf, ph, pw = patch                       # (1, 2, 2) for Wan2.2 video
    f_out = (frames - pf) // pf + 1
    h_out = (lat_h - ph) // ph + 1
    w_out = (lat_w - pw) // pw + 1
    return f_out * h_out * w_out
```

Validated against `Conv3d` output arithmetic across 10 configurations (2 frame counts × 5 resolutions) — **10/10 exact**, while the naive form diverges on the 720×720 cases. With `latent_t = 81` this gives `(81, 22, 22)` → **39,204**, matching the revision's number, so the *value* is right under its own assumption and only the *formula* is wrong.

**This matters more than a typo.** A Level-2 test is the gate meant to catch regressions. If it encodes `F*H*W//(ph*pw)`, it will fail at 720×720 with a confusing 1,802 delta, or worse, be "fixed" by relaxing it until it passes — destroying the gate's value.

**But do not freeze `39,204` either — see §5. That number is itself unverified.**

### 1.2 `audio latent = 157` is asserted as known-good without support

The revision states, in §11:

```text
audio latent → audio_seq_len = 157
```
> "for the relevant Ovi 5-second configuration. The tests should assert those values."

I verified the *mechanism* — `wan/model.py:596` asserts `patch_size == (1,)` for audio, so audio sequence length does equal `a_noise.size(1)`, and 157 is internally consistent with the engine's own allocation. **That part is sound.**

But **the value 157 itself is unverified against any checkpoint.** It appears nowhere except the hardcoded engine literal at `fusion_engine/engine.py:183,211,216`. Upstream selects it per-variant via `model_name`; this repo has no variant system at all. Asserting `157` as a known-good constant would freeze an unvalidated number into a test and give it false authority.

**Fix:** assert the *invariant* (`audio_seq_len == a_noise.shape[1]`, and `a_noise.shape[0] == spec.audio_latent_length`) and let the spec supply the value. Add `157` as a **provisional** expectation clearly marked as pending checkpoint verification.

Note the likely relationship: `audio.json` carries `temporal_rope_scaling_factor: 0.19676`, which upstream presumably uses to derive the audio latent length from the video one. The repo uses this key **nowhere**. That is the missing link that would let `157` be computed from the spec instead of hardcoded — and it is the natural way to remove the guesswork.

### 1.3 `a_noise` channel count contradicts the config — a third channel bug

Not mentioned anywhere in the revision:

```
config audio in_dim = 20, out_dim = 20
engine allocates a_noise = (157, 12)      # fusion_engine/engine.py:183
```

**12 ≠ 20.** The revision's Gate 4 lists *"correct 48/20 channels"* as a fix, which covers this — but its §11 success criteria only spell out the **video** invariant (`[48, 81, 45, 45]`) and treat audio as merely "157". The audio channel mismatch is a distinct defect from the audio sequence length and deserves its own explicit criterion, or it will be missed.

Also note `audio.json` carries `temporal_rope_scaling_factor: 0.19676`, which upstream presumably uses to *derive* the audio latent length from the video one. The repo uses it nowhere. That is the missing link that would let `157` be computed rather than hardcoded.

### 1.4 Sequencing error: "restore keyframe handoff" cannot be done in step 1

The roadmap puts, under step 1 **REPAIR MERGE DAMAGE**:

```text
├── restore ImageEngine
├── restore keyframe handoff
```

The keyframe handoff is **not restorable as code.** Verified:

| | `e1fbbbb` VideoEngine | current VideoEngine |
|---|---|---|
| backend | diffusers `self.pipe` + `self.pipe.unet` | raw `WanModel` + `Wan2_2_VAE` |
| base | SD1.5 + AnimateDiff | Wan2.2 DiT |
| keyframe latent path | `self.pipe.unet.config.in_channels` | no `unet`, no `pipe` |

The old `keyframe_latent` was mixed into **diffusers SD1.5 latents** and validated against `self.pipe.unet.config.in_channels`. The current engine has neither. Transplanting that code would be meaningless.

**What is restorable is the *behavioural contract*, and it is restorable now** — one line, independent of any engine:

```python
# generation_manager.py:221  (currently)
image_path=scene.keyframe_path or conditioning_image_path
# restore the second fallback
image_path=scene.keyframe_path or conditioning_image_path or scene_state.reference_frame_path
```

`reference_frame_path` is already populated by the image path at `generation_manager.py:150`. So step 1 should say *"restore the `reference_frame_path` fallback in the image→video handoff"*, not *"restore keyframe handoff"* — and the latent-seeding mechanism itself belongs in the Ovi I2V step (Gate 3 / K), built against the Wan VAE, not ported from AnimateDiff.

Also note `IdentityLatentEncoder.encode()` is coupled to a **diffusers** pipeline (`pipe.vae`, `pipe.image_processor`, `/8` spatial). It is **not** reusable for Wan latents (`/16`, `z_dim=48`). It will need a Wan-native encoder — another reason this cannot be a step-1 restore.

---

## 2. Ordering observations (not errors)

### 2.1 Provider-rename repair is correctly *not* blocking

Roadmap step 1 lists "repair provider rename corruption." Verified this is **latent, not blocking**:

- `settings.llm.enabled = False` by default
- `generation_manager.py` imports all `llm` symbols under `TYPE_CHECKING` only
- module-level runtime imports are just `__future__`, `typing`, `core.logging.logger`

So the corruption does **not** block Gate 1 (prompt/image). Keeping it in step 1 is fine as cleanup, but it should not be treated as a gate blocker — labelling it as one risks delaying the image path for no gain.

### 2.2 SDXL downgrade is logged but not recorded — supports §12

The revision's "no silent fallback" policy is well-founded. Verified precisely: `model_registry.get()` reassigns `model_id = _SD15_MODEL_ID` and emits `LOG.warning`, but **nothing persists the substitution** — no result metadata, no metric. `core/metrics.py` *has* a `downgraded: bool` field, but it is never set from the registry path. So the policy has a natural home to extend; only the wiring is missing.

### 2.3 Double-ownership confirmed for §6

`fusion_engine/engine.py:128` assigns `self.bundle = self.registry.get(...)`, then lines 144-147 unpack into four locals. The engine owns the bundle *and* the registry holds a reference. `registry.unload()` nulls the registry side only. This confirms the ownership-boundary argument; the `ModelSession` design is the right fix.

---

## 3. What the revision gets right (endorsed unchanged)

- **The confidence split** (architecture ~0.95 / implementation ~0.80 / production ~0). This is the correct framing and I endorse it. The gating claim — *"we still have no evidence the vendored stack survives checkpoint load → denoise → decode"* — is exactly right.
- **P0-A** `output_dir` type error — confirmed, `str / str` → `TypeError`, on the success path.
- **P0-B two-lock analysis** — confirmed. `additional_emb_dim` assert *and* unusable keyframe. Both independently prevent I2V.
- **P0-C ownership nulling** — confirmed, and correctly identified as the crux rather than `empty_cache()`.
- **P0-D variants as first-class contract** — confirmed; `model_name`/`fp8`/`sample_steps` all absent from the repo YAML, and nothing reads it.
- **39,204 as the target sequence length** — the *diagnosis* (`81` is a unit error) is right and worth keeping, but the numeric target is itself unverified; see §1.1 and §5.
- **Section 5** (treat `model_loading_utils.py` as the reference; prohibit a parallel loader) — correct and the highest-ROI item.
- **Section 8** (remove standalone `AudioEngine`, retain Ovi's internal audio branch) — correct distinction.
- **Section 9** environment gate — correct, and sm_120 is a real blocker given the Jan-2024 pins.
- **Gate ordering** G0→G3 before any expensive generation — correct instinct.

---

## 4. Corrected starting point

The revision's roadmap is usable **after four edits**. Nothing else needs to change.

| # | Fix |
|---|---|
| 1 | Replace the `F*H*W//(ph*pw)` invariant with `derive_seq_len()` (§1.1) |
| 2 | Demote `157` from asserted constant to provisional value pending checkpoint verification (§1.2) |
| 3 | Add audio **channel** count (20, not 12) as an explicit Gate-4 + success criterion (§1.3) |
| 4 | Rewrite roadmap step 1's "restore keyframe handoff" → "restore the `reference_frame_path` fallback"; move latent seeding to the Ovi I2V step (§1.4) |

### Corrected first milestone (Gate 0 → Gate 1)

```
1  restore ImageEngine from e1fbbbb            (restore, do not rewrite)
2  restore reference_frame_path fallback       (1-line behavioural fix, engine-agnostic)
3  repair model_loading_utils imports+paths    (upstream reference, not a new loader)
4  make inference_fusion.yaml authoritative     (+ model_name / fp8 / sample_steps)
5  real tokenizer budgeting
6  eliminate global prompt contamination
7  canonical CharacterAsset + keyframe
8  Level-2 contract tests using derive_seq_len()   <- gate that catches §1.1
9  only then: model-contract fixes (H, I, J, K)
```

Note step 2 is now a one-liner and unblocks the image→video contract *before* any Ovi work — which is better sequencing than either the original plan or this revision.

---

## 5. Limitations

- **No forward pass was run.** `torch` is absent and no weights are mounted (`/kaggle` does not exist). The `derive_seq_len` result is derived from `wan/model.py:624` (`Conv3d(kernel=patch_size, stride=patch_size)`) plus the vendored VAE's `patchify(x, patch_size=2)` at `vae2_2.py:785`, not from executing the model.
- **VAE temporal length: the `39,204` target is likely wrong, and this is the largest open unknown.** Re-verified in this pass:
  - `vae2_2.py:896` — `Wan2_2_VAE(temperal_downsample=[False, True, True])` → temporal downsample active at 2 of 3 stages → **4x temporal compression**.
  - `vae2_2.py:1015-1023` — `Wan2_2_VAE` passes `temperal_downsample` straight through to `_video_vae`, so the 4x is not overridden.
  - `vae2_2.py:783-797` — `encode()` does `iter_ = 1 + (t-1)//4`, feeding 1 frame then 4-frame chunks, consistent with 4x temporal compression.

  If temporal is 4:1, the latent for 81 frames is **not** `[48, 81, 45, 45]`. It is closer to `[48, 21, 45, 45]` (Wan's VAE keeps the first frame at full rate: `latent_t = 1 + (iter_-1)`, with `iter_ = 1 + (81-1)//4 = 21`). That gives:

  | assumption | latent | patch grid | `vid_seq_len` |
  |---|---|---|---|
  | engine's (latent_t = 81) | `[48, 81, 45, 45]` | `(81, 22, 22)` | **39,204** |
  | VAE 4:1 (latent_t = 21) | `[48, 21, 45, 45]` | `(21, 22, 22)` | **10,164** |

  Both the revision and my earlier review assumed latent_t = 81. **The spatial 45 is solidly derived** (`patchify(2)` at `vae2_2.py:785` × 3 downsamples = 16x). **The temporal 81 is not** — it was inherited from the engine's own hardcoded assumption, which is exactly the kind of unverified premise this whole exercise is meant to eliminate.

  **Consequence:** do not freeze `39,204` — or `10,164` — into a test. Assert the *invariant* (`vid_seq_len == derive_seq_len(latent.shape[1], latent.shape[2], latent.shape[3], patch)`) and let the VAE's actual output define the value. Settle the number empirically at Gate 0 by encoding a real clip and printing the latent shape.
- Audio `157` and channel count `20` (vs engine's `12`) remain unvalidated against any checkpoint, as noted.