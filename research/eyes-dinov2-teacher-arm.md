# B2-DINOv2T reconnaissance: what re-enabling the RT-DETRv4 teacher costs

**Superseded in part by the run itself.** This page was written before the arm ran. The
arm ran on 2026-09-06 to checkpoint `9ecb4627` and scored AP 0.50:0.95 of 0.8345 on the
person_v2 dev-val against the pre-declared bar of 0.8368, so it was not promoted
(final_models/eyes/README.md). Teacher repository pin
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`, teacher weights sha256
`0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`, load-clean assertion
passed, peak video memory 23,758 MiB against the 30,500 MiB gate. The source findings
below still hold against the pinned upstream tree; the recommendations and the open items
are recorded with their outcomes.

Dated 2026-09-05, processor only; nothing had been run, trained or downloaded when
it was written. The subject is research/eyes-instrument-protocol.md I3 item 3, the gated
DINOv2-B/14 foundation-feature distillation arm.

**Method.** `git clone https://github.com/RT-DETRs/RT-DETRv4` checked out at the
PINS.yaml pin `55fefaaed7efe2a5f72d0a18fd4e05965e35c292`
(third_party/PINS.yaml, `repos.rtdetrv4.pin`), read on the Mac. `third_party/` is a tracked directory
in this repository and a clone directory placed
under it would **not** be covered by any .gitignore rule, so
the clone was placed outside the repository, in a scratch directory, and only
this note comes back. Reproduce with the clone command above; every `RTV4/` path
below names a file in that tree at that SHA, as do the shorter `configs/`, `engine/`
and `tools/` paths and the bare file names, all of which are given relative to that
clone root and none of which is a path in this repository. Each reference names the
element inside the file rather than a line number. DEIM `09d35d53` was cloned and read
the same way where it is cited.

Nothing here is a measurement. Two claims are labelled as estimates; everything else is
source. One claim, the weight file's URL and digest, was labelled as unverified from this
machine when the page was written and was confirmed at first download; it now carries that
confirmation.

---

## Headline: three findings that change the plan

1. **The student's distillation tap is stride-32, not stride-16.** The protocol's
   statement that a 560x560 teacher input gives a 40x40 grid matching the student's
   stride-16 does not describe what the upstream tree does. `use_encoder_idx: [2]` over
   `feat_strides: [8, 16, 32]` (both keys in RTV4/configs/base/dfine_hgnetv2.yml, never
   overridden anywhere in the include chain, and confirmed in the B1 run's own
   dumped config, outputs/pod_backup/train_rtv4_s0_final.log line 4, a run artefact under
   outputs/ that is not in version control) makes
   `encoder_idx_for_distillation = 2` (RTV4/engine/rtv4/hybrid_encoder.py), so
   the projected feature is the **stride-32** map: 20x20 at 640 input. 560x560
   still runs, because the criterion bilinearly resizes the teacher down to the
   student grid (the spatial-resize step of RTV4/engine/rtv4/rtv4_criterion.py), but
   the extra teacher resolution is then averaged away. The aligned, cheaper setting is
   `target_downsample_factor: 32`, which resizes 640 to 280 and yields an exact 20x20
   with **no** resize at all. See the teacher input-size section below for the
   arithmetic.
2. **There is no in-repo micro-batching of the teacher.** The teacher is called
   once on the whole batch, `teacher_model(samples)` under `no_grad`
   (the training step of RTV4/engine/solver/det_engine.py), and it runs outside the
   autocast block that opens immediately after it, that is in **fp32**. The
   micro-batched no-grad teacher the protocol names is a patch that would have to be
   written, not a config key.
3. **The in-repo DINOv2 loader wants a local `facebookresearch/dinov2` clone plus a
   raw `.pth` state dict, not `transformers`, and not `facebook/dinov2-base` as
   published.** `transformers` is listed in RTV4/requirements.txt but is imported by
   **zero** files at this pin (`grep -rn transformers --include=*.py` returns no hits).
   Details and the consequence for the PINS entry are in the loader section below.

---

## 1. Where the teacher gate lives

The gate is the `teacher_model` property of `engine/core/yaml_config.py`.

| what | where |
|---|---|
| the gate itself | the `teacher_model` property of `RTV4/engine/core/yaml_config.py` |
| the gating condition | `if self._teacher_model is None and 'teacher_model' in self.yaml_cfg` |
| type dispatch | the property pops `type`, then dispatches on `"DINOv3TeacherModel"` and on `"DINOv2TeacherModel"`, and **raises** on anything else |
| where the built teacher is attached | `RTV4/engine/solver/_solver.py`, `self.teacher_model = to(cfg.teacher_model, device)` |
| where it is consumed | `RTV4/engine/solver/det_solver.py` passes it into `train_one_epoch`; `RTV4/engine/solver/det_engine.py` calls it |
| where it is kept out of the checkpoint | `RTV4/engine/solver/det_solver.py`, where `state_dict()` skips the key `'teacher_model'` |
| the backing field | `RTV4/engine/core/_config.py` |

The gate is **presence-only**. There is no `enabled: false`; the only way to have no
teacher is for the key to be absent from the whole merged config, which is exactly
what the fork here does and says it does (third_party/configs/rtv4_s_person_v2_noteacher.yml, provenance-card header).
`train.py` routes `--test-only` to `solver.val()`, which never touches the
teacher, so **an evaluation of a B2 checkpoint needs no teacher and no DINOv2 weights**.

> **Latent upstream bug, cosmetic here.** The type dispatch pops `'type'` out of the
> config dict. The DINOv3 branch puts it back on success; the **DINOv2 branch does
> not**, because it restores the key only on the exception path. So after a successful
> DINOv2 build, `yaml_cfg['teacher_model']` has lost its `type` key. Harmless in this
> flow (`train.py` prints the config before the solver is constructed, and yaml_cfg
> is not serialized into the checkpoint), but it means a provenance dump taken
> after setup would under-report the teacher. To keep the type in the run record,
> read it from the local config file, not from `cfg.yaml_cfg`.

## 2. Config keys that enable distillation: all three are required

Distillation is off unless all three of these are true. B1 turned off all three.

| # | key | enabling value | where it is read | B1's value |
|---|---|---|---|---|
| 1 | `teacher_model:` block | present, `type` one of the two known strings | the gate and the type dispatch in `yaml_config.py` | **absent** |
| 2 | `HybridEncoder.distill_teacher_dim` | `> 0`, and **equal to the teacher's embed dim** | the constructor parameter and the projector construction in `hybrid_encoder.py` | **0** |
| 3 | `RTv4Criterion.losses` | must contain `'distill'` | the loss map in `rtv4_criterion.py` plus its distill dispatch | `['mal','boxes','local']` |

Optional fourth: `RTv4Criterion.distill_adaptive_params`, a constructor parameter of
`rtv4_criterion.py` that defaults to None. It is an epoch-wise re-weighting driven by
encoder gradient share, implemented in the adaptive re-weighting block of
`det_solver.py`. With `enabled: True, rho, delta, default_weight` it
retargets `loss_distill` each epoch so the encoder's gradient share sits in
`rho` plus or minus `delta`, clamping the change to a factor inside [0.1, 10] per epoch and
resetting to `default_weight` once the no-aug phase starts.

`distill_teacher_dim: 0` does more than skip the loss: the projector is never
**built**, because its construction in `hybrid_encoder.py` is guarded on that value,
so B1's checkpoint has no projector tensors
at all, confirmed on disk with 794 tensors and no `encoder.feature_projector.*` key
(`outputs/pod_backup/b1_best_stg2_345c30a5.pth`, likewise a run artefact not in version
control, read 2026-09-05).

The upstream S recipe's own block, copied exactly from its teacher and distillation
keys (RTV4/configs/rtv4/rtv4_hgnetv2_s_coco.yml):

```yaml
teacher_model:
  type: "DINOv3TeacherModel"
  dinov3_repo_path: dinov3/
  dinov3_weights_path: pretrain/dinov3_vitb16_pretrain_lvd1689m.pth
  patch_size: 16
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
HybridEncoder:
  distill_teacher_dim: 768
RTv4Criterion:
  weight_dict:
    loss_distill: 5
  distill_adaptive_params: {enabled: True, rho: 11, delta: 1, default_weight: 20}
```

## 3. Which projector and level the S model uses

- **Level:** `use_encoder_idx[-1]` = `2`, so `feat_strides[2]` is **stride 32**
  (the `encoder_idx_for_distillation` assignment in `hybrid_encoder.py`; strides in
  `configs/base/dfine_hgnetv2.yml`). At the
  640x640 train and eval size this is a **20x20** map. The comment beside that
  assignment describes it as the target AIFI output F5, which is
  consistent: the AIFI transformer runs on that same S5 level, and the projector is
  applied to the encoder's output, after the transformer, not to the raw
  backbone feature.
- **Projector:** a single `nn.Linear(hidden_dim, distill_teacher_dim)` with **no
  activation**, because the GELU is commented out inside the projector's
  `nn.Sequential` in `hybrid_encoder.py`. For the S config `hidden_dim` is 256 (the
  `hidden_dim` key of `configs/dfine/dfine_hgnetv2_s_coco.yml`), so
  with a 768-dim teacher it is a linear map from 256 to 768: 256x768 + 768 =
  **197,376 new parameters** on a 10.18 M-param model (+1.9 %).
- **Applied only in training:** `if self.training and self.feature_projector is not
  None and enc_ind == self.encoder_idx_for_distillation`, the condition in the
  encoder's forward. The encoder returns a 2-tuple only in that case, which
  `RTv4.forward` unpacks
  (`RTV4/engine/rtv4/rtv4.py`). At eval and at export the projector is dead
  code, so **the ONNX graph is unchanged by B2**, and the export patch and the
  acceptance-parity harness need no work.
- **No GAM anywhere.** `grep -rni 'gam'` over the whole tree at this pin matches only
  the words Game board in two commented-out Objects365 class lists
  (`engine/solver/_solver.py`). There is no gated or global attention module in
  that tree. The line recording a removed GAM block in the provenance card of
  third_party/configs/rtv4_s_person_v2_noteacher.yml therefore had
  **no referent at this pin**. That line has since been deleted and the correction
  recorded in the same provenance card. It was a documentation error only; the config
  itself was always correct (it removes
  the `teacher_model` block, sets `distill_teacher_dim: 0` and drops `'distill'`,
  all three of which are real).

## 4. The distill loss form and its weight

`loss_distillation` in `RTV4/engine/rtv4/rtv4_criterion.py`. Reading it out:

1. It pulls `student_distill_output` and `teacher_encoder_output` from the model
   output dict (put there by `rtv4.py`). If either is missing the loss is a
   literal `0.0`, **and that zero is hardcoded onto `torch.device('cuda')`**
   in the missing-output branch, which is why the config header records that a CPU
   smoke of the distill-enabled path is impossible.
2. The channel check: **channel dims must match exactly**, else `ValueError`. This is
   the hard constraint that makes `distill_teacher_dim == teacher.embed_dim` mandatory.
3. The spatial-resize step: if the spatial dims differ, the **teacher** is
   `F.interpolate`d (bilinear, `align_corners=False`) down or up to the **student's**
   H by W. Never the
   reverse. This is the mechanism that makes a 40x40 teacher grid legal against a
   20x20 student, and equally the mechanism that discards the extra resolution.
4. The normalize-and-compare block flattens both to `[B, HW, C]`, L2-normalizes both
   along `C`, then applies `F.cosine_similarity(..., dim=-1)` and

   ```
   loss_distill = (1 - cos_sim).mean()
   ```

   that is, **mean per-token cosine distance**, averaged over batch and all spatial
   positions, in `[0, 2]`. (The explicit `F.normalize` in that block is redundant,
   because `cosine_similarity` normalizes internally, but it is harmless.) There is no
   temperature, no logit matching, no masking: every spatial token counts equally,
   background included.

**Weight.** `weight_dict['loss_distill']`, applied in the distill dispatch of
`rtv4_criterion.py` via `_get_distillation_weight_for_epoch()`, which returns the fixed
weight when `distill_adaptive_params` is absent. Note that dispatch is a **special
case**: unlike every other loss it is multiplied by the
weight inside the `if`, and it is skipped when the value is exactly 0. Values in
play:

| where | value |
|---|---|
| `configs/base/rtv4.yml` | `loss_distill: 10.0`, with the comment beside it recording it as 5 times the reported setting due to zero aux loss |
| `configs/rtv4/rtv4_hgnetv2_s_coco.yml` (S and M) | **5** |
| `configs/rtv4/rtv4_hgnetv2_l_coco.yml` and its `_x_` sibling | 15 / 20 |
| protocol's declared value | **5**, matching the published S recipe exactly |

`configs/base/rtv4.yml` also puts `'distill'` in its default `losses` list, so
re-enabling B2 means **removing** this fork's `losses:` override rather than adding
anything. The distill loss is main-branch only: the aux and dn loops in
`rtv4_criterion.py` iterate `self.losses` too, but `loss_distillation` reads keys
that exist only on the top-level output dict, so aux copies return the 0.0 branch,
which is what the base config's zero-aux-loss comment is about.

## 5. Loader situation: the sharp edge

Class `DINOv2TeacherModel` in `engine/rtv4/dinov2_teacher.py`. Its loader is three
calls:

```python
self.model = torch.hub.load(dinov2_repo_path, model_name, source='local')   # hub load
state_dict = torch.load(weights_path, map_location='cpu')                    # weights
msg = self.model.load_state_dict(state_dict, strict=False)                   # not strict
```

- **A DINOv2 loader exists in-repo**, so none has to be written, and `transformers`
  is not needed. `transformers` is listed in RTV4/requirements.txt but imported
  nowhere at this pin.
- **`source='local'` means a local clone of `facebookresearch/dinov2` must be on
  disk** at `dinov2_repo_path`, because `torch.hub.load` needs that repo's
  `hubconf.py`. Same pattern as the hub load in the DINOv3 teacher
  (`dinov3_teacher.py`), which is why the upstream S config says
  `dinov3_repo_path: dinov3/`. **This is a second
  third-party repo to pin in PINS.yaml, with its own licence**, not just a weights
  download.
- **`strict=False` on that third call is the trap.** If the state dict's keys do not
  match the hub model, the load succeeds silently and the teacher runs on **random
  weights**;
  the loader logs the missing and unexpected lists at INFO level, and the trainer's
  logging is not configured to show them. The one Hugging Face artefact the protocol
  names,
  `facebook/dinov2-base`, is the **`transformers` port** (`Dinov2Model`), whose
  parameter names (`embeddings.patch_embeddings.projection.weight`, `encoder.layer.N.…`)
  are not the original repo's (`patch_embed.proj.weight`, `blocks.N.…`). Handing that
  file to the last two calls above would load **zero** tensors and raise nothing.
  So **the download must be the original `facebookresearch/dinov2` weight file
  (`dinov2_vitb14_pretrain.pth` or the `_reg4` variant), not the Hugging Face
  `transformers` checkpoint**, or a conversion step must be written and tested. Both
  are Apache-2.0.
  [Confirmed at first download, 2026-09-06: the fetched file was
  `dinov2_vitb14_pretrain.pth` from the facebookresearch release host, its sha256 is
  `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`, and the load-clean
  assertion passed, so it is the original state dict and not the Hugging Face port.
  third_party/PINS.yaml, `checkpoints.dinov2_vitb14`, holds the URL, the digest and the
  verification note.]
- **A hard assertion protects against the silent-random-weights case anyway**, one
  step later: `dinov2_teacher.py` reads `self.model.embed_dim`, and if that is not
  768 the criterion's channel check in `rtv4_criterion.py` raises on the first
  batch. So a wrong model size fails loudly; only wrong key names at the right
  size fail silently. Mitigation is one line in the launch script: assert that
  `load_state_dict`'s `missing_keys` is empty.
- Defaults to be overridden: the constructor default `model_name='dinov2_vitl14_reg'`
  (ViT-**L**, embed_dim 1024); for DINOv2-B/14 pass `model_name: dinov2_vitb14` (or
  `dinov2_vitb14_reg`) **and** `distill_teacher_dim: 768`. The constructor default
  `patch_size: 14` is already right for every DINOv2 variant.
- The teacher is frozen and eval-mode at construction, and excluded from
  the checkpoint (`state_dict()` in `det_solver.py` skips it), so no DINOv2 tensor is
  ever serialized into a B2 weight file. The C-1 counsel question (third_party/PINS.yaml, `repos.rtdetrv4.license_notes`) is unchanged in
  shape by B2: the run would still train with a foundation teacher, now an
  Apache-2.0 one with no click-wrap, instead of inheriting DINOv3 lineage silently
  through the init.

## 6. Teacher input-size handling: the arithmetic

The forward pass of `dinov2_teacher.py`. The teacher **derives its own input size from
the student's**; there is no 560 setting. Per forward:

```
target_H_out    = H_in / target_downsample_factor      # constructor default tdf = 32
target_H_vit_in = round(target_H_out) * patch_size     # constructor default 14
processed       = TF.resize(normalized, [H_vit, W_vit], BICUBIC, antialias=True)
patch_tokens    = model.forward_features(processed)['x_norm_patchtokens']
                  -> reshaped to [B, C, H_in/tdf, W_in/tdf]
```

with a hard `ValueError` if the token count does not factor as expected.
So for the 640x640 student:

| `target_downsample_factor` | teacher input | teacher grid | student grid (stride-32) | criterion does |
|---|---|---|---|---|
| **32** (default) | 280x280 | 20x20 | 20x20 | nothing, an exact match |
| **16** (the 560 plan) | **560x560** | **40x40** | 20x20 | bilinear **downsample** from 40 to 20 (the spatial-resize step of `rtv4_criterion.py`) |

The 560x560 figure is reachable by setting `target_downsample_factor: 16`, but the
premise attached to it is not: the 40x40 grid does **not** meet a stride-16 student
map, because the student never taps stride-16. It meets a 20x20 map through an
averaging resize.

Making the student actually tap stride-16 would mean `use_encoder_idx: [1]`, which
moves the AIFI transformer itself from S5 to S4. That is an architecture change, not a
config tweak: 4 times the AIFI token count (1600 against 400), a different meaning for the
COCO-pretrained `encoder.encoder.0.*` tensors (they would still load, with identical
shapes, while being semantically wrong), and a violation of the minimal-delta law the
bake-off runs under. **Recommendation: it is not taken.**

**Recommendation for B2 as scoped, and what was taken:** `target_downsample_factor: 32`
(280x280, exact grid match, roughly 4 times cheaper teacher) was recommended as the
primary arm, and it is the value the run used
(third_party/configs/rtv4_s_person_v2_dinov2t.yml). The 560x560 variant would have been a
second pre-declared arm whose stated hypothesis is that a higher-resolution teacher,
average-pooled to the student grid, is a better target than a low-resolution one. That is
a real question, but a different one from what the protocol states, and it doubles the B2
spend; it was not run.

For comparison, the DINOv3 teacher the released weights actually used gets to the same
place differently: a fixed `AvgPool2d(2,2)` in `dinov3_teacher.py` takes 640 to
320, and at `patch_size: 16` that is 20x20, the same stride-32 alignment, with no
dynamic resize at all.

## 7. Checkpoint keys the init would need

Initializing B2 from `rtv4_s_coco.pth` (`-t`, the same path B1 used):

- **New tensors, in the checkpoint of neither init nor B1:**
  `encoder.feature_projector.0.weight` `[768, 256]` and
  `encoder.feature_projector.0.bias` `[768]` (from
  `nn.Sequential(nn.Linear(256, 768))` in `hybrid_encoder.py`).
- **They will appear in the startup line as `missed`, and that is correct.**
  `BaseSolver.load_tuning_state` calls `_matched_state` (in `_solver.py`), which
  puts any key absent from the pretrain dict into `missed` and loads the rest with
  `strict=False`. So the B2 launch check becomes: `missed` must be exactly
  the class heads B1 saw, `decoder.denoising_class_embed.weight` (with `unmatched`
  = `enc_score_head` plus `dec_score_head.0-2`, per
  research/eyes-rtdetrv4-training-record.md, section "Startup check"), **plus exactly the two
  `encoder.feature_projector.0.*` entries, and nothing else**. Anything else in that
  list is a stop.
- The projector is therefore **randomly initialized** and has to catch up while it is
  already being used as the loss's student side. This is upstream's own behaviour
  (their COCO runs start the same way), but it is the mechanical reason a distill
  weight of 5 with a `warmup_iter` of 800 is not obviously the right pairing for a
  48-epoch schedule, and it is the most likely source of an early-epoch regression.
- **B2's own checkpoints will carry those two extra tensors (796, not 794).** Anything
  that consumes B1 and B2 weights together must tolerate that:
  - `scripts/detector/soup_v2.py` will refuse a mix of B1 and B2 on the key-set check,
    which is correct behaviour and worth knowing in advance.
  - `tools/deployment/export_onnx.py` calls `cfg.model.load_state_dict(state)` with
    **default `strict=True`**, so exporting a B2 checkpoint through a
    `distill_teacher_dim: 0` config **fails**. Export B2 with a config that keeps
    `distill_teacher_dim: 768` (the projector is skipped at eval anyway, by the
    training-only condition in `hybrid_encoder.py`, so the graph is identical), or
    strip the two keys first.
    This is the mirror of the consequence the noteacher config header records, that
    with `distill_teacher_dim: 0` the feature_projector is never built and the saved
    checkpoint strict-loads at export time with that same config.
- No teacher tensor is ever in the file (`state_dict()` in `det_solver.py` skips the
  `'teacher_model'` key), so a B2 checkpoint is the same size as a B1 one plus about
  197 k floats (about 0.8 MB).

## 8. Video memory: what the 30,500 MiB gate is really gating

B1 seed 0 peaked at **26,521 MiB** with no teacher (research/eyes-instrument-protocol.md, section I3 item 1),
leaving about 4 GB of headroom against the declared gate. Against that:

- the teacher runs on the **full batch of 32** in one call (the training step of
  `det_engine.py`), and no micro-batching exists to soften the peak;
- it runs **outside** the autocast block that opens after it, that is in **fp32**;
- its weights are about 86 M params, roughly **344 MiB** resident, frozen;
- activations are transient under `no_grad`, and the trainer's torch 2.11.0 scaled
  dot-product attention does not materialize the attention matrix, so the peak is
  dominated by per-layer token
  activations: 32x1605x768x4 B approximately **158 MiB** per stored tensor at 560x560, against
  32x405x768x4 B approximately **40 MiB** at 280x280, with a handful live at once.

[Estimate, not measured: total teacher overhead roughly 1 to 2 GiB at 280x280 and
roughly 3 to 5 GiB at 560x560, the latter plausibly over the 30,500 MiB gate.]
[Estimate: the 4 times cheaper 280x280 arm is the one likely to clear a 15-minute dry run.]
The gate is the right instrument precisely because these are estimates, but the
ordering is not in doubt, and it is another reason to make
`target_downsample_factor: 32` the primary arm.

If the gate fails, the levers in order of least damage: `target_downsample_factor:
32` (if 16 was tried first), then wrapping the teacher call in `torch.autocast` or
calling `.half()` on the frozen teacher, then micro-batching the teacher call (a real
patch: `torch.cat([teacher(chunk) for chunk in samples.split(k)])` at the teacher
call in `det_engine.py`). Reducing `total_batch_size` is **not** on that list: it
changes the recipe and breaks comparability with B1's seeds.

---

## PINS.yaml entries for DINOv2-B/14

Two entries were needed, because the loader in `dinov2_teacher.py` needs a repo **and** a
weight file, and both keep PINS' sha256-at-first-download law. They are live in
third_party/PINS.yaml, which is the record:

- `repos.dinov2` carries the role note (teacher architecture only, training time only, no
  `src/vividly` import, no DINOv2 tensor ever serialized into a checkpoint produced here),
  the URL `https://github.com/facebookresearch/dinov2`, the pin
  `7764ea0f912e53c92e82eb78a2a1631e92725fc8` recorded at first clone, `license:
  Apache-2.0`, and the licence note that the original 2023-04 release was CC-BY-NC-4.0 and
  was relicensed Apache-2.0 in 2023-06, so the pin must be at or after that relicense and
  the LICENSE file in the checked-out tree must be read before any training run.
- `checkpoints.dinov2_vitb14` carries the model line (86M params, embed_dim 768),
  `teacher_feature_dim: 768` with the note that `HybridEncoder.distill_teacher_dim` must
  equal it or the criterion's channel check raises, the weight URL with its verification
  status, the LVD-142M lineage, `license: Apache-2.0`, the sha256 recorded at first
  download, the load notes on the `strict=False` trap and the empty-`missing_keys`
  assertion, and a `config_keys` block giving `model_name: dinov2_vitb14`,
  `patch_size: 14` and `target_downsample_factor: 32`, and recording that the two
  remaining required keys, `dinov2_repo_path` and `weights_path`, are deliberately not
  spelled there because their literal values are pod locations, written in
  third_party/configs/rtv4_s_person_v2_dinov2t.yml.

The same keys as written for the run are in
third_party/configs/rtv4_s_person_v2_dinov2t.yml.

---

## Open items raised here, and how each was settled

1. **The teacher input size was decided before the run was launched.**
   `target_downsample_factor: 32` (exact grid match, roughly 4 times cheaper) was taken
   over `16` (the 560x560 the protocol names, with the teacher then bilinearly downsampled
   from 40 to 20), and third_party/configs/rtv4_s_person_v2_dinov2t.yml records it.
   Amending research/eyes-instrument-protocol.md was an owner decision, because the
   protocol's stated reason for 560 does not hold.
2. **`facebookresearch/dinov2` is pinned and its LICENSE text is checked in the clone.**
   third_party/PINS.yaml records the pin `7764ea0f912e53c92e82eb78a2a1631e92725fc8` and the
   licence gate that reads LICENSE at that tree before the run proceeds. House rule, and
   the pre-2023-06 tree was CC-BY-NC.
3. **The weight file is the original state dict, not the Hugging Face port.** The
   empty-`missing_keys` assertion is in the launch chain
   (`assert_teacher_loads_clean` in scripts/pod/eyes_chain.sh) and it passed on the run
   (final_models/eyes/README.md).
4. **The GAM block line is fixed.** The provenance card of
   third_party/configs/rtv4_s_person_v2_noteacher.yml now records that the line had no
   referent at this pin and has been deleted.
5. **`distill_adaptive_params` was left out.** The published S recipe enables it
   (`rho: 11, delta: 1, default_weight: 20`), but it makes `loss_distill` a moving target
   across epochs, which complicates any claim that the published recipe was run unchanged
   and makes a seed-to-seed comparison noisier. Fixed weight 5 was the simpler
   pre-declaration and is what third_party/configs/rtv4_s_person_v2_dinov2t.yml sets.
