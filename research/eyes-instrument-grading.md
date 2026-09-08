# Eyes B1: I1 precision instrument and I2 ONNX parity, graded on the Mac

Dated 2026-09-05. Mac processor only. The pod's GPU was not touched; all pod traffic
was read-only: `cat` of trainer config/source files, `sha256sum`, one `tar -cf -` of
20 dev-val JPEGs streamed to stdout, and the later pull of the 1,800 dev-val frames
recorded in deviation 3. Nothing was written on the pod.

**Verdicts.**

1. **I1 provenance: pass.** All four sidecars carry the expected checkpoint sha256,
   the expected dev-val annotation file, 540,000 rows over 1,800 images, and a
   `detections_sha256` that matches the file on disk today.
2. **I1 dumps reproduce their checkpoints exactly.** Re-graded on the Mac with
   pycocotools, the identity dump for `345c30a5` scores AP **0.8355529764** against
   the training log's `best_stat` of `0.8355529763877473`, a delta of exactly
   **0.0**. The `--compare-ap` watch item required by the protocol addendum (c) is
   therefore satisfied, after the fact and off the pod.
3. **I2 ONNX parity: pass.** On the sampled dev-val frames the exported ONNX and the
   checkpoint's dump agree on the detection count at every operating point, every
   ONNX box has a dump twin at IoU at or above 0.9964, and the median box disagreement is
   0.006 px. Every rank-order disagreement traces to an adjacent tie between two
   queries whose scores differ by about 1e-5.
4. **Processor ms/frame: 54.8 ms p50 in isolation, 76.3 ms/frame streaming** at 640x640,
   batch 1, onnxruntime 1.28.0 CPU EP, Apple M5 Pro. This is the first measured
   processor figure for this lane.
5. **The TTA and ensemble protocol row loses.** WBF at `fuse.py` defaults costs
   **-0.0423 AP** against the identity view it fuses. It stays what the protocol
   already called it: a protocol row, not a checkpoint, not promotable.
6. **MVOR, internal only: AP 0.1935** (95% clustered CI 0.1671 to 0.2325) for
   `345c30a5`. The bytes were on local disk so nothing was downloaded. The init
   `238a3f65` arm is blocked: no such checkpoint or ONNX exists on this Mac.
7. **A hazard worth more than any number here.** The dev-val frames in this Mac
   checkout's registry tree are a different JPEG encode from the pod's dev-val
   frames. See the section on the Mac frames tree below.
8. **Full-split ONNX AP parity: pass (resolved same day).** With the 1,800
   pod-byte frames pulled, the ONNX scores **0.8355776** against the trainer's
   0.8355530, a delta of +2.5e-5. The whole I2 chain is closed. See deviation 3.

Every number below is a **synthetic SmartSpaces dev-val** number, not deployment
evidence, except the MVOR section, which is internal only.

## Artefacts and hashes (POLICY.md, rule 10)

| artefact | sha256 |
|---|---|
| checkpoint B1 RT-DETRv4-S seed 0 | `345c30a539c2b6bb665302825f418322284b98f6cbcf55d58d84ffeb75b16e7e` |
| checkpoint DEIM-S seed 1 | `e98de401670d34d61d03100974eb72449607b319611ff454e8a865cd07b81147` |
| ONNX export of `345c30a5` | `4026204967aa01a7680f7ce0c6169463d654a519f8093383a10537321f56fd49` |
| GT `person_v2_devval.json` (Mac and pod, identical) | `bc9bedadbcf33cc7d06e5171cd08f091d6a8f1f974ba2e3e1083a34195dd89dc` |
| dump `dets_b1_rtv4s_seed0_identity.json` | `bed59aa2ed63a2800061a941698ab5d2a30032ca1430d6fa2605159619538d82` |
| dump `dets_deim_s_seed1_identity.json` | `3c4da71a9887c22342f8cce7b9adfb4c2d36ea3eeaa0f47324e5173bd044acbf` |
| dump `dets_b1_rtv4s_seed0_hflip.json` | `210e8f2a5493f80d444c9ad50b3c649429d5c7460420b2a54605da37e214d830` |
| dump `dets_deim_s_seed1_hflip.json` | `6dacecb4f96c3acd743c7d9e8c646f1a3e4c464ee707d35d46e27eecf899c14d` |
| MVOR GT `mvor_person_v2_coco.json` | `95295552f38649b1560e9a4503431ac3a238898e04036c5b2749c6c53da21be3` |
| MVOR release annotations `camma_mvor_2018.json` | `64f4270149ef1a738a881a764c4174264d051c2c2e4fd252f9c10fd5b377c0e0` |

---

## 1. Provenance check: pass

Read from the four `.sidecar.json` files beside the dumps.

| dump | checkpoint sha256 | config | ann_file | rows | imgs | hflip | weights |
|---|---|---|---|---:|---:|---|---|
| b1_rtv4s_seed0_identity | `345c30a5…b16e7e` | `RTV4/configs/rtv4/rtv4_s_person_v2_noteacher.yml` | `person_v2_devval.json` (`bc9bedad…dd89dc`) | 540,000 | 1,800 | false | `ema.module` |
| b1_rtv4s_seed0_hflip | `345c30a5…b16e7e` | same | same | 540,000 | 1,800 | true | `ema.module` |
| deim_s_seed1_identity | `e98de401…b81147` | `DEIM/configs/deim_dfine/deim_s_person_v2.yml` | same | 540,000 | 1,800 | false | `ema.module` |
| deim_s_seed1_hflip | `e98de401…b81147` | same | same | 540,000 | 1,800 | true | `ema.module` |

The `config` column names files inside the trainer repositories, not this repository:
`RTV4/` is the RT-DETRv4 tree at its `third_party/PINS.yaml` pin and `DEIM/` is the
DEIM tree at its pin. The vendored copies in this repository are
`third_party/configs/rtv4_s_person_v2_noteacher.yml` and
`third_party/configs/deim_s_person_v2.yml`. The `ann_file` column names the annotation
file each sidecar records under the pod layout described in TRAINING.md; the sha256 is
what carries the evidence, and it is the same on both machines.

Checks that passed:

- Both expected checkpoint hashes present, one per model, on both of that model's views.
- `mode: run` on all four (not `convert`), `score_floor: 0.0`, `limit: null`,
  `n_images_with_detections` 1,800 of 1,800.
- 540,000 = 1,800 by 300, which is what the postprocessor must emit: `num_queries`
  300 by `num_classes` 1, so `topk(300)` returns every query. Confirmed in the files:
  every image carries exactly 300 rows, `category_id` is 0 everywhere.
- `trainer_git_head` matches the PINS.yaml pin for each trainer:
  `55fefaaed7efe2a5f72d0a18fd4e05965e35c292` (RT-DETRv4) and
  `09d35d53d39ee3145a1e61e3a989b28b9468d1dd` (DEIM).
- Each sidecar's `detections_sha256` matches `shasum -a 256` of the file today: all
  four match. The dumps have not been touched since they were written on the pod.
- The GT the sidecars name is byte-identical to the Mac's copy: `sha256sum` on the
  pod returned `bc9bedad…dd89dc` for the pod's copy of `person_v2_devval.json`, and
  `shasum -a 256` on
  `data/train_ok/physicalai_smartspaces/annotations/person_v2_devval.json`
  returned the same. 1,800 images, 33,755 annotations, one category (`id: 0`,
  person), zero `iscrowd`, 30 cameras by 60 frames, all of Warehouse_005.

No dump was refused.

## 2. I1: precision, recall and FP-per-image at fixed score cuts

Grader: `scripts/eval/detector_precision_at_thr.py`, IoU 0.5, no maxDets cap, no
area ranges, camera-depth 2. Ground truth is the file hashed above. All rows carry
the short label `synthetic-devval`. Its long form appears in two shapes across the
four sidecars. `research/eyes-precision-345c30a5-identity.json` carries the grader
default: `synthetic SmartSpaces dev-val; not deployment evidence.` The other three
carry the extended string: `synthetic SmartSpaces dev-val; not deployment evidence
and not acute-event evidence.` Both say the same thing about these rows, which is
that they are synthetic dev-val numbers and not deployment evidence.

```
uv run python scripts/eval/detector_precision_at_thr.py \
  --gt data/train_ok/physicalai_smartspaces/annotations/person_v2_devval.json \
  --det outputs/eyes_b1/dumps/dets_b1_rtv4s_seed0_identity.json \
  --title "B1 RT-DETRv4-S seed0 (345c30a5) identity, dev-val" \
  --out-md outputs/eyes_b1/i1/precision_b1_rtv4s_seed0_identity.md \
  --out-json research/eyes-precision-345c30a5-identity.json
```

### Overall, all six graded surfaces

| checkpoint | view | score at or above | TP | FP | FN | precision | recall | F1 | FP/img |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `345c30a5` | identity | 0.3 | 31951 | 4718 | 1804 | 0.8713 | 0.9466 | 0.9074 | 2.621 |
| `345c30a5` | identity | **0.4** | 31413 | 2302 | 2342 | **0.9317** | **0.9306** | 0.9312 | **1.279** |
| `345c30a5` | identity | 0.5 | 30704 | 1059 | 3051 | 0.9667 | 0.9096 | 0.9373 | 0.588 |
| `e98de401` | identity | 0.3 | 31960 | 4709 | 1795 | 0.8716 | 0.9468 | 0.9076 | 2.616 |
| `e98de401` | identity | **0.4** | 31429 | 2311 | 2326 | **0.9315** | **0.9311** | 0.9313 | **1.284** |
| `e98de401` | identity | 0.5 | 30709 | 1041 | 3046 | 0.9672 | 0.9098 | 0.9376 | 0.578 |
| `345c30a5` | hflip | 0.3 | 31942 | 4618 | 1813 | 0.8737 | 0.9463 | 0.9085 | 2.566 |
| `345c30a5` | hflip | 0.4 | 31452 | 2277 | 2303 | 0.9325 | 0.9318 | 0.9321 | 1.265 |
| `345c30a5` | hflip | 0.5 | 30781 | 1016 | 2974 | 0.9680 | 0.9119 | 0.9391 | 0.564 |
| `e98de401` | hflip | 0.3 | 31972 | 4807 | 1783 | 0.8693 | 0.9472 | 0.9066 | 2.671 |
| `e98de401` | hflip | 0.4 | 31456 | 2355 | 2299 | 0.9303 | 0.9319 | 0.9311 | 1.308 |
| `e98de401` | hflip | 0.5 | 30716 | 964 | 3039 | 0.9696 | 0.9100 | 0.9388 | 0.536 |
| `345c30a5` | TTA-WBF | 0.3 | 30204 | 1692 | 3551 | 0.9470 | 0.8948 | 0.9201 | 0.940 |
| `345c30a5` | TTA-WBF | 0.4 | 26715 | 613 | 7040 | 0.9776 | 0.7914 | 0.8747 | 0.341 |
| `345c30a5` | TTA-WBF | 0.5 | 22720 | 197 | 11035 | 0.9914 | 0.6731 | 0.8018 | 0.109 |
| `e98de401` | TTA-WBF | 0.3 | 30021 | 1772 | 3734 | 0.9443 | 0.8894 | 0.9160 | 0.984 |
| `e98de401` | TTA-WBF | 0.4 | 26151 | 581 | 7604 | 0.9783 | 0.7747 | 0.8647 | 0.323 |
| `e98de401` | TTA-WBF | 0.5 | 22145 | 193 | 11610 | 0.9914 | 0.6561 | 0.7896 | 0.107 |

Per-camera tables (30 cameras by 3 thresholds per surface) are in the JSON files
listed at the bottom; the markdown renderings are under `outputs/eyes_b1/i1/`.

**Read of the demo point (score at or above 0.4, `345c30a5`).** 93.2% of drawn boxes are
real, 93.1% of people are drawn, and an average frame carries **1.28 phantom
boxes**. The two detectors are indistinguishable here: 0.9317 vs 0.9315 precision,
0.9306 vs 0.9311 recall, 1.279 vs 1.284 FP/img. Nothing in this instrument
separates B1 from the leading DEIM model, which is the same conclusion the AP bake-off
reached by a 0.0003 margin.

**Where the FPs live.** Per-camera at 0.4 for `345c30a5`, precision ranges from
0.8906 (Camera_0022, 3.717 FP/img, 1,976 GT boxes) to 0.9873 (Camera_0001, 0.100
FP/img, 467 GT). The worst three FP/img cameras are 0022 (3.717), 0004 (3.100) and
0026 (3.000); the worst three recalls are 0004 (0.8584), 0003 (0.8623) and 0010
(0.8681). Every one of them is a crowded view. The instrument's whole point is that
this is invisible in AP: the overall 1.28 FP/img hides a **55-fold spread** across
cameras, 0.067 (Camera_0007) to 3.717 (Camera_0022).

### COCO AP of every dump, re-graded on the Mac

Not part of I1's declared table, but it is the `--compare-ap` proof the protocol
addendum (c) required, and it costs nothing once the dumps are local. pycocotools
`COCOeval`, same GT.

| dump | AP .50:.95 | AP50 | AP75 | AP_s | AP_m | AP_l | AR100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `345c30a5` identity | **0.8355529764** | 0.9581 | 0.8826 | 0.1277 | 0.7224 | 0.9307 | 0.8596 |
| `e98de401` identity | 0.8341295847 | 0.9580 | 0.8817 | 0.1265 | 0.7207 | 0.9299 | 0.8585 |
| `345c30a5` hflip | 0.8360926449 | 0.9582 | 0.8825 | 0.1170 | 0.7241 | 0.9307 | 0.8602 |
| `e98de401` hflip | 0.8342329073 | 0.9583 | 0.8815 | 0.1298 | 0.7198 | 0.9302 | 0.8581 |
| `345c30a5` TTA-WBF | 0.7932833457 | 0.9342 | 0.8390 | 0.1119 | 0.6718 | 0.8936 | 0.8188 |
| `e98de401` TTA-WBF | 0.7914370291 | 0.9338 | 0.8379 | 0.1160 | 0.6653 | 0.8927 | 0.8178 |

The identity dump for `345c30a5` reproduces the training log's `best_stat`
(`epoch 47, coco_eval_bbox: 0.8355529763877473`,
`outputs/pod_backup/train_rtv4_s0_final.log`) to **every printed digit**;
the delta is exactly 0.0. `e98de401`'s 0.8341 likewise matches its known dev-val
AP. `run` mode did what its header claims: it re-walks the trainer's own eval loop
and persists the list the trainer discards, with no numerical drift at all.

### The hflip and TTA protocol row: pre-declared, and it loses

The protocol declares this as an optional same-session row: hflip-only dumps for the
pre-declared protocol row, at fuse.py defaults, iou_thr 0.55, with no grid search,
reported as an ensemble and TTA protocol row, not a checkpoint and not promotable.

Run with `src/vividly/eval/wbf.py` at `WBFParams()` exactly as declared,
`iou_thr 0.55`, `skip_box_thr 0.05`, `conf_type 'avg'`, all weights 1.0,
`ScoreCal()` identity per view, through a new thin driver,
`scripts/detector/fuse_tta_dumps.py`. No knob was tuned; the driver exposes none.

```
uv run python scripts/detector/fuse_tta_dumps.py \
  --gt data/train_ok/physicalai_smartspaces/annotations/person_v2_devval.json \
  --det identity=outputs/eyes_b1/dumps/dets_b1_rtv4s_seed0_identity.json \
  --det hflip=outputs/eyes_b1/dumps/dets_b1_rtv4s_seed0_hflip.json \
  --out outputs/eyes_b1/dumps/dets_b1_rtv4s_seed0_tta_wbf.json
```

Result: **-0.0423 AP** for `345c30a5` (0.7933 vs 0.8356) and **-0.0427** for
`e98de401` (0.7914 vs 0.8341). The single hflip view on its own is marginally better
than identity (+0.0005 / +0.0001); it is the fusion that destroys the number.

Two mechanisms, both properties of the declared defaults rather than of TTA:

- `conf_type='avg'` rescales each cluster's score by (weight of the contributing
  models) / (total weight). With two equally weighted views a box seen by only one
  view keeps **half** its score. AP is a score-ranking metric, so halving the score
  of every single-view box reshuffles the ranking and costs AP directly.
- The same rescaling means a fixed cut at 0.3 / 0.4 / 0.5 on fused scores is **not
  the same operating point** as that cut on a single view. That is why the TTA rows
  in the table above look like a different model: at 0.4 the fused recall falls to
  0.7914 while precision rises to 0.9776. Those rows are a different threshold in
  disguise, not a precision/recall improvement.

The protocol row is recorded and closed. It is not a checkpoint, it is not
promotable, and on this evidence it is not worth a second look at these defaults.
Anything further would be the grid search the protocol forbids.

**Not run:** the four-way cross-model ensemble (both checkpoints by both views). The
protocol's phrase, ensemble and TTA protocol, does not unambiguously pre-declare a
cross-architecture fusion, and inventing one would be an analysis beyond the
protocol.

---

## 3. I2: ONNX parity against the checkpoint

The identity dump is the checkpoint's reference output: `scripts/detector/dump_detections_v2.py run`
walks the trainer's eval loop with `solver.ema.module`, and
`RTV4/tools/deployment/export_onnx.py` loads the same `checkpoint['ema']['module']`
tensors into `cfg.model.deploy()` + `cfg.postprocessor.deploy()`. Same weights,
same postprocessor, so for one image the two must emit the same 300 boxes in the
same score order, up to fp32 drift.

Harness: `scripts/eval/onnx_parity_v2.py`, CPU EP only (POLICY.md, section 6: the
CoreML EP crashes on these graphs at the pinned onnxruntime version).

**Preprocessing was read out of the trainer, not guessed.** The reads below were
read-only against the pod's trainer tree; every `RTV4/` path names a file in that
tree, not in this repository:

- `RTV4/configs/base/dataloader.yml` `val_dataloader.dataset.transforms.ops` is exactly
  `[{Resize, size: [640, 640]}, {ConvertPILImage, dtype: float32, scale: True}]`,
  and nothing in the include chain
  (`rtv4_s_person_v2_noteacher.yml` to `dfine_hgnetv2_s_coco.yml` to
  `base/dataloader.yml`) overrides it.
- the `Resize` registration in `RTV4/engine/data/transforms/_transforms.py`,
  `Resize = register()(T.Resize)`, is `torchvision.transforms.v2.Resize`, which on a PIL
  input calls PIL's own `Image.resize((640, 640), BILINEAR)`: a fixed square, aspect ratio
  not preserved, resampling done in PIL space.
- `ConvertPILImage`, in that same transforms module, is `pil_to_tensor → .float() → /255.`
  and nothing else. **There is no ImageNet mean/std normalisation in this pipeline.**
- `orig_size` is `[w, h]` (the field the dataset builds in
  `RTV4/engine/data/dataset/coco_dataset.py`) and the postprocessor does
  `bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)` (the box-scaling step in
  `RTV4/engine/rtv4/postprocessor.py`), so `orig_target_sizes = [[1920, 1080]]`
  and boxes come back in original-image pixels.

The exported graph confirms the contract: inputs `images` float32 `[N,3,640,640]`
and `orig_target_sizes` int64 `[N,2]`; outputs `labels` int64 `[N,300]`, `boxes`
float32 `[N,300,4]` (xyxy), `scores` float32 `[N,300]`; ir_version 8, opset 17,
producer `pytorch 2.11.0`, 926 nodes.

**Sample:** 20 dev-val frames, image ids 1, 91, 181, … 1711 (every 90th), which
spans 20 of the 30 cameras and both frame offsets in the split. Pulled from the pod
because, as recorded below, the Mac's own copies are a different encode. Their sha256s
are in `outputs/eyes_b1/parity_sample/SHA256SUMS.txt` and in the per-image records
of the parity JSON; three were cross-checked against `sha256sum` on the pod and
matched.

```
uv run python scripts/eval/onnx_parity_v2.py \
  --onnx outputs/eyes_b1/b1_best_stg2_345c30a5.onnx \
  --gt data/train_ok/physicalai_smartspaces/annotations/person_v2_devval.json \
  --images outputs/eyes_b1/parity_sample \
  --det outputs/eyes_b1/dumps/dets_b1_rtv4s_seed0_identity.json \
  --bench 40 --warmup 8 \
  --out-json outputs/eyes_b1/i2/parity_345c30a5_podbytes.json
```

### Deltas, ONNX `4026204967aa…fd49` vs dump `bed59aa2…538d82` (ckpt `345c30a5…`)

| score floor | ONNX dets | dump dets | images with a count delta | min best-IoU | max IoU-matched score delta | max IoU-matched box delta | p50 score delta | p50 box delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 307 | 307 | 0 | 0.9964 | 0.0558 | 0.391 px | 4.5e-05 | 0.006 px |
| **0.4** | **324** | **324** | **0** | **0.9964** | **0.0558** | **0.676 px** | **5.0e-05** | **0.006 px** |
| 0.3 | 355 | 355 | 0 | 0.9964 | 0.0558 | 0.676 px | 5.9e-05 | 0.007 px |
| 0.05 | 1211 | 1211 | 8 | 0.1815 | 0.1769 | 84.6 px | 1.7e-04 | 0.022 px |
| 0.0 (all 300/img) | 6000 | 6000 | 0 | 0.0 | 0.985 | 1071 px | 2.9e-05 | 0.071 px |

The maximum absolute difference on the top-1 score of each image is **4.40e-05**. Zero label mismatches at
any floor (there is only one class, so this is a shape check, not a semantic one).

**How to read the floor.** The postprocessor emits a fixed top-300 per image
regardless of content; on a dev-val frame only ~20 of those clear 0.3 and the rest
is the model's noise tail, scores down to 1e-4, whose box regressions are
arbitrary. At floor 0.0 the worst case is therefore dominated by tail queries that
mean nothing: a rank swap between two 0.0002-scored boxes prints a box delta of
about 1900 px on a 1920 px frame. The 0.3 / 0.4 / 0.5 rows are the ones that
describe the model as it is actually run.

**Every rank-order disagreement is a tie swap.** Three of the 20 images show a
rank-aligned box delta above 1 px at floor 0.5 (ids 991, 1081, 1261). Each was
traced query by query: in every case ONNX rank *k* is the dump's rank *k plus or minus 1* with
the box matching to <1 px, and the two scores that swapped differ by 2e-5 to 3e-4.
Example, image 1261: ONNX rank 4 (s=0.964470) is the dump's rank 5 (s=0.964425)
and vice versa, same two boxes.

**One genuine numeric outlier, isolated.** Image 1261 has exactly one query whose
score really differs: box `[1024.70, 299.61, 1053.25, 424.03]`, ONNX 0.633848 vs
checkpoint 0.689646 (delta 0.0558, logit delta approximately 0.25). Its box matches to <0.02 px. It is
a duplicate detection of the same person already caught at rank 20 (box
`[1024.51, 299.70, 1053.16, 424.11]`, 0.853906 vs 0.853050, delta 0.0009), that is, a
query sitting on the flat, unstable part of the score surface. Sorting both lists
by box position and pairing them, the *second*-largest score delta on that image is
0.0030 and everything below it is at or below 0.0013. Across all 20 images at floor 0.4,
exactly **1 of 324** matched pairs exceeds an absolute score difference of 0.01; the p99 is 0.0035.

**Call: pass.** Structural disagreement would show as a count mismatch, a missing
box, or a low worst-case IoU. There is none: counts are exact at every operating
point, every ONNX box has a dump twin at IoU at or above 0.9964, boxes agree to a median
0.006 px, and the score distribution agrees to a median 5e-05. The single 0.0558
outlier is one duplicate query out of 324 head detections and does not move any
operating point.

### The Mac frames tree is not the dev-val bytes: read this before grading pixels here

`data/train_ok/physicalai_smartspaces/frames/train/...` on this Mac and the training
pod's person_v2 frames tree hold the same frames at the same 1920x1080 with the same
JFIF metadata keys, but they are **different JPEG encodes**.
`Warehouse_005/Camera_0000/000000.jpg` is `4948232c…` here and `fcfd002b…` there
(229,193 vs 229,181 bytes). Decoded, the two differ on ~44% of pixels with a maximum
absolute difference of 8 to 10 of 255 and a mean absolute difference of approximately
0.69 to 0.75. This is not a metadata difference.

It matters more than it sounds. Re-running the exact same parity harness on the
same 20 image ids, changing only which tree the pixels come from:

| pixels | ONNX dets @0.4 | dump dets @0.4 | images with count delta | min best-IoU | max score delta | max box delta |
|---|---:|---:|---:|---:|---:|---:|
| pod bytes | 324 | 324 | 0 / 20 | 0.9964 | 0.0558 | 0.68 px |
| this Mac's registry tree | 320 | 324 | 6 / 20 | 0.3437 | 0.2211 | 85.0 px |

The encode difference is roughly two orders of magnitude larger than the export
drift. Extended to the whole split (all 1,800 frames, Mac tree, same ONNX, same
reference dump, `outputs/eyes_b1/i2/parity_345c30a5_macbytes_FULLTREE.json`): at
floor 0.4 the ONNX finds 33,699 detections against the dump's 33,715, 374 of 1,800
images disagree on the count, p50 absolute score difference 9.5e-04, p99 0.086, and 3,617
of 33,699 matched pairs exceed an absolute score difference of 0.01.

Consequences, recorded so the finding is not made twice:

- Any pixel-level Mac-side comparison against a pod-produced dump must use pod
  bytes. The 20-frame sample in `outputs/eyes_b1/parity_sample/` exists for exactly
  this reason.
- The full-split Mac-tree numbers in the paragraph above are **encode-confounded**
  and are not an ONNX parity result. They are evidence about the trees.
- Where the two trees came apart is not established here. Both are 1920x1080 JPEG
  with identical JFIF keys, so the likely cause is two separate extraction runs from
  the MTMC source rather than a copy. Worth resolving before anything else on this
  Mac is graded against pod outputs at the pixel level.

### Processor ms/frame

Host: Apple M5 Pro, arm64, macOS-26.6.2, Python 3.11.15, numpy 2.4.6,
**onnxruntime 1.28.0**, `CPUExecutionProvider` (available providers also list
CoreML and Azure; CoreML is barred by POLICY.md, section 6), `intra_op_num_threads` left at the
onnxruntime default.

| model | input | batch | warmup | timed | mean ms | p50 ms | p95 ms | min ms | max ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `b1_best_stg2_345c30a5.onnx` (`4026204967aa…fd49`) | 640x640 | 1 | 8 | 40 | 55.04 | **54.77** | 56.72 | 54.09 | 59.86 |

`session.run` only, the model forward plus the exported postprocessor. Preprocessing
(PIL decode + resize) is not in that figure, and the bench reuses one already-warm
input tensor.

The streaming figure is higher and is the one to plan against. Over the 2,196-image
MVOR pass (640x480 PNG source, same ONNX, same host), `session.run` alone measured
mean 63.5 / p50 **64.8** / p95 70.0 ms, and end-to-end wall including PIL decode and
resize was 167.5 s for 2,196 frames = **76.3 ms/frame**. So: ~55 ms is the model in
isolation, ~65 ms is the model when a decode is interleaved between calls, and
~76 ms is a frame's real cost on this host. Quote 76.3 ms/frame for anything that
looks like a throughput claim, not 54.8.

Comparison note. The earlier in-house precedent, a 6.0 ms detector p50 for
`rtmdet-nano` at 320x320 through rtmlib, comes from a detector generation since
retired and is not comparable with the rows above, which measure a 10.2M-param DETR
at 640x640 with the postprocessor inside the graph. The rule that no performance
claim goes unmeasured (POLICY.md, rule 9) is
satisfied by the row above: it is a measured number on named hardware with the ONNX
hash beside it.

---

## 4. MVOR: the authorised internal-only look

> **Internal only.** MVOR is CC BY-NC-SA 4.0 real hospital footage (registry entry
> `configs/datasets/mvor.yaml`, bucket EVAL_ONLY, CQ-24 open, recorded go-ahead for
> test-only use). It trains nothing. Nothing in this section is ever quoted
> externally, and no MVOR byte or MVOR detection file leaves this machine. The
> detections were produced and graded entirely on the Mac.

**Bytes were already on disk**, so nothing was downloaded. `data/eval_only/mvor/`
in this checkout holds `camma_mvor_dataset/` (1.3 GB extracted, `day{1..4}/cam{1..3}/color/`),
`camma_mvor_dataset.zip`, the release annotations, and the v2-converted COCO file
`annotations/mvor_person_v2_coco.json` (2,196 images at 640x480, 4,699 person
boxes, `category_id` 0, zero `iscrowd`) built earlier by
`scripts/data/convert_mvor.py --config configs/convert/mvor_person_v2.yaml`. The
registry rule is satisfied trivially: the evaluation ran from the only checkout whose
tree physically holds the bytes, and a checkout without `data/eval_only/mvor` could
not have run it. No symlink was created into or out of any bucket tree.

```
uv run python scripts/eval/eval_detector_v2_onnx.py \
  --onnx outputs/eyes_b1/b1_best_stg2_345c30a5.onnx \
  --gt data/eval_only/mvor/annotations/mvor_person_v2_coco.json \
  --images data/eval_only/mvor/camma_mvor_dataset \
  --out-det outputs/eyes_b1/i2/dets_mvor_345c30a5.json \
  --out-json outputs/eyes_b1/i2/mvor_345c30a5.json \
  --cluster-depth 2 --bootstrap 200 --label mvor-internal
```

### `345c30a5` (ONNX `4026204967aa…fd49`) on MVOR, 2,196 images

| metric | value |
|---|---:|
| AP .50:.95 | **0.1935** |
| AP50 | 0.3989 |
| AP75 | 0.1561 |
| AP small / medium / large | 0.0000 / 0.0054 / 0.2487 |
| AR1 / AR10 / AR100 | 0.1609 / 0.3562 / 0.4336 |
| clustered bootstrap, 12 clusters (day by cam), 200 resamples | mean 0.1960, 95% CI **[0.1671, 0.2325]** |

Same detections at the I1 operating points (IoU 0.5, no maxDets cap):

| score at or above | TP | FP | FN | precision | recall | F1 | FP/img |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 3011 | 15917 | 1688 | 0.1591 | 0.6408 | 0.2549 | 7.248 |
| **0.4** | 2800 | 9077 | 1899 | **0.2357** | **0.5959** | 0.3378 | **4.133** |
| 0.5 | 2546 | 5298 | 2153 | 0.3246 | 0.5418 | 0.4060 | 2.413 |

Per day and cam at 0.4, recall spans 0.5000 (day1/cam1) to 0.7206 (day4/cam1) and
precision 0.2061 (day3/cam2) to 0.3206 (day4/cam1); FP/img is 3.4 to 4.9 everywhere.
The failure is uniform across days and cameras, not one bad view.

**Read.** This is the field to beat, and it is a long way from the dev-val numbers
on the same checkpoint: AP 0.8356 to 0.1935, and at the demo point precision
0.9317 to 0.2357 with FP/img 1.28 to 4.13. A detector trained entirely on synthetic
Omniverse warehouse renders transfers poorly to a real operating room: small
people (AP_small 0.0000, AP_medium 0.0054), unfamiliar clothing and equipment,
publisher-blurred faces, and a 640x480 source upscaled to a 640x640 square input.
Recall at 0.4 is 0.60, so the model does find most people; what collapses is
precision, at ~4 phantom boxes per frame.

Nothing here is a promotion or demotion signal, because MVOR trains nothing and gates
nothing. It is the number any future real-footage claim must be measured against,
and it states plainly where this lane currently stands on real clinical-adjacent
imagery.

**Init arm not run.** The protocol authorises one look each for `345c30a5` and the
init `238a3f65`. Only the first was spendable at the time: no `238a3f65` checkpoint
or ONNX existed on the grading machine. See deviation 4.

---

## 5. Deviations, blocks and open items

1. **No scp of dev-val images was needed for ground truth; one was needed for pixels.**
   The dev-val annotation JSON was expected to be absent from the Mac. It is present
   and byte-identical to the pod's. The 20 sample frames did have to come from the
   pod, for the encode reason above, which is a stronger reason than the one
   anticipated.
2. **The parity sample lives in `outputs/eyes_b1/parity_sample/`**, with a
   `SHA256SUMS.txt`. It is engineering scratch: no registry entry, no
   annotations, no symlink into or out of any bucket tree. The dev-val ground truth
   and the full frames tree were read in place from the checkout that physically
   holds them (this one), as the registry rule requires.
3. **The protocol's full-split ONNX AP parity against 0.8356: run and passed, after
   an initial block.** The protocol declares a Mac run of
   `scripts/eval/eval_detector_v2_onnx.py` (onnxruntime CPU, pycocotools, per-camera
   clustered bootstrap) for parity against the trainer's 0.8356. The block, while it
   stood, was this: the script exists and works (it produced the MVOR numbers), but
   the run needs the pod's 1,800 dev-val frames, and the Mac tree's frames are the
   wrong bytes; pod transfer was capped at about 16 to 20 images while a 13 h
   training run was live (pod load average 21.3 at the time of this work). The cost
   estimated at the time was approximately 440 MB scp (Warehouse_005 is 2.2 GB for
   9,000 frames; dev-val is 1,800 of them), then approximately 3 min of Mac processor
   time. The command was:
   ```
   uv run python scripts/eval/eval_detector_v2_onnx.py \
     --onnx outputs/eyes_b1/b1_best_stg2_345c30a5.onnx \
     --gt data/train_ok/physicalai_smartspaces/annotations/person_v2_devval.json \
     --images <pod-byte frames root> \
     --out-det outputs/eyes_b1/i2/dets_devval_onnx_345c30a5.json \
     --out-json outputs/eyes_b1/i2/devval_onnx_345c30a5.json \
     --cluster-depth 2 --bootstrap 200 --label synthetic-devval
   ```
   What was established without it: the dump reproduces the checkpoint's AP to 0.0,
   and the ONNX reproduces the dump per-image. The missing link was only the
   split-level AP of the ONNX itself.

   **Resolved the same day, after this page's first commit.** The 1,800 pod-byte
   frames were pulled read-only into
   `outputs/eyes_b1/devval_frames_podbytes/` (count 1,800; spot sha256 vs the pod
   matched, `fcfd002b…`) and the exact command above ran to completion:

   | quantity | value |
   |---|---:|
   | ONNX full-split AP .50:.95 | **0.8355776016** |
   | trainer `best_stat` (ckpt `345c30a5…`) | 0.8355529764 |
   | delta | **+0.0000246** |
   | AP50 / AP75 | 0.9581 / 0.8826 |
   | clustered bootstrap (30 cameras, 200 resamples) | mean 0.8355, 95% CI [0.8205, 0.8548] |
   | detections sha256 | `cc6d33afe83220f5…` |
   | inference p50 / wall | 55.3 ms / 124.7 s for 1,800 frames |

   The I2 chain is now closed end to end: dump matches checkpoint (delta 0.0), ONNX matches
   dump per-image (median 0.006 px), and ONNX matches checkpoint at split level (delta +2.5e-5,
   the accumulated fp32 drift over the whole split). Verdict unchanged: **pass**.
4. **The MVOR init arm (`238a3f65`) was blocked for want of bytes.** The protocol
   authorises one look each for `345c30a5` and the init `238a3f65`. On 2026-09-05
   no `238a3f65` checkpoint or ONNX existed anywhere on the grading machine
   (`outputs/pod_backup/` held only `b1_best_stg2_345c30a5.pth`,
   `best_stg2_seed1_e98de401.pth` and the pose checkpoints; no `rtv4_s_coco.pth`).
   Exporting it would have needed the pod's trainer venv. The `345c30a5` look was
   spent; the init look is still unspent.

   **The blocker no longer stands, and the look is still unspent.** A graph of
   `238a3f65` was exported afterwards from the pinned RT-DETRv4 source, ONNX
   `b4c3ff04a89b4938bfa6fe39ee458ea00fa9b52e407954fa4b21af5d65e53390`, and run through
   this repository's `rtdetr_onnx` backend on the processor as the comparison detector
   in research/eyes-coupling-probe.md. The MVOR init look is therefore unspent by
   choice rather than for want of an artefact.
5. **This work added three scripts**, ruff-clean (`ruff check` and
   `ruff format --check` both pass):
   `scripts/detector/fuse_tta_dumps.py` (the declared `fuse.py`-defaults TTA row),
   `scripts/eval/onnx_parity_v2.py` (the dump against ONNX comparison), and
   `scripts/eval/eval_detector_v2_onnx.py` (the ONNX grader the protocol named but
   which did not exist).
6. **The protocol page's I2 paragraph names a script that had never been written.**
   `scripts/eval/eval_detector_v2_onnx.py` was declared in
   `research/eyes-instrument-protocol.md` as if it existed. It does now. Its clustered
   bootstrap resamples cameras, not images, and it notes its own conservatism:
   duplicate cluster draws collapse because COCOeval keys on image id, so the
   interval is slightly narrower than a true cluster bootstrap.
7. **`--test-only` wording.** The protocol's I1 paragraph still describes `--test-only`
   dumps; addendum (c) already corrected that and the sidecars confirm
   `mode: run`. No action, noted so the two readings are not mistaken for two runs.

## Files

Written by this work:

- `research/eyes-instrument-grading.md` (this page)
- `research/eyes-precision-345c30a5-identity.json`
- `research/eyes-precision-e98de401-identity.json`
- `research/eyes-precision-345c30a5-tta-wbf.json`
- `research/eyes-precision-e98de401-tta-wbf.json`
- `research/eyes-i1-coco-ap-dump.json`
- `scripts/detector/fuse_tta_dumps.py`
- `scripts/eval/onnx_parity_v2.py`
- `scripts/eval/eval_detector_v2_onnx.py`
- `outputs/eyes_b1/i1/`: per-surface markdown plus the hflip-view JSONs
- `outputs/eyes_b1/i2/parity_345c30a5_podbytes.json`: the parity result plus bench
- `outputs/eyes_b1/i2/parity_345c30a5_macbytes_FULLTREE.json`: the encode evidence
- `outputs/eyes_b1/i2/mvor_345c30a5.json`, `outputs/eyes_b1/i2/dets_mvor_345c30a5.json`,
  `outputs/eyes_b1/i2/precision_mvor_345c30a5.{md,json}`: internal only, stay local
- `outputs/eyes_b1/parity_sample/`: 20 pod-byte dev-val frames plus `SHA256SUMS.txt`
- `outputs/eyes_b1/dumps/dets_{b1_rtv4s_seed0,deim_s_seed1}_tta_wbf.json` + sidecars

The `research/` and `scripts/` entries are in version control. Everything under
`outputs/` and `data/` named on this page is a run directory or a staged corpus that
is not in version control, and the same holds for the `outputs/` paths quoted in the
commands above.
