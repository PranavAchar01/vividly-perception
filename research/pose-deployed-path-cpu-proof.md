# The v2 pose model of record on the deployed path: CPU person-500 ladder

Dated 2026-09-05. **Every row below is eval-smoke provenance, not a Vividly accuracy
claim.** COCO-val is the steering surface; no frozen hospital-style pose exam exists in this
repository, and no shipping claim rides on any number here.

## Why this run exists

The v2 pose model of record's headline number, **COCO val2017 AP 0.774115**
(`research/pose-seed-floor-and-champion.md`), was measured **trainer-side**: the mmpose
evaluation loop on the pod, **ground-truth boxes**, `flip_test=True`, torch weights on CUDA.
The artefact the verifiers actually consume is a different thing: the exported ONNX
`pose_v2_champion_p0e38_6ba96285.onnx`, driven through `RtmlibPose` on **onnxruntime CPU**,
with boxes from a **detector**, and no flip TTA. That artefact had never been scored on
that path. This ladder scores it.

The two measurements are **not** a before-and-after reading of the same quantity and must never be
subtracted from each other. The comparisons this run supports are the ones **inside** the
table: same subset, same detector, same decode, one variable at a time.

## The three rows

Frozen subset `configs/eval/subset_coco_val_person500.json`, fingerprint
`2db41bbad264b7dc`, 500 COCO val2017 images. Boxes for **every** row: stock **rtmdet-m
person `235e8209` at 640, score_thr 0.05** (cached ONNX sha256 prefix `4f4d7e07`; the
`235e8209` token is the upstream `.pth` digest prefix, not the ONNX digest). There are 15,571
detections in every row, because the detector is deterministic and byte-identical across rows,
so it is a control, not a variable.

| # | row | pose ONNX sha8 | flip TTA | boxes | det AP | **kpt AP** | kpt AP 95% CI | kpt AP50 | kpt AR |
|---|---|---|---|---|---:|---:|---|---:|---:|
| 1 | `coco500_pose_body7_e48f03d0` (reproduction) | `5c0a4bf6` | off | stock-m `235e8209` | 0.6823 | **0.7285** | [0.7038, 0.7532] | 0.8940 | 0.7961 |
| 2 | `coco500_pose_v2_6ba96285_noflip` | `6ba96285` | off | stock-m `235e8209` | 0.6823 | **0.7196** | [0.6939, 0.7440] | 0.8846 | 0.7866 |
| 3 | `coco500_pose_v2_6ba96285_flip` | `6ba96285` | **on** | stock-m `235e8209` | 0.6823 | **0.7262** | [0.7015, 0.7508] | 0.8916 | 0.7918 |

det AP is identical to 4 dp in all three rows (0.6823 / AP50 0.9005 / AR@100 0.7664) by
construction. CIs: 1000-resample **per-image** bootstrap, seed 0; det AP CI
[0.6626, 0.7046] in every row.

### Row 1 is the gate, and it passed

Row 1 re-runs a leg that already has a recorded answer:
`coco500_eyes_stockm_20260826_111338_2db41bba` [clean], **det AP 0.682, kpt AP 0.729**
(`research/stats-compendium.md`). Measured now: **det AP 0.6823, kpt AP 0.7285**.
The instrument reproduces its own history to within a thousandth on both metrics, so rows
2 and 3 are interpretable. The gate is declared in the config
(`configs/eval/coco500_pose_body7_e48f03d0.yaml`, `gate:` block) and enforced by the runner,
and a miss would have stopped the ladder before the v2 rows were built.

## What the rows say

Paired bootstrap deltas, computed on the **same** resample draws for both rows (seed 0), so
the shared-image and shared-detector correlation cancels instead of inflating the interval:

| delta | metric | point | 95% CI | P(delta above 0) |
|---|---|---:|---|---:|
| row 1 against row 2 (body7 against v2, both no-flip) | kpt AP | **-0.0089** | [-0.0148, -0.0037] | 0.001 |
| row 2 against row 3 (v2, no-flip against flip) | kpt AP | **+0.0066** | [+0.0030, +0.0109] | 1.000 |
| any row against any row | det AP | +0.0000 | [+0.0000, +0.0000] | 0.000 |

(The det AP row is a sanity check, not a finding: the delta is exactly zero in all 1000
resamples, and `P(Δ>0) = 0.000` for a strictly-greater test on an exactly-zero difference
means never larger, not reliably smaller.)

**1. As the path runs today, the trainer-side ordering inverts.** Trainer-side, the v2
fine-tune beats its body7 init by **+0.0096** (0.774115 against 0.764499, ground-truth boxes, flip on).
On the deployed path with detector boxes and no flip, it **loses to it by 0.0089**, and the
paired interval excludes zero. **That inversion is a defect in the deployed
path's preprocessing, not a property of the model.** See the colour-order section, which
is the most important finding in this document.

**2. Flip TTA recovers about three quarters of that gap, and costs for it.** Turning on
the mmpose-equivalent flip TTA moves the v2 model from 0.7196 to 0.7262 (+0.0066, interval
excludes zero, positive in 1000 of 1000 resamples), leaving it **0.0023 short** of body7,
which is inside the noise and not a recovery of the lead. The price is the pose stage: p50
rises from 344.8 to 488.9 ms per image and from 20.21 to 32.86 ms per person crop
(**1.63x**), for two network forwards per crop instead of one.

**3. Absolute deployed-path numbers are far below the trainer-side one, for all models.**
0.7196 against 0.774115 is not a regression in the model; it is what detector boxes at
score_thr 0.05 plus ONNX and rtmlib preprocessing cost, and the body7 row (0.7285 here
against 0.764499 trainer-side) pays a similar toll. Neither number is a claim about the other.

## The colour-order defect, the headline finding

Every statement in this section describes the path as it ran on 2026-09-05, the date of this
measurement. The correction taken afterwards is recorded in **D** below.

**rtmlib never converts colour.** `RTMPose.preprocess` affine-crops the cv2 image (BGR),
normalises with the ImageNet mean and standard deviation, and feeds that straight to the
graph. There is no `cvtColor` anywhere in the rtmlib pose path (verified by grep in rtmlib
0.0.16). The mmpose RTMPose `data_preprocessor` is **`bgr_to_rgb=True`**, and neither ONNX
graph carries the swap: `scripts/pose/export_rtmpose_onnx.py` wraps `extract_feat` and
`head`, which sit after the preprocessor, and the upstream mmdeploy SDK export does its
`to_rgb` in `pipeline.json`,
which rtmlib does not implement. So the deployed path feeds **every** RTMPose graph
channel-swapped input, and has done so in every CPU proof this repo has ever run.

Two diagnostic rows measure the cost. Same subset, same detector, same decode, same
bootstrap seed: the one difference is `pose.color_order: rgb`, a runner-side knob that
converts the image handed to the pose stage (the detector keeps its own convention).

| # | row | colour fed to pose | flip | **kpt AP** | 95% CI | kpt AR |
|---|---|---|---|---:|---|---:|
| 1 | `coco500_pose_body7_e48f03d0` | BGR (**as the path runs today**) | off | 0.7285 | [0.7038, 0.7532] | 0.7961 |
| 2 | `coco500_pose_v2_6ba96285_noflip` | BGR (as today) | off | 0.7196 | [0.6939, 0.7440] | 0.7866 |
| 3 | `coco500_pose_v2_6ba96285_flip` | BGR (as today) | on | 0.7262 | [0.7015, 0.7508] | 0.7918 |
| Diag 1 | `coco500_pose_body7_e48f03d0_rgb` | RGB (diagnostic) | off | 0.7393 | [0.7139, 0.7636] | 0.8046 |
| Diag 2 | `coco500_pose_v2_6ba96285_noflip_rgb` | RGB (diagnostic) | off | **0.7473** | [0.7242, 0.7715] | 0.8107 |

Paired deltas, all on the same 1000 resample draws (seed 0 and the same image order in both
runs, so rows from the two runs pair exactly):

| delta | point | 95% CI | P(delta above 0) |
|---|---:|---|---:|
| v2: BGR to RGB | **+0.0277** | [+0.0215, +0.0342] | 1.000 |
| body7: BGR to RGB | +0.0108 | [+0.0062, +0.0154] | 1.000 |
| **RGB, no flip: body7 to v2** | **+0.0081** | [+0.0023, +0.0137] | 0.995 |
| v2: flip with BGR to no-flip with RGB | +0.0211 | [+0.0144, +0.0278] | 1.000 |

What this says, in order of importance:

**A. The inversion was the defect.** Fix the colour order and the deployed path reproduces the
trainer-side ordering: v2 beats body7 by **+0.0081** [+0.0023, +0.0137], against +0.0096
trainer-side. The v2 fine-tune's gain does transfer; the path was throwing it away.

**B. The defect costs the v2 model 2.6x what it costs body7** (+0.0277 against +0.0108). A model
fine-tuned harder on one colour convention is more sensitive to being fed the other, which
is why measuring the model of record on the deployed path found something the body7 smoke row
never would have.

**C. The free fix is worth 4.2x what flip TTA is worth.** One `cvtColor` per image, about
1 ms, buys +0.0277; a second network forward per crop (1.63x pose latency) buys +0.0066. A
colour-corrected no-flip run beats a flip-TTA BGR run by **+0.0211** while being cheaper.
Any decision about shipping flip TTA should be re-taken after the colour order is fixed,
not before.

**D. This was a live defect in the shipped path, not a laboratory curiosity.** At the time of
this measurement nothing in `src/vividly/models/rtmlib_backend.py` converted colour, so every
consumer of `RtmlibPose`, including the keypoint extraction that feeds the fall and seizure
verifiers, was fed BGR crops. Rows Diag 1 and Diag 2 are **diagnostics, not ladder rows**: they
describe what the models do if the preprocessing is corrected, not what the path did when
these rows were measured. The correction was taken later, and it was not a change to the pose
backend: it travels as the runner-side `pose.color_order` key set to `rgb` in the v2c
extraction generation, one of that generation's two declared deltas in all seven
`configs/events/extract_*_v2c.yaml`, with `src/vividly/models/rtmlib_backend.py` left
unchanged and the pipeline configurations fenced against setting the key by
`tests/test_pipeline_color_order.py`. It was measured on the event lanes rather than assumed,
because a verifier trained on features extracted with BGR crops may not simply improve when
the extraction changes underneath it: the fall refit on the v2c trees is recorded on
`final_models/fall/README.md` and the seizure refit on `final_models/seizure/README.md`.

## Latency

Apple M5 Pro, onnxruntime **1.28.0**, `CPUExecutionProvider`, **default thread count**,
batch 1. macOS 26.6.2, Python 3.11.15, numpy 2.4.6. The pose stage is billed **per image**
(all of that image's crops) as well as per crop; this subset averages **31.1 person crops
per image** at score_thr 0.05, which is why the per-image pose p50 is two orders above the
per-crop cost.

| row | det p50 / p95 | pose p50 / p95 | pose ms per crop | e2e p50 / p95 | wall |
|---|---:|---:|---:|---:|---:|
| 1 body7, no flip | 212.9 / 285.0 | 347.1 / 2040.5 | 20.36 | 573.3 / 2290.9 | 430 s |
| 2 v2, no flip | 205.7 / 291.3 | 344.8 / 1964.0 | 20.21 | 556.6 / 2210.1 | 428 s |
| 3 v2, flip | 166.0 / 233.3 | 488.9 / 3432.4 | 32.86 | 660.0 / 3652.1 | 603 s |
| Diag 1 body7, no flip, RGB | 130.4 / 190.0 | 236.5 / 1325.3 | 13.71 | 372.8 / 1526.0 | 284 s |
| Diag 2 v2, no flip, RGB | 127.3 / 183.8 | 223.6 / 1302.0 | 13.20 | 348.2 / 1490.0 | 274 s |

**The diagnostic rows are not evidence that RGB is faster.** The detector is byte-identical in all five
rows and its p50 still ranges 127 to 213 ms: rows 1 to 3 shared the laptop with other work,
while rows Diag 1 and Diag 2 ran alone. That spread, roughly plus or minus 25 percent, is the honest
error bar on every latency cell here, and it swamps the difference between the modes it is
not measuring. The one latency comparison that is internal to a single run is rows 2 against
3: flip TTA costs **1.63x** on the pose stage, moving from 20.21 to 32.86 ms per crop, not
2x, because only the network forward is duplicated, while the affine crop, the normalise and
the SimCC decode happen once.

**No Jetson Orin figure exists, and none is quoted until one is measured there
(POLICY.md, rule 9).**

## What was actually run

The pre-declared ladder (rows 1 to 3):

    uv run python scripts/pose/eval_pose_ladder.py \
        --config configs/eval/coco500_pose_body7_e48f03d0.yaml \
        --config configs/eval/coco500_pose_v2_6ba96285_noflip.yaml \
        --config configs/eval/coco500_pose_v2_6ba96285_flip.yaml \
        --bootstrap 1000 --seed 0

The colour-order diagnostic (rows Diag 1 and Diag 2), added afterwards and labelled as such:

    uv run python scripts/pose/eval_pose_ladder.py \
        --config configs/eval/coco500_pose_body7_e48f03d0_rgb.yaml \
        --config configs/eval/coco500_pose_v2_6ba96285_noflip_rgb.yaml \
        --bootstrap 1000 --seed 0

Run artefacts (gitignored; every row, both bootstrap distributions, the paired deltas):
`outputs/eval/pose_ladder_20260905_031458/results.json` and
`outputs/eval/pose_ladder_20260905_034047/results.json`.

All five configs are stock-schema and the stock runner
(`scripts/eval/eval_runner.py --config <config>`) accepts any of them unchanged, returning
the AP without an interval. What the ladder runner adds is the interval, the gate and the
`color_order` knob. **Caveat on the two `_rgb` configs:** the stock runner ignores
`pose.color_order` and would silently run the BGR row instead, so those two must go through
`scripts/pose/eval_pose_ladder.py`. Their rows carry `"pose_color_order": "rgb"` in
`results.json`; rows 1 to 3 predate the key and were BGR.

### How the interval is computed

COCOeval splits into `evaluate()` (per image: match detections to ground truth) and
`accumulate()` (sort all detections by score, sweep the PR curve). Only the second depends
on which images are in the set, so `evaluate()` runs once and each of the 1000 resamples
re-runs a re-implementation of `accumulate` over a resampled image-index list.
pycocotools' own `accumulate` cannot be reused: it derives its image list through `set()`,
which silently drops the duplicates that **are** the bootstrap. The re-implementation
self-checks on every run: resampling the identity index list must reproduce
`COCOeval.stats[0]` to 1e-9 or the runner refuses to publish an interval.

One trap worth recording: **the two COCO forks encode a matched detection differently.**
pycocotools leaves an unmatched detection's `dtMatches` at 0 and tests truthiness;
xtcocotools, which the keypoint path needs for OKS, initialises the row to -1 and tests
`>= 0` (cocoapi PR #332). Reading -1 as truthy turns every false positive into a true
positive, and it produced a keypoint AP of exactly **1.0** on the first attempt here. The
runner detects the convention from the data.

### The flip TTA, precisely

`src/vividly/models/rtmlib_pose_flip.py`, registered as pose backend `rtmlib_flip`. Mirror
the preprocessed crop, run the graph again, permute the COCO L/R channel pairs, reverse the
**X** SimCC vector along its bin axis (the Y vector is permuted but never reversed),
average the two SimCC pairs, then argmax-decode **once**. That is mmpose's own
`flip_vectors` contract. There is deliberately **no half-bin shift**: heatmap flip TTA in
mmpose applies `shift_heatmap`, the SimCC path does not, and adding one would be a silent
half-pixel bias against the trainer-side number this ladder exists to compare with.
Averaging the distributions and decoding once is also not the same as decoding twice and
averaging coordinates, because the SimCC argmax is hard, so the latter would average two
quantised integers. Pinned by `tests/test_pose_flip_tta.py`, including a
mirror-equivariance check against this ONNX.

## Provenance of every row

| | |
|---|---|
| Boxes (all rows) | `rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.zip`, cached as ONNX sha256 `4f4d7e07350b1753299111d1ae500fd64447a5b0e38e4bacbefab6573c742d30`; label `rtmdet-m-person-onnx[train=obj365;ship=no]` |
| Rows 1 and Diag 1, pose | `rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504` ONNX sha256 `5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c`; body7-trained, **ships nowhere** (`final_models/joints/README.md`, section Identity and provenance) |
| Rows 2, 3 and Diag 2, pose | `pose_v2_champion_p0e38_6ba96285.onnx` sha256 `6ba9628506a8899d3a3e940dcc8af5d30342921f09d2605afa505b57453d3527`, from checkpoint prefix `a0a035fe`; COCO-only fine-tune off a **body7 init** (`pt-body7` backbone: AI Challenger, CrowdPose, MPII, sub-JHMDB, Halpe, PoseTrack18). Provenance: `final_models/joints/README.md`, section Identity and provenance. Shippability is a question for counsel (POLICY.md, rule 8). |
| Imagery | COCO val2017, registry entry `coco`, bucket TRAIN_OK, staged on 2026-09-05 as APFS clones at `data/train_ok/coco`, which is gitignored; both val annotation JSONs verified byte-identical to the digests the registry pins |

A caveat on the labels the rows carry: `rtmpose-m-person-v2-onnx-6ba96285[train=COCO;ship=counsel]`
is derived by `_shippability()` from the filename, and `RtmlibPose` hard-codes
`pretrain_data=("AIC", "COCO")` for every RTMPose. For this model the fine-tune data really
is COCO-only, but the backbone pretrain is `pt-body7`, which is a **superset** of AIC+COCO.
The label understates the lineage; `final_models/joints/README.md`, section Identity and
provenance, is the record.

## What this does not establish

- Nothing about hospital or top-view footage. This is COCO val2017, perspective imagery.
- Nothing about the selection of the model of record. `a0a035fe` remains the v2 pose model of record on the
  protocol that selected it; this ladder measures a different protocol and does not
  re-open that outcome.
- Nothing about whether flip TTA should ship. +0.0066 kpt AP for 1.63x pose latency on a
  CPU budget is a trade for the owner to weigh, and rows Diag 1 and Diag 2 say it should be
  re-taken after the colour order is fixed, not before.
- Nothing about what fixing the colour order would do to the event lanes. Rows Diag 1 and Diag 2 are
  COCO keypoint AP. The fall and seizure verifiers were trained on features extracted
  through the BGR path, so changing the extraction is a re-training question, not a
  free upgrade. Measure before changing.
- Nothing about the pose soup (`research/pose-weight-soup.md`), which is unscored and was
  not run here.

## Open follow-ups this run created

1. **Closed, and not as this run proposed.** The colour order was corrected, but no `to_rgb`
   flag was added to `RtmlibPose`: `src/vividly/models/rtmlib_backend.py` carries no colour
   handling at all, and the pose backend is deliberately unchanged. The correction travels
   runner-side as the `pose.color_order` key (`scripts/run_pipeline.py`), which converts the
   image handed to the pose stage, and the seven `configs/events/extract_*_v2c.yaml` set
   `color_order: rgb` as one of that generation's two declared deltas.
   `final_models/joints/README.md` records the operating point as corrected to red-green-blue
   in the newest track-tree generation. Worth +0.0277 kpt AP to the v2 model of record on this
   subset for about 1 ms an image.
2. **Closed.** The event-lane consequence was measured before the correction reached anything
   downstream, rather than a default being flipped under trained verifiers: the v2c
   re-extraction generation is what carries the correction, and the fall refit on those trees
   is recorded on `final_models/fall/README.md`.
3. **Closed in the collector, and the compendium still carries no row.**
   `scripts/events/stats_compendium.py` filtered `outputs/eval/` on directory name prefixes
   that did not include the `pose_ladder_<stamp>/` directory this ladder writes; `pose_ladder_`
   is now in that prefix tuple, under its own joints-lane heading, and the row schema was
   already compatible (`condition`, `det_AP`, `kpt_AP`). `research/stats-compendium.md` is
   frozen at the state of the runs present on 2026-08-28 and is regenerable only from run
   directories that are not in version control, so it carries no row for this ladder, which
   ran after that date.
