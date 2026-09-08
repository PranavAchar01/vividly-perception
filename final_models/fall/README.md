# Fall lane: event verifier

The model of record for the fall lane, what it is measured on, the arms that were dropped,
the operating points, and the track-tree generations its features came from. Every number
on this page carries the hash of the graph that produced it.

## Disclosure, which applies to every number below

This is a motion-signature verifier over track windows, not a fall detector and not a
medical device. It is a prototype-class weight, trained on public licence-reviewed acted
fall footage under the recorded interim carve-out in `POLICY.md`. No number on this page is
clinical detection performance, it is never quoted as such internally or externally, it is
badged internal demonstration only, it is not deployed commercially without counsel sign
off, and it is slated for replacement the moment clinical footage exists. At least one
acted-fall dataset is held out entirely as a test set, and a dataset that trains loses its
test role permanently.

## Identity and provenance

| field | value |
|---|---|
| Model of record | fall_v3 |
| Checkpoint sha256 | `3d2d17eed74b735b4baef6e5f84eef1d9ed6f4c2b02f14ed072c7cbe9f221526` |
| ONNX sha256 | `67954c04eeef301ff2ab3a1238a79dba2bfa397b3eba5a93a1d3102d7e00010e` |
| Shipped as | `final_models/fall/model.onnx`, digest in `final_models/fall/model.onnx.sha256`, run record in `final_models/fall/report.json` |
| Architecture | small temporal convolutional network: three Conv1d blocks (kernel 5, hidden width 64, batch normalisation, ReLU, max-pool 2, dropout 0.15) over the 64 time steps of the eight track channels, mean and max pooled, one linear head; 60 epochs |
| Window | 8.0 s span, 1.5 s stride, 64 steps, `contain` labelling policy, `contain_margin_s` 0.5, `min_present` 0.35 |
| Channels | 8: `vy`, `vx`, `h_rel`, `aspect`, `angle_norm`, `angle_valid`, `kpc`, `present` |
| Calibration | score is the sigmoid of the logit over a temperature of 1.0251, fitted on the train-side holdout, monotonic, degenerate fit false |
| Training | batch 256, learning rate 0.001, weight decay 0.0001, positive weight automatic, seed 20260826, device cpu |
| Training data | 881 windows, 190 positive and 691 negative, from 282 videos |
| Source trees | the v1 generation of `le2i_fdd`, `gmdcsa24` and `caucafall` track sequences |
| Best train-side validation AUC | 0.99589, a train-side holdout number and not an exam number |
| Code revision | b1dfe79 |
| Weight class | prototype |
| Shippable | counsel |

The verifier reads the pipeline's own view of these videos, never their ground-truth boxes,
because the live system does not have those. It loads through
`src/vividly/models/event_verifier.py` on onnxruntime CPU.

One prose value in `final_models/fall/report.json`, the `disclosure` string, was normalised
on 2026-09-07 to the wording the current trainer emits; every metric value in that file is
untouched.

Datasets and their roles. `le2i_fdd`, `gmdcsa24` and `caucafall` train. `ur_fall` and
`multicam_fall` are registry-enforced test only and never enter a training manifest. Each
dataset's licence terms are recorded word for word in its registry entry under
`configs/datasets/`, and any research-only or non-commercial wording is flagged for counsel
rather than absorbed; the summary is in `DATASETS.md`.

Shippability. Every weight in this lane is prototype class under the interim event-model
carve-out. Nothing here ships without counsel sign off.

## Before and after

One deliberate change separates the model of record from the alternative recorded on this
lane: the pose graph that produced the track sequences the verifier reads. Same recipe,
same three training datasets, same architecture, same window geometry.

| | Model of record, fall_v3 | Alternative, fall_v2pose |
|---|---|---|
| verifier ONNX sha256 | `67954c04eeef301ff2ab3a1238a79dba2bfa397b3eba5a93a1d3102d7e00010e` | `b66c3a0abc1ce214d93757a7b5b1269ea5027e3ae48f3cecbd33f7ef515e63e3` |
| checkpoint sha256 | `3d2d17eed74b735b4baef6e5f84eef1d9ed6f4c2b02f14ed072c7cbe9f221526` | `1ab191ae98eeb67d132eac889675f8c199dc0f9d6dc7a6aad4b2194b76bc41d3` |
| pose graph behind the tracks | v1 generation, upstream body7 `5c0a4bf6` | joints lane model of record `6ba96285`, keypoint AP 0.774115 |
| train-side validation AUC | 0.99589 | 0.99558 |

Three held-out exams, one look each, no variant selection on the test side. Window AUC is
the per-window metric; video AUC is the per-video decision metric the run reports carry.

| exam | videos | metric | fall_v3 `67954c04` | fall_v2pose `b66c3a0a` | delta, alternative minus model of record |
|---|---|---|---|---|---|
| multicam, all cameras, headline | 192 | window AUC | 0.79276 | 0.78474 | minus 0.00802 |
| multicam, all cameras, headline | 192 | video AUC | 0.75815 | 0.73234 | minus 0.02581 |
| multicam, 24 picked views | 24 | window AUC | 0.81595 | 0.80814 | minus 0.00781 |
| multicam, 24 picked views | 24 | video AUC | 0.82609 | 0.86957 | plus 0.04348 |
| UR Fall | 52 | window AUC | 0.98413 | 0.97698 | minus 0.00715 |
| UR Fall | 52 | video AUC | 0.96711 | 0.94925 | minus 0.01786 |

Reading. This is a tie, not a win. Five of the six metrics move by less than the
seed-to-seed noise observed at this dataset size, and the one clear move is the picked-view
video AUC, which favours the alternative. On the headline all-cameras exam the model of
record is ahead on both metrics. A genuine joint-accuracy upgrade in the pose front end
(keypoint AP 0.764499 to 0.774115) did not translate into fall gains, because falls are
dominated by gross body motion and trajectory collapse, which the earlier pose already
captured. The headroom in this lane is in the verifier and the data, not the front end.
The alternative is recorded and kept as a reproducible variant on the newer pose, not
promoted.

## Arms measured and rejected

The push that followed the comparison above ran on the alternative's track trees against a
pinned holdout, with the exam budget declared before any run. The gate that governs
promotion has two parts and either part failing drops the candidate regardless of accuracy:
the UR Fall lie-transition false positives at threshold 0.8 must not be above the control
arm's, and the chute24 in-alarm fraction at 0.8 must not be above the control's.

| arm | checkpoint | ONNX | outcome |
|---|---|---|---|
| F0, the unchanged recipe, control | `4dffa823e10a8f04eb764aa17db682ad1e549aaff2ee233258e2b8c93f28b069` | `bb819c308f70909da1e6d1a4afe2f149e1c9227b601446af7430c7312451e587` | control; provided the lane's first seed floor |
| F1, `overlap` labelling at 4.0 s minimum | `2ccc0c15ea6772821559f7466525964e4a7436bb7d98bb14d65df392a375092e` | `916332b21b2cdffed102c4f1faafe69c360f09ce84fe133e728b4bdb782eda7b` | gate failure, dropped |
| F2, keypoint-geometry channels | `0d6f4eb3c5b2820ef1f0cd3833e1a0deb25008999a89cf127d067b81ae24da97` | `e7cb37776f5174158762f8bdb767bc22218407d22028c48d33a0056de8220d95` | gate failure, dropped |
| F3, camera-geometry augmentation, strong at K equals 8 | `2f97dc174ebfffaa1d4143f5200beddd5f7e77a45c6f63220d6ed265e8155c49` | `119c6f84ab10fb0c05e86f4ef8fa04cf2e7615603e38ea74733e7e2f1c7f1b68` | not promoted, no look taken |

Each digest identifies one run: seed 20260826 for F0 and for F2, seed 20260905 for F1, and
the strong setting at K equals 8 at seed 20260901 for F3. The other seeds of each arm
produced their own checkpoints and graphs, which are not quoted here.

**F1, label geometry.** Window AUC on all cameras rose to 0.80012, a gain of 0.0185 over
the control look, which does exceed the 0.00184 seed floor. It was dropped anyway: UR Fall
lie-transition false positives at 0.8 rose to 16 of 46 windows over 11 of 16 videos against
the control's 15 over 9, and the chute24 in-alarm fraction rose to 13 of 654, 0.0199,
against the control's 11 of 654, 0.0168. The UR window AUC fell 0.032 against the control,
which corroborates the trade: `overlap` labelling buys angle recall and pays in
lie-transition specificity.

**F2, keypoint-geometry channels.** Fifteen per-step channels behind a default-off flag,
byte-identical when off and pinned by a golden-hash test. It cleared promotion on the
pinned holdout by a wide margin (holdout mean 0.99787 against the control's 0.98639, a
delta of 0.01148 against a control spread of 0.00442) and produced the push's largest
all-cameras gain, 0.0298 over the control look, reaching window AUC 0.81143. It was dropped
on the same gate: UR lie-transition false positives at 0.8 rose to 22 of 46 windows over 12
of 16 videos against the control's 15 over 9, while the chute24 half passed at 9 of 654,
0.0138. The keypoint channels make a downward movement more convincing, not less.

**F3, camera-geometry augmentation.** Vertical foreshortening between 0.55 and 1.0, a
perspective trapezoid, roll within plus or minus 12 degrees and mirroring, applied to the
raw keypoint and box rows with the window re-featurised and the posture angle recomputed
from the warped keypoints. Eighteen runs, three strengths by two duplication factors by
three seeds, selected on the pinned holdout only. The chosen setting reached a holdout mean
of 0.99068 against the control mean of 0.98639, a gain of 0.00429 against a pre-declared
promotion margin of 0.00442. It missed by 0.00013, so no exam look was taken and the
candidate was not promoted. The narrower three-seed floor of 0.00184 would have cleared it;
the wider five-seed margin was fixed in advance and governs, and rulers are not switched
after the number is seen. A post-hoc identity control, eight byte-identical copies per
positive with no warp, scored 0.98688, which is 0.00025 below the three-seed control mean
of 0.98712 over seeds 20260826, 20260901 and 20260905, the seed set this post-hoc arm ran,
so the movement is the geometry and not positive oversampling.

## Operating points

The exam tables below were recorded at threshold 0.8. What the replay pipeline and the demo
platform actually attach is 0.95, set out after the tables.

| exam | window AUC | events caught at 0.8 | fraction of quiet footage in alarm at 0.8 |
|---|---|---|---|
| multicam, all 8 cameras | 0.79276 | 103 of 200 | 9.1 percent |
| multicam, 24 picked views | 0.81595 | 12 of 25 | 0.0 percent |
| UR Fall | 0.98413 | 14 of 30 | 56.5 percent |

**The all-cameras row is the honest one.** The 24-video exam was found on 2026-08-27 to be
a hand-picked best-view surface. Running all eight cameras raised the catch rate, 56 percent
to 59 percent at threshold 0.5, while costing average area under the curve, which is the
shape a model should have: it degrades gracefully off the good angle instead of collapsing.

**The UR Fall caveat.** The 0.98413 is real but flattering. That set's negatives are mostly
people lying down deliberately, and 15 of its 16 lie-transition clips fire. It is an easy
exam, kept for continuity, and it is why this lane is not judged on area under the curve
alone.

**The coverage failure.** 25 of the 200 all-camera events produced no window at all,
because the tracker never held the person. That is a tracking failure, not a scoring
failure, and no threshold reaches it. On the UR Fall surface the same effect caps the
recorded catch ceiling at 14 of 30 at any threshold, because 16 of the 30 events are
zero-window under the recorded `contain` extraction.

A later threshold study, fitted on exam negatives and therefore provisional until pilot
shadow data exists, recorded the following for the model of record resliced onto the v2pose
track trees. Catches are over the 200 all-camera events; the episode and in-alarm columns
are measured on chute24, the single no-fall scenario, 0.273 hours across 8 cameras, and on
the UR Fall activities-of-daily-living recordings, 0.081 hours over 38 recordings.

| threshold | all-camera catches, chute24 episodes per hour track and camera, chute24 in alarm | UR catches, UR episodes per hour, UR in alarm |
|---|---|---|
| 0.5 | 156 of 200, 65.9 and 47.6, 26.8 percent | 14 of 30, 234.6 and 197.5, 69.8 percent |
| 0.9 | 121 of 200, 14.7 and 14.7, 5.3 percent | 14 of 30, 98.8 and 98.8, 34.9 percent |
| 0.95, recommended for this head | 111 of 200, 7.3 and 7.3, 2.8 percent | 13 of 30, 49.4 and 49.4, 17.4 percent |

The model of record scores UR activities of daily living high at every threshold, so if it
runs anywhere its quiet range starts at 0.95, not 0.9. Persistence is not available as a
lever here: two consecutive windows at any threshold catches nothing on UR Fall, because
the surviving events yield too few consecutive windows under the recorded extraction.

At the point each head is recommended at, the loudness gap is a property of the head and not
of the threshold: on the UR activities-of-daily-living surface the model of record
`67954c04` runs 49.4 episodes per hour at 0.95, against 12.3 for fall_v2pose `b66c3a0a` at
its own recommended 0.9, both measured on the v2pose trees (`research/operating-points.md`).

What the replay pipeline attaches. `configs/pipeline/replay_demo.yaml` runs the fall head
`fall_then_still_v0` in annotate mode against the lane directory `final_models/fall`, graph
`67954c04`, temperature 1.0251, at a confirm threshold of 0.95, which is the operating
point recommended above. The loader reads the window geometry and the temperature from
`final_models/fall/report.json` and refuses the graph unless it hashes to the digest that
report records, so it never scores with unaccounted-for weights; the graph is a Git LFS
object, so `git lfs pull` has to run before the pipeline starts, and removing the
verification block runs the demo without a verifier. Every threshold on this lane is
provisional until the calibration protocol in `docs/CALIBRATION.md` has its preconditions
met.

## Track trees

Verifier features are read from extracted track sequences, so a verifier's identity
includes the stack that produced its tree. Three generations exist. The v1 and v2pose
extraction configurations are not carried in this repository; their settings are recorded
here because the model of record's training tree was built by the first of them.

Settings common to all three generations: tracker backend `bytetrack` with
`track_activation_threshold` 0.4, `high_conf_det_threshold` 0.4, `lost_track_buffer` 30 and
`minimum_iou_threshold` 0.1; detector score threshold 0.3 at input 640x640; track layer
`baseline_window_s` 300.0, `reap_after_s` 30.0, `kp_conf_min` 0.3, `ring_window_s` 300.0
and `baseline_min_samples` 30; `rules_config` `configs/pipeline/rules.yaml`;
`jpeg_quality` 30; `annotated_only` true. The operating point was chosen on 2026-08-22 from
the burn-in run of that date and has not moved since.

| generation | detector | detector ONNX sha256 | pose | pose ONNX sha256 | tracker | declared settings |
|---|---|---|---|---|---|---|
| v1, built the model of record's tree | upstream rtmdet-m person release, token `235e8209`, which is a checkpoint digest prefix and not the graph digest | `4f4d7e07350b1753299111d1ae500fd64447a5b0e38e4bacbefab6573c742d30` | upstream RTMPose-m body7 release, token `e48f03d0`, checkpoint sha256 `e48f03d0cfe1285ee8b6d3457ac3ce33a4594a92b080053ab1ec4a7e300975f2` | `5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c` | bytetrack, gates 0.4 | the common settings above; keypoints not emitted |
| v2pose, built the alternative's tree | unchanged from v1 | `4f4d7e07350b1753299111d1ae500fd64447a5b0e38e4bacbefab6573c742d30` | joints lane model of record, checkpoint `a0a035fe` | `6ba9628506a8899d3a3e940dcc8af5d30342921f09d2605afa505b57453d3527` | unchanged from v1 | exactly two declared deltas from v1: the pose graph becomes the joints lane model of record, and `emit_keypoints` becomes true. Emitting keypoints is neutral for training, because the trainer reads only the nine frozen fields; the keypoints are recorded for later rhythm work. Detector, tracker, track layer and rules unchanged |
| v2c, the current generation, `configs/events/extract_*_v2c.yaml` | eyes lane model of record, checkpoint `345c30a5`, backend `rtdetr_onnx` | `4026204967aa01a7680f7ce0c6169463d654a519f8093383a10537321f56fd49` | joints lane model of record, checkpoint `a0a035fe` | `6ba9628506a8899d3a3e940dcc8af5d30342921f09d2605afa505b57453d3527` | unchanged | exactly two declared deltas from v2pose: the detector becomes the eyes lane model of record on the `rtdetr_onnx` backend, and `pose.color_order` is set to `rgb`. Tracker, track layer, rules, `emit_keypoints` and dataset scope unchanged |

Two gates were passed before any v2c extraction ran. The backend parity gate compared the
new detector backend against the grading path on 50 frames of pod bytes and returned
bit-exact agreement, 0.000000 pixel and 0.000e+00 score differences over 50 of 50 frames
(commit 38309de). The fence gate confirmed that no pipeline configuration under
`configs/pipeline/` sets the colour-order key, so the correction reaches the live path only
through a declared re-extraction (commit d151d17).

### Fall F0 refit on the v2c track trees, measured 2026-09-07

Status: measured, not promoted. The bar was not met and the false-alarm gate failed on its
second part. The same two frozen heads that hold the bar lose about 0.28 of window AUC when
re-sliced onto the v2c trees with their weights untouched, so the movement belongs to the
track trees and not to the refit, evidence below. The model of record is unchanged.

Bar: all-cameras window AUC 0.78474, held by fall_v2pose ONNX
`b66c3a0abc1ce214d93757a7b5b1269ea5027e3ae48f3cecbd33f7ef515e63e3`. Gate: UR Fall
lie-transition false positives at 0.8 not above the control's 15 of 46 windows over 9 of 16
videos, and chute24 in-alarm at 0.8 not above the control's 11 of 654, 0.0168; the control is
the F0 seed 20260826 look on the v2pose trees, checkpoint
`4dffa823e10a8f04eb764aa17db682ad1e549aaff2ee233258e2b8c93f28b069`, ONNX
`bb819c308f70909da1e6d1a4afe2f149e1c9227b601446af7430c7312451e587`.

Protocol as declared: recipes `configs/events/train_fall_F0_v2c_s20260826.yaml`,
`train_fall_F0_v2c_s20260901.yaml` and `train_fall_F0_v2c_s20260905.yaml`, the F0 control
recipe unchanged except for the three source trees; pinned holdout of le2i Home_02, gmdcsa24
S4_ADL and S4_Fall, caucafall S9 and S10; 2,001 windows, 169 positive and 1,832 negative,
under `contain`; the median-validation seed took exactly one look on each of the three exams
through `configs/events/eval_fall_urfall_F0_v2c.yaml`, `eval_fall_multicam_F0_v2c.yaml` and
`eval_fall_multicam_allcams_F0_v2c.yaml`. Selection never touched an exam. Code revision
dfc38da.

| seed | checkpoint sha256 | ONNX sha256 | validation AUC at the best epoch | curve |
|---|---|---|---|---|
| 20260826 | `787d332bd60dd39974164f10c664308df60df0dd067e7a09301752dfa3f80d01` | `4da67acdd2c976c1036c67815dc40816352f0413f70971dad8369130c1e1f6b5` | 0.85832 | `curves/fall_v2c_F0_s20260826.csv` |
| 20260901, median, took the looks | `a5da25da24c1752009bc55a066fe55fec4f450ee660da3ebe34344a1fc66873e` | `baa93141776c05fc862d929a2c8cbda382097dd55c5ed82924589317a07d3fa3` | 0.83552 | `curves/fall_v2c_F0_s20260901.csv` |
| 20260905 | `0adfe8a791dcb8487cd72e943a4e29db92d07140f934a141012a12fa2dfcc8e0` | `032364104e18efc895bc93a9f8c7c1b190083e8cc7d7c1e29b54e941d48092d1` | 0.81498 | `curves/fall_v2c_F0_s20260905.csv` |

The seed spread is 0.04334. On the v2pose trees the same recipe and seeds reached 0.98639
to 0.98823, so the validation surface itself dropped by about 0.15 before any exam was
looked at. The three run reports are `report_v2c_F0_s20260826.json`,
`report_v2c_F0_s20260901.json` and `report_v2c_F0_s20260905.json` in this folder, with the
disclosure string carried in the wording this repository uses and every other value as
written by the trainer.

The looks, one per exam, all on the v2c exam trees:

| head, weights | UR Fall window / video AUC | multicam reference camera window / video AUC | multicam all cameras window / video AUC | UR lie-transition false positives at 0.8 | chute24 in-alarm at 0.8, reference camera | chute24 in-alarm at 0.8, all cameras |
|---|---|---|---|---|---|---|
| F0 refit, seed 20260901, ONNX `baa93141` | 0.81818 / 0.69123 | 0.64961 / 0.52174 | 0.55615 / 0.23600 | 13 of 84 windows | 0 of 205 | 143 of 1644, 0.0870 |
| model of record fall_v3, ONNX `67954c04`, re-sliced | 0.76578 / 0.64035 | 0.40267 / 0.13043 | 0.51600 / 0.23463 | 15 of 84 windows | 57 of 205 | 319 of 1644, 0.1940 |
| fall_v2pose, ONNX `b66c3a0a`, re-sliced | 0.77326 / 0.58421 | 0.43223 / 0.19565 | 0.49847 / 0.24863 | 9 of 84 windows | 44 of 205 | 258 of 1644, 0.1569 |

Reading the table: the candidate's all-cameras window AUC 0.55615 is 0.22859 under the bar,
which is larger than the seed spread, so it is a real loss and not noise. The first part of
the gate passes on the count, 13 against 15, although the v2c UR tree carries 84
lie-transition windows where the v2pose tree carried 46; the second part fails, 0.0870
against 0.0168. The two re-sliced heads scored 0.79283 and 0.78474 on the v2pose trees and
score 0.51600 and 0.49847 here, so the exam surface moved under every head at once, and the
refit is the strongest of the three heads on these trees.

Why the trees moved, measured on the training-side sets and never on an exam: the v2c
generation pairs the eyes model of record with the detector score threshold 0.3 inherited
from the earlier generations. On this side-view footage that detector emits 1.78 boxes per
frame on LE2I and 4.86 on GMDCSA-24 where the v1 detector emitted 1.02 and 1.04, and the
track counts rise five to ten times (LE2I 227 to 1,283, GMDCSA-24 206 to 2,203, CAUCAFall
142 to 635, UR Fall 222 to 773, multicam all cameras 1,953 to 10,735). A frame-level probe
against a stock COCO RT-DETRv4-S detector, and a reading of the disagreement frames by eye,
found the extra boxes on furniture in empty rooms and the lying person after a fall scored
0.30 to 0.46 by the eyes model where the stock detector scored 0.56 to 0.77. The full probe,
with both detectors' box counts at every threshold, is `research/eyes-coupling-probe.md`. The
fix is a further declared extraction generation with a detector that is in domain on ordinary
side-view footage, which is an owner decision; nothing has been re-extracted.

## Figures

- `final_models/fall/validation_curves.png`: per-epoch validation area under the curve for
  every run of this lane on one axis, fall_v2pose picked out in colour and the other 37
  runs in grey behind it. Three of those grey curves are the v2c refit recorded above and
  form the low band, because that refit validates on the v2c holdout, which is a different
  surface. The model of record has no per-epoch record in this repository, so it is not a
  curve on this figure: its best validation area under the curve of 0.99589 is drawn as a
  horizontal dashed reference line. The axis is labelled AUC, not AP: the third column of
  every curve CSV in this lane is named `val_ap` for uniformity across lanes, but for the
  event verifiers it carries the trainer's per-epoch validation area under the curve read
  from each run's report.
- `final_models/fall/champion_vs_baseline.png`: the same train-time holdout axis carrying
  only the compared pair, the fall_v2pose curve against the reference line for the model of
  record at 0.99589. None of the three held-out exams is on this axis, because they are
  measured on held-out video rather than on the train-time holdout; their values are in the
  before and after table above, where the window AUC is 0.79276 against 0.78474 on the
  multicam all-cameras headline, 0.81595 against 0.80814 on the 24 picked views, and
  0.98413 against 0.97698 on UR Fall.

Source CSVs are `final_models/fall/curves/*.csv`, 38 files, 60 epochs each.

## Full digests

Every short form used anywhere above resolves to a full sha256 either in this table, in the
arms table or in the track-tree table, so no number on this page is quoted against a hash
that exists only as a prefix.

| artefact | sha256 |
|---|---|
| Model of record, fall_v3 checkpoint | `3d2d17eed74b735b4baef6e5f84eef1d9ed6f4c2b02f14ed072c7cbe9f221526` |
| Model of record, fall_v3 graph | `67954c04eeef301ff2ab3a1238a79dba2bfa397b3eba5a93a1d3102d7e00010e` |
| Alternative, fall_v2pose checkpoint | `1ab191ae98eeb67d132eac889675f8c199dc0f9d6dc7a6aad4b2194b76bc41d3` |
| Alternative, fall_v2pose graph | `b66c3a0abc1ce214d93757a7b5b1269ea5027e3ae48f3cecbd33f7ef515e63e3` |
| Stock rtmdet-m person release checkpoint, token `235e8209` | `235e820939cb2ff33c505441e71f7e9532958c281636a963c6829d100867aed9` |
| Joints lane model of record checkpoint, token `a0a035fe` | `a0a035fe65909980447560a70ae2f8d13400c1d12490a929e7cf3c5a7b26f102` |
| Eyes lane model of record checkpoint, token `345c30a5` | `345c30a539c2b6bb665302825f418322284b98f6cbcf55d58d84ffeb75b16e7e` |

The four push-ledger arms carry their full checkpoint and graph digests in the arms table.
The fall_v2pose checkpoint digest and every ledger digest were read from the run report
each trainer wrote beside its own artefacts. For every arm that was measured on an exam,
the graph digest is confirmed a second time by the model digest recorded in that exam's own
report.

## Provenance gaps

Recorded so they can be closed rather than rediscovered.

- The fall v1 verifier `080b5d86` is recorded with eight hex characters only. Its graph is
  `outputs/verifier/fall_v1/verifier.onnx`, a build output under the untracked `outputs/`
  tree, so that head exists only where it was trained, and no run report, export sidecar or
  checkpoint manifest available in this repository carries its full digest; it should be
  captured from the training host at the next opportunity. It is the head the index in
  `final_models/README.md` quotes the 2026-08-27 detector-swap probe against, window AUC
  0.821 to 0.675 on multicam and 0.957 to 0.822 on UR Fall, and its run record is
  `research/event-verifier-v1-record.md`. No fall v1 weight is carried in this lane
  directory, and no number in the tables above is quoted against it.

## Detail

- Comparison against the newer pose, with the full exam tables:
  `research/fall-v2pose-comparison.md`.
- Push protocol, the pinned holdout and the append-only exam-look ledger:
  `research/fall-push-v3-protocol.md`.
- Seed floor: `research/fall-noise-floor.md`.
- Camera-geometry augmentation arm: `research/fall-geometric-augmentation-arm.md`.
- Threshold study and the decision-layer definitions: `research/operating-points.md`.
- Negative-side counts and their denominators: `research/event-negatives-bookkeeping.md`.
- Fall dataset licences: `research/fall-dataset-licences.md` and `DATASETS.md`.
- Measurement rules that govern every number above: `POLICY.md`.
