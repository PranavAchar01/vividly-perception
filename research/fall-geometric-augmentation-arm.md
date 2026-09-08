# Fall push v3 arm F3, camera-geometry augmentation

Dated 2026-09-05. The protocol is research/fall-push-v3-protocol.md arm F3, pre-declared.
The track-space augmentation carve-out that admits this arm is recorded in POLICY.md,
rule 4. The machine is a development Mac, CPU only, at most 2 trainings concurrent.

The lever works and is **not promoted**. The best of the six
pre-declared settings (strong, K = 8) lifts the pinned-holdout mean from the F0 control's
**0.98639** to **0.99068**, a gain of **+0.00429** against a promotion bar of **+0.00442**
(the F0 control's 5-seed spread, its two extra seeds trained alongside arm F2). It misses by **0.00013**, so **no
exam look was taken** and the exam budget is untouched. A post-hoc control shows the gain is
camera geometry and not positive oversampling, and that it is bought the same way F1 and
F2 bought theirs: by making the model louder on negatives (holdout negatives at 0.5 or above
rise from 6.0 to 10.7 windows of 151, an increase of 78 percent), which is the exact axis the
false-alarm gate killed both earlier arms on. 22 trainings (18 grid, 3 post-hoc, 1
determinism reproduction), **0 exam evaluations**.

Standing framing unchanged: an interim prototype under the carve-out in POLICY.md, rule 4,
motion signature only, never quoted as clinical detection performance.

## What was built

`src/vividly/verifier/augment.py` gains a second transform family, separate from the three
seizure-lane transforms, which are untouched and still default to off:

- **Camera geometry**, applied to the raw keypoint and box rows of a window and then
  re-featurized. It cannot be done in feature space, because a foreshortened body has a
  different posture angle, box aspect and height profile and none of those is a function
  of the others once `featurize_window` has run.
- One draw is one virtual camera, constant across the window's samples, applied in
  box-height units about the window's own median box centre, in the declared order
  **mirror, then keystone, then vertical foreshortening, then roll**:
  - `mirror` maps u to -u;
  - `trapezoid` applies the projective keystone, mapping u,v to u/(1+gv), v/(1+gv), with
    gain g in [-0.12, 0.12] per box-height (`GEO_TRAPEZOID_MAX`; the denominator is floored at
    0.5 so the vanishing line is never approached, and at g = 0.12 a point one box-height away
    divides by 0.88);
  - `s_y` vertical foreshortening in [0.55, 1.0] (`GEO_SY_MIN`, the protocol floor);
  - `roll` plus or minus 12 degrees (`GEO_ROLL_MAX_DEG`).
- **The posture angle is recomputed from the warped keypoints, never copied.**
  `torso_angles_deg` is a vectorized copy of `vividly.tracks.posture_angle_deg`, with the same
  torso midpoints, the same tilt-magnitude convention and the same gate under which a keypoint
  strictly below `kp_conf_min` fails, and a test asserts it equals that function exactly on
  random keypoints, so the copy cannot drift. Copying the recorded angle would have been the
  one fatal shortcut: foreshortening a standing body is a change of apparent torso tilt (at
  s_y = 0.55 a 45-degree torso reads 61.2 degrees, and an upright body under 12 degrees of
  roll reads 12 degrees).
- **Boxes travel through the same map** (the axis-aligned hull of the four warped corners),
  so box and pose stay mutually consistent. Deriving the box from the warped keypoints alone
  was considered and rejected: keypoint hulls are systematically tighter than detector
  boxes, so every augmented row would have carried an out-of-distribution `h_rel` and
  `aspect`, and the arm would have measured that instead of camera geometry.
- Warped rows are written on the extraction's own numeric grid (kp 0.1 px, box 0.01 px,
  angle 0.01 degrees), so an augmented row is numerically indistinguishable from a recorded one.
  The 0.1 px keypoint grid is worth at most atan(0.1/60) approximately 0.1 degrees of angle quantization, far
  below the pose estimator's own noise.
- `t` and `kpc` are untouched: this family contains no time transform and confidence is not
  a geometric quantity. `present` is therefore identical to the source window's, tested.
- Draws are online, fresh every epoch, from the augmenter's own numpy Generator, on train
  positive rows only, selected after `grouped_split`, so no augmented view of a holdout window
  can exist. That is the scope clause of the augmentation carve-out in POLICY.md, rule 4. K
  views per positive are appended, matching the protocol's K augmented draws per positive
  window; the val surface is byte-identical to the F0 surface, so the arm and the control are
  read on one ruler.

Plumbing that had to exist for any of this to be honest:

- `vividly.verifier.features.WindowSource` and `with_sources=True` on `windows_for_video` and
  `windows_from_extraction`: the labeller hands back the exact in-window rows and the
  **unrounded** `t_end` it featurized with. Re-featurizing from the rounded
  `LabeledWindow.t_end` moves the sample selection at the edges, measured at 9 of 400 le2i
  windows changed, which would have been a silent 2 percent corruption of the augmented set.
- `GeoAugmenter.verify_identity()`, run by the trainer before epoch 1: every source window
  must re-featurize to the byte-identical row the dataset already holds. **0 mismatches in
  all 21 runs** (172 source windows each). Any difference in an augmented row is therefore
  the camera and nothing else.
- `n_fallback_rows_total` in every report: a warp that cost a window its `min_present`
  floor would fall back to the original row and be counted. **0 in all 21 runs.**

**Default off is the same bytes, at training scale.** `configs/events/train_fall_F0_s20260826.yaml`
re-run under the modified trainer into a scratch out_dir reproduces `val_auc_best 0.98675`
and onnx sha256 `bb819c308f70909da1e6d1a4afe2f149e1c9227b601446af7430c7312451e587`,
identical to the recorded F0 seed-20260826 control. (`verifier.pt` differs, as it did in
the seizure reproduction, only because it pickles the config dict and that carries the
scratch `out_dir`.) Every delta below is the lever, not machine drift.

**Tests:** `tests/test_verifier_augment.py`, 37 tests (20 seizure-family and 17 new), all
passing. The new ones cover: identity at strength 0 and at s_y = 1 (byte-identical, not
`allclose`); the angle recompute equalling `posture_angle_deg`; the angle following the
geometry on a synthetic track running upright to mid-fall to fallen, and not equalling the
deliberately wrong recorded angle; foreshortening never reducing apparent tilt and mirror
never changing it, since it is a magnitude; no NaN and every channel inside the featurizer's
declared range over a 10-epoch battery at the protocol's extremes; `present` unchanged;
source-major row order; seed and epoch reproducibility; `verify_identity` catching a tampered
base; every protocol bound refused outside its range; a keypoint-less tree refused outright;
and `with_sources` leaving the windows it builds unchanged. The golden featurization hash,
prefix `14dd1831`, is untouched and still passes.

## The pre-declared ablation, 18 runs

3 strengths by K in {4, 8} by 3 seeds, on the F0 recipe (`contain` labelling, the original 8
channels, both pinned by the false-alarm-gate failures of F1 and F2), pinned holdout,
selection on the pinned holdout only. Strengths are taken word for word from the protocol;
the mild and full trapezoid settings are pinned at half of, and at all of, the module's
declared gain bound. `mirror_p` 0.5 throughout.

| setting | s1 `20260826` (onnx/ckpt) | s2 `20260901` | s3 `20260905` | mean | spread | against control 0.98639 |
|---|---|---|---|---|---|---|
| weak / k4 | 0.98565 `4a61dce9`/`e744a5fc` | 0.98896 `0927231c`/`8d200f72` | 0.99043 `24361671`/`cdeda826` | 0.98835 | 0.00478 | +0.00196 |
| weak / k8 | 0.98675 `4d49fe85`/`1bd65903` | 0.98712 `6535d7f4`/`deb74074` | 0.99043 `55a58911`/`5c588bfa` | 0.98810 | 0.00368 | +0.00171 |
| med / k4 | 0.98639 `40758de9`/`b33d3e1e` | 0.98933 `3d13906c`/`75b4a5ee` | 0.99043 `03b6e890`/`21848fc6` | 0.98872 | 0.00404 | +0.00233 |
| med / k8 | 0.98749 `54e2b038`/`932b4b81` | 0.99043 `ba2c2531`/`5fcc6331` | 0.99191 `122a30ce`/`5a19225a` | 0.98994 | 0.00442 | +0.00355 |
| strong / k4 | 0.98786 `035e2764`/`44e0eed5` | 0.99043 `f2eddfe2`/`f1a7cfec` | 0.99117 `bbf52fad`/`d10b7c1e` | 0.98982 | 0.00331 | +0.00343 |
| **strong / k8** | 0.98859 `87fec2bf`/`484b490b` | **0.99080** `119c6f84`/`2f97dc17` | 0.99264 `14b1dc50`/`045446f0` | **0.99068** | 0.00405 | **+0.00429** |

weak = s_y in [0.85, 1.0], roll plus or minus 5 degrees, no trapezoid. med = s_y in [0.7, 1.0], roll plus or minus 8 degrees,
trapezoid plus or minus 0.06. strong = s_y in [0.55, 1.0], roll plus or minus 12 degrees, trapezoid plus or minus 0.12.
The dataset is identical in all 18: 881 windows (190 positive, 691 negative), 22 groups,
pinned holdout 169 (18 positive, 151 negative), train 712, the same numbers F0 and F2 ran on.
`resolved_pos_weight` is 3.6368 everywhere, because `auto` resolves on the unaugmented
dataset exactly as in F0, so the arms differ only by the appended rows. K = 4 appends 688
rows per epoch and K = 8 appends 1376, on 172 train positives. The best epoch lands at 19 to
58 in every run, with none of the epoch 1 to 2 early-stopping pathology the seizure lane's F2
fold showed. Runs take 34 to 114 s each.

**Chosen setting: strong / K = 8**, by mean holdout AUC, the single pre-declared selection
statistic. Its median-val seed is `20260901` (0.99080, onnx `119c6f84`, ckpt `2f97dc17`).

The ordering is also readable as a dose-response, which is part of why the lever looks real
rather than seed-shaped: mean AUC rises monotonically with strength at K = 8
(weak 0.98810, med 0.98994, strong 0.99068), it rises with strength at K = 4 as well
(0.98835, 0.98872, 0.98982), and every setting is above the control. K itself is not
monotone, since weak/k8 sits below weak/k4, so the dose that matters is the geometry, not the
number of copies, which the post-hoc control below confirms directly.

## Promotion outcome: not promoted, no exam look spent

| reading of the control spread | bar | strong/k8 mean | cleared |
|---|---|---|---|
| **5-seed control spread 0.00442, the pre-declared primary reading and the wider, safer floor** | 0.99081 | 0.99068 | **no, by 0.00013** |
| 3-seed F0 seed floor 0.00184 (research/fall-noise-floor.md, reported only) | 0.98823 | 0.99068 | yes |

The two readings disagree, and the pre-declared rule governs: the look is taken only if the
holdout mean beats the control mean by more than 0.00442. A gain of +0.00429 does not, so
**no look is taken and the candidate is recorded as not promoted.** The shortfall is 3
percent of the 0.00442 promotion margin, so the result sits close enough to the boundary
that it is resolved conservatively on purpose, exactly the way the seizure push resolved
its own ambiguous reading, because an exam look is irreversible. Choosing the narrower
floor after seeing the number would be picking the ruler to fit the result.

The arm's own seed spread (0.00405 at strong/k8) is itself very nearly the size of the whole
gain (0.00429), which is the independent reason not to spend a look: three seeds cannot
separate a +0.004 effect from seed noise on this holdout.

## The negative side, and which way the trade went

F1 and F2 both died on UR lie-transition false positives, so the honest question about F3 is
not its AUC but where it puts the negatives. No exam look is available, so this is measured
where it can be measured honestly: on the pinned holdout, scored with each run's exported
ONNX and its own fitted temperature, which is the same scoring rule the exam instrument uses.
151 holdout negatives, 18 positives. **Higher is worse in every negative column.**

| arm | n | holdout AUC | neg at or above 0.5 | neg at or above 0.8 | neg mean | neg mass | pos at or above 0.8 (of 18) |
|---|---|---|---|---|---|---|---|
| F0 control (5 seeds) | 5 | 0.98639 | 6.80 | 4.60 | 0.0804 | 12.14 | 14.80 |
| F0 control (3 matched seeds) | 3 | 0.98712 | 6.00 | 4.33 | 0.0741 | 11.20 | 14.67 |
| dup control K=8 (post-hoc) | 3 | 0.98688 | 6.67 | 5.00 | 0.0821 | 12.39 | 15.33 |
| weak / k8 | 3 | 0.98810 | 10.00 | 5.00 | 0.1175 | 17.74 | 16.00 |
| med / k8 | 3 | 0.98994 | 7.67 | 4.33 | 0.0885 | 13.36 | 16.67 |
| strong / k4 | 3 | 0.98982 | 9.67 | 5.67 | 0.1059 | 15.98 | 16.33 |
| **strong / k8 (chosen)** | 3 | 0.99068 | **10.67** | 5.00 | **0.1191** | **17.98** | **17.00** |

Per seed, the chosen setting against its own control seed: `20260826` moves from 5 to 10
windows at 0.5 or above, `20260901` from 8 to 13, `20260905` from 5 to 9. The count at 0.8 or
above is flat, moving from 4.33 to 5.00, that is one extra window in three runs, while the
count at 0.5 or above rises by 78 percent and the negatives' mean score rises by 61 percent,
from 0.0741 to 0.1191.

**Reading:** the chosen setting buys its ranking gain by moving the whole score distribution
up, positives further than negatives, which is why AUC improves, but the negative mass moves
with it. Positives above 0.8 go from 14.67 to 17.00 of 18, with the median moving from 0.977
to 0.995. That is the same signature the seizure lane recorded for its own augmentation arm,
an increase of 18 percent in windows at 0.5 or above and of 64 percent at 0.8 or above, out
of fold, and it is the same trade F1 and F2 were dropped for: a more convincing downward
transition, not a better fall. On this evidence the UR lie-transition half of the false-alarm
gate, the half that failed for both earlier arms at 15 of 46 windows for the control, is
where an F3 look would most likely have been spent and lost. Recorded so it is not
re-litigated.

Two caveats stated plainly, because this is holdout evidence and not exam evidence: the
pinned holdout carries only 151 negatives and none of them is a UR ADL lie-transition, since
that class exists only on the exam surface, and the temperature is fitted per run on this very
holdout, so the absolute counts are optimistic for every arm equally. The comparison is
like for like; the extrapolation to the exam is a hypothesis, not a measurement.

## Post-hoc diagnostic: is the gain the geometry or the oversampling? Not pre-declared

K = 8 appends eight copies of every train positive, so part of the effect could be plain
positive oversampling. The identity camera (s_y = 1, no roll, no trapezoid, no mirror,
K = 8) isolates it: every appended row is then a byte-identical copy of its positive, so the
model sees each train positive 9 times per epoch and nothing else changes.
`configs/events/train_fall_F3geo_dupctrl_k8_s*.yaml`, written and run after the grid, taking
no part in selection:

| arm (3 matched seeds) | s1 | s2 | s3 | mean | against the 3-seed control |
|---|---|---|---|---|---|
| F0 control | 0.98675 | 0.98823 | 0.98639 | 0.98712 | reference |
| dup control K=8 | 0.98418 `691ea07e` | 0.98786 `03fdb5f6` | 0.98859 `ee6ad285` | 0.98688 | **-0.00025** |
| strong / k8 | 0.98859 | 0.99080 | 0.99264 | 0.99068 | **+0.00356** |

Positive oversampling on its own is worth **nothing** (-0.00025, and its negative side is
the control's: 6.67 windows at 0.5 or above against 6.00). The whole of the +0.0036 AUC
movement and the whole of the +4.0 windows of extra negative loudness belong to the camera
geometry. The lever is real; it is the bar it fails, not the mechanism.

## Files

- `src/vividly/verifier/augment.py`: the camera-geometry family (`GeoSpec`, `GeoDraw`,
  `torso_angles_deg`, `warp_points`, `warp_track_window`, `GeoAugmenter`,
  `make_geo_augmenter`). The three seizure transforms are untouched.
- `src/vividly/verifier/features.py`: `WindowSource` and `with_sources`, both additive; the
  windows, their order and their contents are unchanged, tested.
- `scripts/events/train_event_verifier.py`: the `geo_augment:` config key, defaulting to off,
  `build_dataset(..., with_sources=)`, the pre-epoch identity guard, and `geo_augment` in
  the report. It refuses `negative_class_weights` and `geo_augment` together, because the
  weight vector is per original row while the lever appends rows.
- `configs/events/train_fall_F3geo_{weak,med,strong}_k{4,8}_s{20260826,20260901,20260905}.yaml`
  (18, pre-declared) and `train_fall_F3geo_dupctrl_k8_s*.yaml` (3, post-hoc).
- `tests/test_verifier_augment.py`: 37 tests.
- `final_models/fall/curves/fall_push_F3geo_*.csv`: 21 per-epoch curves (18 grid, 3
  post-hoc), added to the `RUNS` table in `scripts/analysis/extract_training_curves.py` and
  regenerated.
- `outputs/verifier/fall_F3geo_*/` (21 runs, gitignored; every hash above is from their
  report.json), and `outputs/verifier/_repro/fall_F0_s20260826_geo_module/` (the byte-identity
  reproduction).

## Provenance gaps

Recorded so they can be closed rather than rediscovered.

This page carries 37 digests recorded with eight hex characters and nothing longer: the ONNX
and checkpoint pair of every one of the 18 grid runs except the chosen setting, and the three
ONNX digests of the post-hoc duplicate control. None of the 37 resolves to a full digest
anywhere in this repository, because the lane page carries the chosen setting alone and
the `outputs/verifier/fall_F3geo_*/report.json` files every one of them was read from are
gitignored. They should be captured from those run directories at the next opportunity. Until
they are, a prefix on this page identifies a run rather than a graph, and no number here may
be attributed to a graph on the strength of one.

Two exceptions are complete. The chosen setting's checkpoint
`2f97dc174ebfffaa1d4143f5200beddd5f7e77a45c6f63220d6ed265e8155c49` and graph
`119c6f84ab10fb0c05e86f4ef8fa04cf2e7615603e38ea74733e7e2f1c7f1b68` are carried on
`final_models/fall/README.md`, and the full value behind the golden featurization hash
`14dd1831` is asserted in `tests/test_verifier_augment.py`.

## What is not claimed

- Not that camera-geometry augmentation does not work: it moves the pinned holdout by
  +0.0043 with a clean dose-response and a clean attribution, on 3 seeds. The claim is that
  this movement is smaller than the control spread the protocol fixed as the price of a look.
- No number here is an exam number. ur_fall, multicam and multicam_allcams were not scored
  for any F3 run; the push's exam budget is where F2 left it, at 12 looks.
- No number here is clinical detection performance. Every model is an interim prototype under
  the carve-out in POLICY.md, rule 4.
