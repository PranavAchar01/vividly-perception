# Push 2 items 2-3: hard-negative weights and track-space augmentation

Dated 2026-09-05. The protocol is research/seizure-push-2-protocol.md (items 2 steps 2-3, and item 3),
authorised by research/seizure-push-2-out-of-fold-gate.md (item 2 trigger met: 75.4% of
top-decile false-alarm mass is class (a), pat04 seizure-free recordings) and
the track-space augmentation carve-out now recorded in POLICY.md, rule 4.
The machine is a development Mac, processor only, torch 2.7.1 / numpy 2.4.6, about 75 to 100 s per training.

Both levers are recorded and **neither is promoted**. Hard-negative
weighting fails the F2 gate at alpha 2 and at alpha 4 (and triples the fold's seed spread);
track-space augmentation clears F2 by +0.0103 but moves the full 4x3 grid by **+0.00129**
against a pre-declared bar of +0.03075, and out of fold it is measurably louder on the very
seizure-free recordings the push set out to quiet (+18% windows at or above 0.5, +64% at
or above 0.8).
**No exam look was spent; the push's two-candidate budget is still fully unspent.**
20 full trainings (18 arm runs + 2 determinism reproductions), 0 exam evals.

Standing framing unchanged: interim prototype, motion signature only, never quoted as
clinical detection performance, and no claim that the system detects seizures.

**Selection is train-side only.** Every run below trains on the eight rotation-pool
patients (pat01/02/04/05/09/11/13/14) and is judged on that fold's own held-out
patients. Exam patients, E1 pat06/08/10/12 and E2 pat03/07, are never built, never
scored, and no exam eval config runs anywhere in the selection phase.

## Pre-declarations (written before any arm result was read)

**Baseline of record.** Arm-B grid mean **0.8013** over 4 folds x 3 seeds
(configs/events/rot_train_conv_rotB_f*_s*.yaml), per-fold means / seed spreads:

| fold | held out | s1 | s2 | s3 | mean | spread |
|---|---|---|---|---|---|---|
| F1 | pat01+pat04 | 0.81623 | 0.83886 | 0.83567 | 0.83025 | 0.023 |
| F2 | pat02+pat09 | 0.75479 | 0.77060 | 0.75569 | **0.76036** | 0.016 |
| F3 | pat05+pat11 | 0.71629 | 0.70875 | 0.70215 | 0.70906 | 0.014 |
| F4 | pat13+pat14 | 0.90633 | 0.87013 | 0.94000 | 0.90549 | 0.070 |
| **grid** | | | | | **0.80129** | |

**Determinism proof (ran first).** `rot_train_conv_rotB_f2_s1.yaml` was re-run on that machine
with the unmodified trainer and again with the modified trainer, both into scratch
out_dirs: `val_auc_best 0.75479` and onnx sha256 `b9e04d03...` **both times, identical
to the recorded arm-B f2_s1 / final B checkpoint**. The default-off path is therefore
not merely equivalent; it is the same bytes, and every delta below is a real effect
of the lever, not machine drift. (`verifier.pt` differs only because it pickles the
config dict, which carries the scratch `out_dir`.)

**Fold-F2-first rule (protocol item 3 order).** F2 has the tightest seed spread
(0.016). An arm advances to the full 4-fold x 3-seed grid only if its F2 mean over the
three seeds beats the F2 baseline mean **0.76036 at all**. The combined arm runs only
if both single levers clear that F2 gate.

**Promotion bar (fixed here, before any grid ran).** The protocol requires the grid
mean to beat arm B by more than the pooled fold spread. That phrase is read
conservatively, and the conservative reading governs, because an exam look is irreversible and the push allows at most two
promoted candidates of which the ensemble consumed none:

- **pooled fold spread = mean of the four recorded per-fold spreads = (0.023 + 0.016 +
  0.014 + 0.070)/4 = 0.03075.**
- **Promotion bar: grid mean > 0.80129 + 0.03075 = 0.83204.**
- Reported alongside but never used to override: the in-quadrature reading
  sqrt(sum s_i^2)/4 = 0.01917 (bar 0.82046), and the arm's own measured seed-wise
  grid-mean spread (the spread of the three per-seed grid means, which is a direct
  estimate of grid-mean seed noise rather than an assumed one).
- If an arm's grid mean lands between the two readings, the verdict is **not
  promoted**, and the ambiguity is recorded explicitly rather than resolved in favour
  of spending a look.

**pos_weight** is `auto` in every config, identical to arm B, and resolves to
`min(10.0, 6975/387) = 10.0`, pinned at 10.0 in every run, recorded as
`resolved_pos_weight` in each report.json.

## What was built

### Lever 1: per-window hard-negative weights (item 2 steps 2-3)

`scripts/events/train_event_verifier.py`, additive and default-off:

- `build_dataset(cfg, classify_negatives=...)` tags every window with the negative
  class the OOF diagnosis used: `seizure_free` (a: the recording carries no event at
  all), `owner_offevent` (b), `nonowner_track` (c), `forced_negative` (d: any source
  with `force_label`, the AJILE12 diet). (d) is kept separate on purpose: the AJILE12
  metas also carry an empty `events_t`, and folding them into (a) would silently
  reweight the whole 2250-window diet. The class resolution reuses
  `features.normalize_events` and `features._event_track`, the same functions that
  produce the labels, so the classes cannot drift from the labelling.
- **Verification against the OOF diagnosis:** on the arm-B sources the classifier
  returns 5112 WU windows / 387 positive / `seizure_free` **3219**, `owner_offevent`
  **163**, `nonowner_track` **1343**; every count is identical to the item-2 table in
  research/seizure-push-2-out-of-fold-gate.md.
- `negative_class_weights: {seizure_free: alpha}` (absent by default) sets a per-window
  loss weight; the loss becomes `(elementwise_bce * w).sum() / w.sum()` per batch. The
  **normalization by the batch's own weight sum is deliberate**: with the plain
  `(bce*w).mean()` an alpha of 2 on 3219 of 6460 train windows inflates total gradient
  magnitude about 1.5 times, which is a hidden learning-rate change; normalizing makes
  alpha a pure re-weighting so the ablation measures the lever and not an LR confound.
- Weighting is applied to train rows only. The val surface stays the unweighted
  patient-held-out AUC arm B was selected on, so both arms are read on one ruler.
- For fold F2 all 3219 `seizure_free` windows are train-side (they are pat04's, and
  F2 holds out pat02+pat09), mean train weight 1.4983 at alpha 2.

### Lever 2: track-space augmentation (item 3)

`src/vividly/verifier/augment.py`. This arm adds exactly three track-space transforms and no
more (the camera-geometry family in the same module was added later, by the fall arm):

1. **mirror**: negate `vx` only (channel resolved by name, never by index).
2. **time-warp**: bounded in-window resample about the window centre, factor
   r in [0.85, 1.18], enforced by `AugmentSpec.__post_init__` (a config outside the
   declared band raises). Nearest-step sampling, so `present`/`angle_valid`/every
   `*_valid` stay binary and r = 1.0 is the exact identity. Velocity channels are
   scaled by 1/r (a warped window is a different time base) and re-clipped to the
   featurizer's own bounds, so no augmented window leaves the trained value range.
3. **contiguous observation dropout**: one contiguous run of steps is marked
   unobserved, writing exactly what `featurize_window` writes at `present = 0`
   (0.0 everywhere except `h_rel`/`aspect` = 1.0). The run length is bounded by
   `floor(present_sum - 0.5 * n_steps)`, which keeps the window's `present` fraction
   at or above 0.5 no matter which steps the run covers; on a window already below 0.5
   (min_present is 0.35) the transform is a no-op.

Online per-epoch draws, train rows only, built after `grouped_split` and fed only the
train slice, so no augmented view of a holdout window can exist (the augmentation scope
clause in POLICY.md, rule 4).
The augmenter draws from its own numpy Generator and never the global numpy/torch RNG,
so the batch-permutation stream is the baseline's. Not implemented in this arm, by protocol:
track shatter, positive-only jitter, amplitude noise, camera-geometry warps (the
camera-geometry family reached the module later, with the fall arm).

### Tests (written before the arms ran)

`tests/test_verifier_augment.py`, the 20 tests this arm added, **20 passed**:

| claim | tests |
|---|---|
| default off | `make_augmenter` returns None for `None`/`{}`/`False`/`enabled: false`; `build_dataset(cfg)[3] is None` |
| featurization byte-identity | golden sha256 `14dd1831...` of the featurizer's output on a fixed fixture (features.py is untouched by this change) |
| identity round-trip | zero probabilities are byte-identical (`array_equal`, not `allclose`); time-warp r = 1.0 is exact; mirror is involutive and touches vx only; input array never mutated |
| label invariance | row order and count preserved (fingerprint in an untouched scene column); augmenter signature takes no label argument |
| no NaN | 30 epochs x 16 windows with all three transforms at p = 1: all finite, vy/vx within +/-10, `present`/`angle_valid` binary |
| present floor | never crossed over 20 epochs; no-op on a window already below the floor |
| protocol bounds | warp range outside [0.85, 1.18] refused; unknown augment key refused |
| negative classes | class names are the OOF's four; typo'd class name and alpha at or below 0 refused; a `force_label` source is never called `seizure_free` |

## F2 ablation (single fold, 3 seeds each)

Fold F2 = held out pat02 + pat09, 902 val windows / 136 positive. All three arms use the
same three seeds as arm B (20260826 / 20260901 / 20260905) and the same held-out val
surface, so the columns are directly comparable. Selection statistic = `val_auc_best`,
the same statistic arm B was selected on.

| arm | s1 `20260826` | s2 `20260901` | s3 `20260905` | mean | spread | against baseline 0.76036 | F2 gate |
|---|---|---|---|---|---|---|---|
| baseline arm B | 0.75479 `b9e04d03` | 0.77060 `12aeb901` | 0.75569 `4e0890a2` | **0.76036** | 0.016 | not applicable | not applicable |
| HN alpha 2 | 0.76947 `52e73573` | 0.74650 `b4e3d2e1` | 0.73416 `bc5953f0` | 0.75004 | 0.035 | **-0.01032** | **fail** |
| HN alpha 4 | 0.76836 `03244c7a` | 0.77189 `3f0dce42` | 0.72608 `fb007a0d` | 0.75544 | 0.046 | **-0.00492** | **fail** |
| AUG (track-space) | 0.77765 `ea34d95a` | 0.78426 `5636606e` | 0.75008 `dac7542a` | **0.77066** | 0.034 | **+0.01030** | **pass** |

Hashes are the onnx sha8 of each run; checkpoint sha8s are in each run's report.json
under `outputs/verifier/conv_p2{hn2,hn4,aug}/f2_s*/`. `resolved_pos_weight` is 10.0 in
every row.

**Verdicts.**

- **Hard-negative weighting (item 2 steps 2-3) is closed.** Neither alpha improves the
  F2 fold: alpha 2 is -0.0103 and alpha 4 is -0.0049 against the baseline fold mean,
  and both roughly triple the fold's seed spread (0.035 / 0.046 against 0.016). Weighting
  the class that carries 75.4% of the out-of-fold false-alarm mass makes this recipe
  less stable, not more discriminative, on unseen patients. Per the pre-declared order
  neither alpha advances to the full grid.
- **The combined arm was not run.** The protocol conditions it on both single levers
  clearing F2; only one did. `configs/events/rot_train_conv_p2combo*` were never
  generated and no combined run exists.
- **Track-space augmentation advances** to the full 4-fold x 3-seed grid on the strength
  of +0.01030 on the F2 fold mean, which is what the pre-declared gate, an improvement on
  the F2 fold at all, asks for, while noting up front that +0.0103 is smaller than the
  fold's own baseline seed spread (0.016) and the arm's own spread is 0.034, so the F2
  signal alone certifies nothing. That is exactly why the promotion decision is taken
  on the full grid against a spread-sized bar, not here.

**Structural note that governs how any HN grid mean would have read.** For fold F1 the
held-out patients are pat01 + **pat04**, and pat04 owns every `seizure_free` window in
the pool. Train-side `seizure_free` counts per fold: **F1 0**, F2 3219, F3 3219, F4 3219.
The HN lever is therefore a structural no-op on F1 (its loss reduces to the unweighted
one up to float-summation order), so an HN grid mean would have been three-quarters
lever and one-quarter baseline. Recorded because it also bounds how much the lever could
ever have moved a grid mean, independent of the F2 result above.

**Selection-instrument caveat (applies to arm B as much as to these arms).** On fold F2
`val_auc_best` is reached at **epoch 1-2 in every arm, baseline included** (arm B f2_s1
epoch 2, f2_s2 epoch 1, f2_s3 epoch 2; HN2 f2_s1 epoch 2), after which val AUC decays to
about 0.67 by epoch 60. The recipe's F2 number is therefore a very-early-stopping statistic
on a noisy curve, which is a plausible source of the large seed spreads seen here. This
is a property of the arm-B recipe that this ablation surfaced; it is not a reason to
change the selection rule mid-protocol, and none was changed.

## Full grid: track-space augmentation (4 folds x 3 seeds)

Same 4 pre-declared folds, same 3 seeds, same recipe as arm B in every respect except
the `augment:` block. 12 runs, all under `outputs/verifier/conv_p2aug/f*_s*/`.

| fold | held out | s1 `20260826` | s2 `20260901` | s3 `20260905` | AUG mean | arm-B mean | delta | AUG spread |
|---|---|---|---|---|---|---|---|---|
| F1 | pat01+pat04 | 0.80110 `a652fdbc` | 0.84540 `cd35ad4c` | 0.86036 `429da878` | 0.83562 | 0.83025 | +0.00537 | 0.059 |
| F2 | pat02+pat09 | 0.77765 `ea34d95a` | 0.78426 `5636606e` | 0.75008 `dac7542a` | 0.77066 | 0.76036 | +0.01030 | 0.034 |
| F3 | pat05+pat11 | 0.68587 `f0911bb2` | 0.69385 `6a936f88` | 0.69599 `2e99ab3f` | 0.69190 | 0.70906 | **-0.01716** | 0.010 |
| F4 | pat13+pat14 | 0.89468 `7e04ac8e` | 0.90329 `92a6e051` | 0.93848 `229ec079` | 0.91215 | 0.90549 | +0.00666 | 0.044 |
| **grid** | | | | | **0.80258** | **0.80129** | **+0.00129** | |

Checkpoint sha8s: f1 `ec3b65f8`/`738b56a0`/`0a50909c`, f2 `8143cffb`/`c6f9c2a3`/`db9fa913`,
f3 `d53d6134`/`5ea967eb`/`03576158`, f4 `abb99c49`/`ce00247d`/`8c12ea1a`.
`resolved_pos_weight` 10.0 in all 12.

### Promotion decision: not promoted, no exam look spent

| reading of the pooled fold spread | value | bar | AUG grid mean | cleared |
|---|---|---|---|---|
| mean of the four per-fold spreads (**the pre-declared primary**) | 0.03075 | 0.83204 | 0.80258 | **no** |
| in quadrature, sqrt(sum s_i^2)/4 (secondary, reported only) | 0.01917 | 0.82046 | 0.80258 | **no** |

The gain is **+0.00129**, which is 4% of the primary pooled fold spread of 0.03075 and
7% of the in-quadrature reading of 0.01917. There is no ambiguity between readings to
adjudicate: the arm fails both by more than an order of magnitude. And the arm's own
measured grid-mean seed noise is larger than its gain in both arms: per-seed grid means
are 0.78982 / 0.80670 / 0.81123 for AUG (spread **0.02140**) and 0.79841 / 0.79708 /
0.80838 for arm B (spread 0.01129). The +0.00129 is 6% of the augment arm's own seed
spread.

**Verdict: no candidate is promoted from push-2 items 2-3. No exam look is taken;
E1 and E2 were never scored, and the push's two-candidate budget is still fully
unspent (the ensemble gate consumed none, and neither does this).**

Sign consistency is also absent: augmentation helps F1/F2/F4 by +0.005 to +0.010 and
hurts F3 by -0.017, which is the pattern of reseeding rather than of a lever. It also
widens the seed spread on the two folds where it helps (F1 0.023 -> 0.059, F2 0.016
-> 0.034) while the fold it hurts is the one where it tightens (F3 0.014 -> 0.010).

### The false-alarm read, which is the point of the push

No exam look is available, so the false-alarm question is answered where it can be
answered honestly: on the selection surface, out of fold. Fold F1 holds out pat01 +
**pat04**, and pat04 owns every seizure-free recording in the pool, so the F1 3-member
ensembles of each arm give a proper out-of-fold read of exactly class (a), the class
that carries 75.4% of the top-decile false-alarm mass and authorised this whole item.
Instrument: `outputs/verifier/push2_oof/run_push2_aug_oof.py` (same window build, same
mean-of-calibrated-logits rule as the item-1/2 runner), output
`outputs/verifier/push2_oof/aug_oof_summary.json`.

**Regression first:** the arm-B column reproduces the recorded gate-doc numbers exactly
(pat04__free mass 980.105 / 1058 at or above 0.5 / 44 at or above 0.8; pat04__no-Sz2P
615.489 / 874 / 1; class (a) total mass 1595.594 against the recorded 1595.6), which is a
**pass on both recordings**, so the augment column below is measured on a
verified-identical surface.

Class (a) label-0 windows (3219 of them; higher is worse in every column):

| arm (F1 members) | score mass | n at or above 0.5 | n at or above 0.8 | pat04__free mean | pat04__no-Sz2P mean | F1 ens AUC |
|---|---|---|---|---|---|---|
| arm B `48aea67c`/`d4e2443f`/`df7b4de1` | 1595.59 | 1932 | 45 | 0.4685 | 0.5461 | 0.84414 |
| AUG `a652fdbc`/`cd35ad4c`/`429da878` | **1802.38** | **2283** | **74** | 0.5389 | 0.5989 | 0.85720 |
| change | **+13.0%** | **+18.2%** | **+64.4%** | +0.070 | +0.053 | +0.013 |

Per recording, arm B -> AUG: pat04__free mass 980.1 -> 1127.4, at or above 0.5 1058 ->
1294, at or above 0.8 44 -> 66; pat04__no-Sz2P mass 615.5 -> 675.0, at or above 0.5
874 -> 989, at or above 0.8 1 -> 8. The on-video non-event windows move the same way
(pat01__Sz2PG at or above 0.8 10 -> 14).

**This is the finding that matters.** Track-space augmentation buys a ranking gain that
is inside seed noise (+0.0013 grid) and pays for it by making the model measurably
louder on the exact seizure-free recordings the push set out to quiet: +18% more
windows over 0.5 and +64% more over 0.8. A candidate that raises AUC by shifting the
whole score distribution upward is not a false-alarm improvement, and this one would
have arrived at the exam with a worse negative side than final B. Promoting it would
have spent an irreversible look on a regression.

### Files written

- `src/vividly/verifier/augment.py`: the three track-space transforms, default off.
- `scripts/events/train_event_verifier.py`: `negative_class_weights:` and `augment:`
  keys, negative-class tagging, `resolved_pos_weight` in the report. Additive; the
  default path is byte-identical (proved by the `b9e04d03` reproduction above).
- `tests/test_verifier_augment.py`: the 20 tests this arm added, all passing.
- `configs/events/rot_train_conv_p2{hn2,hn4}_f2_s{1,2,3}.yaml` (6, F2 only, the
  lever closed there) and `rot_train_conv_p2aug_f{1,2,3,4}_s{1,2,3}.yaml` (12).
- `outputs/verifier/conv_p2{hn2,hn4}/f2_s*` (6 runs), `outputs/verifier/conv_p2aug/f*_s*`
  (12 runs), gitignored; every hash above is from their report.json.
- `outputs/verifier/push2_oof/run_push2_aug_oof.py` + `aug_oof_summary.json`, also gitignored.
- `final_models/seizure/curves/seizure_p2_*.csv`: 18 per-epoch curves, added to
  `scripts/analysis/extract_training_curves.py` RUNS and regenerated.

### What is not claimed

- Nothing here says augmentation cannot work; it says that this augmentation, at these draw
  probabilities, on this recipe and these 8 patients, moves the rotation grid by less
  than seed noise and costs false-alarm quiet. The three transforms and their bounds
  were fixed in advance and were not tuned after seeing a result, and no probability, no
  warp bound and no dropout cap was changed between the F2 ablation and the grid.
- The hard-negative result is not a finding that hard negatives are useless; it is that per-window
  reweighting of class (a) at alpha 2 and 4, with pos_weight pinned at 10.0 and the
  loss re-normalized, does not improve unseen-patient ranking on the tightest fold and
  triples its seed spread.
- No number in this document is clinical detection performance. Every model is an
  interim prototype under the carve-out in POLICY.md, rule 4.

## Provenance gaps

Recorded so they can be closed rather than rediscovered.

This page carries 35 digests recorded with eight hex characters and nothing longer: the ONNX
digests in the F2 ablation table and in the full grid table, the 12 checkpoint digests listed
beneath the grid, and the three arm B member digests quoted in the class (a) table. None of
the 35 resolves to a full digest anywhere in this repository, because no run behind them
was promoted, so the lane page carries none of them, and the `report.json` files under
`outputs/verifier/conv_p2{hn2,hn4,aug}/` and `outputs/verifier/conv_rotB/` that hold the full
values are gitignored. They should be captured from those run directories at the next
opportunity. Until they are, a prefix on this page identifies a run rather than a graph, and
no number here may be attributed to a graph on the strength of one.

The one exception is the baseline reproduction `b9e04d03`, whose full digest is carried on
`final_models/seizure/README.md`.
