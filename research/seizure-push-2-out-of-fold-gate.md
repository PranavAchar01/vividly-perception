# Push 2 items 1 and 2: within-fold seed-ensemble gate and out-of-fold hard-negative diagnosis

Dated 2026-09-05. The protocol is research/seizure-push-2-protocol.md (pre-declared). The
runner is outputs/verifier/push2_oof/run_push2_oof.py, on a development Mac, processor only
(onnxruntime 1.28.0, CPUExecutionProvider). Per-window cache:
outputs/verifier/push2_oof/windows.jsonl; machine-readable summary:
outputs/verifier/push2_oof/summary.json.

Standing framing: interim prototype, motion signature only, never clinical detection
performance.

**Selection surface only.** Every window comes from
outputs/events/wu_sahzu_tracks_v2pose restricted to the eight rotation-pool patients
(pat01/02/04/05/09/11/13/14). Exam patients, E1 pat06/08/10/12 and E2 pat03/07, were
never built and never scored; no exam eval config ran. Zero exam looks spent by this job.

Window building reuses `vividly.verifier.features.windows_from_extraction` with the
arm-B WindowSpec taken unchanged from the 12 fold configs (window_s 12.0, stride_s 3.0,
n_steps 64, policy overlap, overlap_min_s 4.0, min_present 0.35, scene_channels true;
all 12 `configs/events/rot_train_conv_rotB_f*_s*.yaml` verified identical except
`holdout_locations` and `train.seed`), the same code path
scripts/events/train_event_verifier.py and eval_event_verifier.py use. Labels come
from each meta's `events_t` under the overlap policy, exactly as in training.

## Regression check (ran first, gate read only after it passed)

Single member f2_s1 (onnx sha8 `b9e04d03`, T = 1.0251) scored on its own fold-2
held-out windows (pat02+pat09, 902 windows / 136 positive):

| | target (report val_auc_best) | measured | \|delta\| | tolerance | outcome |
|---|---|---|---|---|---|
| f2_s1 `b9e04d03` window AUC | 0.75479 | 0.754790 | 2.9e-08 | 1e-3 | **pass** |

Beyond the pinned check, all 12 recomputed single-member AUCs match their own
report.json `val_auc_best` to every recorded decimal (see the report column in the
tables below), so the window build and scoring path reproduce the training-time
validation surface exactly.

## Item 1: per-fold seed-ensemble gate

Ensemble rule (fixed by the protocol): per window,
score = sigmoid(mean over the fold's 3 members of (logit / T_member)), T from each
member's report.json calibration. Every member's report records a temperature, so no
T=1 fallback was needed; three members (f1_s2, f1_s3, f3_s2) record T = 1.0 with
`degenerate_fit: true`, and that recorded 1.0 is what was used. Mirror-vx TTA
(vx channel negated, probability averaged with the plain pass) is the separate
column; it never enters the gate decision.

### F1, held out pat01 and pat04 (3826 windows, 55 positive)

| member | onnx sha8 | T | window AUC | report | +TTA |
|---|---|---|---|---|---|
| f1_s1 | `48aea67c` | 1.8114 | 0.81623 | 0.81623 | 0.82389 |
| f1_s2 | `d4e2443f` | 1.0 (degenerate) | 0.83886 | 0.83886 | 0.84007 |
| f1_s3 | `df7b4de1` | 1.0 (degenerate) | 0.83567 | 0.83567 | 0.84553 |
| mean member | | | 0.83025 | | |
| **ensemble** | | | **0.84414** | | 0.84903 |

delta = +0.01389 against seed spread 0.023: **fail**

### F2, held out pat02 and pat09 (902 windows, 136 positive)

| member | onnx sha8 | T | window AUC | report | +TTA |
|---|---|---|---|---|---|
| f2_s1 | `b9e04d03` | 1.0251 | 0.75479 | 0.75479 | 0.76538 |
| f2_s2 | `12aeb901` | 0.5385 | 0.77060 | 0.77060 | 0.77358 |
| f2_s3 | `4e0890a2` | 0.9284 | 0.75569 | 0.75569 | 0.76091 |
| mean member | | | 0.76036 | | |
| **ensemble** | | | **0.76842** | | 0.77497 |

delta = +0.00806 against seed spread 0.016: **fail**

### F3, held out pat05 and pat11 (255 windows, 146 positive)

| member | onnx sha8 | T | window AUC | report | +TTA |
|---|---|---|---|---|---|
| f3_s1 | `2762e668` | 3.2813 | 0.71629 | 0.71629 | 0.70925 |
| f3_s2 | `ea54b814` | 1.0 (degenerate) | 0.70875 | 0.70875 | 0.71252 |
| f3_s3 | `cb928653` | 2.5618 | 0.70215 | 0.70215 | 0.70221 |
| mean member | | | 0.70906 | | |
| **ensemble** | | | **0.71453** | | 0.71660 |

delta = +0.00547 against seed spread 0.014: **fail**

### F4, held out pat13 and pat14 (129 windows, 50 positive)

| member | onnx sha8 | T | window AUC | report | +TTA |
|---|---|---|---|---|---|
| f4_s1 | `cea0c7f4` | 1.1601 | 0.90633 | 0.90633 | 0.90734 |
| f4_s2 | `b150f9a8` | 0.9057 | 0.87013 | 0.87013 | 0.87114 |
| f4_s3 | `bbd8ee55` | 1.0771 | 0.94000 | 0.94000 | 0.94253 |
| mean member | | | 0.90549 | | |
| **ensemble** | | | **0.91544** | | 0.92025 |

delta = +0.00996 against seed spread 0.070: **fail**

### Gate outcome

| fold | delta (ensemble minus mean member) | seed spread | pass |
|---|---|---|---|
| F1 | +0.01389 | 0.023 | no |
| F2 | +0.00806 | 0.016 | no |
| F3 | +0.00547 | 0.014 | no |
| F4 | +0.00996 | 0.070 | no |

**0 of 4 folds pass, 3 were required, so the gate fails. The ensemble lever is closed,
P2-ENS is not authorised, and no exam look is spent on it.**

Reading: the ensemble helps on every fold (all deltas positive, +0.005 to +0.014,
and mirror-vx TTA adds a further +0.002 to +0.007 on top in 11 of 12 member columns
and all 4 ensemble columns), but the gain never exceeds what reseeding alone
produces. It beats the fold's best single member only on F1. A consistent
seed-noise-sized gain is exactly what the pre-declared pass rule exists to screen
out: it could not be certified on E2 (Hanley-McNeil SE approximately 0.035) and does not warrant
an exam look.

## Item 2: out-of-fold hard-negative diagnosis (same scoring pass)

Every rotation-pool window carries the mean-of-3 ensemble score of the fold that
held its patient out (F1 members `48aea67c`/`d4e2443f`/`df7b4de1` score pat01+pat04,
F2 `b9e04d03`/`12aeb901`/`4e0890a2` score pat02+pat09, F3
`2762e668`/`ea54b814`/`cb928653` score pat05+pat11, F4
`cea0c7f4`/`b150f9a8`/`bbd8ee55` score pat13+pat14). Surface: 5112 windows total,
387 positive, 4725 label-0.

False-alarm mass among the 4725 label-0 windows, by negative class:

| class | n windows | score mass (sum) | n at or above 0.5 | n at or above 0.8 | mean score |
|---|---|---|---|---|---|
| (a) pat04 seizure-free recordings | 3219 | 1595.6 | 1932 | 45 | 0.496 |
| (b) owner-track pre-onset / post-event on seizure recordings | 163 | 41.5 | 12 | 1 | 0.255 |
| (c) non-owner (staff/other) tracks on seizure recordings | 1343 | 386.1 | 308 | 26 | 0.287 |
| (d) AJILE12 diet windows | 0 | none | none | none | none |

**(d) confirmation:** AJILE12 does not flow through this path. The out-of-fold surface
is built solely from outputs/events/wu_sahzu_tracks_v2pose; the arm-B configs list
outputs/events/ajile12_neg_tracks_v1 as a second training source (force_label 0),
and no window from it was built or scored here. WU only, as the protocol requires.

Class (a) is entirely pat04 (the only rotation-pool patient with seizure-free
recordings), so all of it is scored by the F1 ensemble, which held pat04 out, and that
is a proper out-of-fold read. By recording:

| recording | n windows | score mass | n at or above 0.5 | n at or above 0.8 |
|---|---|---|---|---|
| pat04__free | 2092 | 980.1 | 1058 | 44 |
| pat04__no-Sz2P | 1127 | 615.5 | 874 | 1 |

Scene-channel breakdown (mean in-window active-track count from the window's own
scene_n_tracks channel; the extraction meta carries no separate scene level):

| mean scene tracks | n windows | score mass | n at or above 0.5 | n at or above 0.8 | mean score |
|---|---|---|---|---|---|
| at or below 1.5 | 78 | 15.6 | 0 | 0 | 0.200 |
| 1.5 to 3 | 179 | 74.6 | 65 | 4 | 0.417 |
| 3 to 6 | 4346 | 1880.5 | 2136 | 67 | 0.433 |
| > 6 | 122 | 52.5 | 51 | 1 | 0.430 |

### Class-concentration answer (the trigger for steps 2 and 3)

Top decile of label-0 windows by ensemble score = 473 windows, total mass 356.3:

| class | top-decile mass | share | n windows |
|---|---|---|---|
| (a) pat04 seizure-free recordings | 268.5 | **75.4%** | 360 |
| (c) non-owner tracks | 85.4 | 24.0% | 110 |
| (b) owner off-event | 2.4 | 0.7% | 3 |

**Yes: 75.4% of the top-decile false-alarm mass sits in one class, (a) pat04 seizure-free
recordings.** That is at or above the protocol's 60%, so its trigger for steps 2 and 3
(per-window weights alpha in {2,4}, curriculum, train-side selection only) is met. Within
class (a) the mass splits pat04__free 204.8 against pat04__no-Sz2P 63.7; within class (c)
the largest single contributor is pat01__Sz2PG non-owner tracks (46.8). Note the base
rate behind the trigger: the F1 ensemble puts 60% of pat04's seizure-free windows above 0.5
(mean score 0.496), mostly mid-range scores, with only 45 of 3219 clearing 0.8.

## Provenance gaps

Recorded so they can be closed rather than rediscovered.

The twelve ensemble members are identified by eight hex characters of their ONNX digest, and
eleven of the twelve resolve to nothing longer anywhere in this repository: `48aea67c`,
`d4e2443f`, `df7b4de1`, `12aeb901`, `4e0890a2`, `2762e668`, `ea54b814`, `cb928653`,
`cea0c7f4`, `b150f9a8` and `bbd8ee55`. None of the eleven was promoted, so the lane page
carries none of them, and the `report.json` files under `outputs/verifier/conv_rotB/` that
hold the full values are gitignored. All eleven should be captured from those run directories
at the next opportunity. Until they are, a prefix on this page identifies a run rather than a
graph. The twelfth member, `b9e04d03`, is complete: its full digest is carried on
`final_models/seizure/README.md`.

## Files

Everything below sits under outputs/, which is gitignored, so these are build outputs of the
recorded run rather than files carried in this repository.

- outputs/verifier/push2_oof/windows.jsonl: 5112 rows carrying fold, key, location,
  track_id, t_end, label, owner_track, neg_class, per-member scores, ensemble score,
  TTA score and scene mean tracks.
- outputs/verifier/push2_oof/summary.json: everything above, machine-readable.
- outputs/verifier/push2_oof/run_push2_oof.py: the runner (reuses
  vividly.verifier.features; no training code touched).

No exam surface was built or scored. Item 1 outcome: **the lever is closed**.
Item 2 outcome: **the trigger is met, and steps 2 and 3 are authorised under train-side
selection**.
