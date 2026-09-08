# H2h: the AJILE12 channel-masking control

Dated 2026-09-05. Scope: report only, pre-declared in
`research/seizure-push-2-protocol.md` item 4, which admits a channel-masking control of
about 20 minutes at the 2h dose, named H2h, as report-only work. **No exam look was taken
and none was possible:** every run below is scored on the arm-B fold-F2 patient holdout
(pat02, pat09), and no exam patient (pat06, pat08, pat10, pat12, pat03, pat07) appears in
any source, split or metric on this page. No model of record changes here; the broad
seizure-motion head stays final B `b9e04d03` and the convulsive head stays the model of
record `07c399be`. These are interim prototype weights under the prototype carve-out in POLICY.md
rule 4 throughout, so nothing here is clinical detection performance.

## The worry this tests

`research/event-negatives-bookkeeping.md` records the caveat exactly: arm B's training diet
carries the 2 h AJILE12 proxy-negative dose, and **every proxy row carries the fingerprint
`angle_valid` 0 together with a constant `kpc` of 0.5294**, while real WU-SAHZU rows do not.
A net can learn that a row carrying the fingerprint is a negative and never look at the
behaviour. If arm B's validation AUC rests on that shortcut, the number is an artefact of
the assembler, not evidence about the diet.

Measured, before anything was trained, at row level over the trees on disk:

| population | rows | kpc | angle null, that is `angle_valid` 0 |
|---|---:|---|---:|
| `ajile12_neg_tracks_v1` (the 2 h diet) | 218,746 | **exactly 0.5294 on 218,746 of 218,746 rows** | **218,746 / 218,746 = 100%** |
| `wu_sahzu_tracks_v2pose`, F2 train patients | 506,770 | min 0.0536, p10 0.1558, **median 0.3615**, p90 0.6690, max 0.8789 | 282,791 / 506,770 = 55.8% |

0.5294 is not a near-constant, it is a **single value**, and the mechanism is exact:
`assemble_ajile12_diet.py` sets `kpc = present / 17`, the deposit carries 9 upper-body
joints, and the deposit's missingness is **per frame, not per joint**. That was checked on
sub-01_ses-3, whose 2,592,000 frames split into 2,456,338 with all 9 joints and 135,662
with zero, and nothing in between. So every row the assembler keeps has exactly 9 present
and 9/17 = 0.5294, and no other value is reachable. The angle is null on every row because
`posture_angle_deg` needs hips and AJILE12 has none.

## What the control changes, and what it deliberately does not

A one-off **fingerprint-harmonised copy** of the diet was written to the gitignored
`outputs/events/ajile12_neg_tracks_v1_h2h` and consumed through a cloned config. Nothing
committed changes behaviour: the assembler, the trainer, the featuriser and the original
tree are untouched, and the only new committed files are the four `configs/events/h2h_*`
configs and this page.

Per row, **only `kpc` and `angle` are replaced.** `f`, `t`, `track`, `cx`, `cy`, `w`, `h`
and the `kp` block are copied unchanged, so `vy`, `vx`, `h_rel`, `aspect`, `present` and
both scene channels are bit-identical between the two arms.

**Mechanism: donor transplant, not parametric imputation.** The pre-declared sketch was to
resample `kpc` from the WU distribution, leave `angle_valid` at 0 and impute `angle_norm`.
That was not run, because `features.featurize_window` derives both channels from the same
field: `angle_norm` is `clip(angle, 0, 90) / 90` with nulls filled to 0, and `angle_valid`
is `isfinite(angle)`. **An `angle_valid` of 0 therefore implies an `angle_norm` of 0 in
every real row of every tree.** Imputing `angle_norm` under a pinned `angle_valid` of 0
would have manufactured a state that occurs nowhere in the WU data, swapping one fingerprint
for a stranger one. Instead each AJILE12 track is handed the `(kpc, angle)` time series of
one real WU track, so the two channels stay jointly consistent and both marginals become
WU's by construction:

- **Donor pool:** tracks of `wu_sahzu_tracks_v2pose` restricted to the F2 **train**
  patients (pat01, pat04, pat05, pat11, pat13, pat14) with at least 64 rows, giving 291
  tracks and 504,045 rows. The F2 validation patients, pat02 and pat09, are excluded on
  purpose so no validation-side statistic reaches a train row.
- **Draw:** one donor per AJILE12 track, sampled with probability proportional to donor
  length, so the transplant reproduces the WU **row**-level marginal rather than its
  track-level one, and read from a random wrapped offset. Seeded `20260905`; 42 distinct
  donors were drawn for the 61 segments.
- **Result:** 218,746 rows rewritten. kpc p10 0.1648, median 0.4450, p90 0.6808, with 6,993
  distinct values where there had been 1; angle null at 48.3% where WU train is 55.8%. The
  residual gap is 61-draw sampling noise and is stated rather than tuned away.

The window geometry is unchanged by construction and was checked: both trees produce
**2,250 windows** under the arm-B `WindowSpec`, and both training runs report the identical
dataset, that is 7,362 windows, 387 positive, 6,975 negative, across 20 videos.

### How identifiable the source actually is, channel by channel

Rank AUC of each channel's window mean for the target of whether a window came from the
AJILE12 tree, against the F2 train patients' WU windows, 2,250 AJILE12 windows against
4,210 WU windows. A value of 0.500 is indistinguishable; 0.000 and 1.000 are both perfect
source flags.

| channel | source AUC, original | source AUC, harmonised | reading |
|---|---:|---:|---|
| `kpc` | 0.6926 | **0.5586** | declared fingerprint, largely removed |
| `angle_valid` | 0.1741 | **0.5583** | declared fingerprint, largely removed |
| `angle_norm` | 0.1905 | **0.5827** | moves with `angle_valid`, the same field |
| `vy` | 0.5197 | 0.5197 | untouched by design |
| `vx` | 0.4897 | 0.4897 | untouched by design |
| `h_rel` | 0.5250 | 0.5250 | untouched by design |
| `present` | 0.5424 | 0.5424 | untouched by design |
| `aspect` | 0.3147 | 0.3147 | untouched, and a real artefact of the assembler's box synthesis |
| `scene_births` | 0.2812 | 0.2812 | untouched |
| `scene_n_tracks` | **0.0017** | **0.0017** | untouched, and the strongest flag on the page |

Two exact-form checks stand behind those numbers:

- **Before H2h the fingerprint is a perfect source flag.** 144,000 of 144,000 AJILE12
  window-steps carry `kpc` exactly 0.5294, against 33 of 269,440 WU steps, which is 0.012%,
  and the within-window standard deviation of `kpc` is close to 0 on every AJILE12 window
  against a WU median of 0.0364. After H2h that standard deviation is 0.0407, which is
  WU-like.
- **A stronger flag survives H2h untouched.** `scene_n_tracks` is 0.125 on every AJILE12
  window, because the assembler emits exactly one track per segment and 1/8 is the
  channel's normaliser, and only 14 of 4,210 WU windows sit at or below it. The rule that
  `scene_n_tracks` at or below 0.125 implies AJILE12 has **precision 0.9938 and recall
  1.000** on this pool. It cannot be removed at the data layer: a ghost track long enough
  to raise the density would itself generate windows and change the dataset, and a short one
  would spike `scene_births` instead. Removing it needs a code change, which the report-only
  scope forbids.

## Runs

Recipe: `configs/events/rot_train_conv_rotB_f2_s1.yaml`, arm B, fold F2, on CPU, 60 epochs,
holdout pat02 and pat09. The baseline arm's s2 and s3 rows are the **recorded** runs on
disk (`outputs/verifier/conv_rotB/f2_s{2,3}`), not re-run, and determinism is proven below.
Every out_dir is gitignored under `outputs/verifier/h2h/`.

**Execution environment, and why the reports carry no git commit.** The working tree held
concurrent uncommitted edits to `src/vividly/verifier/{features,augment}.py` and
`scripts/events/train_event_verifier.py` while this control ran. The four runs recorded
below were therefore executed against a **pristine copy of `src/` and `scripts/` at git
HEAD `08aa327`**, produced with `git archive HEAD src scripts` into a scratch directory,
with `configs/`, `data/` and `outputs/` symlinked back to this repository, so no uncommitted
file could reach any run and all four arms saw byte-identical code. The `git_commit` field
of `report.json` is `null` for exactly that reason: the snapshot sits outside the
repository, so the trainer's `git rev-parse` finds none. The commit is `08aa327`, recorded
here instead. An earlier pass of the same four runs against the live working tree is kept
for comparison and is **bit identical** to the snapshot pass, with the same validation AUC,
the same ONNX sha256, the same fitted temperature and the same 60-epoch history on all four
runs, so the in-flight edits are demonstrably behaviour-neutral on this path and no arm was
advantaged by when it ran.

| arm | seed | config | val AUC (best) | onnx sha8 |
|---|---|---|---:|---|
| baseline reproduction | 20260826 | `h2h_conv_rotB_f2_s1_baseline.yaml` | **0.75479** | `b9e04d03` |
| baseline, recorded | 20260901 | `rot_train_conv_rotB_f2_s2.yaml` | 0.77060 | `12aeb901` |
| baseline, recorded | 20260905 | `rot_train_conv_rotB_f2_s3.yaml` | 0.75569 | `4e0890a2` |
| **harmonised** | 20260826 | `h2h_conv_rotB_f2_s1_harmonized.yaml` | **0.78876** | `06b43397` |
| **harmonised** | 20260901 | `h2h_conv_rotB_f2_s2_harmonized.yaml` | **0.78954** | `70afe929` |
| **harmonised** | 20260905 | `h2h_conv_rotB_f2_s3_harmonized.yaml` | **0.77629** | `dafe0cbf` |

**Regression check: cleared.** The unchanged reproduction returns val_auc_best **0.75479**
and onnx **`b9e04d03`**, the pre-declared expectation, with a per-epoch history identical
to the committed run at all 60 epochs and the same fitted temperature 1.0251. This is the
third determinism proof of this recipe, after `outputs/verifier/_repro/{pre,post}_f2_s1`
taken at commits 7044af6 and b05a0e3, and the first one **after** the augment module landed
in the trainer at 2547927, so the augment refactor is confirmed byte-neutral on the
default-off path. The checkpoint `.pt` sha differs (`a748ed7f` against `24b56799`) because
that file serialises the config dict, which carries the new `out_dir`; the ONNX graph, the
history and the calibration are identical, so the weights are the same weights.

## Result

| | s1 | s2 | s3 | mean | spread |
|---|---:|---:|---:|---:|---:|
| baseline (arm B, F2) | 0.75479 | 0.77060 | 0.75569 | **0.76036** | 0.01581 |
| harmonised (H2h) | 0.78876 | 0.78954 | 0.77629 | **0.78486** | 0.01325 |

- **Delta of means: +0.02450**, against the fold's own declared seed spread of **0.016**
  (`research/seizure-push-2-protocol.md` item 1 fixes F2 at 0.016; the recorded seeds give
  0.01581). The delta is 1.55 times the spread.
- **The separation is clean, not a mean artefact:** the worst harmonised seed, 0.77629,
  beats the best baseline seed, 0.77060, by +0.00569. Three of three harmonised runs sit
  above three of three baseline runs.
- The direction is **up**. Masking the fingerprint did not cost the model anything.

## Reading: the shortcut is not confirmed on the declared channels, and is inconclusive as a general claim

Stated as narrowly as the evidence allows:

1. **The declared fingerprint is not what arm B's fold-F2 validation AUC rests on.** The
   shortcut hypothesis makes one prediction: take the fingerprint away and the number falls.
   It did not fall; it rose by +0.0245 of mean AUC with no seed overlap. On the specific
   worry recorded in `research/event-negatives-bookkeeping.md`, the answer is **not
   confirmed**.
2. **This is not a certificate that the AJILE12 dose is shortcut-free.** The control removes
   `kpc`, `angle_valid` and `angle_norm` as source flags and leaves `scene_n_tracks`, which
   is a *better* source flag than either of them ever was, at precision 0.9938 and recall
   1.000, plus a weaker one in `aspect`. Source identifiability was reduced, not eliminated,
   so **on the general question the control is inconclusive**, and it is inconclusive in a
   specific direction: whatever shortcut capacity remains is identical in both arms, which
   is exactly why the +0.0245 cannot be explained as the shortcut having been taken away.
3. **The most economical explanation of the gain is that the harmonised negatives are
   harder.** The validation surface is 100% real WU windows from two held-out patients.
   Giving the proxy negatives WU-like confidence and angle texture forces the decision
   boundary onto channels the validation surface actually varies over, instead of letting
   two constant columns carry the proxy class. That is a claim about cause, and it is not
   tested here.
4. **What this does not license.** It is one fold and three seeds, validation side only. It
   is not an E1 or E2 result, it does not move any model of record, and it does not promote
   the harmonised diet: H2h is a control, and the harmonised tree is a report-only artefact.
   If the diet is ever to be re-cut with WU-like fingerprint channels, that is a new
   pre-declared step, and it would have to clear the same grid the push-2 arms did.
5. **Follow-on work worth its cost:** the `scene_n_tracks` constant is an assembler
   artefact, one synthesised track per segment, not a fact about AJILE12 footage, and it is
   the largest remaining source flag in the arm-B diet. Closing it needs either a
   multi-track assembler or a featuriser that can be told a source has no scene context,
   which is a code change and therefore a separate declared step.

## Artefacts

- Configs (committed): `configs/events/h2h_conv_rotB_f2_s1_baseline.yaml`,
  `configs/events/h2h_conv_rotB_f2_s1_harmonized.yaml`,
  `configs/events/h2h_conv_rotB_f2_s2_harmonized.yaml`,
  `configs/events/h2h_conv_rotB_f2_s3_harmonized.yaml`.
- Runs (gitignored): `outputs/verifier/h2h/a_baseline` and
  `outputs/verifier/h2h/b_harmonized{,_s2,_s3}`, each with `report.json`, `verifier.pt` and
  `verifier.onnx`.
- Harmonised tree (gitignored): `outputs/events/ajile12_neg_tracks_v1_h2h`, whose
  `h2h_harmonization.json` records the summary above plus the donor track, offset and row
  count for all 61 segments; each `meta/*.json` carries the same record under
  `provenance.h2h_control`.
- **The harmoniser itself is a scratch instrument, not a committed script**, because this
  pass was scoped to add no script outside `scripts/analysis/`. The tree is nevertheless
  reproducible from the record without it: `h2h_harmonization.json` pins the seed
  `20260905`, the donor pool definition, and the exact donor track and offset for every one
  of the 61 segments. If the harmonised diet is ever wanted for more than a control, the
  instrument should be promoted into `scripts/data/` under its own declared step.

## Registry bookkeeping fixed in the same pass

`configs/datasets/ajile12.yaml` still read `status: requested` with
`hashes_unavailable_reason: "streaming harvest in progress (2026-08-28)"` long after the
harvest finished. All 55 of the 55 declared sessions are on disk, that is 55 `*.pose.npz`
and 55 `*.epochs.json` totalling 6.2 GiB, `manifest.jsonl` carries a sha256 for each of
those 110 files, and the diet assembled from them has been training arm-B verifiers since
that diet was admitted. The entry now reads `status: available`, pins `manifest.jsonl`
(`e78d623d…`, 22,380 bytes) in `files:` the way a derived corpus is pinned, since the 110
session files the manifest names each carry their own sha256 inside it, and
clears `hashes_unavailable_reason`. This is documentation only: no data moved and no
behaviour changed, except that `Registry.dataset_root("ajile12")` now returns the path
instead of raising `DatasetNotAvailable` for a set that is demonstrably present. Moving to
`available` also armed the registry's own attribution test
(`test_attribution_required_entries_are_credited`), which requires a live
`attribution_required` entry to be credited in README.md; the AJILE12 citation was added to
the README's Attributions list, and `tests/test_registry.py` is green.
