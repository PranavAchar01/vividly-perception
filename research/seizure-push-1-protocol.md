# Seizure push: pre-declared protocol

Dated 2026-09-05. The push was authorised as four items, with the re-identification work
deferred, and all four seizure items were adopted, including the patient-rotation
protocol change (item 4).

This document is written before any new training runs, so selection rules and exam
allocation are fixed in advance. Nothing here changes the standing framing: interim
prototype weights, motion signature only, never quoted as clinical detection
performance, and no claim that the system detects seizures. Those rules are rule 4 and
section 5 of POLICY.md.

## Diagnosis being addressed (evidence in research/seizure-v2pose-comparison.md plus breakouts)

1. **Positive starvation.** 4 train patients and about 8 convulsive clips, so models
   memorise patients (the `convulsion_v2pose` run, ONNX `614b92fa`: max_score pinned
   0.487 to 0.548 on every unseen-patient video).
2. **Negative starvation.** The model of record, ONNX `07c399be`, fires 0.89 to 0.98 on the
   3 seizure-free recordings; prior negative-diet attempts (v4 ward, v6 home-room) proved
   the diet must be context-matched (registry ajile12.yaml notes).

   Provenance note on both ranges: each was read off the per-video scores in the run
   directory of the model named beside it, and those directories are build outputs under
   `outputs/` that are not in version control, so neither range can be recomputed from the
   tree. Each is a diagnosis of the starting position, and no bar or promotion rests on it.
3. **Exam composition.** The permanent exam (pat06/08/10/12) contains no
   convulsive (PG) seizures: all 12 positives are focal. The retiring testsplit was
   7 of 13 focal. The convulsive class had no clean exam.
4. **Variance.** 6 prior variants span testsplit 0.41 to 0.71; single runs are
   uninterpretable. All selection below uses multi-fold, multi-seed means.

## Exam allocation (supersedes the testsplit exam; the permanent focal exam is untouched)

- **E1 (focal exam, permanent):** pat06, pat08, pat10, pat12, unchanged, never
  trains, all-focal. E1 is scored on the v2pose track tree, 1,430 windows with 177
  positive. The two historical reference points sit on different window pools and are not
  one bar, as `final_models/seizure/README.md` records: the model of record `07c399be`
  scored **0.64270** through `configs/events/eval_wusahzu_unallocated_v2.yaml` on
  `outputs/events/wu_sahzu_tracks_v2`, 1,386 windows with 178 positive, and the single run
  `convulsion_v2pose` `614b92fa` scored **0.69921** through
  `configs/events/eval_wusahzu_unallocated_v2pose.yaml` (window AUC). Because the pools
  differ, the 0.64270 figure is not a look at the E1 surface and does not stand against a
  number measured on it.
- **E2 (convulsive exam, new, permanent):** **pat03 and pat07**, which never enter any
  training manifest from this point forward. Neutral selection rule, fixed before
  any new run: pat03 is the only PG patient with a seizure-free recording, which gives E2 a
  false-alarm side; pat07 carries the largest PG footage volume among the remaining PG
  patients. Honesty note: pat03 and pat07 scores were visible in the two historical testsplit
  looks (model of record and v2pose); they were never trained on, and no selection in this push
  may use any exam surface, so E2 remains a valid held-out exam under the same
  training-leakage standard as before.
- **Retired as an exam:** the testsplit (pat03/07/14/04/09/11). pat14/04/09/11
  move into the rotation pool below. Old testsplit numbers remain in the record as
  historical context only; they are no longer comparable bars.

## Rotation pool and folds (item 4)

Pool (8 patients): PG = pat01, pat02, pat05, pat13, pat14; focal and negatives = pat04
(including its 2 seizure-free recordings), pat09, pat11.

4 pre-declared folds, leave-2-out, PG and focal paired deterministically by patient id
order (leftover PG patients pair together):

| Fold | Val (held-out) patients | Train patients |
|---|---|---|
| F1 | pat01, pat04 | pat02, pat05, pat09, pat11, pat13, pat14 |
| F2 | pat02, pat09 | pat01, pat04, pat05, pat11, pat13, pat14 |
| F3 | pat05, pat11 | pat01, pat02, pat04, pat09, pat13, pat14 |
| F4 | pat13, pat14 | pat01, pat02, pat04, pat05, pat09, pat11 |

The train side grows from 4 to 6 patients per fold, roughly 50% to 75% more positive
clips than the old split.

## Seeds, selection, and exam discipline

- **Seeds per fold:** 3 (20260826, 20260901, 20260905), giving 12 runs per arm.
- **Arms:** A = WU-SAHZU rotation only (this baseline isolates the rotation effect);
  B = A plus the AJILE12 negative diet (item 1) once assembled.
- **Selection metric:** mean patient-held-out val window-AUC across the 4 folds
  (mean of 12 runs per arm; per-fold spread reported). Selection never touches E1 or E2.
- **Exam policy:** E1 and E2 are run once per promoted candidate (at most one candidate
  per arm), after selection is frozen. Every exam look is recorded here.
- **Per-epoch curves** are stored for every run, under a standing directive, via the
  existing extractor; grid runs log the same history JSON.

## AJILE12 diet (item 1, label-integrity machinery)

Harvest: `scripts/data/harvest_ajile12.py` (streaming pose and epochs only), running
2026-09-05. Diet assembly follows the registry diet rule exactly as written: negative
windows only from positively-labelled normal spans; blocklist and unlabelled time
excluded; patients' seizure spans never become positives. Assembly lands as a sequences
directory consumed with `force_label: 0`; the assembler script and its config are
committed before arm B trains. AJILE12 is train-only; it never appears in any exam.

## Final candidates and exam looks (declared 2026-09-05, before any exam run)

Grid outcome (patient-held-out val window-AUC, 12 runs per arm): arm A mean 0.7981
(sd 0.0772) over `configs/events/rot_train_conv_rot_f*_s*.yaml`, arm B mean 0.8013
(sd 0.0756) over `configs/events/rot_train_conv_rotB_f*_s*.yaml`, a tie on ranking. Both arms promote
one candidate each, because the diet's target effect, false alarms on seizure-free
footage, is visible on E2's negative side and not in val AUC, so both take exams.

Final-candidate rule (fixed here, before exams): per arm, retrain the recipe on
the full 8-patient pool with `holdout_locations: [pat02, pat09]` (the median
fold's pair, used only for checkpoint selection) and seed 20260826:
`rot_train_conv_finalA.yaml` and `rot_train_conv_finalB.yaml`. Exam looks: E1 and E2, once
per candidate. The model of record 07c399be also runs E2 once as the comparison bar
(it never trained on pat03 or pat07; its scores on them were already visible in the
historical testsplit look, so this re-slices an old look rather than burning a
new surface). All looks recorded here; no further exam runs without a new
declared decision.

## Exam looks taken (2026-09-05) and arm C declaration

E1 (focal, permanent): final A (ONNX `a373127b`) **0.83772**, final B (ONNX `b9e04d03`)
**0.87422**. The two historical reference points above, the model of record `07c399be` at
0.64270 and the single run `614b92fa` at 0.69921, sit on different window pools and are not
comparators for these two numbers. E2 (convulsive): the re-slice of the model of record
`07c399be` **0.72916**, final A `a373127b` 0.58105, final B `b9e04d03` 0.57469. Reading:
rotation put focal seizures into the positive diet, which broadened the model on the focal
exam at the cost of convulsive sharpness, so the model of record, an all-PG-trained
specialist, keeps E2. The two new exams
expose this axis for the first time.

**Arm C (declared before running, the last exam-burning variant of this push):**
a convulsion specialist. Sources = WU PG patients only [pat01, pat02, pat05, pat13,
pat14], with focal patients excluded entirely, their labels untouched and not
repurposed, plus the AJILE12 diet (force_label 0). Cross-validation: 5
leave-one-patient-out folds
by 2 seeds (20260826, 20260901), the ten recipes
`configs/events/rot_train_conv_rotC_f*_s*.yaml`, selection = mean held-out val AUC. Final C:
all 5 PG patients, holdout_locations [pat02] (checkpoint selection only), seed
20260826; then one look each at E2 and E1. After arm C the push closes with a
full verdict regardless of outcome; further variants need a new declared decision.

**Arm C looks taken:** final C (ONNX `d3e6d8c6`) E1 **0.58613**, E2 **0.51008**, which is
chance. Both are recorded here, so this ledger holds all seven looks of the push and
agrees with research/seizure-push-1-verdict.md and
research/event-negatives-bookkeeping.md.

## Measurement fix (item 2)

`wu_sahzu_breakout.py` (PG, focal and negative split) is now part of every exam
report's presentation. Headline sensitivity = PG only; focal reported separately;
negatives give the false-alarm side.
