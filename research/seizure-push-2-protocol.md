# Seizure push 2: pre-declared protocol

Dated 2026-09-05. Push 2 is authorised, together with the track-space augmentation arm
recorded in item 3. This satisfies the closing clause of
research/seizure-push-1-protocol.md, which requires a new declared decision before further
variants, and the model-of-record section of research/seizure-push-1-verdict.md. Written
before any new run or exam look.

Standing framing unchanged: interim prototype, motion signature only, never quoted as
clinical detection performance, and no claim that the system detects seizures.

**Stated up front, per the push-1 evidence:** 0.95 convulsive AUC is out of reach on
5 PG patients (three independent proofs that 4 to 5 patients memorise; E2's size means a
single look cannot even certify 0.729 to 0.80, Hanley-McNeil SE approximately 0.035). The push-2
targets are: focal E1 0.88 to 0.89 hoped (0.90 is upside, not a promise), convulsive E2
model of record unchanged unless a candidate clears it decisively.

## Exams and bars (unchanged from push 1)

- E1 focal (pat06/08/10/12, the unallocated patients held exam-side): bar = final B
  `b9e04d03` **0.87422**.
- E2 convulsive (pat03+pat07, locked): bar = the model of record `07c399be` **0.72916**.
- Selection never touches E1 or E2. Budget: **at most two promoted candidates**, each gets
  one E1 and one E2 look, recorded here. Nothing else looks at an exam.

## Item 1: zero-training within-fold seed-ensemble gate (runs first)

Members: the 12 existing conv_rotB checkpoints (4 folds by 3 seeds; f2_s1 is final B,
sha `b9e04d03`; siblings `12aeb901`, `4e0890a2`; other folds as recorded in the
push-1 grid). No new training.

- Combination rule (fixed here): per-window score = mean over the fold's 3 members of
  (logit / T_member), T from each member's report.json temperature; sigmoid after the
  mean. Mirror-vx TTA is computed as a separate column, never mixed into the gate.
- Surface: each fold's own held-out patients' windows, rebuilt once from the gitignored
  build tree outputs/events/wu_sahzu_tracks_v2pose with the arm-B WindowSpec.
- Regression check: single-member f2_s1 must reproduce val 0.75479 before the gate is
  read.
- **Pass rule (fixed here):** ensemble-minus-mean-member delta must exceed that fold's
  own seed spread (F1 0.023, F2 0.016, F3 0.014, F4 0.070) on **at least 3 of 4 folds**.
  A pass promotes candidate P2-ENS, the merged 3-member ONNX of the full-pool finalB
  recipe seeds, to one E1 and one E2 look. A fail closes the ensemble lever, and no
  exam look is spent.

## Item 2: out-of-fold hard-negative diagnosis (step 1 only, no exam contact)

Same scoring pass as item 1: every rotation-pool window gets the mean score of the 3
members of the fold that held its patient out. Tabulate false-alarm mass by negative
class, that is pat04 seizure-free recordings, pre-onset windows and non-owner (staff)
tracks, and by scene-channel level. Output: a table in this document plus a
windows.jsonl cache. Steps 2 to 3 (per-window weights alpha in {2,4}, curriculum) run only if
at or above 60% of the top-decile false-alarm mass sits in one class, and only with train-side
selection.

## Item 3: track-space augmentation arm (broad head lineage only)

Label-preserving transforms of already-admitted real recorded tracks are augmentation
rather than synthesis, applied to train rows only and after the split (POLICY.md, rule 4).
Transforms (fixed here; the three that survived adversarial review): mirror (negate vx
only), bounded in-window time-warp 0.85 to 1.18, contiguous observation dropout with
present at or above 0.5 floor. Online per-epoch draws, train rows only, applied after
grouped_split; pos_weight pinned to 10.0 so loss weighting matches the 12 conv_rotB
baselines. Explicitly excluded: track shatter, positive-only jitter (collides with the
synth generator's negative class).

- Order: single-fold F2 ablation (about 12 runs) first; if F2 val mean improves at all,
  the full 4-fold by 3-seed grid under the finalB recipe (AJILE12 diet kept at 2h).
- Selection: mean patient-held-out val AUC against the push-1 arm-B mean 0.8013 and per-fold
  spreads. If the grid mean beats arm B by more than the pooled fold spread, one final
  candidate (full pool, holdout_locations [pat02, pat09], seed 20260826) takes one
  E1 and one E2 look.
- The convulsive crown is not the target (rotation val does not track E2: arm A/B/C
  val 0.798/0.801/0.836 to E2 0.581/0.575/0.510). Any E2 gain is a bonus, not a claim.

## Item 4: negative-side bookkeeping (report only, every seizure candidate)

metrics_deep plus decision-layer curves over the recorded eval windows of every push-1
candidate and the model of record (in-alarm at 0.5/0.8/0.9, per-camera episodes/hr, persist:2,
catches; denominators stated: pat03__free 662.3 s; E1 has zero negative videos and its
video_fp 0 is vacuous). AJILE12 fingerprint caveat recorded (angle_valid 0, constant
kpc 0.5294 on every proxy row): a channel-masking control of about 20 minutes at the 2h
dose (H2h) is authorised as report-only.

## Exam looks ledger (append-only)

**Budget status: 2 of 2 candidate slots still unspent; zero exam looks taken under
push 2.** Every item has now resolved without exam contact:

| item | lever | outcome | record | look spent |
|---|---|---|---|---|
| 1 | within-fold seed ensemble | gate fail on 0 of 4 folds, lever closed | research/seizure-push-2-out-of-fold-gate.md | no |
| 2 step 1 | out-of-fold hard-negative diagnosis | trigger met (75.4% class (a)), steps 2-3 authorised | research/seizure-push-2-out-of-fold-gate.md | no |
| 2 steps 2-3 | per-window weights on class (a), alpha 2 / 4 | **F2 gate fail on both** (-0.0103 / -0.0049 against 0.76036, seed spread tripled), lever closed, no grid | research/seizure-push-2-arms.md | no |
| 3 | track-space augmentation | F2 gate pass (+0.0103) -> full grid **0.80258 against 0.80129 = +0.00129**, against a pre-declared bar of +0.03075, so **rejected**; and out-of-fold it fires more often on class (a) (+18% windows at or above 0.5, +64% at or above 0.8) | research/seizure-push-2-arms.md | no |
| (combined) | weights + augmentation | not run, because it was conditioned on both single levers clearing F2 and only one did | research/seizure-push-2-arms.md | no |
| 4 | negative-side bookkeeping | report only, recorded | research/event-negatives-bookkeeping.md | no |
| 4-H2h | AJILE12 fingerprint channel-masking control (the item-4 authorised report-only control) | shortcut not confirmed on the declared channels; no candidate, no exam contact. Detail below the table | research/seizure-ajile12-negative-control.md | no |

Item 4-H2h in full. The baseline reproduces f2_s1 exactly (0.75479 / `b9e04d03`). The
harmonized diet, whose kpc and angle series are donated from F2 train-side WU tracks, scores
**0.78486 against 0.76036 = +0.02450** over 3 seeds, which is 1.55 times the F2 seed
spread, and the worst harmonized run beats the best baseline run. The number therefore
rose with the fingerprint removed, so the shortcut is not confirmed on the declared
channels.
`scene_n_tracks`, constant 0.125 to source, precision 0.9938 and recall 1.000, remains an
open flag that cannot be closed at the data layer. The val surface was pat02 and pat09 only,
so no candidate was declared and no exam was contacted. A harmonized-diet training arm would
be a new declared push, not this one.

The models of record are therefore unchanged by push 2: the broad seizure-motion head
stays final B `b9e04d03`, and the convulsive head stays the model of record `07c399be`.
