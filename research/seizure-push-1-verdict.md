# Seizure push 1: outcome (2026-09-05)

The push produced a new best broad seizure-motion model. Final B (`b9e04d03…`) sets the
permanent focal exam bar at 0.87422 window AUC. The previous holder's 0.64270 was measured
on an older track tree over a different window pool, so it is not a comparator and no delta
between the two is quoted; `final_models/seizure/README.md` records that position. The
earlier model of record `convulsion_v2_real` (`07c399be…`) keeps the convulsive lane
(E2 0.72916 against 0.58105, 0.57469 and 0.51008 for the three new arms). The convulsive-specialist
ceiling is set by data (5 PG patients), not by recipe.

Protocol, folds, diet rule, and every exam look: research/seizure-push-1-protocol.md
(pre-declared throughout; selection never touched an exam). Interim prototype, motion
signature only, never clinical performance, and detection of seizures is never claimed.

## Exam scoreboard (window AUC; one look each, all recorded)

| Model (onnx sha256) | Trained on | E1 focal exam (pat06/08/10/12, permanently exam-side) | E2 convulsive exam (pat03/07) |
|---|---|---|---|
| model of record `convulsion_v2_real` (`07c399be…`) | 4 PG patients | 0.64270, historical: measured on the older `wu_sahzu_tracks_v2` tree through `configs/events/eval_wusahzu_unallocated_v2.yaml`, 1,386 windows with 178 positive, which is a different surface from E1 | **0.72916**, keeps the convulsive lane |
| earlier single run `convulsion_v2pose` (`614b92fa…`) | 4 PG patients | 0.69921, historical: measured through `configs/events/eval_wusahzu_unallocated_v2pose.yaml`, again not the E1 surface | not measured |
| final A, rotation (`a373127b…`) | 8 pool patients, pat02 and pat09 held out for checkpoint selection | 0.83772 | 0.58105 |
| **final B, rotation plus AJILE12 diet (`b9e04d03…`)** | 8 pool patients plus the 2 h AJILE12 negative diet, pat02 and pat09 held out for checkpoint selection | **0.87422**, new best | 0.57469 |
| final C, PG specialist (`d3e6d8c6…`) | 5 PG patients plus the AJILE12 diet, pat02 held out for checkpoint selection | 0.58613 | 0.51008 (chance) |

## What was learned (each claim traces to a run above or the grids)

1. More patients per model transfers; recipes do not. Rotation (4 to 6 training
   patients) took unseen-patient validation from a flat 0.5 collapse to a 0.798 mean
   (12 runs, `configs/events/rot_train_conv_rot_f*_s*.yaml`), and set the focal exam bar
   at 0.87422 on the E1 surface. The previous holder's 0.64270 sits on a different tree
   and window pool and is not a comparator for it. Every gain in this push came from
   data composition.
2. The AJILE12 negative diet earns its place. Arm B ties arm A on ranking (validation
   +0.003) but wins the focal exam by +0.037 (0.874 against 0.838), consistent with its
   design goal of teaching normal hospital-bed motion. 55/55 sessions harvested, 121.5
   min of labelled-normal spans, and the diet rule enforced by the assembler.
3. The specialist path is data-capped. Arm C self-validated at 0.836 mean on its own
   5 PG patients (10 runs, `configs/events/rot_train_conv_rotC_f*_s*.yaml`) yet scored
   chance on both exams: 5 patients still memorises, exactly
   like the original 4. No convulsive model beats the model of record until more
   convulsive patients exist.
4. The exam split itself was a finding: the old permanent exam contains zero
   convulsive events (all focal), so progress on the convulsion verifier was being graded
   on the wrong class. E2 (pat03/07, never trained from now on) fixes that permanently.

## Models of record (v2 lane, prototype class)

- Broad seizure-motion head, final B, `b9e04d03…` (config
  `configs/events/rot_train_conv_finalB.yaml`; curves
  `final_models/seizure/curves/seizure_push_finalB.csv`).
- Convulsive head, the model of record `convulsion_v2_real` `07c399be…`, unchanged.
- Final C recorded, not promoted. Further variants require a new declared decision;
  the true unlock remains more clinical patients (CQ-21 pilot).

## Provenance gaps

Recorded so they can be closed rather than rediscovered.

- Two rows of the scoreboard carry eight hex characters of their ONNX digest and nothing
  longer: final A `a373127b` and final C `d3e6d8c6`. Neither prefix resolves to a full
  digest anywhere in this repository, because neither arm was promoted, so neither has a
  lane page entry, and the run directories under `outputs/verifier/` that hold the full
  values in their `report.json` are gitignored. Both should be captured from those run
  directories at the next opportunity. Until they are, the two numbers each row quotes
  travel with a prefix rather than with a full digest.
- The other three rows are complete. The full digests of final B `b9e04d03`, the model of
  record `convulsion_v2_real` `07c399be` and `convulsion_v2pose` `614b92fa` are carried on
  `final_models/seizure/README.md`.
