# Limb channels, second and final attempt, and the fold-power problem it exposed

Dated 2026-08-28. The retry configurations are
`configs/events/lopo_limbgate{4,5}_pat*.yaml`.

The first limb-channel run lost on all 8 comparisons (mean leave-one-patient-out val AUC
0.767 against the control's 0.863, with pat13 collapsing from 0.992 to 0.717). This is the
retry that tests the measured explanation for that failure. The retry confirms the
explanation, does not save the feature, and turns up something more important than either:
two of the four leave-one-patient-out folds this lane has been ranking recipes on cannot
support the numbers read off them.

## The explanation, measured

`LIMB_CONF_MIN` was hardcoded at 0.2, below the confidence of a typical joint in this
footage, so the spectral limb channels were measuring the spectrum of estimator noise.
Measured over all 976,659 rows of `wu_sahzu_tracks_v2kp` (1,140 tracks, 36 recordings).
`research/pose-noise-profile.md` recomputes the same tree at 983,104 rows and a 0.303
median, a 0.7 percent row difference it attributes to the earlier count dropping the rows
`tracks_from_rows` discards, so the two counts are one population measured twice, not a
contradiction:

| population | wrist+ankle conf median | > 0.5 | > 0.8 | share of frames |
|---|---|---|---|---|
| every track | 0.305 | 25.1% | 1.8% | 100% |
| tracks with mean kpc at or above 0.4 | 0.447 | 42.8% | 3.4% | 49.8% |
| tracks with mean kpc at or above 0.5 | 0.511 | 51.4% | 4.4% | 30.3% |
| wrists alone, kpc at or above 0.5 | **0.579** | 62.9% | not measured | 30.3% |

The pooled 0.305 is an artefact of averaging real people together with spurious tracks
(bedding, chairs, shadows). Limbs on tracks that are actually people are visible.

A gate is only affordable if it keeps the event. It does: over the 8 seizure-positive
recordings of the four training patients, in-event and out-of-event rows sit at an
identical 0.402 median track quality, so a quality gate is close to label-neutral, and
every recording retains rows (12.6% to 48.3%, 23.7% overall).

## The retry

One variable changes from the failed run, `window.limb_conf_min`. Same seed (20260826),
same architecture, same window geometry, same patients, same extraction tree. Configs in
`configs/events/lopo_limbgate{4,5}_pat*.yaml`; the bar was written into them before the
runs.

**Pre-declared gate:** mean leave-one-patient-out val AUC must beat the control's 0.8634,
and no fold may sit more than 0.03 below its own control fold.

**Provenance gap.** The sixteen validation AUCs in the next table were not recorded with
their per-run ONNX digests, and the run directories they came from are build outputs under
`outputs/` that are not in version control, so the digests cannot be recovered now. The
numbers are usable as a within-page comparison of four legs run under one recipe and one
seed, and are not quotable on their own the way a digest-carrying number is. No model of
record rests on them.

| fold | control (no limb) | gate 0.2 (failed run) | gate 0.4 | gate 0.5 |
|---|---|---|---|---|
| pat01 | 0.8981 | 0.8873 | **0.9111** | 0.8669 |
| pat02 | 0.9508 | 0.8902 | 0.9325 | 0.9007 |
| pat05 | 0.6129 | 0.5746 | 0.5746 | 0.5604 |
| pat13 | 0.9917 | 0.7167 | 0.9833 | 0.9875 |
| **mean** | **0.8634** | 0.7672 | 0.8504 | 0.8289 |

| leg | mean against 0.8634 | worst fold delta against 0.03 | verdict |
|---|---|---|---|
| gate 0.4 | 0.8504, below the bar | -0.0384, outside the tolerance | rejected |
| gate 0.5 | 0.8289, below the bar | -0.0526, outside the tolerance | rejected |

## How much of a difference these folds can actually carry

Before reading the table above as a verdict, the folds have to be able to support one.
Window counts per patient, from the shared corpus (1,172 windows, 173 positive):

| fold | positive windows | negative windows | positive rate | AUC 95% CI half-width |
|---|---|---|---|---|
| pat01 | 57 | 548 | 9.4% | plus or minus 0.054 |
| pat02 | 44 | 399 | 9.9% | plus or minus 0.062 |
| pat05 | 64 | **22** | **74.4%** | plus or minus 0.064 |
| pat13 | **8** | 30 | 21.1% | **plus or minus 0.149** |

(Hanley and McNeil standard error, evaluated at AUC 0.9; the interval widens toward 0.5.)

Two of the four folds are thin. pat05 is scored against **22 negative windows** and is 74%
positive where every other patient is 9% to 21%, because its extraction spans only 88 to
118 s of footage and the seizure occupies 76 to 90 s of that. pat13 carries **8 positive
windows**, so nothing smaller than plus or minus 0.15 is legible in it at all.

This is a property of the protocol, not of any recipe, and it applies retroactively to
every leave-one-patient-out number this lane has produced.

## Outcome: limb channels stay out, but not for the reason the raw means suggest

Read against the intervals above:

| leg | all 4 folds | pat01+pat02+pat13 | pat01+pat02 only |
|---|---|---|---|
| control (no limb) | 0.8634 | 0.9469 | 0.9245 |
| gate 0.2 (failed run) | 0.7672 | 0.8314 | 0.8887 |
| gate 0.4 | 0.8504 | 0.9423 | 0.9218 |
| gate 0.5 | 0.8289 | 0.9184 | 0.8838 |

- **The original diagnosis is confirmed and is large.** Moving the gate from 0.2 to 0.4
  moves the mean from 0.767 to 0.850, and pat13 recovers from 0.7167 to 0.9833, a swing far
  outside even pat13's plus or minus 0.149. Admitting sub-0.4 joints demonstrably damaged the model.
- **A correctly gated limb feature is not worse than the control; it is
  indistinguishable from it.** Gate 0.4 trails the control by 0.0130 across all four
  folds, by 0.0046 on the three folds that exclude pat05, and by 0.0027 on the two
  thickest folds. Every one of those gaps sits
  well inside the confidence intervals above. The earlier phrasing of this
  result, that three of four folds regress, read structure into noise and is withdrawn.

**The feature still does not go in.** Six extra channels that cannot be shown to help,
measured on the best folds available, are six channels of cost and risk for no
demonstrated gain. The accurate statement is that the effect is not measurable, not that
the feature is harmful, and if the fold-power problem below is fixed the question is cheap
to reopen.

## What the retry establishes

1. **A gate near 0.4, not higher.** 0.5 is worse than 0.4 (0.9184 against 0.9423 excluding
   pat05) and the frame counts explain it: 30.3% of frames survive at 0.5 against 49.8% at
   0.4. Past a point the gate discards signal faster than noise.
2. **pat05's 0.61 is not a model weakness, and should stop being quoted as one.** It is
   22 negative windows and a 74% positive rate. It has dragged every leave-one-patient-out
   mean this lane has reported, including the ones used to rank recipes before this
   retry, and the differences it appeared to show were never resolvable.
3. **The fix is extraction coverage, not features.** pat05's source recordings run 300 to
   403 s while its extraction spans 88 to 118 s. Re-extracting pat05 and pat13 across
   their full footage would convert two unusable folds into usable ones. That is the
   highest-value next action in this lane, and it is data preparation, not modelling.

No model of record changes: `convulsion_v2_real` (`07c399be`, window AUC 0.69841 on the
wusahzu testsplit) still holds the seizure lane. The `limb_conf_min` knob stays in `WindowSpec`, defaulting to the historical
0.2 so every existing config is byte-identical.
