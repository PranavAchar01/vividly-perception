# Rule threshold calibration protocol

**Status: not yet executed.** What it waits on is pilot shadow-mode data, an agreed
alerts-per-nurse-per-shift ceiling and a site data agreement, the three preconditions in
section 2 that are still outstanding. The protocol is written down so the provisional
thresholds in `configs/pipeline/rules.yaml` have a named destination, and so that nobody has
to invent one under time pressure later.

**Scope: `configs/pipeline/rules.yaml` only.** The rule identifiers named in section 3 are the
whole subject. The provisional re-identification values in `configs/pipeline/reid.yaml`, that is
`tau`, `max_per_identity`, `top_k`, `max_proposals_per_track`, `confidence_at_tau` and
`enroll_every_s`, point here as their destination, but no protocol for them is written yet:
nothing below covers `tau`, the detection-and-identification rate against false-accept rate, or
the session gallery, and section 6 puts re-identification out of bounds as a calibration input.
Until a re-identification section exists, the interim helper is `vividly.reid.calibrate`, which
picks `tau` at a stated false-accept budget and records the achieved false-accept rate,
detection-and-identification rate and false-merge rate next to whatever it recommends. Its output is a recommendation, not a calibrated threshold, and the
`reid.yaml` values stay provisional regardless.

**Activates when all four preconditions in section 2 exist.** The first, fine-tuned weights for
the lane whose outputs feed the rule, is met and recorded by hash. The rules read boxes from the
eyes lane, RT-DETRv4-S person v2 checkpoint `345c30a5`, ONNX `40262049`, and keypoints from the
joints lane, RTMPose-m person v2 checkpoint `a0a035fe`, ONNX `6ba96285`, both indexed in
[final_models/README.md](../final_models/README.md). Pilot shadow-mode data does not exist.
Until it does, every threshold stays provisional, and the replay layer stays what
[PIPELINE.md](PIPELINE.md) says it is: structure, not a tuned product.

**Interim measured candidates, 2026-09-05, recorded before this protocol.**
`research/operating-points.md` records measured provisional verifier operating points from
recorded artefacts. The seizure broad-head threshold was fitted on the non-exam out-of-fold
rotation pool and only reported on the exam artefacts. The convulsive head's threshold is
transferred, not fitted. **Every fall number is fitted on exam negatives**, because no non-exam
fall surface exists, and it is provisional until pilot shadow-mode data replaces it. None of
that executes the protocol below, none of it is a false-alarm rate, and every value travels with
its ONNX sha8 and its denominator in hours, as the hash rule requires.

---

## 1. Why nothing is calibrated yet

Two inputs were missing when this protocol was written, in this order. The first now exists;
the second does not.

**Weights.** Every rule input is derived from detections, pose keypoints and track continuity.
Stock models are trained on clean web imagery and degrade on hospital-camera imagery: heavy
compression, low frame rate, multiple people, hours seated. Nothing was fitted while stock
estimators were still underneath the rules, because a posture threshold fitted on top of a
posture estimator that is then replaced fits the estimator's error, not anyone's behaviour.
The estimators are now the lane models of record named above, and that same reasoning is why a
threshold is refitted whenever either of them moves.

**Labelled events.** No recording of a real waiting room with known outcomes exists, and the
event-data rule forbids manufacturing one: no simulated or acted medical events in any training
set, synthetic renders included ([../POLICY.md](../POLICY.md), rule 4). The trajectory fixtures
in `src/vividly/rules/fixtures.py` are unit-test material for wiring and onset arithmetic. They
are not evidence about people and they are not the labelled set this protocol needs.

## 2. Activation preconditions

All four, before step 1 runs. The first is met; the other three are not.

1. **Fine-tuned weights** for the lane feeding the rule, with the checkpoint recorded by hash in
   the provenance record. No accuracy number travels without that hash, and that applies to a
   threshold fitted against those weights just as much. Met: eyes `345c30a5` with ONNX
   `40262049` and joints `a0a035fe` with ONNX `6ba96285`, each on its lane page under
   `final_models/`.
2. **Shadow-mode data from a pilot site.** Shadow mode means log everything and act on nothing,
   for 4 to 8 weeks.
3. **An agreed alerts-per-nurse-per-shift ceiling**, defined before any new alert type ships. It
   is an input to this protocol, not an output of it.
4. **A site data agreement** covering that footage and the labels derived from it, recorded at
   registry-entry time with the agreement reference in the entry metadata. The site's terms sit
   on the counsel checklist next to the evaluation-only licences.

### Before step 1: the negative control

Step 1 starts from a measured baseline on real footage that contains no medical events. Replay
the rules over such footage and report firings per camera-hour per rule. Such a run satisfies
none of the outstanding preconditions above, so its output is a count, never a false-alarm rate,
and it is labelled that way wherever it appears. It exists so that every firing on real footage
can be looked at before a single threshold moves.

## 3. Step 1: per-rule threshold sweep against labelled shadow-mode events

Run per rule identifier, independently: `fall_then_still_v0`, `seizure_candidate_v0` and
`egress_without_discharge_v0`, whose three clocks live in the egress head. The ranking criterion
is asymmetric by policy: a grid point that trades detection latency or marginal recall for fewer
false positives wins, and the false-positive rate per camera-day is reported first at every
point.

- **Freeze the inputs first.** One checkpoint pair, one tracker configuration, one shadow-data
  window, all three recorded next to the result. Sweeps run on different weights are not
  comparable and are not plotted on the same axes.
- **Replay offline.** Sweep by replaying the shadow window through `scripts/run_pipeline.py`
  with the rule's own thresholds varied over a grid declared in a sweep configuration, every
  other rule's thresholds held at their current values. Deterministic seeking and
  faster-than-realtime playback make this a batch job over recorded data, not a live experiment.
- **Score against the labels.** At each grid point count firings that match a labelled event
  within a stated onset tolerance, firings with no event behind them, and events with no firing.
  Report onset error as its own number, not folded into the match rate: `onset_t` is what the
  product promises the nurse, the timestamp when the change began, so a rule that finds a real
  event late is failing differently from one that misses it.
- **Report latency beside accuracy** at every grid point, as every other evaluation in this
  repository does.
- **Output a curve per rule, not a number.** The operating point is chosen in step 2.

## 4. Step 2: apply the alerts-per-nurse-per-shift ceiling

The ceiling is a product constraint fixed before the sweep, and it is not relaxed afterwards to
make a rule look better.

Convert each candidate operating point into alerts per nurse per shift using the shadow window's
own census and staffing, then discard every point above the ceiling before comparing detection
quality among the survivors. The rules share one budget: the ceiling applies to the union of
firings a nurse actually sees, so a new rule tightens the existing ones rather than adding
headroom of its own.

The reason the ceiling leads and sensitivity follows: documented hospital monitor false-alarm
rates run 72 to 99 percent with positive predictive value as low as 27 percent, and alarm
fatigue is how products in this category die. A rule that cannot hold its share of the budget
does not ship as an alert.

## 5. Step 3: feedback-label-driven iteration

Agree and false-alarm clicks write the feedback field of `AlertRecord` through the dashboard,
which is the labelling loop and the governance requirement in one. Once an operating point is
running in shadow mode:

- Re-run step 1 on a rolling window with the feedback labels as the target, keeping the step 2
  ceiling fixed.
- Track per-rule agreement rate over time. A drop is either drift in that site's population or a
  change upstream in weights or tracker settings, and the frozen-input record from step 1 is what
  tells the two apart.
- Version every threshold change: bump the rule's `rule_version` so each `AlertRecord` states
  which version fired, and re-baseline instead of comparing across versions.
- The governance question stays open for counsel and is not an engineering call: fine-tune on
  site, against exporting only labels and derived features and never raw video.

## 6. Out of bounds for this protocol

- No calibrating against simulated or acted medical events, and no synthetic renders of them,
  ever.
- No evaluation-only dataset in any calibration set that feeds shipped weights or shipped
  thresholds. A `LEGAL_HOLD` entry is not touched at all.
- No facial recognition of any kind: no face detection, embedding, matching or face-localisation
  branch. Re-identification is a separate case and a narrower rule. It does exist here, as
  `src/vividly/reid/` plus the session-scoped gallery wired into the replay pipeline, and what
  this protocol requires is that the stage stays disabled for the whole sweep, which is already
  its shipped default: `configs/pipeline/reid.yaml` is off by default. No re-identification
  output is admitted as identity for calibration. Track identity during a sweep is whatever the
  tracker produced, gaps included, and gap behaviour is itself something the sweep measures.
- No vital-signs proxy admitted as a rule input.
- No threshold quoted outside the repository before it has been through steps 1 to 3, and none
  quoted without the checkpoint hash it was fitted against.

## 7. Where the results land

When this protocol activates: sweep configurations go in `configs/pipeline/` next to
`rules.yaml`; per-rule curves and the chosen operating point go in `research/` beside the other
evidence documents; and the operating point itself goes back into `configs/pipeline/rules.yaml`
with its provisional comment replaced by a pointer to the result file and the checkpoint hash.

Until then, those provisional comments stay exactly as they are, and so does this file's status
line.
