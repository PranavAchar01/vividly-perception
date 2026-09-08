# Operating points v1: decision-layer candidates for the event-verifier heads

Dated 2026-09-05. Scope: report and recommendation only. No training, and no new
model inference on any footage: every number below is computed from recorded per-window
scores (`windows.jsonl`), recorded eval summaries (`metrics_deep.json`,
`decision_layer_study.json`, `breakout.json`), and the extraction metadata already on
disk (`outputs/events/wu_sahzu_tracks_v2pose/meta/*.json`). All weights are interim
prototypes; **no number here is clinical detection performance**, and every value is a
count on recorded windows, never a ward false-alarm rate.

**Calibration status.** Nothing here executes docs/CALIBRATION.md. Of its preconditions
(section 2), the weights half is met by the lane models of record; pilot shadow-mode data,
an alerts-per-nurse-per-shift ceiling and a site agreement are still unmet. Every operating
point below therefore stays
**PROVISIONAL**, including the recommended ones. What this pass adds is *measured*
provisional points in place of the unmeasured 0.5 default, under the fitting discipline
declared per head below.

Instrument definitions, identical to `scripts/events/decision_layer_study.py`, which the
recomputations below reproduce **byte-exactly** on every recorded grid row before extending
it; reproduction checks passed for `conv_finalB/eval_E2`, `conv_champion_reslice/eval_E2`,
`fall_v2pose/eval_multicam_allcams` and `fall_v2pose/eval_ur_fall`:

- **single:thr** fires on any window at or above thr. **persist:k@thr** requires k
  consecutive same-track windows at or above thr, with a gap of at most 3.5 s.
  **hyst:hi|k@lo** fires immediately at or above hi, or on k consecutive windows at or
  above lo.
- **catch**: the first fire of any track in [onset, event_end + 30 s]; latency is the fire
  t_end minus the onset, at window level, so it includes the window span itself.
- **eps/h**: distinct false-alarm episodes per negative-recording hour, with fires less
  than 30 s apart merged, reported at track level and at camera level, where camera level
  merges all tracks on one recording first.
- **in-alarm**: the fraction of the negative footage covered by the union of
  [fire minus 12 s, fire] spans. The instrument hard-codes the 12 s span, so the fall rows,
  which use 8 s windows, are uniformly overstated by up to 4 s per fire; comparisons
  between points are unaffected.

Fitting discipline, stated per surface:

| head | fitting surface | exam artefacts (report only, never refit) |
|---|---|---|
| seizure broad finalB `b9e04d03` | **non-exam** out-of-fold rotation pool (`outputs/verifier/push2_oof/windows.jsonl`, 5112 windows, pat01/02/04/05/09/11/13/14) | E2 `conv_finalB/eval_E2` and E1 `conv_finalB/eval_E1` |
| seizure convulsive model of record `07c399be` | none exists for this model, because the out-of-fold pool is finalB-family, so the **threshold is transferred from the out-of-fold fit**, not refit | E2 `conv_champion_reslice/eval_E2` |
| fall `b66c3a0a` and the fall_v3 model of record `67954c04` | **exam side**: no non-exam fall negative surface exists, so every fall number is **fitted on exam negatives, provisional until pilot data** (docs/CALIBRATION.md) | chute24 `eval_multicam_allcams` and UR ADL `eval_ur_fall` |

Denominators, stated once and repeated per table:

- **Out-of-fold pool negatives: pat04__free 1781.6 s plus pat04__no-Sz2P 955.7 s, which is
  0.7604 h** across 2 seizure-free recordings, with durations taken as
  `n_frames_container / fps` from the extraction meta. Out-of-fold events: 17, of which 10
  are convulsive_PG and 7 focal, on the 8 rotation patients.
- **E2 negatives: pat03__free, 662.3 s, which is 0.184 h**, one recording on one camera.
- **E1 has no negative recordings at all**, so per-hour columns are not applicable there
  and only catch cost is reported.
- **chute24: 0.273 h**, the 8 cameras of the one no-fall chute, `eval_multicam_allcams`.
- **UR ADL: 0.081 h**, 38 ADL recordings. 16 of the 30 UR fall events are zero-window under
  the recorded contain extraction, so **14/30 is the recorded catch ceiling at any
  threshold**, and those misses are extraction coverage rather than scoring.

---

## 1. Seizure broad head: finalB, onnx sha8 `b9e04d03`

### Fit (non-exam out-of-fold pool, mean-of-3 fold out-of-fold member score, 0.7604 h negatives)

The out-of-fold score is the mean of the three per-fold rotB members' calibrated
probabilities (`member_scores` in `push2_oof/windows.jsonl`), which is the model-agnostic
shape of the finalB family and not the model of record. The full 23-rule grid is not carried
as a file; section 7 says how it regenerates. The decisive rows are:

| rule (out-of-fold fit) | catches (PG, focal) | med. latency | eps/h track, camera | in-alarm |
|---|---|---|---|---|
| single:0.5 (current provisional) | 16/17 (9/10, 7/7) | 21.8 s | 101.3, 2.6 | **99.0 %** |
| single:0.75 | 14/17 (9/10, 5/7) | 47.0 s | 21.0, 13.2 | 6.8 % |
| **single:0.8 (recommended)** | **12/17 (9/10, 3/7)** | **48.2 s** | **5.3, 5.3** | **2.2 %** |
| persist:2@0.8 | 9/17 (7/10, 2/7) | 62.2 s | 2.6, 2.6 | 1.1 % |

- At 0.5 the pool's alarm is lit 99.0 % of the seizure-free footage; the camera eps/h of
  2.6 is the alarm-never-turns-off inversion, not quiet. Moving from 0.5 to 0.8 keeps
  **every convulsive-PG catch** (9/10 at both points; the one miss, pat05__Sz3PG, is missed
  at 0.5 too) and pays entirely in focal catches, 7/7 down to 3/7, for an in-alarm cut of
  roughly 45 times. Persistence is strictly worse here: persist:2@0.8 gives up 2 PG catches
  for a 2 times episode saving, so it is rejected under the minimal-catch-cost rule, and
  the live engine has no persistence anyway, as section 5 records.
- Combiner robustness: rescoring with `score_ens`, the calibrated-logit mean, moves the
  same point to 14/17 at 28.9 and 15.8 eps/h with 10.6 % in-alarm, so the elbow between 0.7
  and 0.8 is not an artefact of the mean-of-3 choice.
- The AJILE12 fingerprint caveat (research/event-negatives-bookkeeping.md) applies: arm-B
  training saw proxy negatives with a source fingerprint, so quietness on its own family's
  out-of-fold pool certifies shape, not ward behaviour.

### Report at the fitted points (exam artefacts, recorded, not refit)

E2 covers pat03 and pat07, with negatives from pat03__free at 0.184 h, taken from the
recorded `decision_layer_study.json` for the 0.5 and 0.8 rows and from the byte-exact
recompute for the 0.75 row:

| point | E2 catches | med. latency | eps/h track, camera | in-alarm | E1 catches (12 focal, no negative hours) |
|---|---|---|---|---|---|
| single:0.5 | 4/4 | 39.0 s | 114.1, 27.2 | 68.7 % | 10/12, median 12.6 s |
| single:0.75 | 3/4 | 28.8 s | 59.8, 27.2 | 23.7 % | not recorded at 0.75 |
| **single:0.8** | **3/4** | **28.8 s** | **54.4, 21.7** | **17.2 %** | **7/12, median 29.8 s** |

Reasoning in one line: **the out-of-fold elbow sits at 0.8, where PG catches are unchanged
from 0.5 while the seizure-free surface goes from alarm-always-on at 99 % to 2.2 %
in-alarm, and the exam side confirms the direction, 68.7 % to 17.2 %, at a cost of 1 of the
4 E2 catches and 3 of the 12 E1 catches.** The E2 catch lost at 0.8 is pat03__Sz2PG, event
max score 0.6211; the E1 cost falls on the focal exam, which is exactly the trade the
out-of-fold fit predicted, focal 7/7 down to 3/7.

## 2. Seizure convulsive head: the model of record `07c399be` (E2 reslice of convulsion_v2_real, temperature 2.377)

No non-exam surface exists for this model, because the out-of-fold pool scores are
finalB-family. The threshold is therefore **transferred** from the out-of-fold fit,
single:0.8, and the exam rows below are a report of the recorded artefacts, not a fit. E2
negatives are pat03__free at 0.184 h.

| point | E2 catches | med. latency | eps/h track, camera | in-alarm |
|---|---|---|---|---|
| single:0.5 (current provisional) | 4/4 | 6.0 s | 81.5, 27.2 | 75.2 % |
| **single:0.8 (recommended, transferred)** | **4/4** | **9.0 s** | **54.4, 32.6** | **39.3 %** |
| single:0.85 (exam-side observation only) | 4/4 | 9.0 s | 38.0, 21.7 | 25.2 % |

Reasoning in one line: **at the transferred 0.8 the model of record keeps 4 of 4 PG catches, on
both the decision-layer and the metrics_deep event definition, with median latency 9.0 s,
and cuts in-alarm from 75.2 % to 39.3 %.** The 0.85 row is listed because it is recorded
and strictly better on this one 0.184 h recording, 4 of 4 at 25.2 % in-alarm, but choosing
it would be fitting on the exam, so it stays an observation until pilot shadow data exists.
Note that the calibration of the model of record, T = 2.377, differs from the finalB family at
T of about 1.03, so the transferred threshold is a policy choice, not a score-scale
identity.

## 3. Fall head: fall_v2pose, onnx sha8 `b66c3a0a`

**Every row is fitted on exam negatives and stays provisional until pilot data**
(docs/CALIBRATION.md). Two surfaces, both recorded: chute24 at 0.273 h across 8 cameras,
and UR ADL at 0.081 h across 38 recordings, with a 14/30 recorded catch ceiling. The grid
rows come from the two `decision_layer_study.json` files, and the 0.85 and 0.95 rows are
recomputed byte-compatibly from the recorded `windows.jsonl`.

| point | chute24: catches, eps/h track and camera, in-alarm | UR: catches, eps/h track and camera, in-alarm |
|---|---|---|
| single:0.5 (current provisional) | 141/200, 25.6 and 14.7, 9.5 % | 14/30, 197.5 and 172.8, 60.1 % |
| single:0.8 | 116/200, 14.7 and 14.7, 5.2 % | 13/30, 61.7 and 61.7, 20.6 % |
| **single:0.9 (recommended)** | **100/200, 14.7 and 14.7, 5.0 %** | **11/30, 12.3 and 12.3, 4.1 %** |

Reasoning in one line: **persistence is off the table, because persist:2 at any threshold
catches 0 of 30 on UR, the surviving events yielding too few consecutive windows under the
recorded extraction, so the head lives on a single threshold; and 0.9 is where the UR ADL
surface finally goes quiet, 197.5 down to 12.3 eps/h and 60 % down to 4.1 % in-alarm, at a
cost of 3 UR catches out of the 14-catch recorded ceiling and 41 of 200 chute24 catches.**
That is the direction the acute-event rule set requires: false-positive rate first,
marginal recall second. single:0.8 is the catch-lean fallback if the pilot ceiling allows
about 5 times the UR burden. Median catch latency is flat across the family, 2.4 to 2.6 s
on UR and 3.5 to 5.3 s on chute24.

A measured but currently unbuildable note: hyst:0.9|2@0.65 dominates single:0.9 on chute24,
115 of 200 at the same 14.7 eps/h, and degrades exactly to single:0.9 on UR, 11 of 30 at
12.3. It is the one recorded combination that would justify the persistence and hysteresis
code change in section 5, which the engine could not express when this was written; see
section 8, where that seam landed and `hyst:0.9|2@0.65` became expressible as
`persist_k: 2, confirm_threshold: 0.65, hyst_hi: 0.9`.

## 4. Fall head: the fall_v3 model of record, onnx sha8 `67954c04` (reslice through v2pose tracks)

**Fitted on exam negatives, provisional until pilot data.** No recorded
`decision_layer_study.json` exists for these two eval directories, so every row below is
recomputed from the recorded `windows.jsonl` per-window scores with the byte-verified
instrument logic, reusing per-window scores with no new inference. The denominators are the
same as in section 3.

| point | chute24: catches, eps/h track and camera, in-alarm | UR: catches, eps/h track and camera, in-alarm |
|---|---|---|
| single:0.5 (current provisional) | 156/200, 65.9 and 47.6, 26.8 % | 14/30, 234.6 and 197.5, 69.8 % |
| single:0.9 | 121/200, 14.7 and 14.7, 5.3 % | 14/30, 98.8 and 98.8, 34.9 % |
| **single:0.95 (recommended)** | **111/200, 7.3 and 7.3, 2.8 %** | **13/30, 49.4 and 49.4, 17.4 %** |

Reasoning in one line: **the model of record scores UR ADLs high at every threshold, still 34.9 %
in-alarm at 0.9, and 0.95 is the only recorded point that halves that again, to 49.4 eps/h
and 17.4 %, while keeping 13 of the 14 UR recorded ceiling and 111 of 200 chute24 catches
at 2.8 % chute24 in-alarm.** If this head ships anywhere, the UR surface says its quiet
range starts at 0.95, not 0.9, and even there it is 4 times louder on UR than fall_v2pose
at 0.9, which is a model observation for the model record in final_models/README.md, not a
threshold fix.

---

## 5. What the engine supported at the analysis pass, and what needed a small code change (see section 8 for what later landed)

Read against `src/vividly/rules/engine.py`, `src/vividly/rules/verify.py`,
`src/vividly/models/event_verifier.py`, `src/vividly/events/store.py`,
`src/vividly/dashboard/app.py` in this repository at f945dac.

**Supported at that revision, by configuration alone and with no code change:**

- **Per-head confirm threshold**: `verify.heads.<rule_id>.confirm_threshold` in a pipeline
  config (`OnnxEventVerifier.from_config`). A score at or above the threshold confirms, a
  score below rejects, and the score is always stamped as the verdict confidence. The
  default is 0.5 (`DEFAULT_CONFIRM_THRESHOLD`, PROVISIONAL). **No in-tree pipeline config
  carried a `verify:` block at that revision**, so the moment a head was wired the
  effective point was the 0.5 code default, which is the provisional point every table
  above leads with. `configs/pipeline/replay_demo.yaml` carries one now, at the points
  section 8 records.
- **Verify policy**: `verify.policy: annotate | require`, that is fail-open with a stamp
  against fail-closed.
- **Per-rule, per-track debounce**: `refire_after_s: 900` in
  `configs/pipeline/rules.yaml`, so a track re-alerts at most every 15 minutes per rule.
- **Evaluation cadence**: `eval_every_s: 1.0`.
- **Window abstention**: thin evidence scores as unavailable, never as a guess.

**Not supported at that revision. Each is a small, local change, and none was recommended
for this pass's operating points; section 8 records what landed on 2026-09-05:**

1. **Persistence and hysteresis (persist:k@thr, hyst:hi|k@lo).** The live path scores
   exactly one window per firing: `OnnxEventVerifier.verify` calls `featurize_window` once
   at `onset_t` minus `pre_pad_s`, and there is no k-consecutive-windows knob anywhere. The
   cheap change, if it were ever measured to pay: add `persist_k: int = 1`, and optionally
   `hyst_hi: float | None`, to `VerifierHead` and `from_config`, and in `verify()` score the
   k windows at `t_end + i * spec.stride_s` for i from 0 to k minus 1 from the same rings,
   confirming when all k are at or above the threshold, or when any is at or above
   `hyst_hi`. That is about 20 lines and one head-config field, with no engine change. The
   measurements above say: negative value for fall, where UR persist catches 0 of 30;
   negative for the seizure broad head at the elbow, where persist:2@0.8 drops 2 PG catches
   on the out-of-fold pool; and positive only for the chute24-side hysteresis noted in
   section 3. Build it when that trade is worth 15 catches.
2. **Per-camera episode merge.** Debounce is keyed on rule id and track id only, so two
   tracks on one camera alert independently, and neither the sqlite store nor the dashboard
   deduplicates alerts; the dashboard's only merge is re-identification identity
   confirmation. The camera-level eps/h columns above are therefore an *instrument* view,
   not a live behaviour. The cheap change, if the pilot ceiling needs it: a second debounce
   map in `RuleEngine._admit` keyed by rule id alone with its own `camera_refire_after_s`,
   which works because the pipeline opens one camera per process, so rule scope is camera
   scope. That is about 10 lines plus one RuleConfig field and one rules.yaml entry. The
   measured value at the recommended points: the E2 model of record at 0.8 saves 54.4 down to
   32.6 eps/h; out-of-fold finalB at 0.8 saves nothing, 5.3 to 5.3; and the fall surfaces at
   0.9 save close to nothing, because track and camera already agree. Real but modest, so
   defer it to pilot data.
3. **A verifier-threshold home in rules.yaml.** `load_rule_config` builds a `RuleConfig`
   that rejects unknown keys, so `confirm_threshold` cannot live in
   `configs/pipeline/rules.yaml` without widening RuleConfig; it belongs in the pipeline
   config's `verify:` block, where the backend already reads it. The proposed diff below
   therefore touches `replay_demo.yaml`, and rules.yaml only gains a comment pointer.

## 6. Proposed diff (not applied by the analysis pass; see section 8 for what later landed)

`configs/pipeline/replay_demo.yaml`, wiring the v2-lane heads at the recommended points,
using the interim run directories under `outputs/`, which is gitignored. For the models of
record the same block points at `convulsion_v2_real` with 0.8 and `fall_v3` with 0.95:

```diff
--- a/configs/pipeline/replay_demo.yaml
+++ b/configs/pipeline/replay_demo.yaml
@@ (after the `rules_config:` line)
 rules_config: configs/pipeline/rules.yaml

+# --- event verification (the verifier seam, interim prototype heads) -----------
+# PROVISIONAL measured operating points, research/operating-points.md, 2026-09-05.
+# These are not docs/CALIBRATION.md output: its preconditions are still unmet.
+# The seizure threshold was fitted on the non-exam out-of-fold rotation pool, the
+# finalB family, mean of 3 fold out-of-fold; the fall threshold was fitted on exam
+# negatives, chute24 0.273 h and UR ADL 0.081 h, because no non-exam fall surface
+# exists. Both stay provisional until pilot shadow-mode data. Quote no number
+# without the ONNX sha8 beside it.
+verify:
+  policy: annotate
+  heads:
+    seizure_candidate_v0:
+      run_dir: outputs/verifier/conv_finalB      # onnx sha8 b9e04d03 (T 1.0251)
+      confirm_threshold: 0.8   # was the 0.5 default; out-of-fold: PG 9 of 10 kept, in-alarm
+                               # 99.0% to 2.2% (0.7604 h negative); E2 report: 3 of 4,
+                               # 17.2% in-alarm (0.184 h)
+    fall_then_still_v0:
+      run_dir: outputs/verifier/fall_v2pose      # onnx sha8 b66c3a0a (T 1.486)
+      confirm_threshold: 0.9   # was the 0.5 default; exam-fitted provisional: UR ADL
+                               # 197.5 to 12.3 eps/h (0.081 h), chute24 100 of 200 at
+                               # 14.7 eps/h (0.273 h); persist:k unusable, UR catches 0 of 30
+
 # --- outputs ------------------------------------------------------------------
```

Equivalents for the models of record, same block shape:
`seizure_candidate_v0: convulsion_v2_real @ 0.8` (sha8 `07c399be`, E2 4 of 4, in-alarm
75.2 % to 39.3 %, 0.184 h) and `fall_then_still_v0: fall_v3 @ 0.95` (sha8 `67954c04`,
chute24 111 of 200 at 7.3 eps/h, UR 13 of 30 at 49.4 eps/h, exam-fitted provisional).

`configs/pipeline/rules.yaml`, a comment pointer only, because RuleConfig rejects new keys,
so there is no numeric change:

```diff
--- a/configs/pipeline/rules.yaml
+++ b/configs/pipeline/rules.yaml
@@ (header comment block, after the docs/CALIBRATION.md pointer)
 # protocol that eventually replaces them, including the four preconditions it needs before
 # step 1 can even start, is docs/CALIBRATION.md. The replay-mode stance around it is
 # docs/PIPELINE.md.
+#
+# The verifier confirm thresholds do not live here, because RuleConfig rejects unknown
+# keys: they are per-head `verify.heads.<rule_id>.confirm_threshold` in the pipeline config.
+# Measured PROVISIONAL candidates, with the fitting surface and onnx sha8 each number
+# is tied to, are in research/operating-points.md (2026-09-05).
```

`docs/CALIBRATION.md`, caveat lines to add under the status stub:

```diff
--- a/docs/CALIBRATION.md
+++ b/docs/CALIBRATION.md
@@ (end of the status block)
 > structure, not a tuned product.
+
+**Interim measured candidates, 2026-09-05, recorded before this protocol.**
+`research/operating-points.md` records measured provisional verifier operating points from
+recorded artefacts. The seizure broad-head threshold was fitted on the non-exam out-of-fold
+rotation pool and only reported on the exam artefacts. The convulsive head's threshold is
+transferred, not fitted. **Every fall number is fitted on exam negatives**, because no non-exam
+fall surface exists, and it is provisional until pilot shadow-mode data replaces it. None of
+that executes the protocol below, none of it is a false-alarm rate, and every value travels with
+its ONNX sha8 and its denominator in hours, as the hash rule requires.
```

## 7. Blockers and limits of this pass

- **Denominators are tiny.** 0.7604 h out-of-fold, 0.184 h on E2, 0.273 h on chute24 and
  0.081 h on UR ADL. Single-episode granularity is 1.3 to 12.3 eps/h depending on surface,
  so treat every episode rate as a count with its hours, not as a rate estimate.
- The seizure head of record has no non-exam surface. Its 0.8 is a transferred policy,
  and its better-looking 0.85 row is exam-observed only.
- **All fall numbers are exam-fitted**, as sections 3 and 4 and every proposed caveat line
  say. The UR persist result is additionally bounded by extraction coverage, 16 of 30
  zero-window events, so persistence for falls is unmeasured rather than merely poor.
- **E1, the focal exam, has no negative recordings**, so the broad head's focal catch cost
  at 0.8, 10 of 12 down to 7 of 12, has no burden column to trade against.
- decision_layer_study latencies are window-granular and include the window span, and the
  fall in-alarm figures inherit the instrument's 12 s alarm span on 8 s windows.

The byte-exact reproduction checks and the full 23-rule grids per surface are not carried as
a file. They regenerate from the recorded artefacts named above with
`scripts/events/decision_layer_study.py`, whose logic the recomputations here reproduce.

---

## 8. Applied 2026-09-05

The section 6 diff went into the tree on that date, and the two section 5 gaps have code
seams, **both off by default**, because sections 1 and 3 measured them as losses at the
recommended points. Nothing below changes any number above: the thresholds are the same
PROVISIONAL, exam-fitted or out-of-fold-fitted values, and turning either seam on is still a
config edit nobody has data to justify. This section records the state on 2026-09-05; the
`verify:` block was later repointed to the models of record, as the last bullet of this
section says.

**Config, the section 6 diff as proposed.**

- `configs/pipeline/replay_demo.yaml`, the `verify:` block. `policy: annotate`;
  `seizure_candidate_v0` points at `outputs/verifier/conv_finalB` at **0.8** (sha8
  `b9e04d03`, T 1.0251); `fall_then_still_v0` points at `outputs/verifier/fall_v2pose` at
  **0.9** (sha8 `b66c3a0a`, T 1.486). Verified to load: `OnnxEventVerifier.from_config` in
  this repository builds both graphs, sha-checked, and reports
  `fall_then_still_v0:b66c3a0a+seizure_candidate_v0:b9e04d03`.
- `configs/pipeline/rules.yaml`, the verifier-threshold comment pointer in the header
  block, with no numeric change.
- `docs/CALIBRATION.md`, the caveat block inside the status section.
- `configs/pipeline/synthetic_demo.yaml` and `configs/pipeline/egress_demo.yaml`, the
  header comment pointers only, saying why those two stay verifier-less: their fixtures are
  abstract points, and the egress head's subject is a reaped track that any verifier must
  answer as unavailable. Neither had a `verify:` block and neither gains one.

Two additions to the proposed diff text, both non-numeric. First, the block records that
`outputs/` is gitignored, so the two run directories exist only where the interim
event-verifier training runs were done, and the pipeline refuses to start without them,
raising `VerifierBackendError` rather than scoring with unaccounted-for weights; deleting
the block returns the demo to running without a verifier. Second, the section 6
equivalents on the lane of record (`convulsion_v2_real` at 0.8, sha8 `07c399be`, and
`fall_v3` at 0.95, sha8 `67954c04`) ride along as a comment so the model-of-record numbers travel with the
file.

**Seam 1, persistence and hysteresis (section 5, item 1), off by default.**

- `src/vividly/models/event_verifier.py`: `VerifierHead.persist_k`, default 1, at :184 and
  `hyst_hi`, default None, at :188, both checked by `validate_decision_policy` from the
  post-init at :191; `verify()` scores the k windows at
  `firing.onset_t + i * head.spec.stride_s` at :400-408, with the per-window
  graph call factored into `_score` at :415. It confirms when all k are at or above
  `confirm_threshold`, or immediately when any is at or above `hyst_hi`, which is the
  study's persist:k@thr and hyst:hi|k@lo, so section 3's `hyst:0.9|2@0.65` is now
  expressible as `persist_k: 2, confirm_threshold: 0.65, hyst_hi: 0.9`.
- Off means byte-identical: at `persist_k == 1` the anchor offset is skipped
  arithmetically, the graph is called once with the same array, and the verdict carries the
  same confidence float. Confidence under persistence reports the binding score: the
  minimum over the k for an all-k confirmation, and the maximum for the hysteresis leg and
  for a rejection.
- One judgement the section 5, item 1 sketch did not specify: if any of the k windows cannot be
  featurised, the verdict is **unavailable, not rejected**, because an incomplete pattern
  is missing evidence, which is the same abstention rule the single-window path follows.

**Seam 2, camera-level episode merge (section 5, item 2), off by default.**

- `src/vividly/rules/base.py`, `RuleConfig.camera_refire_after_s`, default **0.0**, which
  is off, validated as non-negative by the same class's post-init check.
  `configs/pipeline/rules.yaml` carries it at 0.0 under the `camera_refire_after_s` key,
  with the PROVISIONAL, acute-event and CALIBRATION.md comment the yaml pin requires.
- `src/vividly/rules/engine.py`, the second debounce map `_last_fired_camera` keyed by rule
  id, the counter `n_camera_suppressed`, the check inside `_admit`, and the accessor
  `last_fired_camera_t`. Two behaviours are worth stating because section 5, item 2 did not: the
  camera check runs after the per-rule, per-track check and before the per-track timestamp
  is written, because a firing it suppresses did not happen and must not start that
  person's own 900 s refire window; and its map deliberately survives the reap cleanup,
  because its subject is the camera, not the person.
- Correction to the parenthetical in section 5.2: the one-camera-per-process fact is true,
  since a `Pipeline` opens exactly one source, but it is not what the camera-singleton
  guard of the time enforced; that guard kept exactly one fisheye camera model in the
  camera package. Neither the guard nor the package exists in this repository: the fisheye
  rule is now an unconditional ban, enforced by
  `scripts/guards/check_no_fisheye_primitives.py`. The conclusion, that rule scope is camera
  scope, stands; only the citation was wrong.

**Tests.** `tests/test_operating_point_seams.py`, 25 tests over 62 cases, all green, with
no graph loaded, using scripted `session.run` fakes. It pins the applied thresholds and
their provenance strings, both default-off byte-identity claims, k-consecutive and
hysteresis behaviour, persistence abstention, camera-debounce suppress-then-admit across
two tracks, per-rule scoping, survival of a reap, and the config refusals. Also green after
the change: `test_event_verifier_backend.py`, `test_rules.py`, `test_egress.py`,
`test_dashboard.py`, `test_demo_platform.py`, `test_alert_path_raw_signal.py` and
`test_verifier_features.py`; `lint-imports` passes 6 of 6 and the guards are clean.

**Hardening that came with the seams**, not in section 6 and with no effect on any valid
config: `verify.heads.<rule_id>` now refuses unknown keys (`HEAD_CONFIG_KEYS`) instead of
ignoring them, because a mistyped `confirm_treshold` silently keeping the 0.5 default is
the same failure `RuleConfig.from_mapping` already refuses for rule thresholds; and
`confirm_threshold` is range-checked to [0, 1]. `load_head` now validates the decision
policy before opening the graph, next to the calibration check.

**Still proposal only, or unchanged.**

- Every threshold above remains PROVISIONAL. None of the four preconditions in
  docs/CALIBRATION.md is met, so this is a measured provisional point, never a calibrated
  one.
- The block for the models of record, `convulsion_v2_real` at 0.8 and `fall_v3` at 0.95,
  was a comment at the time of this pass rather than a wired config. It is wired now:
  `configs/pipeline/replay_demo.yaml` attaches `seizure_candidate_v0` from
  `final_models/seizure` at 0.8 and `fall_then_still_v0` from `final_models/fall` at 0.95,
  which are the graphs `07c399be` and `67954c04` this section names.
- Both seams stay off. Sections 1 and 3 say why: persist:2@0.8 costs 2 convulsive-PG
  catches on the out-of-fold pool, UR ADL persist:2 catches 0 of 30, and the camera merge
  saves close to nothing at the fall and out-of-fold points, its one measured win being the
  E2 model of record at 0.8, 54.4 down to 32.6 eps/h on 0.184 h. The code exists so the trade can
  be taken on pilot data, not before.
- Section 5.3 is unchanged and was not attempted: `confirm_threshold` still cannot live in
  `rules.yaml`, and `RuleConfig` was not widened to let it.
- No re-measurement was done in this pass. Every number in sections 1 to 4 is the one the
  analysis pass recorded; nothing was re-run, re-fitted or re-scored.
