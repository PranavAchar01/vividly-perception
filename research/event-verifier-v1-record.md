# Event verifiers, first generation: training and held-out test numbers (2026-08-26 to 2026-08-28)

**Disclosure, which applies to every number below.** These are interim prototype event
verifiers (`vividly_weight_class: prototype`), internal demonstration only. They are trained
on public acted-fall footage (le2i_fdd, registered `train_interim`) and on synthetic
pose-space convulsion motion. No number here is clinical detection performance; the test sets
are engineering-only under the recorded grants, and every number travels with its checkpoint
sha256, as rule 10 of `POLICY.md` requires. Run directories named below sit under
`outputs/`, which is not in version control, so those paths locate a run rather than a
tracked file.

The three demonstration event models recorded as final_v1:

| model | trained on | checkpoint (onnx sha256, first 8) |
|---|---|---|
| fall final_v1 | le2i_fdd track windows (130 videos, 45,370 rows, extraction c6f1100) | `080b5d86` |
| convulsion final_v1 (= finalv1synthetic) | synthetic convulsion motion (820 seqs) + le2i real-motion negatives | `9ed32801` |
| egress_without_discharge_v0 | rule head, not a trained model | n/a |

Feature stack for both trained heads: 8-channel per-track windows
(vy, vx, h_rel, aspect, angle_norm, angle_valid, kpc, present), n_steps=64, extracted at the
detector, pose and tracker operating point in use when these heads were trained
(rtmdet-m person `4f4d7e07`, rtmpose-m body7 `5c0a4bf6`, bytetrack trackers 2.6.0). The
replay pipeline has since moved to the models of record on the lane pages under
`final_models/`.

Vocabulary, so every later section resolves against the lane table in README.md: in this
record the word eyes names the person detector that produced a track tree, which is the eyes
lane, and the pose model is always named separately by its own hash. Phrases such as stock
eyes, chained eyes and lr25 eyes name successive weights of that one lane, never a pose
model. The same shorthand appears in the extraction configurations under
`configs/events/` and carries the same meaning there.

## fall final_v1: training (le2i_fdd, registered train_interim)

- 541 windows (163 pos / 378 neg) over 122 videos, holdout 0.2 by video.
- val AUC best **0.99924** (in-domain: same dataset as training, held-out videos).
- val operating points: thr 0.5 gives recall 1.00 / precision 0.85; thr 0.8 gives recall
  0.97 / precision 0.89.
- run: `outputs/verifier/fall_v1/report.json`, git c6f1100, pt `7d21e906`.

## fall final_v1 against UR Fall: held-out test (never trained, registered test_only)

`configs/events/eval_fall_urfall_v1.yaml`, model `080b5d86`, git c6f1100:

- **window AUC 0.957 / video AUC 0.931** (52 videos with valid windows, 104 windows).
- **Zero missed falls at every measured threshold**: video_fn = 0 at thr 0.5, 0.8,
  and 0.9 across the 14 scored fall videos (all scored at or above 0.95, most at or above
  0.99).
- The false positives are UR Fall's deliberate lie-down ADL sequences: at thr 0.9,
  15 of the 17 video FPs carry the dataset's own lie-transition flag. The verifier
  sees an 8 s upright-to-lying motion signature; separating intentional lying-down from
  a fall is the job of the live `fall_then_still_v0` cascade (onset dynamics plus
  prolonged stillness) that this head plugs into, not of the verifier window alone.
- Coverage caveat, stated plainly: only 14 of the 30 extracted fall videos produced a
  window under the contain policy (window_s 8.0, min_present 0.35), because UR Fall clips are
  short and tracks fragment. 52 of 70 extracted videos scored overall. The number
  covers what it covers; widening window coverage is future evaluation work, not a reason to
  restate this run.
- Deep metrics (2026-08-26, `metrics_deep.json`): PR AUC 0.720; window sensitivity
  1.0 at every threshold up to 0.9 (spec 0.711, F1 0.519 at 0.9; best F1 0.609 at
  0.95 with spec 0.80). All 14 scored falls fire within **2.1 to 3.1 s of onset**
  (median 2.55 s) at thr 0.8. Class separation: falls median-max 0.999,
  lie-transition ADLs 0.982 (the confounder, 15/16 over 0.9), other ADLs 0.070
  (only 2/22 over 0.9). The model separates falls from normal activity almost
  perfectly and cannot separate them from deliberate lie-downs, which is the
  cascade's job.

## convulsion final_v1 = finalv1synthetic: training

- Positives: synthetic pose-space convulsion motion only (no acted seizures
  exist and acting one is barred by rule 4 of `POLICY.md`). Negatives: synthetic kinds plus
  every le2i real track window force-labelled 0 (real falls as convulsion-negatives, by
  design).
- val AUC best **0.99681** (in-domain synthetic and le2i holdout; this number says the
  head learned the synthetic signature, nothing more).
- run: `outputs/verifier/convulsion_v1/report.json`, git c6f1100, onnx `9ed32801`.

## convulsion final_v1 against WU-SAHZU-EMU: the real-seizure holdout

`configs/events/eval_convulsion_wusahzu_v1.yaml`, model `9ed32801`, run before any
real-data training existed (pre-registered). The recorded result:

- **0 of 14 convulsive (PG) seizures caught at any threshold.** Every PG event
  video's max score is at or below 0.16; ten of fourteen are at or below 0.005. Window AUC
  0.389 (events-only preliminary; the 3 non-seizure recordings were still extracting).
- Not an extraction artefact: 1,874 windows over the 33 videos, 564 positive; every
  PG video produced 14-150 windows; tracks and features flow normally.
- The four videos the model does flag (0.91 or above) are all focal (P) events,
  pat08 and pat09, so it fires on other ward motion, not on the convulsions.
- Reading, stated plainly: the synthetic pose-space convulsion signature does not
  transfer to real bed-bound EMU seizures (blanketed patients, distant wall camera,
  tonic-then-clonic evolution). This branch was pre-authorised: the three-variant
  comparison (finalv1synthetic / finalv1synthetic+real / finalv1real) on a
  disjoint-patient split proceeded, the winner took the demonstration slot, and it was
  retrained on a re-extracted track tree. The seizure model of record is now
  convulsion_v2_real, ONNX `07c399be`, recorded on `final_models/seizure/README.md`.

## Full-grid and coverage pass (2026-08-28)

Instrument pass over the frozen windows.jsonl of all 10 recorded evaluations, with no new
test looks (no model rescored, nothing selected; more statistics from the same
frozen predictions). Changes: `metrics_deep.py` now applies the evaluation's own
location filter, iterates the extraction metas for event denominators (an event in a
zero-window video counts as missed and never vanishes, the hole class that made
the first v4 run void), and emits a `coverage` block; `decision_layer_study.py`
adds per-camera episode collapse next to per-track; the compendium renders
coverage and alarm-experience lines for every evaluation. Regression: every previously
published ROC, PR, catch and FP figure reproduced exactly.

**Correction, UR Fall denominators.** The dataset holds 70 videos (30 falls),
not the 52 (14 falls) later summaries implied. 18 videos produce zero windows,
all of them clips of 1.8 to 6.2 s, structurally shorter than the fall lane's 8.0-s
window under its `contain` policy (a track must span the full window), so they
cannot be tested under deployment geometry at all; tracks exist, the eyes saw
the person, and the clips simply cannot hold a deployment window. (Correction to the
first publication of this note: the min-presence floor of 4.2 s quoted there was the
seizure lane's 12.0 times 0.35 arithmetic, and `fall-09` at 6.2 s is structural
like the rest, not a tracking shortfall.) The v1 UR section
recorded this caveat on day one and summaries dropped it; the instrument now
forces it: every UR event row reads **14/30**, not 14/14. Both fall models
share the identical hole (same extraction), so nothing on either lane changes. Multicam: 4
of 25 fall events have no overlapping window (chute04__cam6, chute19__cam7,
chute20__cam8, chute23__cam8 ev1), which are coverage gaps, not low scores. Seizure
exams are clean: 16/16 videos scored, 13/13 events windowed, footage coverage
0.88 to 0.95.

**New rows measured on frozen scores** (see `research/stats-compendium.md` for the full
grid): per-camera alarms/h and negative time-in-alarm for v1/v1synthreal/v2/v3/v4
and both fall models on both fall exams; fall v2's first deep metrics, with UR
ROC/PR 1.000/1.000, multicam PR 0.634 (v1: 0.487), UR in-alarm 46.9% @0.5 and
4.1% @0.9 against v1's 92.0% and 74.0%. v4's per-camera column exposes the inversion
plainly: 6.4 cam-alarms/h @0.5 against 22.2 @0.9 only because at 0.5 the alarm never
turns off (95.7% in-alarm).

**Supplementary exam, the never-tested patients.** pat06/08/10/12 (label-uncertain in the
registry, all focal, never in any train or test set) were sitting
extracted in tracks_v2. Pre-declared configs
`eval_wusahzu_unallocated_{v2,v4}.yaml` scored the 9 available videos once
each: **v2 `07c399be`: 8/9 @0.5** (median 15.6 s), ROC 0.707 / PR 0.182;
**v4 `6d205fee`: 7/9 @0.5, 6/9 @0.8**, ROC 0.778 / PR 0.339, six of nine event
maxima at or above 0.975, so v4's focal mastery transfers to unseen patients. Caveats
travel with it: focal events are outside the convulsive product target of the seizure rule,
onset labels carry the recorded label uncertainty (latencies indicative), no negative footage
exists for these patients, and this supplement is not the primary exam and retires if the
patients are ever moved train-side. tracks_v2 holds only 9 of their 12 videos
(pat06 Sz2P/Sz3P and pat08 Sz1P missing; extraction queued, tracks_v1 has all 12).

**Supplement completed, 12 of 12, the same day.** The 3 missing videos extracted
through the chained eyes and the surface re-ran uniformly (coverage 12/12 for
all three models). Final rows on the second exam surface, events @0.5 / @0.8 and window
ROC/PR:
**v2 `07c399be` 11/12 / 6/12, 0.643/0.170** (miss: pat08__Sz3P);
**v4 `6d205fee` 10/12 / 9/12, 0.726/0.262** (misses: pat10 Sz1P and Sz2P);
**v6 `10cdbe78` 12/12 / 10/12, 0.531/0.130**, where v6's home-quiet diet, rejected
on the primary exam, is the only model with perfect recall on the fresh
patients (its low ROC is the same score-saturation chatter measured there).
The exam-side allocation is already returning value: three models, three different miss
patterns on data none of them trained on.

**What followed this pass.** Two of the five declared expansions ran. The multicam
all-cameras extraction covered all 192 camera-videos through stock eyes and became an
additional fall exam surface, with one uniform look per fall model, recorded below. The
3 missing unallocated extractions landed through chained eyes, so the supplement covers
12/12, also recorded below. Three items were declared and never run: a short-clip stress
instrument for sub-window clips (UR's 16 dark falls), reported as its own labelled row and
never merged with deployment-geometry numbers; a full-test-split detector sweep beside a
frozen instrument whose sets rule 1 of `POLICY.md` has since retired; and a `fall-09`
single-video tracking diagnosis.

## The seizure lane is information-starved, not only data-starved (2026-08-28)

Measured on this date, and it reframes every seizure result above. The hypothesis under
test: a convulsion is rhythmic motion at about 1 to 5 Hz while the normal in-bed movement it
must be told apart from (rolling, scratching, adjusting bedding) has the same
amplitude and no periodicity, and nothing in `CHANNELS` expresses periodicity, so the
model was never given the one feature that defines the event.

Native-rate rhythmicity channels were built for this (`RHYTHM_CHANNELS`, opt-in,
computed on raw in-window samples before the step grid; `src/vividly/verifier/features.py`).
They were then measured against the real WU-SAHZU test split, 4,880 labelled
windows, 285 positive:

| signal, alone | ROC |
|---|---|
| rhythm_strength (box centre) | 0.429 |
| rhythm_freq | 0.494 |
| rhythm_band_ratio | 0.412 |

Chance is 0.500. The rhythm of the box centre carries nothing, and the sampling
rate is not the explanation: the extraction runs at a median **29.9 Hz**, so the whole
clonic band is representable with room to spare.

A second probe asked whether any other surviving signal carries the rhythm, since
during a convulsion the patient stays put and the limbs jerk (5,455 windows, 891
positive): width 0.387/0.405, aspect 0.382/0.395, area 0.396/0.407, height
0.435/0.456, torso angle 0.530/0.514, keypoint confidence 0.347/0.365 (peak and band
share each). Nothing separates, and several inverted: convulsion windows are
*less* periodic in box shape than negatives, consistent with the tracker shattering
(17-33 fragments against 1-2 when calm) and turning the box trace into noise.

**The cause, confirmed in the extractor on 2026-08-28.** At the time of this measurement
`scripts/events/extract_track_sequences.py` had the 17 keypoints in hand (`obs.keypoints`)
and wrote only two derived scalars from them, a torso angle and the mean confidence,
discarding every joint position. So limb motion, which is where a convulsion actually
lives, was destroyed at extraction time and no verifier trained on those sequences could
see it. The rhythm hypothesis was not refuted; it was **unmeasurable from what the
extraction persisted**. This finding is what motivated the extractor's `emit_keypoints`
option, which writes the full COCO-17 keypoints into the track rows.

Corroborating measurement from the same probe: on this surface a single raw channel,
`vx` window-mean, scores **ROC 0.659** against the trained v2 model's 0.698. Six
model versions separate by about 0.04 above one hand-computed feature. That is the
signature of a model with almost nothing to learn from, not of a model that needs another
recipe.

Consequence for the plan that followed: the AJILE12 diet and this finding composed rather
than competed, because AJILE12 ships 9 upper-body keypoints **including both wrists**, exactly
the limb channels this probe identifies as missing, in the matched EMU context. The next
seizure attempt was therefore declared as persisting keypoints in extraction, then deriving
limb-relative motion and its rhythmicity, then training against an AJILE12 negative diet
expressed in the same 9 joints. Nothing in that chain was a new model recipe, which
is precisely why it was worth doing before another one. Both steps were run and are
recorded elsewhere: `research/seizure-limb-channel-gate.md` holds the second and final
limb-channel attempt, which did not save the feature, and
`research/seizure-ajile12-negative-control.md` holds the AJILE12 diet's channel-masking
control.

Standing rule reaffirmed at the time: `RHYTHM_CHANNELS` stayed off by default and no model
claimed them until a keypoint-bearing extraction existed, because a channel that measures
the wrong signal is worse than a missing one. That extraction exists now, under
`emit_keypoints` in `scripts/events/extract_track_sequences.py`, and the keypoint-bearing
channel groups are `LIMB_CHANNELS` and `KP_CHANNELS` in `src/vividly/verifier/features.py`.
All three groups remain off by default and are enabled per training configuration.

## seizure v6 (matched-geometry quiet diet): rejected, v2 keeps the slot (2026-08-28)

Pre-declared recipe (commit 544bfab): v2 unchanged plus the matched-geometry sets' quiet clips
as force_label:0 negatives through the same chained eyes (gmdcsa24 ADL locations and
caucafall no_event_only). This was the recorded request for quiet footage, built to obey the
measured finding with matched camera geometry (side-view and upper-corner home rooms).
Train: 1,485 windows, with positives identical to v2 (164) and negatives 1,321 = v2's
942 plus only 379 quiet windows (the matched-geometry clips are short, and the 12-s seizure
window extracts few windows per clip). Val 0.983, T=0.625 (healthy). Verifier
`10cdbe78`. One look per surface, consumed here; instruments complete from
birth (coverage clean: 16/16 and 9/9 videos scored).

| | v2 @0.5 | v2 @0.9 | v6 @0.5 | v6 @0.8 | v6 @0.9 |
|---|---|---|---|---|---|
| events /13 | **12** | 6 | 11 | 11 | 11 |
| convulsive /6 | **6** | 3 | 5 | 5 | 5 |
| median latency | 6.95 s | 31.2 s | 6.7 s | 10.2 s | 10.2 s |
| FP windows /h | 962.6 | 57.2 | 1628.7 | 903.3 | 649.1 |
| FP eps/h per-track | 103.8 | 13.8 | 110.2 | 120.8 | 100.6 |
| per-camera alarms/h | 21.2 | 9.5 | 9.5, see note | 20.1 | 27.5 |
| **neg time-in-alarm** | 72.4% | **8.1%** | **90.1%** | 68.9% | 57.3% |

The v6 @0.5 per-camera figure is the alarm-never-off artefact again: fewer per-camera episodes
only because the alarm barely turns off, so the in-alarm row is the truth. Window ROC 0.674 /
PR 0.184 (v2 0.698/0.115); event against quiet medians 0.910/0.284; scores saturate
(every caught event max at or above 0.993, catches flat at 11 across all thresholds, because
T=0.625 sharpens toward the extremes).

**The two misses:** pat09__Sz1P (0.19; v2 also misses it at 0.31) and, the
disqualifier, **pat03__Sz1PG at 0.079, a new convulsive miss** that v2 catches.
Something in the home-ADL negative manifold sits close to this event's motion;
forcing it to 0 taught v6 to zero a real convulsion. The recipe's own header
named this exact risk before the run.

**Supplement on the second exam surface:** 9/9 @0.5, the best unallocated-patient recall
of the three models (v2 8/9, v4 7/9); 7/9 @0.8, ROC 0.669 / PR 0.167, with
the same chatter caveat.

**Outcome.** v6 is rejected; **final_v2 `07c399be` keeps the demonstration slot at
0.8**, unchanged. The recorded target, catches at the 0.5 level with about 8% of quiet
footage in alarm, is missed by roughly 7 times. The standing finding, on its third
measurement, is now sharpened: a negative diet must match the deployment context, not merely
the camera geometry. Overhead wards (v4), side-view home rooms with actors (v6) and the fall
mirror (fall v2) all bought the wrong quiet. What EMU rooms contain, patients in beds,
caregivers at bedside and monitor clutter, is what none of the available quiet footage shows.
Every remaining lever converges on one item: **acquiring real negative footage resembling an
EMU room, which requires an owner decision** (raised since the v2 verdict, now backed by
three converging measurements). Secondary recorded option: a v7 ablation
(v6 minus caucafall, gmdcsa-only diet) is not pre-declared, because the dose is too
small (379 windows) to expect a different finding.

Artefacts: `outputs/verifier/convulsion_v6_real/{report.json,
eval_wusahzu_testsplit/*, eval_wusahzu_unallocated/*}`.

## fall v3 (matched-geometry positive diet): takes the fall slot (2026-08-28)

Pre-declared recipe (commit b78fc4d, committed before any run): v1's recipe plus
gmdcsa24_tracks_v1 and caucafall_tracks_v1 as positive-side diversity through the
same stock eyes, with no retired overhead ward negatives. It is the mirror experiment to
v2, obeying the measured finding that negatives must match deployment geometry; here the
lever is matched-geometry positives instead. Train: 881 windows (190 pos / 691 neg)
from 282 of the 389 indexed videos, val AUC 0.996 (train-side, never
accuracy). Verifier `67954c04`, T=1.025 (healthy). One look per exam, consumed
here. UR denominators use the corrected 14-of-30 coverage accounting.

| | v1 `080b5d86` | v2 `e9cebe4d` | v3 `67954c04` |
|---|---|---|---|
| UR window ROC / PR | 0.957 / 0.720 | 1.000 / 1.000 | 0.984 / 0.815 |
| UR falls (of 14 scorable) @0.5 to 0.9 | 14 at all | 14 at all | **14 at all** |
| UR FP videos @0.5 / 0.8 / 0.9 | 21 / 19 / 17 | **11 / 4 / 1** | 17 / 13 / 6 |
| UR lie-transition FPs @0.9 (of 16) | 15 | **1** | 6 |
| UR neg time-in-alarm @0.5 / 0.9 | 92.0% / 74.0% | 46.9% / 4.1% | 74.3% / 26.2% |
| multicam window ROC | 0.821 | 0.846 | 0.816 |
| multicam fall videos @0.5 (of 23) | **21** | 15 | 20 |
| multicam fall videos @0.9 | **20** | not measured | 14 |
| multicam misses @0.5 (max score) | 07:0.37, 19:0.40 | 8 videos, 0.02 to 0.08 | 02:0.06, 07:0.27, 19:0.11 |
| chute24 crouch confounder max | 0.92 (fires @0.9) | 0 FP | **0.54** (fires @0.5 only) |
| multicam neg time-in-alarm @0.5 | 46.6% | 0% | **14.7%** |

**Outcome: fall v3 takes the fall slot** (demonstration map updated,
`VERIFIER_RUN_DIRS` and confirm_threshold 0.8, which later measurement moved to 0.95;
`final_models/fall/README.md` carries the operating point of record). The reasoning, stated
plainly:
v1's only remaining advantages, chute02__cam4 (caught at 0.53) and 5 more
on-moment event overlaps, exist solely at operating points carrying 46.6%
multicam and 92% UR time-in-alarm, which no deployment would run. At matched
catches (20/23), v3 @0.5 equals v1 @0.9 with a weaker confounder and keeps
quiet range v1 never had (UR 14/14 @0.9 at 26.2% in-alarm against v1's 74.0%). The
matched-geometry mission is measured: matched-geometry positives bought back 5 of the 6
multicam videos the v2 negative diet lost while keeping most of v2's UR quiet.
fall_v1 and fall_v2 stay banked on disk.

**Residuals, recorded:** (1) high and overhead cameras remain the weak axis,
since all 3 misses are cam4/6/7 and @0.9 multicam recall drops to 14/23; the
declared all-cameras multicam expansion (24 to 192 videos) measures exactly
this axis next. (2) On-moment event overlap regressed (14/25 multicam events
@0.5 against v1's 19/25) while video catches held, because v3's fires shift toward the
post-fall aftermath (CAUCAFall label spans include the aftermath); within the
30 s alerting tail it catches 20/25 at 2.65 s median, and the live cascade
alerts on exactly that tail. (3) The pod train venv has no onnxruntime, so fall
exams run on the Mac; the v3 evaluation chain died on this and was completed on the
Mac (evaluations are deterministic ORT CPU, and nothing about the look changed).

**Allocation decision, 2026-08-28.** The unallocated
patients pat06/08/10/12 are allocated exam-side. The
`eval_wusahzu_unallocated_*` supplement becomes a permanent second seizure
exam surface (12 focal events once the 3 missing extractions land), the
primary 13-event exam is unchanged, and the earlier train-side proposal is
closed: these patients never enter a training manifest. Every future seizure
version gets one look at both surfaces.

## multicam all-cameras exam: the picked-view bonus measured, the fall-slot outcome survives (2026-08-28)

The full-testing expansion's fall surface landed: all 192 Montreal Multi-Cam
views (184 fall videos / 200 fall events plus chute24's 8 confounder views),
extracted once with the fall lane's stock eyes at the frame shifts probe-validated
in `src/vividly/data/convert/multicam_fall.py`, and each fall model given its one
pre-declared look (`eval_fall_multicam_allcams_fall{v1,v2,v3}.yaml`; runs under
`outputs/verifier/fall_*/eval_multicam_allcams/`). 4,588 windows, 609 positive.

**First finding: the frozen 24-video exam was a picked-view exam.** Its keys
are one hand-chosen camera per scenario (`chute01__cam8`, `chute02__cam4`, and so on),
not camera 1: the original curation took the best view of each fall. The
all-cams surface removes that bonus and adds the 7 unpicked views per scene.
Reproduction on the 25 shared (chute,cam) events is solid, at 23/25 (v1), 20/25
(v2) and 20/25 (v3) agreeing within 0.15; the drifters are borderline single tracks
that flip between extraction runs, in both directions.

| model | picked-view AUC | picked ev@0.5 | all-cams AUC | all-cams ev@0.5 |
|---|---|---|---|---|
| fall v1 `080b5d86` | 0.821 | 19/25 (76%) | 0.778 | **151/200 (75.5%)** |
| fall v2 `e9cebe4d` | 0.846 | 14/25 (56%) | 0.770 | 86/200 (43%) |
| fall v3 `67954c04` | 0.816 | 14/25 (56%) | **0.793** | 118/200 (59%) |

- **v1's raw catch rate survives the angle axis untouched** (76% to 75.5%): the
  Le2i-only recipe generalises across views, but it pays 25.5% time-in-alarm
  on the chute24 confounder views (persist:2@0.5: 163/200 caught, 44 fires/h).
- **v2 is the angle casualty** (56% to 43%), including a dead camera: 1/25 on
  cam1 with maxima 0.00-0.11. Its quiet is real (0 to 1.2% in-alarm) but the
  multicam recall loss that cost it the slot deepens off the picked views.
- **v3 keeps the best AUC on both surfaces and slightly improves its catch
  rate off the picked views** (56% to 59%). Decision layer on the confounder:
  persist:2@0.5 gives 127/200 at 6.6% in-alarm; persist:2@0.65 gives 117/200 at 2.7%;
  the demonstration operating point persist:2@0.8 gives 108/200 at 1.5%. **The fall-slot
  outcome survives its first out-of-sample angle test.**
- Per-camera catches @0.5 (of 25 each): v1 spreads 15-22 across all 8 cams;
  v3 runs 13-20 except cam5 (8/25, its one weak angle); v2 runs 1-16.
  Negative-diet models lose specific angles; the no-diet model loses quiet.
  The same finding again, now on the angle axis.
- **Coverage honesty:** 25/200 events (12.5%) had no overlapping window in
  this extraction, which are structural misses inside every denominator above (worst:
  cam4 with 7, the view where tracks most often fail to span the fall);
  footage_covered_frac 89.8%. These are extraction-layer losses, not model
  scores, and they cap every model's ceiling at 87.5% on this surface.
- Alarm-experience caveat: the negative base is one staged confounder scene
  seen from 8 cameras (about 13 min total), so fires/h and in-alarm % here rank
  models on the crouch confounder; they are not ward false-alarm rates.

### Labelled study: the price of the contain policy, measured (2026-08-28)

Declared stress study, slot holder only, never an exam row
(`study_fall_urfall_overlap_fallv3.yaml`,
`study_fall_multicam_allcams_overlap_fallv3.yaml`): fall v3 re-scored with
window policy `overlap` (geometry, stride and n_steps untouched; partial-presence
windows are in-distribution, since the `present` channel and min_present 0.35 are
exactly as trained). Question: how much of the structural-miss ceiling is the
windowing policy against tracking depth?

| surface | contain (exam) | overlap (study) | still dark under overlap |
|---|---|---|---|
| UR Fall | 14/30 @0.5, 16 windowless | **17/30**, 13 windowless | 13 |
| multicam all-cams | 118/200 @0.5, 25 windowless | **124/200**, 19 windowless | 19 |

- Every newly scoreable UR event (3) is caught; on all-cams, overlap also adds
  extra windows over already-covered events (5 more catches that way).
- Discrimination is unharmed: window AUC 0.985 (UR) / 0.790 (all-cams,
  against 0.793 under contain); PR AUC rises from 0.363 to 0.391 on all-cams.
- **Alarm cost: none measurable.** The decision layer on the chute24 views is
  unchanged at every operating point: the demonstration point persist:2@0.8 goes from
  108/200 to 109/200 at the identical 1.5% in-alarm; persist:2@0.5 goes from
  127 to 130 at the identical 6.6%.
- The residual 13 (UR) and 19 (all-cams) dark events have tracks shorter than
  the 2.8 s presence floor (0.35 times 8.0 s): that ceiling belongs to tracking
  depth (tracks failing to span the fall), not to windowing, and no window
  policy recovers it.

**Correction, 2026-08-28.** The earlier recommendation in this section, to
switch the fall cascade's deployed window policy to `overlap`, was incorrect and is
withdrawn. `policy` is a labelling concept, not an inference one:
`windows_for_video` drops ambiguously labelled windows before featurising
(`if label is not None`, in `src/vividly/verifier/features.py`), and `featurize_window` never
reads `spec.policy` at all. The live cascade assigns no labels, since it scores every
window as it arrives, so those partially overlapping windows were already
being scored in deployment. Flipping the deployed policy would change nothing.

What the study actually measured is therefore a property of the exam, not of the
model, and it splits the 25 dark events cleanly:

- **6 of 200 (3%)** were never a model failure at all: `contain` refused to
  label them, so the exam could not see catches the deployed system would make.
  Fall v3's true recall is modestly better than its contain-policy numbers say.
- **19 of 200 (9.5%)** stay dark under either policy, being tracks that never span
  the presence floor. That residue is real, and it is a tracking-depth lever.

Standing consequence: the contain-policy exam numbers are a mild under-estimate
and stay the quoted ones (changing an exam to flatter a model is the thing the
protocol exists to prevent); the tracking-depth work is the honest lever; and
the deployed cascade config is unchanged because there was never a change to
make. Recorded rather than quietly deleted, because the incorrect version was
circulated before the code was traced.

## fall v2 (retired overhead ward-negative diet): split verdict, UR Fall dominated and multicam recall lost (2026-08-27)

The pre-declared fall v2 (commit 0915eb1) was v1's recipe plus the six retired
overhead ward recordings as force_label:0 negatives through the fall stack's own
stock eyes, plus the bounded-calibration instrument. Its training configuration is not
carried in this repository: the negative diet came from the quarantined
overhead-fisheye corpora, which rule 1 of `POLICY.md` bars from any training
manifest, so the arm is a record here and not a runnable recipe. Verifier
`e9cebe4d`, T=1.903 (healthy fit), train 5,894 windows (163 pos, positives identical to
v1), train-side val AUC 0.998. One pre-declared look on each held-out
set (same instrument as v1's numbers):

| | v1 `080b5d86` | v2 `e9cebe4d` |
|---|---|---|
| UR Fall window / video AUC | 0.957 / 0.931 | **1.000 / 1.000** |
| UR Fall videos (tp, fp) @0.5 / @0.8 / @0.9 | (14, 21) / (14, 19) / (14, 17) | (14, **11**) / (14, **4**) / (14, **1**) |
| UR Fall lie-transition FPs @0.9 (of 16) | 17 including transitions | **1** |
| multicam window / video AUC | 0.821 / 0.826 | 0.846 / 0.826 |
| multicam videos (tp, fp) @0.5 | **(21, 1)** | (15, 0) |
| multicam lowest fall-video maxima | 0.37 / 0.40 / 0.53 | **0.017 / 0.045 / 0.058 / 0.077** |

**UR Fall is dominated outright**: the same 14/14 catches at every threshold with
false-video counts collapsing from 21 to 11 @0.5 and from 17 to 1 @0.9, because the ward
negatives taught it to ignore exactly the lie-down-controlled motion that plagued v1.

**Multicam loses structurally.** v2 catches 15/23 fall videos against v1's 21/23
@0.5, and no threshold recovers them: the missed videos' score maxima sit at
0.02 to 0.08 (v1: 0.37 and above). The lost videos concentrate on high and overhead camera
angles (chute07__cam6, chute11__cam7, chute19__cam7, chute02__cam4 and others). Those
negatives are overhead ward footage, so overhead-looking falls were taught
into the negative class. The window AUC still rose because the bulk of the
pool ranks better; the video-level operating reality regressed.

**Outcome.** Neither model dominates, so **fall_v1 `080b5d86` keeps the fall
slot** (recall first: a detector blind to a third of falls from overhead-leaning
angles is not deployable, however quiet). fall v2 is banked as the UR Fall
proof and as the second measurement of the same finding the seizure v4 run
produced: **a negative diet buys in-domain quiet and costs out-of-domain recall; negatives
must match the deployment geometry.** The fall levers pre-declared at this point were the
second licence-reviewed fall dataset, for angle diversity on the positive side, and
angle-balanced negatives. The first of the two landed as gmdcsa24 and caucafall, and the
fall v3 section above records what it bought.

Artefacts: `outputs/verifier/fall_v2/{report.json, eval_ur_fall/, eval_multicam/}`.

## seizure v5 (pos_weight cap): discarded on the train-side validation split, no test look (2026-08-27)

The pre-declared v5 (v4 plus `pos_weight: 6.0`, one knob) trained and was judged
on the same train-side validation pool as v4, per the selection protocol: val AUC
0.972 against v4's 0.981, recall@0.5 0.854 against 0.927 (35 against 38 of 41 val positives)
for only 96 to 94 FP, and worse at 0.9 as well. The cap traded recall away without
buying quiet, because the EMU-negative score inflation is a domain problem, not a
class-weight problem. **v5 never touched the test set.** Verifier weights are
retained on disk (`convulsion_v5_real/`), unevaluated.

## seizure v4 (retired overhead ward-negative diet): recall ceiling proven, quiet not reached (2026-08-27)

The pre-declared v4 experiment (commit 537eb78) was v2's recipe unchanged plus
the six retired overhead train-split ward recordings as force_label:0 negatives
through the same chained `55c2c242` eyes. Its training configuration is not carried in this
repository, for the same reason as fall v2 above: the negative diet came from the
quarantined overhead-fisheye corpora barred by rule 1 of `POLICY.md`. Train pool:
4,852 windows (164 pos, positives identical to v2, plus 3,746 ward-negative
windows on top of v2's 942). Verifier `6d205fee`, T=1.346 (healthy fit,
degenerate_fit false), train-side val AUC 0.981. Exam: the identical
285-positive / 4595-negative tracks_v2 window pool as v2, so every number below is
directly comparable. (A first v4 run trained against a pod tracks_v2 tree that
silently held only 24 of 33 videos, that is 10 of 13 exam events; it is void,
preserved at `convulsion_v4_real_VOID_incomplete_tree`, and the redo on the
repaired tree is the declared test look. Lesson recorded: count metas before
trusting same-named extraction directories across machines.)

**What the negative diet bought, the first full sweep:**

| | v2 @0.5 | v4 @0.5 | v4 @0.8 |
|---|---|---|---|
| events caught (of 13) | 12 | **13** | **13** |
| convulsive_PG (of 6) | 6 | **6** | **6** |
| median latency | 6.95 s | 7.24 s | 24.0 s |
| event score maxima | 0.31 to 0.99 | **0.83 to 1.00** | not measured |
| FP windows /h | 962.6 | 1902.9 | 800.6 |
| FP episodes /h (per-track) | 103.8 | 109.1 | 104.9 |
| **negative time-in-alarm** | **72.4%** | **95.7%** | **73.7%** |

Every event max is at or above 0.826, and the focal events v2 missed at 0.308 now score
0.85 to 1.00. 13/13 holds even at 0.8. The recall problem is solved and the
margins are real.

**What it did not buy: quiet.** The new time-in-alarm instrument (fraction of
seizure-free footage with any window over threshold, union across tracks) shows
v4@0.5 is in alarm 95.7% of the time: its per-camera figure of 6.4 episodes/h is an
artefact of the alarm almost never turning off, and per-track episode rates
invert meaning at these fire rates. The honest frontier: **v4@0.8 is close to v2@0.5's
alarm load (73.7% against 72.4%) with one more catch and 3 times the latency.** The
recorded target, catches at the 0.5 level with quiet at the 0.9 level, that is about 8%
in-alarm, is not met at any v4 operating point (v4@0.9: 9/13, 46%).

**Mechanism and the data insight that matters.** Those ward recordings are overhead
fisheye waiting rooms; the exam negatives are side-view EMU patient rooms with
caregivers. The diet taught the head that ward bustle is not a seizure and sharpened
true-event margins, but EMU-room caregiver motion, the actual FP source, became relatively
easier to fire on (pos_weight auto rose from v2's 5.7 to 28.6 with the diet,
inflating all scores; FP windows per hour up 2 to 6 times at matched thresholds). The lever
for quiet is now precise: negatives that look like deployment rooms (side-view
patient rooms, caregivers at bedside), not more overhead wards.

**Outcome.** v4 is banked as the recall-ceiling proof and is not shipped: **final_v2
`07c399be` keeps the demonstration slot at 0.8** (33% in-alarm against v4@0.8's 74%; a
demonstration that alarms three-quarters of the time is not usable). v4's one pre-declared
test look is consumed.

**Pre-declared next steps.**
- **v5, then in flight: v4 plus `pos_weight: 6.0`**, one knob, capping the
  class weight at v2's effective level, aimed exactly at the measured score
  inflation. Train-side validation decides whether it earns its test look.
- **Proposal recorded, requiring an owner decision and not run:** pat06/08/10/12, four
  unallocated patients with 9 annotated focal seizures, were excluded by the registry note
  as label-uncertain for convulsion training. v4 proves the model can master
  focal events given the right diet, and these patients are the only untapped real
  positives. Proposal: admit them train-side only (exam untouched) with a
  widened positive window to absorb onset uncertainty. It requires an owner decision
  because it amends a registry note.

Artefacts: `outputs/verifier/convulsion_v4_real/{report.json,`
`eval_wusahzu_testsplit/{report,metrics_deep,decision_layer_study}.json}`;
the void run is preserved beside it.

## seizure v3 (lr25-eyes re-extraction): rejected (2026-08-27)

The pre-declared v3 experiment: `train_convulsion_v3_real.yaml` is
`train_convulsion_v2_real.yaml` with exactly one line changed, because `sources.dir`
points at `wu_sahzu_tracks_v3`, the WU-SAHZU-EMU re-extraction through the
chained-lr25 eyes (`1e9445c5`) instead of the chained prototype eyes (`55c2c242`).
Same patients (train pat01/02/05/13, registry-enforced), same window geometry,
same seed, same pose (`e48f03d0`). Verifier: `d85f1a20`
(sha256 d85f1a203f5a7d and following), T=1.0 (the calibration fit was degenerate, so the
identity fallback applied; see the instrument note below). Trained twice, the second run
after the calibration-grid fix, and both runs produced the same picture.

**The train side looked like a gain; the held-out exam disagreed.**

| | v2 `07c399be` | v3 `d85f1a20` |
|---|---|---|
| train windows (pos) | 1106 (164) | 935 (142) |
| train-side val AUC | 0.772 | **0.937** |
| held-out window ROC / PR | **0.698** / 0.115 | 0.484 / 0.123 |
| events caught @0.5 (of 13) | **12** | 10 |
| convulsive_PG (of 6) | 6 | 6 |
| median latency | **6.95 s** | 32.4 s |
| event score maxima | 0.31 to 0.99 | 0.494 to 0.522 |
| FP windows /h @0.5 | 962.6 | 707.4 |
| best window F1 | not measured (0.698 AUC) | 0.141 |

v3's lower @0.5 rate is not cleanliness: barely any v3 score clears 0.5 in
either direction, and it has no operating range above it, with 0 windows at or above 0.6
where v2 still has a real tail. The FP/h denominator is 0.944 h, the corrected
figure; see the denominator-correction note below.

v3's held-out scores collapse into [0.494, 0.522], which is the final_v1 failure mode
again (0 exact 0 or 1 among 3869 windows, so this is genuine margin collapse, not
the float-saturation artefact suspected at first). 0.937 to 0.484 is the largest
train-to-held-out generalisation gap measured on this programme: the model
memorised the 4 training patients and transfers nothing.

**Mechanism, measured rather than assumed.** Over the 15 videos both extractions
share, the lr25 eyes produce **18% fewer person-observations** (628,848 to
513,227 rows; `pat01__Sz1PG` literally halves, 34,216 to 17,542) while creating
more track fragments (533 to 590 tracks), so fewer detections and choppier tracks.
WU-SAHZU-EMU is a side-view wall camera, and the lr25 fine-tune is a
side-view downgrade, the same specialist collapse the fall probes measured on
the same date (multicam 0.821 to 0.675, UR Fall 0.957 to 0.822 under the same eyes
swap). Third dataset, same finding. Thinner, choppier input also shrinks the
window pool (min_present drops more windows: 1106 to 935 train, 4880 to 3869 eval),
and what remains supports patient-specific shortcuts (train val 0.937) over
transferable convulsion signal (held-out chance).

**Capacity sweep, protocol upheld.** Three pre-declared variants
(h128 / d4 / h128e90) trained on the same v3 windows; all scored below the
base run's exact 0.93659 on train-side validation, so under the selection protocol none
earned a test look. Discarded unevaluated.

**Calibration instrument note.** The first v3 run fitted T=0.02, saturating
sigmoids to exact 0 and 1 ties. The trainer now bounds the temperature grid to
[0.25, 8], and a boundary fit records `degenerate_fit: true` with a T=1.0
fallback. The full retrain and re-evaluation under the fixed instrument reproduced ROC
0.48426 exactly, so the collapse is in the model, not in the calibration instrument.

**Outcome.** v3 is rejected; **final_v2 `07c399be` remains the seizure model
and keeps the demonstration slot** (confirm_threshold 0.8, unchanged). Extended domain
finding: the chained-lr25 eyes (`1e9445c5`) are barred from side-view event
extraction.
Seizure v4's pre-declared lever is the one final_v2's verdict named:
more seizure-free negative footage (the retired overhead ward recordings through
the same `55c2c242` eyes as tracks_v2, train-split recordings only), not new eyes.

Artefacts: `outputs/verifier/convulsion_v3_real/{report.json,`
`eval_wusahzu_testsplit/{report,metrics_deep,decision_layer_study}.json}`.

### Denominator correction (2026-08-27): every false-positive-per-hour rate below is restated by a factor of three

The v2 verdict and both studies divided false-positive counts by a hardcoded
2.8309 h whose origin cannot be reconstructed. The three negative recordings'
true container duration, the sum of `n_frames_container / fps` from their metas,
and what `metrics_deep` itself reports as `recorded_hours`, is **0.944 h**
(the registry confirms the recordings are real-time 30 fps, so container time
is wall time; the metas' `elapsed_s` is extraction runtime, not footage).
Every FP/h figure published before this note was therefore **three times too
favourable**. Catches, PG counts, latencies, AUCs, score distributions and
every ratio (the rule-family outcome, the ensemble outcome, the per-camera factor of 2.2)
are unaffected. The tables below are corrected in place;
`decision_layer_study.py` and `ensemble_study.py` now derive the denominator
from `metrics_deep.negatives.recorded_hours` instead of any constant, and the
study JSON files were regenerated 2026-08-27.

## seizure final_v2: full verdict (2026-08-26, all 3 negatives landed; FP/h corrected 2026-08-27)

final_v2 (`07c399be`, T=2.377, 10ch, tracks_v2 eyes) against final_v1 (`42dff31c`) on
the complete held-out set: 13 seizure videos plus 3 seizure-free ward recordings
(0.944 h of negatives). Window basis: v2 4880 windows (285 positive / 4595 negative), v1 2163
(216 positive / 1947 negative). The chained eyes track more people, so the pools differ by
construction and event-level rows are the like-for-like comparison.

**Events (metrics_deep, onset-timed):**

| | v1 @0.5 | v2 @0.5 | v2 @0.8 | v2 @0.9 |
|---|---|---|---|---|
| events caught (of 13) | 10 | **12** | 8 | 6 |
| convulsive_PG caught (of 6) | 5 | **6** | 4 | 3 |
| median latency | 47.1 s | **6.95 s** | 19.1 s | not measured |
| FP windows /h (negatives) | 144.0 | 962.6 | 272.2 | 57.2 |
| FP episodes /h (over 30 s gap = new episode) | 28.6 | 103.8 | 39.2 | 13.8 |

v1 has no rows above 0.5 anywhere: its scores collapse into [0.49, 0.53], so
0.5 is its only operating point (0 FP above 0.8 is score collapse, not
cleanliness). Window-level: v2 ROC 0.698 / PR 0.115 on its own pool (v1 0.715 /
0.193 on its smaller pool).

**Verdict.** Neither model dominates; they trade different currencies. final_v2
is the better event detector, with 2 more events including `pat14__Sz2PG` (0.731), the
event every prior variant missed; 6/6 convulsive; about 7 times faster; and thresholds
that mean something (event maxima 0.31-0.99). Its cost is additional firing on negative
footage: at every threshold that preserves its catch advantage it fires more than v1,
because the chained eyes now see and score the caregiver and visitor motion v1's eyes never
tracked. Outcome: **final_v2 takes the demonstration slot at confirm_threshold 0.8**
(measured menu in `src/vividly/demo/jobs.py`, `VERIFIER_CONFIRM_THRESHOLDS`);
final_v1 stays on disk as the quieter fallback. The FP levers for a final_v3
are recorded plainly: train-side negatives are only 4 patients' worth, so more
seizure-free footage (more patients, more wards) is the single biggest lever;
staff-track filtering and episode-level debounce tuning are the others.

Full artefacts: `eval_wusahzu_testsplit/{report,metrics_deep,breakout}.json`,
full-precision `windows.jsonl`.

### Decision-layer study (2026-08-26): catches at the 0.5 level with quiet at the 0.9 level

`scripts/events/decision_layer_study.py` replays the full-precision window
stream through candidate alerting rules (single-threshold, k-consecutive
persistence, hysteresis). Persistence was rejected on v1 for score flicker;
v2's calibrated margins earn it a re-measurement. Headline rows
(catches of 13; PG of 6; median latency; FP episodes per negative hour):

| rule | caught | PG | median lat | FP eps/h |
|---|---|---|---|---|
| single 0.5 | 12 | 6/6 | 7.2 s | 103.8 |
| **persist 2@0.5** | 11 | 5/6 | 9.7 s | 76.3 |
| persist 3@0.5 | 11 | 5/6 | 12.7 s | 62.5 |
| persist 2@0.65 | 10 | 5/6 | 18.0 s | 50.8 |
| single 0.9 | 6 | 3/6 | 31.2 s | 13.8 |

Hysteresis (fire immediately at 0.9 or above, otherwise k firings at the low threshold) adds
nothing over plain persistence, because every event at 0.9 or above also has an earlier
run of two consecutive windows at 0.5. Outcome: the rule family
buys a modest reduction in false positives at a real cost in catches and cannot reach the
recorded target on final_v2's scores; the separation must come from the model. Track-age
gating was considered and rejected on principle without measurement ambiguity:
convulsions shatter the patient track into fresh fragments (v2's own learned
signature), so an age gate suppresses exactly the true-positive windows. The
funded lever was the tracks_v3 (lr25-eyes) re-extraction and retrain, which the seizure v3
section above records as measured and rejected. No final_v3 followed, so the study was
never rerun; the seizure model of record is convulsion_v2_real, ONNX `07c399be`.
Full grid: `eval_wusahzu_testsplit/decision_layer_study.json`.

### Two-model ensemble study (v1 corroborating v2): measured and closed

Question recorded 2026-08-27: whether combining v1 (conservative, slow) with v2 (fast,
high recall) yields each one's best trait. `scripts/events/ensemble_study.py`
time-aligns the two window streams per video (they come from different
extractions, so track ids never correspond; agreement tolerance 6 s either side, and the
alert waits for the corroboration so latency stays causal) and measures OR, AND
and tiered gates. Episodes here collapse per video, not per track, which is the operator's
view, since simultaneous fires from many tracks on one camera are one alarm. The
v2 baselines are re-derived under the same definition, so this table is only
comparable within itself.

| combo | caught /13 | PG /6 | median lat | FP eps/h (per-video) |
|---|---|---|---|---|
| **v2 alone @0.5** | **12** | **6/6** | **7.2 s** | 21.3 |
| v2 alone @0.9 | 6 | 3/6 | 31.2 s | 9.6 |
| v1 alone @0.5 | 10 | 5/6 | 60.8 s | 10.6 |
| OR gate: v2@0.9 or v1@0.5 | 11 | 5/6 | 24.0 s | 13.8 |
| AND gate: v2@0.5 and v1@0.5 (within 6 s) | 6 | 3/6 | 33.3 s | 10.6 |
| tier: v2@0.8 solo, else AND | 10 | 5/6 | 21.8 s | 21.2 |

Outcome: every combination loses to v2 alone. The AND gate collapses catches
because v1's 60.8 s median decision time can almost never corroborate a 7 s
v2 fire inside any causal window; the OR gate only re-adds v1's slow catches, which
v2@0.5 already has, at 3 times the latency. The ensemble lever is closed, and v1
retires as a component. Full grid: `eval_wusahzu_testsplit/ensemble_study.json`.

Side finding that reframes the FP target: counted per camera-episode (the
nurse's experience) instead of per track-episode (the engine's raw event
count), v2@0.5's false alarms are 21.3/h, not 103.8/h, because crowded negative
moments fire about 5 tracks at once and the per-track count multiplies them. The
recorded target (catches at the 0.5 level with quiet at the 0.9 level) reads as 21.3
against about 9.6/h under this definition. Still real work, but the gap is a factor of
2.2, not 7.5. Deployed alerting should debounce per camera, not only per track.

## seizure final_v2: preliminary test-split results (2026-08-26, superseded by the full verdict above)

final_v2 (run `outputs/verifier/convulsion_v2_real`, onnx `07c399be`, T=2.377,
10 channels) against final_v1 (`42dff31c`) on the same held-out patients, with v2 scored
through its own tracks_v2 extraction (chained prototype eyes), 13 seizure videos;
the 3 seizure-free ward recordings are still extracting, so FP/hr is pending
and everything here is preliminary.

- **Events caught @0.5: v1 10/13, v2 12/13**, including `pat14__Sz2PG`
  (v2 max 0.731), the event every comparison variant missed. All 6 convulsive_PG
  caught. The one v2 miss is `pat09__Sz1P` (focal, max 0.308); focal events are
  outside the convulsive product target of the seizure rule, but the miss is recorded.
- **Detection latency @0.5: median 47.1 s becomes 6.95 s** (mean 46.7 becomes 17.3;
  pat07 events 74 to 4 s and 61 to 6 s).
- **Margins exist now.** v1 scored its whole world into [0.49, 0.53], with zero
  events above 0.8, which is score collapse, not cleanliness. v2 event maxima span
  0.52-0.99 with 8/13 events above 0.8; @0.8 it still catches 8/13 with zero
  v1 equivalent.
- **Window-level AUC moved the other way: ROC 0.695 to 0.558, PR 0.532 to
  0.318** (same-subset v1 baseline; v2 has 943 windows against v1's 536 because the
  chained eyes track more people). In-video negative windows crossing 0.8:
  v2 22.9%, and 6.7% cross 0.9. Much of this is caregiver response during
  seizure recordings, but the honest read is that v2 trades window-level
  separability for event-level catches, latency and margins. The deciding
  number, FP/hr on true seizure-free ward footage at 0.8 and 0.9, lands with the
  negative extractions.

Numbers: `eval_wusahzu_testsplit/{report,metrics_deep}.json` under the run directory.

## Three-variant comparison: final verdict (2026-08-26, all 3 negatives landed)

Complete test split: 16 videos (6 convulsive-PG events, 7 focal-P events, 3
non-seizure recordings totalling 0.94 h of ward footage), 2,163 windows (216
positive). Same instrument, same models, all negatives in.

| model | sha8 | window ROC | PR AUC | PG @0.5 | neg videos flagged @0.5 | FP win/hr @0.5 |
|---|---|---|---|---|---|---|
| finalv1synthetic | `9ed32801` | 0.408 | 0.099 | 0/6 | 2/3 | not measured |
| finalv1synth+real | `a95616f2` | 0.559 | 0.132 | 5/6 | 3/3 (two max at 1.0) | 210 (101 even @0.9) |
| **finalv1real** | `42dff31c` | **0.715** | **0.193** | 5/6 | 3/3 (max 0.526) | 144 (0 above 0.53) |

**Winner: finalv1real.** Best window discrimination by a wide margin (its ROC
rose from 0.66 to 0.72 as negatives were added, because the ward footage ranks below its
events), best PR AUC, and the only variant whose failure is structural calibration
(everything scored in [0.49, 0.53]) rather than confusion (synth+real scores the empty
ward at 1.0). Under the recorded amendment to the comparison protocol it took the
demonstration slot (`VERIFIER_RUN_DIRS`) and proceeded to a retrain on a re-extracted
track tree. Stated plainly: no variant is deployable, since
144 FP-windows/hr at 0.5 with a very thin margin is
a prototype, and the number that matters is the data-scaling curve
(0.41 synthetic, 0.56 mixed, 0.72 real). Recalibration (fit on its own training
validation split, never on test) was carried into the final_v2 retrain, which fitted
T=2.377 on the train-side holdout (`final_models/seizure/README.md`).

## Three-variant comparison: deep metrics (2026-08-26, 1 of 3 negatives landed; superseded above)

Test split (patients pat03/07/14 PG plus pat04/09/11 P, 13 events, plus `pat03__free`
11 min ward negative; `pat04__free` and `pat04__no-Sz2P` still extracting): 925
windows, 216 positive. Instrument: `scripts/events/metrics_deep.py` over each
evaluation's `windows.jsonl`; full JSON in each model's `eval_wusahzu_testsplit/`.

| model | sha8 | window ROC | PR AUC | events @0.5 | median latency | ward-neg FP win/hr @0.5 |
|---|---|---|---|---|---|---|
| finalv1synthetic | `9ed32801` | 0.429 | 0.216 | 0/13 | not measured | 49 (and 38/hr even @0.9) |
| finalv1synth+real | `a95616f2` | 0.516 | 0.254 | 10/13 | 24.9 s | 391 (234 @0.9) |
| finalv1real | `42dff31c` | 0.662 | 0.320 | 10/13 | 47.1 s | 272 (0 @0.8 and above) |

Findings, stated plainly:

- **finalv1real is score-collapsed**: every window it scores, positive or negative,
  lands in [0.491, 0.531] (pos median 0.494, neg median 0.493). Its ROC 0.662 is
  real ranking information, but there is no margin: threshold 0.5 slices through the
  middle of the cloud, and its 0 FP/hr at 0.8 is vacuous, because it cannot score anything
  above 0.54. This is the expected failure mode of 8 training videos; it is not a deployable
  calibration.
- **finalv1synth+real has genuine margins** (caught events max 0.75 to 1.0) and the best
  latency, but fires on the empty-ward negative constantly (391 FP windows/hr at
  0.5; the negative video maxes 1.0). Sensitivity with poor specificity so far.
- **finalv1synthetic flags the ward negative at every threshold while catching zero
  events**, which is anti-correlated with reality; the variant is confirmed unusable.
- Both real-data variants catch 5/6 convulsive PG events at 0.5 but miss different
  ones (synth+real misses pat07 Sz2, real misses pat14 Sz2); both miss pat11 Sz1
  (focal). Ensemble and threshold headroom exists, but 8 training videos is the binding
  constraint. The recorded conclusion is that real data helps monotonically
  (ROC 0.43, 0.52, 0.66) and the corpus is far too small, not that one of these is
  shippable.
- Latency definition: the first window with score at or above the threshold ending at or
  after labelled clinical onset, minus onset; a 12 s window plus a tonic-phase lead-in makes
  tens of seconds expected, not alarming.
- Caveat: specificity numbers rest on one 11-minute negative so far; the final verdict
  comes after all 3 negatives land. The single negative recording is 0.18 h, so FP/hr from
  it is an extrapolation, not a measured hour.
- Fusion tested and rejected (2026-08-26, window-dump analysis, no retraining):
  score-mean AUC 0.531, score-max 0.662, rank-average 0.657, none of which beats
  finalv1real alone (0.662); the models' errors correlate only 0.275, but
  synth+real's noise drags every average down. Decision rules at 0.5: OR catches
  5/6 PG (not 6/6, because both models miss pat14 Sz2) at 527 FP-windows/hr on the ward
  negative; AND drops to 4/6 PG at 136 FP/hr. No fusion of these two is a gain;
  the levers were more real data and a stronger detector generation, not
  combination.
- Extraction provenance disclosure: the wu_sahzu_tracks_v1 extraction was produced
  by two machines racing the index, the Mac (onnxruntime CPU EP) and the pod
  (CUDA EP, under a v1 extraction configuration not carried in this repository, with device
  the only difference). The EPs are not bit-identical: on the two videos both machines
  happened to extract (pat08 Sz3,
  pat09 Sz1) row counts differed marginally. Same operating point, same models,
  same config geometry; the mixed-EP provenance is noted here rather than hidden.

## The detector generations behind these extractions (2026-08-26 to 2026-08-27)

The track trees these heads trained and were scored on came from successive weights of an
earlier person-detector generation, and that generation's accuracy was graded on held-out
sets that rule 1 of `POLICY.md` has since retired, so those measurements are not restated
here; the names used above, stock eyes, chained eyes (`55c2c242`) and chained-lr25 eyes
(`1e9445c5`), identify which of those weights produced a given extraction, and the
per-lane verdicts on swapping between them are recorded in the sections above and below.
The eyes lane of record is now RT-DETRv4-S `345c30a5` (ONNX `40262049`), recorded on
`final_models/eyes/README.md`.

## fall final_v1 against multicam_fall: second held-out test

`configs/events/eval_fall_multicam_v1.yaml`, model `080b5d86`, extraction 1b02d2e
(24 scenarios, rate-corrected reference cameras, the overhead-adjacent domain):

- **window AUC 0.821** (533 windows, 80 positive) / video AUC 0.826 (24 videos).
- **21 of 23 staged falls caught at thr 0.5** (20 at thr 0.9). Missed: chute07 cam6
  (0.37) and chute19 cam7 (0.40); chute02 cam4 borderline (0.53).
- The video-level AUC is nearly meaningless here and is reported only for
  completeness: the set has one negative video, chute24, the crouch and moving-down
  confounder designed to fool fall detectors, and it scores 0.92, outranking 4 of
  the falls. The verifier window sees a person descending and staying low; the live
  cascade's prolonged-stillness condition is the component built to separate a crouch
  (the person moves on) from a fall, and this scenario is the argument for it.
- Domain-shift note, stated plainly: Le2i trains side-view rooms; multicam is
  ceiling wide-angle. The window-AUC drop from 0.957 to 0.821 between UR Fall and multicam
  is the honest measure of that gap for the first-generation eyes, and was the baseline a
  re-extraction through later eyes would have had to beat. The probe below measured that
  swap and rejected it.
- **First-lever probe (2026-08-27), measured and closed, eyes swap rejected.** The
  same model 080b5d86 re-tested on chained-lr25 eyes re-extractions of
  multicam and UR Fall (pod run), in which only the detector changed from the v1
  extractions, with pose, tracker and operating point held identical: multicam
  window AUC **0.821 to 0.675** (852 windows) and UR Fall **0.957 to 0.822**,
  with only **34 of 52** UR Fall videos yielding valid windows at all, because the
  overhead specialist frequently fails to form tracks on normal-height cameras,
  exactly as the coco500 specialist-collapse control predicted (stock 0.682 against
  lr25 0.438 on normal-perspective frames). Both arms lost, so the drop is not
  test-time detection quality; the eyes stay stock for the fall stack, and the
  fall levers named at the time were (a) retrain with calibration, a capacity sweep and
  retired overhead ward-footage negatives through stock eyes, and (b) a second licence-reviewed
  fall dataset for angle diversity; the fall v2 and fall v3 sections above record what each
  became. Runs:
  `outputs/verifier/fall_v1/eval_multicam_lr25eyes/`, `eval_ur_fall_lr25eyes/`.
- Deep metrics (2026-08-26, `metrics_deep.json`): PR AUC 0.487; best F1 0.464 at
  thr 0.9 (sens 0.60 / spec 0.826 window-level). Event-level: 19 of 25 staged fall
  events have an overlapping window at or above 0.5 (median latency 2.6 s); of the 6 event
  misses, 4 are window-coverage gaps (no window overlapped the fall moment at all)
  and only 2 scored low (chute07 0.37, chute10 0.06). Video-level 21/23 stays the
  product-question number; the event-level gap is mostly extraction coverage, the
  same lever as the UR Fall caveat.

## egress_without_discharge_v0

Rule head (engine attachment), with no trained weights and no labelled egress dataset in
the repository, so there is no accuracy number to quote, and inventing a proxy would violate
the reporting discipline. Verification status: unit and integration tests green
(engine, zones and demonstration wiring); burn-in style negative-control numbers are the only
honest instrument available and have not been run for egress.
