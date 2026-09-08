# Seizure lane: event verifier

The model of record for the seizure lane, the standing bars it and its sibling head are
measured against, the historical ladder, the arms that were rejected, the operating points,
and the track-tree generations its features came from. Every number on this page carries
the hash of the graph that produced it.

## Disclosure, which applies to every number below

This is a convulsive-motion signature verifier over track windows. It is not a seizure
detector and not a medical device, and the phrase detects seizures is never claimed for it.
It is a prototype-class weight trained on real clinical footage under the registry grant
recorded in `configs/datasets/wu_sahzu_emu_trainsplit.yaml`: bucket `TRAIN_PROTOTYPE`, event
grant role `train_interim`, under the owner grant of 2026-08-26 as amended. No number on this
page is clinical detection performance, it is never quoted as such internally or externally,
it is badged internal demonstration only, and it is not deployed commercially without counsel
sign off (counsel item CQ-23). The
corpus is public: an ungated repository under Apache-2.0 with faces masked, patient consent
and hospital ethics approval, counsel-cleared 2026-08-26.

## Identity and provenance

| field | value |
|---|---|
| Model of record | convulsion_v2_real |
| Checkpoint sha256 | `715d172c9fd8523c3da43faba34644762f847b8a9d3eaa9dc293efe7dda1830b` |
| ONNX sha256 | `07c399be24f89a44670262ea75879cc67f60f46c5cd52e5d2cc4218e914b7e22` |
| Shipped as | `final_models/seizure/model.onnx`, digest in `final_models/seizure/model.onnx.sha256`, run record in `final_models/seizure/report.json` |
| Architecture | small temporal convolutional network: three Conv1d blocks (kernel 5, hidden width 64, batch normalisation, ReLU, max-pool 2, dropout 0.15) over the 64 time steps of the ten track and scene channels, mean and max pooled, one linear head; 60 epochs |
| Window | 12.0 s span, 3.0 s stride, 64 steps, `overlap` labelling policy, `overlap_min_s` 4.0, `min_present` 0.35, scene channels on |
| Channels | 10: `vy`, `vx`, `h_rel`, `aspect`, `angle_norm`, `angle_valid`, `kpc`, `present`, `scene_n_tracks`, `scene_births` |
| Calibration | score is the sigmoid of the logit over a temperature of 2.377, fitted on the train-side holdout, monotonic and neutral for area under the curve |
| Training | batch 256, learning rate 0.001, weight decay 0.0001, positive weight automatic, seed 20260826, device cpu |
| Training data | 1106 windows, 164 positive and 942 negative, from 8 videos: patients pat01, pat02, pat05 and pat13 |
| Source tree | the v2 generation of the clinical corpus track sequences |
| Recipe | `configs/events/train_convulsion_v2_real.yaml` |
| Best train-side validation AUC | 0.77152, a train-side holdout number and not an exam number |
| Code revision | 8fbebe2 |
| Weight class | prototype |
| Shippable | counsel |

A second head of record exists on this lane and is not the same model. Final B is the broad
seizure-motion head: checkpoint
`535141c19f978ad53b51a74add2dec9227d085561ae6dafc293954cf0a52aacd`, ONNX
`b9e04d034fd751f6f087b2c884f033c2a754061cd5719d226011d5839552aecc`,
code revision bc61cab, recipe `configs/events/rot_train_conv_finalB.yaml`, trained on the
eight-patient rotation pool from the v2pose tree plus a wall-camera negative diet, with
patients pat02 and pat09 held out for checkpoint selection, seed 20260826. It holds the
focal bar; the model of record holds the convulsive bar.

Patient allocation is permanent and registry-enforced. Patients pat06, pat08, pat10 and
pat12 are exam-side only and never enter a training manifest; the corresponding earlier
proposal to move them train-side is closed.

One prose value in `final_models/seizure/report.json`, the `disclosure` string, was
normalised on 2026-09-07 to the wording the current trainer emits; every metric value in
that file is untouched. That string is the trainer's fixed boilerplate for every event
verifier, and its acted-fall and synthetic-motion sentence describes the fall lane rather
than this model, whose corpus is the real clinical footage recorded above. The lineage of
record for counsel item CQ-23 is this page and `final_models/seizure/model.json`, not that
boilerplate, and correcting the string at the trainer is owed.

## Before and after

One deliberate change separates the model of record from the alternative recorded on this
lane: the pose graph behind the track sequences. Same recipe, same real clinical training
patients, same detector, same architecture, same window geometry.

| | Model of record, convulsion_v2_real | Alternative, convulsion_v2pose |
|---|---|---|
| verifier ONNX sha256 | `07c399be24f89a44670262ea75879cc67f60f46c5cd52e5d2cc4218e914b7e22` | `614b92fa729b670738e1521b2c3a7f04046d32444f16a08fdf7e58d7740eb0d4` |
| checkpoint sha256 | `715d172c9fd8523c3da43faba34644762f847b8a9d3eaa9dc293efe7dda1830b` | `9367365c35bafe368377861e2300bb734848bfc00d33cc112b4ee0cc6dc3be42` |
| detector behind the tracks | chained prototype detector `55c2c242` | unchanged, `55c2c242` |
| pose graph behind the tracks | v1 generation, upstream body7 `5c0a4bf6` | joints lane model of record `6ba96285`, keypoint AP 0.774115 |
| train-side validation AUC | 0.77152 | 0.74420 |

Two held-out exams over patient sets disjoint from training, one look each. Window AUC is
the metric for this lane: per-video area under the curve is meaningless here, because the
video counts are tiny and almost every video is a positive, 13 of 16 and 12 of 12.

| exam | videos | windows | model of record `07c399be` | alternative `614b92fa` | delta, alternative minus model of record |
|---|---|---|---|---|---|
| testsplit, primary, patients pat03, pat07, pat14, pat04, pat09, pat11 | 16 | 4,880 with 285 positive | 0.69841 | 0.53728 | minus 0.161 |
| unallocated, patients pat06, pat08, pat10, pat12, focal with uncertain onsets | 12 | 1,386 with 178 positive | 0.64270 | 0.69921 | plus 0.056 |

The unallocated exam and the E1 bar recorded under "Standing bars" below cover the same four
patients through different track-tree generations, so they are two surfaces and not one; the
note under that section states which tree carries which number.

Reading. This is a loss on the primary exam, and the lane is too data-poor for a single run
to settle anything. The primary exam fell to 0.537, barely above chance at 0.5, while the
secondary rose to 0.699. With four training patients the tiny validation holdout swung
between 0.38 and 0.56 across the last ten epochs, so the checkpoint selection is itself
noisy, and the other measured variants on this lane span a wider band than this result
moves. As on the fall lane, better joints did not buy better events: the binding constraint
here is data, four patients, not the front end. The alternative is recorded as the run of
record on the newer pose and is not promoted. The real unlock is more clinical data, not another
architecture or another front-end swap.

Precision on the primary exam is low in absolute terms: the model of record's area under
the precision-recall curve on the testsplit is 0.11484.

## Standing bars

Two exams carry pre-registered bars that any candidate on this lane must clear, and
selection never touches either of them.

| exam | definition | bar | held by |
|---|---|---|---|
| E1, focal | patients pat06, pat08, pat10, pat12; 1,430 windows with 177 positive | window AUC 0.87422 | final B, ONNX `b9e04d034fd751f6f087b2c884f033c2a754061cd5719d226011d5839552aecc` |
| E2, convulsive | patients pat03 and pat07, locked; 919 windows with 70 positive | window AUC 0.72916 | the model of record, ONNX `07c399be` |

Context on the same surfaces: final B scores 0.57469 on E2, and the model of record is the
E2 bar holder through a reslice of its scores onto the newer tracks. The two heads are
complementary, which is why both bars stand.

Which track tree carries which number. Every extraction tree named in this paragraph sits
under `outputs/`, a gitignored build tree that this repository does not carry. Both bar
surfaces are measured on `outputs/events/wu_sahzu_tracks_v2pose`
(`configs/events/rot_eval_E1_finalB.yaml` and `configs/events/rot_eval_E2_finalB.yaml`).
The unallocated exam in the before-and-after table above is the same four patients on an
older tree: the model of record was scored there on
`outputs/events/wu_sahzu_tracks_v2` (`configs/events/eval_wusahzu_unallocated_v2.yaml`, the
1,386 windows with 178 positive quoted in that row) and the alternative on
`outputs/events/wu_sahzu_tracks_v2pose` (`configs/events/eval_wusahzu_unallocated_v2pose.yaml`),
each through the tree it was trained on. E1 is the v2pose tree at 1,430 windows with 177
positive. The window pools therefore differ, so the model of record's 0.64270 is not a look at
the E1 bar surface and does not stand against 0.87422; the E2 bar it does hold was measured by
resliced scores on the v2pose tracks, which is the reslice named above.

## Historical ladder and rejected arms

Every row is one pre-declared look at the primary testsplit exam, the same 16 videos with 13
of them carrying events in every row, each variant scored through the track tree generation
its own extraction produced, so the window pool differs between generations: 2,163 windows
on the v1 tree, 4,880 on v2 and 3,869 on v3. The six other measured variants,
convulsion_v1, convulsion_v1_real, convulsion_v1_synthreal, convulsion_v3_real,
convulsion_v4_real and convulsion_v6_real, span testsplit window AUC 0.408 to 0.715, and
where an unallocated number was also recorded it ranges from 0.531 to 0.726; the spread
across attempts is wider than the gap between the model of record and the alternative above,
which is the point.

| variant | ONNX | checkpoint | testsplit window AUC | outcome |
|---|---|---|---|---|
| convulsion_v1 | `9ed32801b4bab9a43ea31251cd2288d6af93b4133dce5573457da2f3777c0a36` | `77847a1cde520402f5260328474336c6105462a7ce97575c15966d4c16e933d2` | 0.40761 | superseded |
| convulsion_v1_real | `42dff31c4bd6aa2a25eca838e742aa4f03ca93cddc7d73aa1e61c50c89c8c8cb` | `aeb813eab73ea77c0cc2ba33ff4bba21ef1cfb293c40e54688c6341e538852b0` | 0.71543 | superseded |
| convulsion_v1_synthreal | `a95616f2cbdc33ff29fa5175b7b08c5610676374eb2454a697b53e73799b9d69` | `094ddddc33046e11ac67eca02c40d714460dff1ca688a495024c830c08b0bb46` | 0.55937 | superseded |
| convulsion_v2_real, model of record | `07c399be24f89a44670262ea75879cc67f60f46c5cd52e5d2cc4218e914b7e22` | `715d172c9fd8523c3da43faba34644762f847b8a9d3eaa9dc293efe7dda1830b` | 0.69841 | promoted |
| convulsion_v3_real | `d85f1a203f5a7d119a0d1eb8e0e9370c5bd7974ffaf22e2990a9cdd816fe2afc` | `a0804c51828623de0141154c5b961401ffa2b7db28752c5830052e09f5692c1a` | 0.48426 | rejected |
| convulsion_v4_real | `6d205fee5254acac52c9fff7e592035146f4d7a833aa9f1773d013954b19acba` | `727fc7263da936a6db76857c268cf486b636867f5212e7e177339f8aaa8fc030` | 0.66889 | banked, then barred, see below |
| convulsion_v5_real | `c8bc5c7035b53d764272c0dfacfe219d508e29a682bfe9d660253500164bb204` | `b2c49bee74bd820dc93a14f0799ed83b4fcffdd7e337a054bf88b5ef062c9696` | no exam look taken | discarded on the train-side split, then barred |
| convulsion_v6_real | `10cdbe78e0cf046ebd4ffd3805d1f6db70912a352354d17e751a3fd6ad2404e0` | `9ec2bf47131ec4757bf59d1f920fa5a098d1fe84511191757e27fe45e1d81903` | 0.67447 | rejected |

**v3, a detector swap.** Exactly one line changed against the model of record's recipe: the
source tree pointed at a re-extraction through a different prototype detector. The
train-side picture looked like a win and the held-out exam did not agree: 0.48426, below
chance, with zero of 13 events caught at threshold 0.8. Rejected.

**v4 and v5, barred by the imagery rule.** v4 was the model of record's recipe unchanged
apart from a larger negative diet: six overhead-fisheye ward recordings from the training splits
of two fisheye corpora, extracted through the same detector, added as forced negatives.
Train pool 4,852 windows, 164 positives identical to the model of record plus 3,746 added
negative windows. It proved a recall ceiling, 13 of 13 events caught at both 0.5 and 0.8
with every event maximum at 0.826 or above, and it bought no quiet at all: 95.7 percent of
seizure-free footage in alarm at 0.5 and 73.7 percent at 0.8, against the model of record's
72.4 percent at 0.5. v5 was v4 with the positive class weight capped at 6.0, one knob; it
scored a train-side validation AUC of 0.97211 against v4's 0.98139 with worse recall, so it
was discarded on the train-side split and never touched an exam.

Both arms are now barred rather than merely unpromoted, because their negative diet was
sourced from overhead-fisheye extractions and no fisheye imagery, real or rendered, enters
any training or evaluation manifest in any lane. Those extractions cannot be rebuilt in
this repository and the arms are not reproducible here. The current heads carry no such
reference: the model of record's recorded training source is the clinical corpus tree
alone, with locations pat01, pat02, pat05 and pat13, and final B's sources are the v2pose
clinical tree plus a wall-camera pose-and-track negative set that contains no imagery at
all. The two barred training configurations, and the one superseded fall configuration that
shared their negative diet, were deleted on 2026-09-07; the extraction configurations that
produced those trees and the registry entries for the corpora behind them had already left
the repository under the same rule. The surviving configuration set therefore names no
such negative source, and `scripts/guards/check_config_registry.py` resolves every training
configuration to a registered dataset with no violation. The measured record of both arms
stays on this page and in `research/event-verifier-v1-record.md`; only the recipes are gone.

**Push 3, the negative-diet training arm.** The one remaining question at the data layer
was whether a harmonised wall-camera negative diet survives as a training arm and produces
a candidate worth an irreversible exam look. Five seeds per arm on one fold, one variable,
with three levers explicitly off because an earlier push measured all three and promoted
none. The push is resolved and not promoted: zero exam looks were taken against the push 3
budget, the single candidate slot went unspent and does not carry over, and nothing on the
lane changed. The three looks dated 2026-09-07 that are recorded further down this page were
declared under the coupling protocol and are charged to that separate budget; they are
carried in the append-only exam-looks ledger of `research/seizure-push-3-protocol.md` for the
record, not as push 3 candidate slots. The broad seizure-motion head stays final B and the
convulsive head stays the model of record.

## Operating points

The demo threshold on this lane is 0.8. The recorded behaviour of the model of record on
the primary exam is the following, where the in-alarm column is the fraction of
seizure-free footage covered by the alarm.

| threshold | events caught | fraction of quiet footage in alarm |
|---|---|---|
| 0.5 | 12 of 13 | 72.4 percent |
| 0.8, the demo point | 8 of 13 | 33.2 percent |
| 0.9 | not recorded | 8.1 percent |

At 0.5 the model sits lit for more than two hours of every three hours of footage with
nothing happening. The demo runs at 0.8 because that trade, catching 8 of 13 instead of 12
of 13 in exchange for cutting the alarm time to a third, is the only recorded operating
point that survives being watched by a person for five minutes. On the second exam surface,
the focal patients with uncertain onsets, the same model drops to 0.64270 and catches 6 of
12 at 0.8.

A later threshold study, fitted for final B on a non-exam out-of-fold pool and transferred
to the model of record because no non-exam surface exists for it, recorded the following on
the E2 exam. The negative denominator is a single seizure-free recording of 662.3 seconds,
0.184 hours.

| threshold | E2 catches | median latency | episodes per hour, track and camera | in alarm |
|---|---|---|---|---|
| 0.5 | 4 of 4 | 6.0 s | 81.5 and 27.2 | 75.2 percent |
| 0.8, recommended and transferred | 4 of 4 | 9.0 s | 54.4 and 32.6 | 39.3 percent |
| 0.85, an exam-side observation only | 4 of 4 | 9.0 s | 38.0 and 21.7 | 25.2 percent |

At the transferred 0.8 the model of record keeps all four convulsive catches with a median
latency of 9.0 seconds and cuts the in-alarm fraction from 75.2 percent to 39.3 percent.
The 0.85 row is recorded and is strictly better on that single 0.184-hour recording, but
choosing it would be fitting on an exam, so it stays an observation until pilot shadow data
exists. The model of record's calibration temperature, 2.377, differs from the final B
family's, near 1.03, so a transferred threshold is a policy choice and not a score-scale
identity. For final B the same study recommends 0.8, where the out-of-fold pool moves from
99.0 percent in alarm at 0.5 to 2.2 percent while keeping all 9 of its 10 convulsive catches
on that pool, at a cost of 4 of the 7 focal catches.

What the replay pipeline attaches. `configs/pipeline/replay_demo.yaml` runs the verifier in
annotate mode and attaches this lane directly: the head `seizure_candidate_v0` takes
`final_models/seizure` as its run directory, graph `07c399be` with temperature 2.377, at a
confirm threshold of 0.8, which is the operating point recorded above. The demo platform
attaches the same directory at the same threshold. No build output is needed to load it:
`load_head` in `src/vividly/models/event_verifier.py` builds a head from any directory
holding `report.json` beside the graph the report names, falling back to `model.onnx` in the
same directory, and opens that graph only when its sha256 equals the digest the report
records. This folder's `report.json` names `verifier.onnx` and records
`07c399be24f89a44670262ea75879cc67f60f46c5cd52e5d2cc4218e914b7e22`, and
`final_models/seizure/model.onnx` carries exactly that digest, so the committed lane
directory is itself a valid run directory. A graph whose digest differs from the report is
refused rather than scored with unaccounted-for weights. Final B, graph `b9e04d03`, is wired
nowhere and is recorded on this page as a measured arm only. Every threshold on this lane is
provisional until the calibration protocol in `docs/CALIBRATION.md` has its preconditions met.

One caveat travels with any quietness claim for the final B family: its training diet
includes a proxy-negative dose whose rows carry a recognisable source fingerprint, so a
network can learn to separate the source rather than the behaviour. Its quietness on its
own training diet certifies shape, not ward behaviour. The exam tables above are real
clinical windows with no proxy rows, so those are the honest measurement.

## Track trees

Verifier features are read from extracted track sequences, so a verifier's identity
includes the stack that produced its tree. Five generations exist on this lane, four of them
tabulated below and the fifth described after the table. The v1, v2 and v2pose extraction
configurations are not carried in this repository; their settings are recorded here because
the model of record's training tree was built by the second of them.

Settings common to all four tabulated generations: tracker backend `bytetrack` with
`track_activation_threshold` 0.4, `high_conf_det_threshold` 0.4, `lost_track_buffer` 30 and
`minimum_iou_threshold` 0.1; detector score threshold 0.3 at input 640x640; track layer
`baseline_window_s` 300.0, `reap_after_s` 30.0, `kp_conf_min` 0.3, `ring_window_s` 300.0
and `baseline_min_samples` 30; `rules_config` `configs/pipeline/rules.yaml`;
`jpeg_quality` 30; `annotated_only` true. The operating point was chosen on 2026-08-22 and
has not moved since; the eyes change across generations, the operating point does not.

| generation | detector | detector ONNX sha256 | pose | pose ONNX sha256 | tracker | declared settings |
|---|---|---|---|---|---|---|
| v1 | upstream rtmdet-m person release, token `235e8209`, which is a checkpoint digest prefix and not the graph digest | `4f4d7e07350b1753299111d1ae500fd64447a5b0e38e4bacbefab6573c742d30` | upstream RTMPose-m body7 release, token `e48f03d0`, checkpoint sha256 `e48f03d0cfe1285ee8b6d3457ac3ce33a4594a92b080053ab1ec4a7e300975f2` | `5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c` | bytetrack, gates 0.4 | the common settings above; keypoints not emitted |
| v2, built the model of record's tree | chained prototype detector `55c2c242` | `6a6e4d90a3774522d7e1127122a5a00be41a947e93c8940a3fc6a0c09f97c3ce` | unchanged from v1 | `5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c` | unchanged from v1 | exactly one declared delta from v1: the detector. The operating point is unchanged, so a v1-against-v2 verifier comparison is attributable to the detector alone |
| v2pose, built the alternative's and final B's trees | unchanged from v2, `55c2c242` | `6a6e4d90a3774522d7e1127122a5a00be41a947e93c8940a3fc6a0c09f97c3ce` | joints lane model of record, checkpoint `a0a035fe` | `6ba9628506a8899d3a3e940dcc8af5d30342921f09d2605afa505b57453d3527` | unchanged | exactly two declared deltas from v2: the pose graph becomes the joints lane model of record, and `emit_keypoints` becomes true. Emitting keypoints is neutral for training, because the trainer reads only the nine frozen fields; the keypoints are recorded for later rhythm work. Each event lane kept its own historical detector at this step so the verifier delta is pose-only |
| v2c, the current generation, `configs/events/extract_wu_sahzu_v2c.yaml` | eyes lane model of record, checkpoint `345c30a5`, backend `rtdetr_onnx` | `4026204967aa01a7680f7ce0c6169463d654a519f8093383a10537321f56fd49` | joints lane model of record, checkpoint `a0a035fe` | `6ba9628506a8899d3a3e940dcc8af5d30342921f09d2605afa505b57453d3527` | unchanged | exactly two declared deltas from v2pose: the detector becomes the eyes lane model of record on the `rtdetr_onnx` backend, and `pose.color_order` is set to `rgb`. Tracker, track layer, rules, `emit_keypoints` and dataset scope unchanged |

The fifth generation is the v3 tree, `outputs/events/wu_sahzu_tracks_v3`, a re-extraction of
the same clinical corpus through a retired prototype detector generation, with the pose,
tracker and operating point of the v2 row held identical, so the v2 against v3 delta is
attributable to the detector alone. It is not tabulated above because no full digest of that
detector is on record in this repository, and nothing standing on this lane rests on it: it
holds no model of record and no bar. It is the tree behind the convulsion_v3_real row of the
ladder, graph `d85f1a20`, which is why that row is scored over 3,869 windows where the v2
tree carries 4,880. The run is recorded in `research/event-verifier-v1-record.md` and
`research/stats-compendium.md`.

The chained prototype detector `55c2c242` behind the v2 and v2pose generations is a
fisheye-era prototype weight. Its provenance is recorded here because the checkpoint
provenance rule applies to every checkpoint that produced a number on this page, including
retired ones. That weight does not ship in this repository, is carried nowhere in
`final_models/`, and trains nothing.

| field | value |
|---|---|
| Checkpoint | `best_coco_bbox_mAP_epoch_10.pth`, sha256 `55c2c242f0da4012e5c63dcf8a93e6951da0bc191b869d6fe97b7df088cca653` |
| ONNX sha256 | `6a6e4d90a3774522d7e1127122a5a00be41a947e93c8940a3fc6a0c09f97c3ce`, opset 11 |
| Training data | retired overhead-imagery training splits, none of them registered here and none re-acquirable, mixed with a degradation-processed COCO person set. Learning rate 1e-04 |
| Backbone pretrain lineage | the upstream rtmdet-m person release, token `235e8209`, checkpoint sha256 `235e820939cb2ff33c505441e71f7e9532958c281636a963c6829d100867aed9`, trained on COCO person and Objects365 person, then a degradation-lineage detector run of this repository |
| Export parity | worst absolute difference between ONNX and torch 6.556510925292969e-07, tolerance 0.001; exported by mmdeploy at commit `bc75c9d6c8940aa03d0e1e5b5962bd930478ba77` |
| Weight class | prototype |
| Shippable | counsel. The Objects365 lineage it inherits is refused by the checkpoint law in `third_party/PINS.yaml`, and the corpora behind it are retired |
| Latency | never measured, so no figure is quoted |

Two gates were passed before any v2c extraction ran. The backend parity gate compared the
new detector backend against the grading path on 50 frames of pod bytes and returned
bit-exact agreement, 0.000000 pixel and 0.000e+00 score differences over 50 of 50 frames
(commit 38309de). The fence gate confirmed that no pipeline configuration under
`configs/pipeline/` sets the colour-order key, so the correction reaches the live path only
through a declared re-extraction and a refit of the verifiers on the new trees (commit
d151d17). Extraction of the clinical corpus on a rented machine is permitted under the
standing rule recorded on 2026-09-05
(`research/eyes-instrument-protocol.md`), which requires the raw footage to be deleted
after the de-identified track sequences are retrieved and never made public.

### Seizure final B refit on the v2c clinical track tree, measured 2026-09-07

Status: measured, not promotable. E1 misses its bar outright. E2 clears its bar on window
AUC, but at the declared alarm threshold the refit catches none of the four convulsive
events that the model of record catches, so it cannot replace it. The two frozen heads
re-sliced onto the v2c tree moved as well, in both directions, evidence below. The model of
record is unchanged.

Bar: two standing bars, one look each. E1 focal window AUC 0.87422, held by final B, ONNX
`b9e04d034fd751f6f087b2c884f033c2a754061cd5719d226011d5839552aecc`, checkpoint
`535141c19f978ad53b51a74add2dec9227d085561ae6dafc293954cf0a52aacd`. E2 convulsive window
AUC 0.72916, held by the model of record, ONNX
`07c399be24f89a44670262ea75879cc67f60f46c5cd52e5d2cc4218e914b7e22`, with the negative side
at 0.8 not regressed: the model of record catches 4 of 4 convulsive events at 331.57
false-positive windows per hour, final B 2 of 4 at 206.55.

Protocol as declared: recipe `configs/events/rot_train_conv_finalB_v2c.yaml`, the final B
recipe unchanged except for the clinical track tree, which is the v2c generation, with the
wall-camera negative set unchanged; holdout patients pat02 and pat09 for checkpoint
selection only; seed 20260826; 5,274 windows, 154 positive and 5,120 negative; exactly one
E1 look and one E2 look through `configs/events/rot_eval_E1_finalB_v2c.yaml` and
`rot_eval_E2_finalB_v2c.yaml`, and one look per exam for each of the two frozen heads
through the `rot_eval_E*_champion_v2c.yaml` and `rot_eval_E*_finalBv2pose_v2c.yaml` files.
Selection never touched either exam. Code revision fd2ec81.

Candidate: checkpoint
`90a8734c7b4238b66ed71843b352dda09c28c24b93152d05de0ee84ccb81a846`, ONNX
`978a1c3193937e894da9679213505472f5cdb1b8514e2b323e1b53ba6beae9bb`, validation AUC at the
best epoch 0.88910 against 0.75479 for the same recipe on the earlier tree; curve
`curves/conv_v2c_finalB_s20260826.csv`; run report `report_v2c_finalB.json` in this folder,
with the disclosure string carried in the wording this repository uses and every other
value as written by the trainer.

The looks, all on the v2c exam tree. E1 has no negative videos, so it carries no video AUC
and no false-positive rate; E2 has 0.184 negative hours.

| head, weights | E1 window AUC | E1 focal events caught at 0.8 | E2 window / video AUC | E2 convulsive events caught at 0.8 | E2 false-positive windows per hour at 0.8 | E2 negative windows over 0.8 |
|---|---|---|---|---|---|---|
| final B refit, ONNX `978a1c31` | 0.69787 | 0 of 12 | 0.88047 / 0.50000 | 0 of 4 | 0.00 | 0 of 621 |
| model of record convulsion_v2_real, ONNX `07c399be`, re-sliced | 0.60001 | 9 of 12 | 0.65902 / 0.25000 | 3 of 4 | 951.23 | 175 of 621, 0.2818 |
| final B, ONNX `b9e04d03`, re-sliced | 0.72499 | 7 of 12 | 0.65681 / 0.25000 | 4 of 4 | 1739.39 | 320 of 621, 0.5153 |

Reading the table: on E1 the refit is 0.17635 under the bar and no focal event reaches 0.8.
On E2 the refit's window AUC is 0.15131 over the bar while its video AUC is 0.5, no
convulsive event reaches 0.8 and no negative window does either, so the ranking within the
exam improved but the score scale moved off the declared operating point. The mechanism is in
`report_v2c_finalB.json`: its calibration block carries `degenerate_fit` true with the
temperature falling back to 1.0, and its own train-side holdout operating points at threshold
0.8 already record 0 true positives and recall 0.0. A verifier that raises no alarm on any of
the four events is not a replacement for one that raises all four.
Final B, which holds the E1 bar at 0.87422 on the earlier tree, scores 0.72499 here with its
weights untouched; the model of record, which holds E2 at 0.72916, scores 0.65902 here. Both
re-sliced heads also grow noisier on E2: the model of record moves from 331.57 to 951.23
false-positive windows per hour and final B from 206.55 to 1739.39, roughly three times and
eight times their earlier rates. The exam surface moved under every head at once.

Why the tree moved: the v2c generation pairs the eyes model of record with the detector
score threshold 0.3 inherited from the earlier generations, and on the clinical footage the
track count of a free-running pat03 video rises from 54 to 331. The detector probe that
measured this behaviour on side-view footage, and the reading of its disagreement frames, is
`research/eyes-coupling-probe.md`; the fall lane recorded the same movement on the same day.
The fix is a further declared extraction generation with a detector that is in domain on
this footage, which is an owner decision; nothing has been re-extracted.

## Figures

- `final_models/seizure/validation_curves.png`: per-epoch validation area under the curve
  for every run of this lane on one axis, convulsion_v2pose and final B picked out in colour
  and the other 31 runs in grey behind them. One of those grey curves is the v2c refit
  recorded above, `curves/conv_v2c_finalB_s20260826.csv`, which validates on the v2c clinical
  tree holdout rather than on the surface the rest of the axis carries, so its 0.88910 does
  not stand against the curves beside it; the same recipe on the earlier tree reached
  0.75479. The model of record has no per-epoch record in this repository, so it is not a
  curve on this figure: its best validation area under the curve of 0.77152 is drawn as a
  horizontal dashed reference line. The axis is labelled
  AUC, not AP: the third column of every curve CSV in this lane is named `val_ap`
  for uniformity across lanes, but for the event verifiers it carries the trainer's
  per-epoch validation area under the curve read from each run's report.
- `final_models/seizure/champion_vs_baseline.png`: the same train-time holdout axis carrying
  only the compared runs, the convulsion_v2pose curve and the final B curve against the
  reference line for the model of record at 0.77152. Neither held-out exam is on this axis
  and neither standing bar is drawn on it, because both are measured on held-out video; the
  exam values are in the before and after table above, and the bars are E1 0.87422 held by
  final B and E2 0.72916 held by the model of record, as recorded in the standing bars
  section above.

Source CSVs are `final_models/seizure/curves/*.csv`, 33 files, 60 epochs each, and the figure
above carries all 33 of them.

## Full digests

Every short form used anywhere above resolves to a full sha256 either in this table, in the
ladder table or in the track-tree tables, so no number on this page is quoted against a
hash that exists only as a prefix.

| artefact | sha256 |
|---|---|
| Model of record, convulsion_v2_real checkpoint | `715d172c9fd8523c3da43faba34644762f847b8a9d3eaa9dc293efe7dda1830b` |
| Model of record, convulsion_v2_real graph | `07c399be24f89a44670262ea75879cc67f60f46c5cd52e5d2cc4218e914b7e22` |
| Final B checkpoint, the E1 bar holder | `535141c19f978ad53b51a74add2dec9227d085561ae6dafc293954cf0a52aacd` |
| Final B graph, the E1 bar holder | `b9e04d034fd751f6f087b2c884f033c2a754061cd5719d226011d5839552aecc` |
| Alternative, convulsion_v2pose checkpoint | `9367365c35bafe368377861e2300bb734848bfc00d33cc112b4ee0cc6dc3be42` |
| Alternative, convulsion_v2pose graph | `614b92fa729b670738e1521b2c3a7f04046d32444f16a08fdf7e58d7740eb0d4` |
| Chained prototype detector checkpoint | `55c2c242f0da4012e5c63dcf8a93e6951da0bc191b869d6fe97b7df088cca653` |
| Chained prototype detector graph | `6a6e4d90a3774522d7e1127122a5a00be41a947e93c8940a3fc6a0c09f97c3ce` |
| Joints lane model of record checkpoint, token `a0a035fe` | `a0a035fe65909980447560a70ae2f8d13400c1d12490a929e7cf3c5a7b26f102` |
| Eyes lane model of record checkpoint, token `345c30a5` | `345c30a539c2b6bb665302825f418322284b98f6cbcf55d58d84ffeb75b16e7e` |

Every verifier digest above, and every digest in the ladder table, was read from the run
report each trainer wrote beside its own artefacts. For every variant that was measured on
an exam, the graph digest is confirmed a second time by the model digest recorded in that
exam's own report. One gap remains on this lane and it is a measurement gap rather than a
hash gap: convulsion_v5_real has no exam number, because no exam look was taken.

## Detail

- Comparison against the newer pose, with the full exam tables:
  `research/seizure-v2pose-comparison.md`.
- Standing bars, the one-variable declaration and the append-only exam-look ledger:
  `research/seizure-push-3-protocol.md`, with the earlier pushes in
  `research/seizure-push-1-protocol.md`, `research/seizure-push-1-verdict.md`,
  `research/seizure-push-2-protocol.md`, `research/seizure-push-2-arms.md` and
  `research/seizure-push-2-out-of-fold-gate.md`.
- The historical ladder, the barred negative-diet arms and the information-starvation
  finding: `research/event-verifier-v1-record.md` and `research/stats-compendium.md`.
- Threshold study and the decision-layer definitions: `research/operating-points.md`.
- Negative-side counts, their denominators and the proxy-fingerprint caveat:
  `research/event-negatives-bookkeeping.md` and
  `research/seizure-ajile12-negative-control.md`.
- Why limb channels did not rescue the lane: `research/seizure-limb-channel-gate.md`.
- Measurement rules that govern every number above: `POLICY.md`.
