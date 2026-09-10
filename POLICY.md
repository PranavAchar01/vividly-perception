# Project policy

Every standing rule for this repository, stated once, in the imperative. This file is the
single source. Where a rule is enforced by code, the enforcing file is named by path, and the
path is the authority on the exact behaviour.

Nothing here is legal advice. Questions reserved for counsel are recorded in
[docs/COUNSEL_QUESTIONS.md](docs/COUNSEL_QUESTIONS.md), keyed on the counsel-question
identifiers that the dataset registry entries cite.

## 1. Scope

The product is privacy-preserving awareness for hospital waiting rooms, built on the cameras
already on the ceiling. Five model lanes serve it: a person detector (eyes), a pose estimator
(joints), a fall verifier, a seizure verifier, and session-scoped re-identification.

The training target domain is ordinary perspective and top-view security-camera footage, that
is wall-mounted or ceiling-mounted rectilinear closed-circuit television. It is not
180 degree overhead fisheye. Compute is rented single-GPU pods; the pods are Blackwell class,
which sets the floor on the training stack.

The runtime layer is a replay pipeline: a frame source feeds detection and pose, a tracker
assigns identifiers, per-track state accumulates each person's own baseline, rule heads read
that state, firings become alert records in SQLite, and a local dashboard renders the overlay
and captures reviewer feedback. It runs against replayed files and synthetic fixtures.
Real-time capture from a site's own cameras is bring-up work, not a deployment.

The rule layer detects acute events with verification. Two candidate rules read live track
state, `fall_then_still_v0` (a fast fall confirmed by a prolonged down-and-still period) and
`seizure_candidate_v0` (sustained convulsive motion). A third head,
`egress_without_discharge_v0`, is an engine attachment rather than a rule, because its subject
is a track the book has already reaped. A trained verifier confirms a candidate before any
alert is stored.

The defect class that matters is the false positive. Rules are multi-condition cascades that
abstain on thin, stale or occluded evidence, and they never attempt to detect gradual
deterioration. Prevention-era detectors (stillness, postural drift, variance collapse, staff
proximity) were removed and are not reintroduced without an owner decision.

Out of scope: deployment, live capture as a product path, automatic identity binding of any
kind, cross-visit or cross-site identity, learned temporal models beyond the event verifiers,
real chart binding, real staff identification, alert delivery to messaging systems, and
anything patient-facing.

## 2. The standing rules

### Rule 1. No fisheye imagery

No fisheye dataset is fetched, re-downloaded, or licence-requested. Requesting that a person
fill in a fisheye licence form is itself a violation of this rule. No fisheye imagery, real or
rendered, enters any training or evaluation manifest in any lane. No fisheye camera model,
render harness or rendered corpus exists in the tree, and none is added.

Enforced by `scripts/guards/check_no_fisheye.py` and the `no-fisheye` pre-commit hook, which
fail on any new fisheye reference in configurations, scripts, source, tests or continuous
integration. The guard reads its frozen inventory from
`scripts/guards/no_fisheye_allowlist.txt`. Extending that inventory is an owner decision, never
the way to land a change.

The primitives that would produce the imagery are banned outright rather than inventoried.
`scripts/guards/check_no_fisheye_primitives.py` and the `no-fisheye-primitives` pre-commit hook
fail on any use of `cv2.fisheye`, `cv2.remap` or `cv2.initUndistortRectifyMap` under `src/`,
`scripts/` or `tests/`, read from the syntax tree so that an aliased import is followed and a
`from cv2 import remap` cannot launder the call. There is no allowlist and no exempt directory
for these three: a change that needs one of them is a change to this rule.

One recorded consequence: acceptance criteria for re-identification that were declared on
fisheye surfaces are void, not failed and not passed. Acceptance for the top-view domain is
declared from scratch on non-fisheye surfaces, and any new embedder trains on non-fisheye data.

One recorded history, carried so it is not rediscovered: two seizure arms trained on a
negative diet extracted from fisheye recordings were measured and are barred, their recipes
were deleted on 2026-09-07, the record stays on `final_models/seizure/README.md`, and no
surviving configuration reaches such rows. No extraction from a fisheye source is permitted.

### Rule 2. No facial recognition

No face detection, no face embedding, no face matching, and no face-localisation branch in any
model, anywhere, in any phase. Ten distributions are denied outright: `face_recognition`,
`insightface`, `deepface`, `retinaface`, `facenet-pytorch`, `mtcnn`, `dlib`, `mediapipe`,
`face-alignment`, `facexlib`.

Enforced by `scripts/guards/check_face_deps.py`, which fails on any face-biometric
distribution in `pyproject.toml` or in `uv.lock`, transitive arrivals included, and by the
`no-face-biometrics` contract in `.importlinter`. The task vocabulary in
`src/vividly/models/base.py` is closed to `person_detection`, `body_pose`, `tracking` and
`appearance_embedding`, so there is no face task to select.

Any dataset package whose contents are face imagery is refused at registration. A dataset
whose distribution includes such a package is registered without it, and the package is never
fetched.

### Rule 3. No vital-signs extraction

No photoplethysmography, no heart rate, no respiration, no acuity scoring, and no proxy for
any of them admitted as a rule input.

### Rule 4. No simulated or acted medical events in training

No simulated or acted medical event enters a training set, and generative patient imagery is
barred without exception. Two carve-outs are recorded, and both are stated in full because
they are what makes the current event models admissible.

**Carve-out (a), engineering-only test sets.** Public, licence-reviewed acted-event datasets
may be registered as internal test sets. They train nothing. No number they produce is ever
quoted, internally or externally, as detection performance. The registry mechanism is an
`EVAL_ONLY` entry with licence review completed before a byte arrives, carrying an
engineering-only marker.

**Carve-out (b), interim prototype event models.** An interim event model may train on public,
licence-reviewed acted-fall datasets and on convulsion motion synthesised in pose space and
track space. Every such weight is prototype class, is badged internal demonstration only, is
never quoted internally or externally as clinical detection performance, is never deployed
commercially without counsel sign-off, and is slated for replacement the moment clinical
footage exists. At least one acted-fall dataset is held out entirely as the test set, and a
dataset that trains loses its test role permanently. Synthesis stays at motion level, that is
skeleton and track trajectories. Generative patient imagery remains barred, and self-recorded
footage remains barred.

**Augmentation is not synthesis.** Label-preserving geometric and temporal transforms of
already-admitted real recorded event tracks are augmentation: mirroring, bounded time warp,
contiguous observation dropout, and camera-geometry warps such as foreshortening, perspective
and roll, with pose scalars recomputed from the transformed keypoints. The label rides along
because the underlying event is a recorded fact. Augmentation is allowed in both event lanes,
focal tracks included, applied to training rows only and only after the split, so no augmented
view of a holdout row ever appears in training. Evaluation-only and exam-side data are never
transformed into anything. The implementation home is `src/vividly/verifier/augment.py`, behind
an `augment:` configuration key that defaults to off; identity round-trip, label-invariance and
not-a-number tests are required before any run.

**Two interpretations, recorded so they are not re-asked.** A live demonstration is not a
dataset: a person lying down in front of a running system to show an alert fire is not a
training or evaluation set, and nothing from a live demonstration is recorded into any corpus.
Self-recorded footage is a separate matter and stays barred: no test-camera purchase, no
mounting, no recording sessions, no labelling loop over footage recorded for this project, and
no camera-zoo hardware.

**Candidate rules train on nothing.** The geometric rule heads are written from clinical
literature and geometry. Closed-form unit-test trajectories, which are abstract points on a
plane tracing threshold-crossing curves and never renderings of people, sit outside this rule
and never enter any manifest.

**Licence recording obligation.** Each fall dataset's licence terms are recorded word for word
in its registry entry, and any research-only or non-commercial wording is flagged for counsel
rather than silently absorbed.

**Label integrity.** Where a negative diet is drawn from a corpus that also contains events,
the event-containing spans are excised using the corpus's own annotations before any window
enters a negative manifest. Annotated event windows never become positives: the labels belong
to the publishing group and are not repurposed.

**Permanent allocations.** A patient, subject or recording allocated to the exam side stays on
the exam side permanently and never enters a training manifest. The recorded allocation is
patients `pat06`, `pat08`, `pat10` and `pat12` of the epilepsy-monitoring corpus, which form
the seizure exam surface of focal, label-uncertain onsets, and patients `pat03` and `pat07`
of the same corpus, which form the locked convulsive exam surface.

### Rule 5. AGPL isolation

Code under the GNU Affero General Public License, which today means `ultralytics` and
`boxmot`, is importable only from `src/vividly/eval/agpl_ref.py`, and that module runs as a
separate process in a separate virtual environment. Numbers come back as serialised JSON. The
two trackers are never imported into one process.

Enforced three ways: the `conflicts` table in `pyproject.toml` under `[tool.uv]`, which makes
co-installation a hard resolver error rather than a silent downgrade; the `agpl-quarantine` and
`runner-never-imports-agpl-ref` contracts in `.importlinter`; and
`scripts/guards/check_agpl_imports.py`.

An architecture under that licence is never adopted into a lane, the YOLO family included.
A candidate architecture is preferred under Apache 2.0, MIT or BSD, and any other licence is
recorded as a counsel flag before its code is pulled.

Numbers produced by such code stay in internal engineering documents and are labelled with
their source. No pseudo-labels are generated from them, so no shipped weight has seen a label
produced by that code.

### Rule 6. Evaluation-only data never trains

An `EVAL_ONLY` dataset never enters a training manifest or a training configuration. The
registry raises, the directory trees are physically separate under `data/`, and
`scripts/guards/check_train_data_paths.py` resolves every path in every training configuration,
following symbolic links, and fails if a path lands outside the training trees.
`scripts/guards/check_config_registry.py` requires that every dataset a training configuration
reaches is a registered entry. The bucket vocabulary is `TRAIN_OK`, `TRAIN_PROTOTYPE`,
`EVAL_ONLY` and `LEGAL_HOLD`, defined in `src/vividly/data/registry.py`.

`TRAIN_PROTOTYPE` trains prototype weights only. A configuration that reaches into the
prototype tree must declare the prototype weight class, and the guards reject it otherwise. A
training-grade corpus may never derive from the prototype tree, and no prototype weight is
deployed commercially without counsel sign-off.

An exam is frozen after the data has been inspected, and exam scenes never train.

### Rule 7. Commodity cameras only

Only cameras a site already owns. No component may assume cloud connectivity at inference
time. The dashboard serves inline markup and script with no external assets and no content
delivery network, which is the local consequence of this rule.

### Rule 8. Checkpoint provenance is recorded, never assumed

Every checkpoint carries a provenance record with its sha256, its training data and its
backbone pretrain data. The record lives on the lane page under `final_models/<lane>/README.md`
for a model of record, in `configs/checkpoints/registry.yaml` for the baseline checkpoints that
file scopes itself to, and in `third_party/PINS.yaml` for the trainer-side initialisation
checkpoints. `src/vividly/models/base.py` types shippability as `yes`, `no` or `counsel`
and `assert_shippable()` raises for anything that is not `yes`.

Four provenance facts are recorded because they govern shippability and are easy to assume
wrongly:

- No published RTMPose checkpoint is free of the AI Challenger corpus. Every one of them, the
  checkpoint upstream calls COCO-only included, is fine-tuned on top of a backbone that was
  pose-pretrained on AI Challenger plus COCO.
- The RTMDet person detectors used Objects365. Objects365-derived checkpoints are not assumed
  to be commercially cleared, and every artefact carrying that lineage is refused by the
  checkpoint law in `third_party/PINS.yaml`.
- RT-DETRv4 initialises from COCO train2017 on an ImageNet backbone, with no Objects365 lane.
- Shippability is a question for counsel, and is never resolved inside this repository.

### Rule 9. No unmeasured performance claim

No performance number is quoted that has not been measured on this stack. In particular, no
Jetson Orin frame-rate figure is quoted: no upstream figure exists for that module, so any such
number would be invented. Every evaluation logs latency in milliseconds beside every accuracy
number.

### Rule 10. Every accuracy number carries its checkpoint hash

A number without its hash is not quotable, because one value can belong to two different
checkpoints. This applies to a threshold fitted against a checkpoint exactly as it applies to
an accuracy figure.

Every re-identification number is provisional and is quoted with the embedder sha256 beside
it. The literal marker `PROVISIONAL` is asserted in sixteen places across eight test files
and stays in code and in configuration comments.

### Rule 11. Weight provenance for re-identification

Banned appearance-embedding weight sources are refused at load time and at export time, and
the refusal message names the licence allowlist. The rules are recorded in
[docs/LICENCE_ALLOWLIST.md](docs/LICENCE_ALLOWLIST.md) and enforced by
`scripts/guards/check_reid_weight_sources.py`, by `src/vividly/models/reid_backend.py` at load,
and by `scripts/reid/export_osnet_onnx.py` at export.

The denylist covers LUPerson-derived lineages, which are never shippable and are benchmarking
material only, and the checkpoints derived from Market-1501, MSMT and Duke. Appearance weights
come only from the canonical MIT-licensed distribution, ImageNet-pretrained variant preferred.

### Owner decision, 2026-09-09: the Clip Library on hospital premises

Event-only Clip storage on hospital premises under the `clip_library` pipeline profile is in
scope; continuous recording remains out of scope; nothing from the founder's rig enters any
corpus; the rig subject is the founder alone unless a written consent decision for a named
adult non-patient participant is recorded here first. The profile
(`configs/pipeline/clip_library.yaml`) may run exactly two rule heads, `fall_then_still_v0`
and `egress_without_discharge_v0`, keeps its own event store in memory, persists no
re-association, runs no verifier head, and writes one spool record per firing for the
separate `vividly-clip` repository to cut, encrypt and retain; the third head and the
periodicity analyzer stay on the device track. `scripts/guards/check_clip_library_config.py`
and `tests/test_clip_emitter.py` enforce this; the design and its counsel questions are
`~/vividly-camera/clip/CLIP-LIBRARY-DESIGN.md`.

## 3. Dataset acquisition and licensing

**Acquisition channel, standing across all lanes.** Datasets are acquired only by self-serve
anonymous download. No dataset owner is ever contacted. No email is sent, no sign-up is
completed, no form is submitted, and no licence-clarification request is made, for any lane,
ever. Any plan step that requires such contact is dead on arrival and is not re-proposed.

**Licence ambiguity.** Where a publisher states two contradicting licences, or states none, the
conservative reading governs and the contradiction is recorded word for word in the registry
entry's conflict note. Ambiguities are resolved by counsel at ship time, never by asking the
publisher.

**Registration before bytes.** A registry entry lands at status `requested` with its licence
text, its licence URL and its acquisition record before anything is fetched. Fetching is double
locked: an explicit flag and an environment variable, and gated sets are refused outright. The
entry flips to `available` with verified sizes and hashes after arrival.

**Use-based restrictions are read as binding.** A licence that prohibits surveillance or the
re-identification of individuals is treated as reaching a patient-monitoring product even in
engineering-only use. Such a set stays out unless counsel affirmatively clears it, and no bytes
are fetched in the meantime.

**Redistribution.** No registered dataset is redistributed from this repository. Clinical and
clinical-adjacent footage stays on the machines that process it: no cloud synchronisation, no
object-store copy, no redistribution.

**Prototype corpus admission.** A corpus admitted for prototype training trains interim
prototype weights only. Those weights are internal demonstration material, and they are never
deployed, distributed or quoted externally without counsel sign-off on the conflict note in the
entry. Counsel-question identifiers are added to an entry's `counsel_questions` only together
with a real `counsel_cleared_utc`, in one edit.

**Synthetic corpora are candidates, not anchors.** A synthetic training slice enters a bake-off
as a pre-training or augmentation candidate that must beat the plain COCO-pretrained baseline
on the frozen exams. Where a synthetic release annotates visible parts only, the box convention
is normalised before anything trains on it.

The registry entries under `configs/datasets/` are the legal record: the licence text as
published, the licence URL, conflict notes, counsel-question identifiers, terms-acceptance
records and event grants live there. [DATASETS.md](DATASETS.md) summarises them and never
replaces them.

## 4. Re-identification scope

Re-identification is session-scoped appearance embedding from image crops. Nothing else.

- A gallery holds appearance embeddings for one visit, is never persisted across visits, is
  never a cross-site identifier, and is cleared at session end.
- Beyond the tracker's own short re-association window the system never auto-binds. It proposes
  a match, a person confirms or rejects it, and no code path joins two track identifiers
  without that confirmation. Both answers are stored as labels.
- No pose signal, no gait signal, no body-shape or anthropometric signal, and no face signal of
  any kind is used for identity.
- The headline measure is the false-merge rate, not rank-1, because a false merge joins two
  patients' identities.
- No re-identification output enters the shipped path before the freedom-to-operate opinion
  recorded as CQ-20.

## 5. Measurement discipline

- A model is compared only on an exam it never trained on.
- Variants are selected on the train-side validation split, never on the exam.
- Each version gets one pre-declared look at the test set.
- An exam is never edited to flatter a model. Frozen exams stay frozen.
- A multi-seed noise floor is measured before any comparison of two arms is called a result.
- Recipes are committed before runs. Every number travels with its checkpoint hash, its seed
  and its configuration.
- A recipe change is adopted through a pre-declared gate, a run that changes one variable, and
  one look at a frozen exam. A configuration declared as a fixed number of changes from a
  previous one states those changes in its own header before the run.
- A firing count from a replay run is a count, never a false-alarm rate. Turning one into the
  other is the protocol in [docs/CALIBRATION.md](docs/CALIBRATION.md).
- Every rule threshold lives in `configs/pipeline/rules.yaml`, every one of them is provisional,
  and none is quoted outside this repository before it has been through that protocol.
- Claim wording is fixed: the system flags event candidates that a verification stage confirms.
  It is not described as detecting seizures or strokes.

## 6. Toolchain rules

- Python is pinned to 3.11. `xtcocotools` publishes no wheel for 3.12 or later and none for
  Linux on aarch64 at any version.
- The shippable tracker is `trackers`, Apache 2.0. `supervision` is used only for its detection
  container and its annotators, never for keypoints.
- TrackEval comes from the maintained PyPI distribution, never from the upstream git URL, which
  still uses a removed NumPy alias and crashes on empty-tracker sequences.
- Video processing shells out to ffmpeg with an explicit argument vector, because the literal
  command is recorded in the sidecar. The `ffmpeg-python` wrapper is banned.
- On macOS, ONNX inference runs on the CPU execution provider. The CoreML provider crashes on
  both detector graphs at the pinned runtime version, so no default configuration selects the
  Metal device.
- The OpenMMLab installer is not used: `mim install` is never run and `openmim` is never added.
- The GPU extra is a no-op on macOS. It installs nothing and exits zero, and that is not a
  working training environment.

## 7. Engineering conventions

- Every runner, trainer, extractor and evaluator under `scripts/` that reads a recorded recipe
  takes `--config`, and the recipe rather than the script source carries every dataset path,
  hyperparameter and threshold. Guards, analysis tools, instrument graders, weight soups,
  parity probes and the demo platform launcher sit outside the rule and take plain flags.
- Model backends are swappable behind configuration. Detector, pose, tracker and appearance
  embedder are protocols in `src/vividly/models/`.
- Public functions carry type hints. Metric and media helpers are pure functions.
- No secrets in code. Credentials come from the environment; the shape is in `.env.example`.
  Experiment tracking is offline-first and no runner may hard-depend on a hosted service.
- Colour order reaches the live path only through a full track-tree re-extraction, never as a
  backend hotfix, because every verifier was trained on features extracted in the current
  colour order.
- No temporal smoothing exists anywhere in the alert path and none is added. Keypoints, boxes
  and centres reach a verifier exactly as the models emitted them. A five-frame moving average
  at 15 frames per second places an exact spectral null on 3.0 Hz, the centre of the clinically
  validated 2 Hz to 6 Hz convulsive band, so the smoother would erase the event it exists to
  confirm. The one admissible filter inside a verifier's feature extractor is an explicit band
  pass that keeps that band. Smoothing for display lives on the dashboard branch and never
  feeds back. Enforced by `tests/test_alert_path_raw_signal.py`.
- Nesting outside `src/vividly/` is at most one group directory deep.
- Commits are small and conventional, and each task ends on a green commit.

## 8. Enforcement index

| Rule | Mechanism |
|---|---|
| 1, no fisheye | `scripts/guards/check_no_fisheye.py`, `scripts/guards/no_fisheye_allowlist.txt`, the `no-fisheye` pre-commit hook; `scripts/guards/check_no_fisheye_primitives.py` and the `no-fisheye-primitives` pre-commit hook, which ban `cv2.fisheye`, `cv2.remap` and `cv2.initUndistortRectifyMap` outright, with no allowlist |
| 2, no facial recognition | `scripts/guards/check_face_deps.py`, the `no-face-biometrics` contract in `.importlinter`, the closed four-member task vocabulary in `src/vividly/models/base.py` |
| 3, no vital signs | rule inputs are declared in `configs/pipeline/rules.yaml`; no such input exists |
| 4, event data | `src/vividly/data/registry.py` event grants and exclusive roles, `src/vividly/verifier/augment.py` train-rows-only augmentation |
| 5, AGPL isolation | the `conflicts` table in `pyproject.toml`, the `agpl-quarantine` and `runner-never-imports-agpl-ref` contracts in `.importlinter`, `scripts/guards/check_agpl_imports.py` |
| 6, evaluation-only never trains | `src/vividly/data/registry.py`, `scripts/guards/check_train_data_paths.py`, `scripts/guards/check_config_registry.py`, the separate trees under `data/` |
| 7, commodity cameras | no network dependency in the dashboard or the runner |
| 8, checkpoint provenance | `configs/checkpoints/registry.yaml`, `final_models/<lane>/README.md`, `assert_shippable()` in `src/vividly/models/base.py`, the checkpoint law in `third_party/PINS.yaml` |
| 9, no unmeasured claim | latency logged beside every accuracy number in `src/vividly/eval/` |
| 10, hashes travel with numbers | `final_models/README.md` index, the `PROVISIONAL` marker asserted by the test suite |
| 11, re-identification weights | `docs/LICENCE_ALLOWLIST.md`, `scripts/guards/check_reid_weight_sources.py`, `src/vividly/models/reid_backend.py`, `scripts/reid/export_osnet_onnx.py` |

Guards are proved by `bash scripts/guards/selftest.sh`, which runs a rejection case for each
one and fails if fewer than its floor of cases run.

## 9. Changing this file

A rule leaves this file only on an owner decision. Except on such a decision,
`counsel_cleared_utc` is never set, a registry entry is never moved out of `LEGAL_HOLD`, a
shippability question is never resolved, and a guard's allowlist is never widened to land a
change.
