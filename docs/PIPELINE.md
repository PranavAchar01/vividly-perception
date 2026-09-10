# The replay pipeline

The replay layer is the end-to-end path through the product. A frame source feeds detection and
pose, a tracker assigns identifiers, per-track state accumulates each person's own baseline,
rule heads read that state, firings pass through a verifier before anything is stored, surviving
firings become alert records in SQLite, and a dashboard renders the overlay, per-person timers
and an alert feed with agree and false-alarm buttons.

Read alongside [../POLICY.md](../POLICY.md), which carries the rules this layer implements, and
[CALIBRATION.md](CALIBRATION.md), which is where the thresholds go once there is data to fit
them to. Every rule threshold is provisional; section 5 says why.

---

## 1. What this layer is and is not

The layer commits to structure: interfaces, wiring, the persistence shape, and the capture of
reviewer feedback. It commits to nothing about alert quality, because no threshold in it has
been fitted to data.

Two properties make it useful before a pilot site exists. The file source supports deterministic
seeking and faster-than-realtime playback, so the harness runner and the pipeline are the same
code path. The rule heads are developed against programmatically generated trajectory fixtures
rather than video, so their unit tests are deterministic and need no footage.

## 2. Why replay only

Two standing rules govern the layer.

**No footage is recorded for this project.** There is no test-camera purchase, no mounting, no
recording session, and no labelling loop over footage recorded for this project.
Camera-to-camera variation is handled by the licensed corpora and by pilot-site cameras later.
Existing ceiling cameras at hospitals remain the product; what is barred is recording hardware
and workflows owned by this project. The network video recorder in
[LIVE_CAMERA.md](LIVE_CAMERA.md) is not an exception to that bar: it is a live inference
bring-up, nothing it sees is recorded into any corpus, and no frame from it enters any manifest
or any training or evaluation set.

**Thresholds stay provisional until pilot shadow-mode data exists.** The weights precondition
is met, by the eyes and joints models of record and their recorded hashes; pilot shadow-mode
data is one of the three still outstanding. All four preconditions are listed in
[CALIBRATION.md](CALIBRATION.md).

Read together: the only footage this pipeline sees before a pilot is licensed third-party clips
and generated fixtures. Live mode means a site's own camera, at their site, later. That is why
`RTSPSource` exists in the codebase as an interface stub whose every method raises
`NotImplementedError`; the working body for bring-up against real hardware is described in
[LIVE_CAMERA.md](LIVE_CAMERA.md).

One live path exists, on an owner decision recorded in `POLICY.md` on 2026-09-09: the Clip
Library profile (`configs/pipeline/clip_library.yaml`) reads a camera through
`source: {rtsp_env: <ENV VAR>, camera_id: ...}`, which `LiveRTSPSource` serves with a wall-clock
stamp (`wall_utc`) on every frame and a `source_epoch_utc` re-based on every reconnect or
stall, blacks out the configured exclusion masks before detection, keeps its event store in
memory, runs exactly the two rule heads named in its mandatory `rules.enabled` allowlist, and
hands each firing to `src/vividly/clips_emitter.py`, which writes a spool record for the
separate Clip Library after the store commits. It records event Clips only, never a continuous
stream; the ring that feeds the cut lives in the library's memory and is not this pipeline's.
`scripts/guards/check_clip_library_config.py` holds every profile that writes a spool to those
rules, whatever the file is called.

## 3. Component map

    sources  ->  tracks  ->  rules  ->  verifier  ->  events  ->  dashboard

**Sources (`src/vividly/sources/`).** `FrameSource` is a protocol with `open()`, `frames()`,
`seek()` and `close()`. `open()` returns `SourceInfo` (fps, width, height, n_frames) and
`frames()` yields `Frame` records (index, t_source, BGR image). `ReplaySource(path, rate)` is
the concrete file source: `rate=None` plays as fast as the machine allows, `rate=1.0` paces to
the wall clock, seeking is deterministic through the OpenCV frame-position property, and every
frame is yielded with `t_source = index / fps` exactly, so two runs over the same file produce
the same timeline. `RTSPSource(url)` is the stub described above.
`SyntheticDetectionSource(path)` is deliberately not a `FrameSource`: it reads a JSONL fixture
and yields per-frame lists of `TrackObservation` from `observations()`, which lets track state,
rules, the store and the dashboard run with no footage and no model weights.

**Image sequences (`src/vividly/sources/image_sequence.py`).** `ImageSequenceSource` plays a
directory of numbered frames as a `FrameSource` at a rate the caller declares. No frame
directory records its own rate, registered corpora ship in that shape (CAUCAFall keeps
per-frame stills beside each sequence video, and MVOR ships numbered colour stills per camera),
and their rates come from the dataset pages as recorded in `configs/datasets/*.yaml`.
Natural-sort order, pattern filtered (`*.jpg` by default; some corpora keep a sidecar file
beside every frame), `t_source = index / fps` exactly as `ReplaySource`, frame-exact seek, and
a decode failure raises rather than shifting every later timestamp. The runner takes it as
`source: {images: <dir>, fps: <Hz>, pattern, rate}`, and `fps` is mandatory there, because a
defaulted rate would put every rule window on a guessed clock.

**Tracks (`src/vividly/tracks/`).** A `TrackObservation` carries track_id, time, box centre and
size, and optionally 17 keypoints as (x, y, conf). `TrackState` keeps ring buffers per track and
exposes `positions()`, `posture_angles()`, `movement_variance()` and `baseline()`, each over a
caller-supplied window. Posture angle is the angle from vertical of the torso axis, the line
from the shoulder midpoint (COCO keypoints 5 and 6) to the hip midpoint (11 and 12), and is
undefined when any of those four keypoints falls below the configured keypoint-confidence
floor. `baseline()` returns `None` until the baseline window has elapsed and then returns
`BaselineStats` (movement variance, posture mean and standard deviation, mean vertical position,
sample count, completeness flag); the window is each person's own first five minutes
(`baseline_window_s`, 300 s by default), and it is configurable. `TrackBook` owns the set of live
tracks, ingests observation batches, and holds the three manual bindings: `set_staff()`,
`set_discharged()` (read by the egress head) and `bind_mrn()`. A track absent longer than the
configured reap window is dropped; a track that leaves and returns inside that window keeps its
identifier and its baseline.

**Rules (`src/vividly/rules/`).** A `Rule` has a `rule_id`, a `rule_version` and
`evaluate(track, now)` returning a `RuleFiring` or `None`. Rules are pure functions over track
state and hold no mutable state of their own. Debounce lives in `RuleEngine`, which fires a
given (rule_id, track_id) pair at most once per configured refire interval. A `RuleFiring`
separates `onset_t`, when the condition began, from `detected_t`, when the tick noticed, and
carries the baseline snapshot plus a details dictionary.

Two rule heads ship. `fall_then_still_v0` is a fast upright-to-down transition with real
downward displacement, confirmed by the person staying down and still for a prolonged period;
both stages are required, so a trip followed by getting up fires nothing. `seizure_candidate_v0`
is sustained, high-amplitude, direction-reversing rhythmic motion. Both abstain on thin, stale
or occluded evidence, because the false positive is the defect class that matters. The engine's
book hook remains for a future room-context rule even though no default rule uses it.
`fixtures.py` holds the deterministic trajectory generators the unit tests run on.

**Egress head (`src/vividly/rules/egress.py`).** `egress_without_discharge_v0` is the third head
and deliberately not a `Rule`: its subject is a track the book has already reaped, so it cannot
be evaluated per live track. `EgressMonitor` is attached to the engine
(`RuleEngine(..., egress=monitor)`) and on every tick diffs the book against its previous
snapshot. A track that vanished becomes a departure candidate only if its own last samples
(inside `egress_exit_window_s`) put it in a configured `ExitZone`, it had been present for at
least `egress_min_presence_s`, it was not staff tagged, and it was not tagged discharged
(`TrackBook.set_discharged()`, the dashboard's discharge tag, which rides identity carry-forward
like the staff tag). A candidate is retired by `mark_discharged()` from the dashboard's
departures panel, or by a confirmed return, that is a `merge_identity()` whose
`reassociated_from` names the departed identifier. Otherwise it fires once, when the person has
been absent `egress_absent_min_s`, with `onset_t` at the last sighting, and the firing takes the
same debounce and verifier path as the rule heads (the verifier receives `track=None`). Exit
zones are polygons in the `egress:` block of the pipeline configuration, off by default; the
monitor refuses to build unless that block is enabled with at least one zone. A mid-room loss, a
short visit, a staff member, a discharged patient, a confirmed return and an ordinary hour all
fire nothing, by test (`tests/test_egress.py`).

**Verifier seam (`src/vividly/rules/verify.py`).** Every firing that survives debounce is
dispatched through an `EventVerifier` when one is attached. The verdict set is closed:
`confirmed`, `rejected` and `unavailable`. A rejected firing never reaches the store. Under the
default `annotate` policy an `unavailable` firing flows through with that answer stamped on it;
under the fail-closed `require` policy it is dropped and counted. The trained implementation is
`src/vividly/models/event_verifier.py`, one ONNX head per rule identifier, built from a lane
directory under `final_models/` or from a training run directory, so the window geometry comes
out of the report the directory carries rather than being restated in the pipeline
configuration, and refusing a graph whose sha256 does not match that report. A firing whose rule
identifier has no head, egress firings included, gets `unavailable`, never a guess.
`AbstainingVerifier` is the placeholder when no head is attached.

Every model this backend loads is a prototype weight ([../POLICY.md](../POLICY.md), rule 6, the
`TRAIN_PROTOTYPE` paragraph): internal demonstration material only and never quoted as clinical
detection performance, which is the badge rule 4's carve-out (b) puts on every such weight and
the citation `src/vividly/models/event_verifier.py` carries at the point of loading. What such
a model may be trained and evaluated on stays open as CQ-21 in
[COUNSEL_QUESTIONS.md](COUNSEL_QUESTIONS.md). Nothing in the alert path smooths the signal; the
reason is in [../POLICY.md](../POLICY.md), engineering conventions.

**Events (`src/vividly/events/`, `src/vividly/schemas/`).** `AlertRecord` is a pydantic v2 model
with `schema_version` pinned to 1, carrying alert identifier, track identifier, rule identifier
and version, onset and detected timestamps, creation time, the baseline snapshot, a details
dictionary, an initially empty feedback label, and the mock chart binding. `EventStore` is
standard-library `sqlite3` in write-ahead-log mode with `PRAGMA user_version=3`, exposing
`insert_alert()`, `set_feedback()`, `alerts(limit)` and `close()`. The two version numbers count
different things: `AlertRecord.schema_version` is the record's wire format and is still 1, while
`user_version` is the version of the file, at 3 since `reassoc_proposals` gained its `run_id`
column and the two partial-unique indexes over its confirmed rows
(`src/vividly/events/migrations.py`). The alert identifier is derived, not random:
`f"{rule_id}-{track_id}-{int(onset_ts*1000)}"`, so replaying the same clip twice does not
manufacture a second alert for the same onset.

**Dashboard (`src/vividly/dashboard/`).** `create_app(state)` returns a FastAPI application.
`GET /` serves inline markup and script with no external assets and no content delivery network,
which is the local consequence of the commodity-camera rule. `WS /ws` streams the encoded frame,
current tracks, current alerts, per-track timers since first seen, and per-stage latency. The
POST routes write back: alert feedback (agree or false alarm) into the store, the staff flag and
the discharge tag onto a track, the chart binding, and the discharge of a departure already on
the egress clock (`POST /departures/{track_id}/discharged`; `GET /api/departures` and the
socket's `departures` key list what the egress head is holding). The feedback route is as much
the point of the dashboard as the overlay is: it doubles as the labelling interface and
satisfies the reviewer-response governance requirement.

**Runner (`scripts/run_pipeline.py`).** One script wires the chain from a `--config` file and
runs it either behind the dashboard or headless. It times the stages source, detect, pose,
track, rules and render, and writes p50 and p95 on exit, which is the rule the rest of the
repository follows: latency is logged next to every other number. Setting `detections_from` in
the configuration swaps the model for `SyntheticDetectionSource`, which is what makes the
zero-footage flow possible. Weights are a configuration swap by construction, through the model
factory.

## 4. The acceptance flows

Each flow is run by hand or from a script. None is a unit test: the unit tests in this
layer are deterministic, network-free and model-free by rule, and anything that needs real
detector weights is an acceptance run instead.

### 4.1 Sample clip with the dashboard

    uv sync
    git lfs pull
    uv run python scripts/run_pipeline.py --config configs/pipeline/replay_demo.yaml

Then open the dashboard at the host and port set in that configuration, 8600 as shipped. The
source is `tests/fixtures/sample_clip.mp4`: 20 frames at 15 fps, 512x384, built as a slideshow
of COCO val2017 stills that are all CC BY 2.0
([tests/fixtures/ATTRIBUTION.md](../tests/fixtures/ATTRIBUTION.md)). Those numbers are read off
the committed file.

Two things to expect. The flow opens four committed graphs under `final_models/`, which are Git
LFS objects, so `git lfs pull` has to have run on a fresh clone or the run fails on a text
pointer instead of a graph; and on macOS it must run on the CPU execution provider, because the
CoreML provider crashes on both detector graphs in onnxruntime 1.28.0. And at 20 frames and
15 fps the clip is about 1.3 seconds of source time, orders of magnitude below any rule window,
so the expected alert count is zero. This flow proves that frames, detections, poses, tracks,
ticks, the store and the socket all connect, and it produces the per-stage latency table. It
proves nothing about rules, and the clip is not waiting-room footage, so no firing or
non-firing on it is evidence about behaviour.

### 4.2 Synthetic fixture, headless

    uv run python scripts/run_pipeline.py --config configs/pipeline/synthetic_demo.yaml --headless

This is the flow for exercising rules. The configuration points `detections_from` at a JSONL
fixture, so there is no video decode, no model weights, no download and no network. Alerts and
the latency JSONL land under `outputs/pipeline/<run>/`. Add `--max-frames N` to truncate a long
fixture. Because the fixture generators put known trajectories in, the expected firings are
known going out, which is what makes this flow checkable and the clip flow not.

### 4.3 Egress fixture, headless

    uv run python scripts/run_pipeline.py --config configs/pipeline/egress_demo.yaml --headless

The third head on a 220 s synthetic scene at 2 Hz: track 1 sits for 60 s, walks out through the
`main door` exit zone over 10 s and is last seen at t = 70; track 99 patrols the room and never
enters the zone. The configuration shrinks the two egress clocks (absence 600 s to 90 s,
presence 120 s to 45 s, marked as shrunk for the demonstration in that file and only there) so
the one expected alert, `egress_without_discharge_v0` with onset 70 s, lands at t = 160 inside
the fixture. Tagging track 1 discharged before it leaves, or discharging the pending departure
from the departures panel, makes the same run write nothing; `tests/test_egress.py` runs both
through the real runner.

### 4.4 Evidence packs and the false-alarm report

    uv run python scripts/eval/extract_evidence.py --run-dir outputs/pipeline/<run> --images <frames dir> --fps <Hz>
    uv run python scripts/eval/extract_evidence.py --run-dir outputs/pipeline/replay_demo --clip tests/fixtures/sample_clip.mp4

For every alert of a run: `<out>/<alert_id>/frames/*.jpg` from `onset - 10 s` to
`detected + 5 s`, both configurable, plus `manifest.json` with the alert record, the reviewer's
label if the dashboard has one, the window, the rate and the run's provenance. `index.json`
lists the packs. `false_alarms.md` is the per-rule report from the agree and false-alarm labels
(firings, labelled, agreed, false alarms, unlabelled, and per camera-hour when the camera time
is known), or, with `--negative-control`, every firing counted as a false positive. This is the
human half of the verification harness: a person looks at exactly what the rule saw. It is also
the sample format a trained verifier consumes. The run is idempotent, and `--force` rewrites.
The pure logic is in `vividly.eval.evidence`.

## 5. Every threshold is provisional

All thresholds live in `configs/pipeline/rules.yaml`. Not one of them has been fitted to data.
Each carries a comment saying so and pointing at [CALIBRATION.md](CALIBRATION.md).

Two independent reasons, either one sufficient:

1. A threshold fitted on top of one estimator fits that estimator's error, not anyone's
   behaviour, so nothing was fitted while stock estimators sat underneath the rules and a
   threshold is refitted whenever the upstream weights move.
2. The published shapes of the rules, a fast upright-to-down transition confirmed by a
   prolonged down-and-still period, a sustained direction-reversing motion band, and a
   departure through an exit zone followed by a long absence, describe the form of each rule.
   They were never presented as calibrated values.

So a threshold from this repository is not quoted as a product number, and a firing count from
any run here is not read as a false-alarm rate. Calibration begins when the preconditions in
[CALIBRATION.md](CALIBRATION.md) are met.

## 6. Staff tagging is a manual tag

Staff designation is a manual dashboard tag. Real staff identification is a later problem.

Mechanically that is `TrackBook.set_staff()`, driven by the dashboard's staff route. No rule
head reads the tag. The egress head reads it in one direction only, so that a staff-tagged track
that leaves is never a departure candidate; otherwise it remains a reviewer affordance, in that
the page distinguishes staff from patients and the tag rides identity carry-forward in
`merge_identity`, as does the discharge tag. There is no staff classifier, no uniform or badge
detector, and no face-based anything, now or ever: the no-facial-recognition rule is permanent.

## 7. Chart binding is a labelled mock

The chart-bind route assigns a deterministic fabricated identifier of the form `MRN-` plus the
zero-padded track identifier, and the interface labels it as a mock. There is no health-record
connection, no patient data and no lookup of any kind. The real mechanism is an open question,
whether a kiosk token, a staff tap at triage or a zone assignment, and full health-record
integration is out of pilot scope.

## 8. Demo material rule

Externally shown demonstrations use only cleanly licensed footage: the sample clip and the
synthetic fixtures. Evaluation-only footage never appears in demonstration material.

Operationally, `tests/fixtures/sample_clip.mp4` and the JSONL fixtures are the only sources that
may appear on a screen outside this repository. Frames from any registered evaluation corpus,
and any overlay, screenshot or clip rendered on them, are internal evaluation artefacts and stay
internal. A `LEGAL_HOLD` entry is not fetched at all. The rule is about where a frame is shown,
not about who is in the room when it is shown, so a screen share counts.

## 9. Re-association is proposed, never performed

The session-scoped re-identification stage changes one thing, and it is worth stating precisely,
because the change is easy to overstate in both directions.

**Inside the tracker's own short re-association window, nothing changed.** A person absent for
less than `reap_after_s` keeps the same `TrackState`, the same baseline, the same staff tag and
the same mock chart binding, exactly as before the stage existed. The tracker joins the
fragments, no gallery is consulted, and nobody is asked anything.

**Beyond that window the system never auto-binds. It proposes.** The stage enrols the appearance
of everyone who leaves, probes everyone who arrives, and when a newcomer looks like somebody who
has already gone it writes a pending row with a confidence and the gallery size that confidence
was computed against. The dashboard shows that row as a card, side by side, with a confirm
button and a reject button, and until a person presses one of them nothing has happened to
anybody's identity. There is no code path in this repository that joins two track identifiers
without that press.

**Both answers are labels.** Confirm and reject are written through the versioned alert store
(`reassoc_proposals`, introduced at file schema version 2 and current at version 3) and exported
by `EventStore.labels_as_rows()` as the score the model gave beside the answer a person gave,
which is the pair a threshold sweep needs. A rejection is not a dismissal: it says two people
who look alike are not the same person, which is the label a false-merge rate is measured
against. A decision is written exactly once; a second click on the same card is a conflict,
never an overwrite.

**The gallery is session scoped.** It holds appearance embeddings for one visit, is never
persisted across visits, is never a cross-site identifier, and is cleared at session end
(`ReidStage.close` calls `SessionGallery.clear`). The proposal rows outlive it on purpose: a row
holds two integer track identifiers, a score, a count and a person's answer, and no embedding, no
crop and no appearance vector of any kind. So after a session ends the dashboard still serves the
history from SQLite, and the panel header says so rather than leaving a reviewer to infer it.
That is the engineering half of CQ-16 in [COUNSEL_QUESTIONS.md](COUNSEL_QUESTIONS.md); whether it
is legally sufficient on its own, or whether a written retention policy is also required, is
counsel's question and it is open.

The headline measure is the false-merge rate, not rank-1, because a false merge joins two
patients' identities and in a chart-bound deployment that is a patient-safety event. Every
threshold in the mechanism lives in `configs/pipeline/reid.yaml` and every one of them is tagged
PROVISIONAL there: `tau`, `max_per_identity`, `top_k`, `max_proposals_per_track`,
`confidence_at_tau` and `enroll_every_s`. `min_gap_s` is not one of them: it ships as `null` and
takes the track layer's `reap_after_s`, rather than holding a second copy that could disagree
with the first. The stage is off by default: a run that wants it has to say so. Every number a
run produces is quoted with the embedder sha256 beside it, and no re-identification output
enters the shipped path before the freedom-to-operate opinion recorded as CQ-20.

What is still out, inside re-identification itself: no face detection, embedding or matching of
any kind, no gait, no pose-sequence identity signal, and no body-shape or anthropometric signal.
Those are gated on CQ-12, CQ-13 and CQ-18.

## 10. Still out of scope

Out of scope here and unchanged: RTSP ingest as a product path, appliance and deployment work,
automatic identity binding of any kind, cross-visit or cross-site identity, the biometric
identity signals above, rare-event detectors, learned temporal models beyond the event
verifiers, real chart binding, real staff identification, alert delivery to messaging systems,
and anything patient-facing.
