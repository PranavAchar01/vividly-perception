"""Per-track state: ring buffers, torso geometry, per-track baselines, and the TrackBook.

Replay-pipeline structure, recorded 2026-08-07 (POLICY.md, section 1). This layer
sits between the tracker and the rules: it turns a stream of per-frame TrackObservations
into the small set of derived quantities every rule is allowed to ask for. Rules are pure
functions over a TrackState, so anything a rule needs has to be computable from here.

Import rule: this module imports numpy, cv2, torch and rtmlib nowhere, not even lazily.
Linux CI syncs the default fork, where numpy and cv2 are darwin-marked and therefore
absent, and it collects this file. Every computation below is stdlib float math.
tests/test_tracks.py parses this module and asserts that it carries no module-level import
of numpy, cv2, torch, rtmlib, trackers, supervision or onnxruntime; that check reads the
module level only, so the absence of lazy imports inside function bodies is a rule this
file keeps rather than one the test enforces. The same rule is why the COCO-17 torso
indices are re-declared here instead of imported from vividly.models.base, which imports
numpy at module level.

Threshold status: every threshold default in TrackConfig is PROVISIONAL. No threshold in
this layer has been fitted to anything. The eyes and joints lanes do now have measured
fine-tuned weights (final_models/eyes/README.md, RT-DETRv4-S person v2, checkpoint
345c30a5, ONNX 40262049, AP 0.50:0.95 0.8356; final_models/joints/README.md, RTMPose-m
person v2, checkpoint a0a035fe, ONNX 6ba96285, keypoint AP 0.774115), but no pilot
shadow-mode data exists to fit these numbers against. Defaults are placeholders chosen to
be structurally sane, not clinically meaningful. See docs/CALIBRATION.md for the protocol
that replaces them.

Clock: `t` is source time in seconds (frame index / fps for a replay source), not wall
clock. Windows are anchored at the track's last_seen_t because the contract methods take
no `now`; a caller that wants a window anchored at the pipeline clock must reap first.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from itertools import pairwise
from typing import Any

# COCO-17 keypoint indices. Re-declared, not imported: see the import rule above.
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
TORSO_KEYPOINTS: tuple[int, int, int, int] = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
NUM_KEYPOINTS = 17

# A torso shorter than this many pixels carries no usable direction; the angle is None
# rather than an arbitrary value out of atan2(0, 0).
MIN_TORSO_PX = 1e-9

# 17 x (x, y, conf), absolute pixels in the frame the observation came from.
Keypoints = tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class TrackConfig:
    """Sizing and gating for the track layer.

    Every threshold default below is PROVISIONAL: it is a placeholder, not a calibrated
    value, and it stays provisional until pilot shadow data exists. See
    docs/CALIBRATION.md. `ring_size` is a sizing choice rather than a threshold and is
    marked as such. Pipeline configs override these in configs/pipeline/*.yaml.
    """

    # PROVISIONAL: 5 minutes of presence before a track has a baseline worth trusting.
    # Long enough that a person settling into a chair is not the whole window, short
    # enough that a 20 minute wait still gets 15 minutes of scored time.
    baseline_window_s: float = 300.0

    # PROVISIONAL: how long a track may be missing before it is dropped. The shipped
    # tracker's own lost_track_buffer is 1800 frames, and the library counts that buffer at
    # 30 fps whatever the stream rate, so it is 60 s of identity memory
    # (src/vividly/models/tracker.py, configs/pipeline/replay_demo.yaml): on the shipped
    # path a short occlusion behind a passing person is absorbed by the tracker before this
    # book ever sees a gap. The recorded extraction configs run the tracker at 30 frames,
    # one second (configs/events/extract_*_v2c.yaml), and there this window is the wider of
    # the two and is what keeps a brief occlusion from costing a baseline.
    reap_after_s: float = 30.0

    # PROVISIONAL: per-keypoint confidence floor for the four torso keypoints. RTMPose
    # confidences on this project's own top-view footage have never been measured, so this
    # number is a guess with the right shape, not a fitted operating point.
    kp_conf_min: float = 0.3

    # Ring sizing, not a threshold: ring buffer length, in samples, for positions and
    # posture angles. 54000 = 3600 s at 15 fps, generous headroom over the longest rule window
    # (collapse_search_s, 120 s), kept large so a config that widens a window later does
    # not silently truncate its evidence. Shrink it in a config when memory matters:
    # a ring shorter than a rule's window silently truncates that rule's evidence.
    ring_size: int = 54_000

    # PROVISIONAL: minimum samples inside the baseline window, for positions and for
    # posture angles independently, before a baseline is marked complete. 30 samples is
    # 2 s at 15 fps, i.e. a track that was present for a token fraction of the window
    # gets a baseline object with complete=False rather than confident-looking numbers.
    baseline_min_samples: int = 30

    def __post_init__(self) -> None:
        if not self.baseline_window_s > 0:
            raise ValueError(f"baseline_window_s must be > 0, got {self.baseline_window_s}")
        if not self.reap_after_s > 0:
            raise ValueError(f"reap_after_s must be > 0, got {self.reap_after_s}")
        if not 0.0 <= self.kp_conf_min <= 1.0:
            raise ValueError(f"kp_conf_min must be in [0, 1], got {self.kp_conf_min}")
        if self.ring_size < 2:
            raise ValueError(f"ring_size must be >= 2, got {self.ring_size}")
        if self.baseline_min_samples < 2:
            raise ValueError(f"baseline_min_samples must be >= 2, got {self.baseline_min_samples}")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TrackConfig:
        """Build from a config block (configs/pipeline/*.yaml `tracks:`).

        Unknown keys raise, because a typo in a threshold name that silently kept the
        default is exactly the failure this project cannot afford. `ring_size` may be
        given directly, or derived from `fps` plus `ring_window_s` (seconds of history
        to keep), which must be supplied together.
        """
        field_names = {f.name for f in fields(cls)}
        derived = {"fps", "ring_window_s"}
        unknown = sorted(set(data) - field_names - derived)
        if unknown:
            raise ValueError(
                f"unknown TrackConfig keys: {unknown}; known keys are "
                f"{sorted(field_names | derived)}"
            )

        kwargs: dict[str, Any] = {}
        for name in ("baseline_window_s", "reap_after_s", "kp_conf_min"):
            if name in data:
                kwargs[name] = float(data[name])
        for name in ("ring_size", "baseline_min_samples"):
            if name in data:
                kwargs[name] = int(data[name])

        if "ring_size" not in kwargs:
            fps = data.get("fps")
            ring_window_s = data.get("ring_window_s")
            if (fps is None) != (ring_window_s is None):
                raise ValueError("fps and ring_window_s must be given together to derive ring_size")
            if fps is not None and ring_window_s is not None:
                if float(fps) <= 0 or float(ring_window_s) <= 0:
                    raise ValueError("fps and ring_window_s must both be > 0")
                kwargs["ring_size"] = max(2, math.ceil(float(fps) * float(ring_window_s)))
        return cls(**kwargs)


@dataclass(frozen=True)
class TrackObservation:
    """One tracked person in one frame. Immutable, so a rule can never edit history.

    Raw-signal rule (the 2026-08-31 review, research/v2-training-stack.md section 7c;
    enforced by tests/test_alert_path_raw_signal.py): the values here are stored exactly as
    the detector and pose model emitted them, never smoothed, averaged, or interpolated on the way
    in. The rules and the event verifiers band-pass or window raw history themselves;
    a smoothed ring would put a spectral null on the seizure band for every consumer
    at once. Display smoothing belongs to the dashboard, downstream of the store.
    """

    track_id: int
    t: float
    cx: float
    cy: float
    w: float
    h: float
    keypoints: Keypoints | None = None

    def __post_init__(self) -> None:
        if self.keypoints is None:
            return
        if len(self.keypoints) != NUM_KEYPOINTS:
            raise ValueError(
                f"keypoints must hold {NUM_KEYPOINTS} COCO-17 entries, got {len(self.keypoints)}"
            )
        for i, kp in enumerate(self.keypoints):
            if len(kp) != 3:
                raise ValueError(f"keypoint {i} must be (x, y, conf), got {len(kp)} values")


@dataclass(frozen=True)
class BaselineStats:
    """What this track looked like during its own first baseline_window_s.

    Per-track and self-referential on purpose: the comparison a rule makes is a person
    against their own earlier behaviour, never against a population norm. `complete` is
    False when the track was present for too little of its own baseline window, or when
    too few frames yielded a usable torso angle; a rule that needs a trustworthy floor
    must check it.
    """

    movement_var: float
    posture_deg_mean: float
    posture_deg_std: float
    cy_mean: float
    n_samples: int
    complete: bool


@dataclass(frozen=True)
class TrackIdentity:
    """The identity fields of one track, snapshotted so they can outlive the track.

    A TrackState is deleted the moment its track is reaped, and with it the staff tag and
    the mock chart binding a human attached to that person. This record is the small,
    immutable copy of exactly those fields, taken at reap time by whoever is watching, so
    that a later human decision, confirming that this new track is that person, can carry
    them forward. It holds no appearance vector, no geometry and no history: it is what a
    reviewer typed, and nothing the system inferred.

    Frozen because it is a snapshot. Mutating one would mean editing the past.
    """

    track_id: int
    is_staff: bool = False
    mrn_binding: str | None = None
    #: The prior track this one was already confirmed to continue, if any. Present so a
    #: chain of confirmed re-associations reads back as a chain rather than collapsing.
    reassociated_from: int | None = None
    #: The dashboard's discharged tag, snapshotted like the staff tag: a reviewer's
    #: statement that this person was released. Read by the egress head
    #: (vividly.rules.egress): a discharged patient leaving is not an event.
    is_discharged: bool = False


def posture_angle_deg(keypoints: Keypoints | None, kp_conf_min: float) -> float | None:
    """Torso tilt in degrees from vertical, in [0, 90].

    Torso axis = midpoint(shoulders 5, 6) to midpoint(hips 11, 12). The angle is measured
    from the image vertical, so an upright torso is 0 and a torso lying along the image
    x axis is 90. Both components are taken in absolute value, which makes the result
    independent of which end is up and of left/right lean direction: this is a tilt
    magnitude, not a heading.

    Returns None when keypoints is None, when any of the four torso keypoints has
    confidence below kp_conf_min (strictly below; equal passes), or when the torso has no
    measurable length. Raises ValueError if the keypoint array is not COCO-17.
    """
    if keypoints is None:
        return None
    if len(keypoints) != NUM_KEYPOINTS:
        raise ValueError(
            f"keypoints must hold {NUM_KEYPOINTS} COCO-17 entries, got {len(keypoints)}"
        )
    for idx in TORSO_KEYPOINTS:
        if keypoints[idx][2] < kp_conf_min:
            return None

    shoulder_x = (keypoints[LEFT_SHOULDER][0] + keypoints[RIGHT_SHOULDER][0]) / 2.0
    shoulder_y = (keypoints[LEFT_SHOULDER][1] + keypoints[RIGHT_SHOULDER][1]) / 2.0
    hip_x = (keypoints[LEFT_HIP][0] + keypoints[RIGHT_HIP][0]) / 2.0
    hip_y = (keypoints[LEFT_HIP][1] + keypoints[RIGHT_HIP][1]) / 2.0

    dx = hip_x - shoulder_x
    dy = hip_y - shoulder_y
    if math.hypot(dx, dy) < MIN_TORSO_PX:
        return None
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def displacement_variance(points: Sequence[tuple[float, float, float]]) -> float | None:
    """Population variance of per-step displacement magnitudes over (t, cx, cy) samples.

    Steps are taken between consecutive samples in the sequence given, so a windowed call
    never reaches back across the window edge. Population variance (divide by N), not the
    sample estimator: these are all the steps there were, not a sample of a larger set.
    Returns None when there are fewer than two samples, i.e. no step exists at all.

    Pure python by requirement, not by preference: this module has to import on a Linux
    runner with no numpy installed.
    """
    if len(points) < 2:
        return None
    steps = [math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in pairwise(points)]
    mean = sum(steps) / len(steps)
    return sum((s - mean) ** 2 for s in steps) / len(steps)


class TrackState:
    """Rolling history of one tracked person.

    Ring buffers are collections.deque with maxlen=cfg.ring_size, so memory per track is
    bounded no matter how long someone waits. The baseline accumulators are deliberately
    not rings: they collect only samples inside the track's own baseline window and then
    freeze, so a baseline can never be corrupted by ring eviction or by later behaviour.

    first_seen_t and last_seen_t are nan until the first update. Every state handed out by
    a TrackBook has had at least one update, so a rule never sees the nan state.
    """

    def __init__(self, track_id: int, cfg: TrackConfig | None = None) -> None:
        self.cfg = TrackConfig() if cfg is None else cfg
        self.track_id = int(track_id)
        self.first_seen_t: float = math.nan
        self.last_seen_t: float = math.nan

        # Manual dashboard tag, v0. Real staff identification is a later problem and is
        # explicitly not an identity claim: no re-identification, no biometrics, no
        # facial recognition of any kind.
        self.is_staff: bool = False
        # Clearly labelled mock chart binding. There is no real chart integration here.
        self.mrn_binding: str | None = None
        # Manual dashboard discharged tag, v0: a reviewer saying this person was released.
        # No chart integration sets it. The egress head (vividly.rules.egress) treats a
        # discharged person leaving through an exit as nothing at all, which is the single
        # biggest false-positive guard that head has.
        self.is_discharged: bool = False
        # The earlier track a human confirmed this one continues, or None. Written by
        # TrackBook.merge_identity and by nothing else: no automatic path in this repo sets
        # it, because beyond the reap window below there is no automatic identity binding
        # at all. See vividly.reid.gallery and vividly.reid.stage.
        self.reassociated_from: int | None = None

        self.n_observations: int = 0
        self.last_observation: TrackObservation | None = None

        self._positions: deque[tuple[float, float, float]] = deque(maxlen=self.cfg.ring_size)
        # (t, w, h, kpc) per observation, appended in lockstep with _positions (same
        # maxlen, unconditional append) so the two rings stay index-parallel. kpc is the
        # mean confidence over all 17 keypoints, nan when the frame carried no keypoints:
        # the same quantity the event-verifier training extraction records, which is
        # the point, because the trained verifier rebuilds its input rows from these rings
        # plus _postures, and a quantity that exists only at training time would make
        # train and inference silently diverge.
        self._boxes: deque[tuple[float, float, float, float]] = deque(maxlen=self.cfg.ring_size)
        self._postures: deque[tuple[float, float]] = deque(maxlen=self.cfg.ring_size)
        self._baseline_positions: list[tuple[float, float, float]] = []
        self._baseline_angles: list[float] = []
        self._baseline: BaselineStats | None = None
        # Cleared for good the first time a sample arrives out of order. While it holds,
        # windowed reads scan backwards and stop early instead of walking the whole ring.
        self._ordered: bool = True

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"TrackState(track_id={self.track_id}, n={self.n_observations}, "
            f"first_seen_t={self.first_seen_t}, last_seen_t={self.last_seen_t}, "
            f"is_staff={self.is_staff})"
        )

    def identity(self) -> TrackIdentity:
        """Snapshot the human-supplied identity fields. See TrackIdentity.

        Taken by a reap watcher so the tag and the mock binding survive the TrackState.
        """
        return TrackIdentity(
            track_id=self.track_id,
            is_staff=self.is_staff,
            mrn_binding=self.mrn_binding,
            reassociated_from=self.reassociated_from,
            is_discharged=self.is_discharged,
        )

    # -- ingest -------------------------------------------------------------
    def update(self, obs: TrackObservation) -> None:
        """Append one observation. Non-decreasing `t` is expected; a late sample is still
        stored, and the state permanently falls back to full-ring window scans so windowed
        reads stay correct."""
        if obs.track_id != self.track_id:
            raise ValueError(
                f"observation for track {obs.track_id} handed to TrackState {self.track_id}"
            )
        if self.n_observations == 0:
            self.first_seen_t = obs.t
            self.last_seen_t = obs.t
        else:
            if self._positions and obs.t < self._positions[-1][0]:
                self._ordered = False
            if obs.t > self.last_seen_t:
                self.last_seen_t = obs.t

        self.n_observations += 1
        self.last_observation = obs
        self._positions.append((obs.t, obs.cx, obs.cy))
        kpc = sum(kp[2] for kp in obs.keypoints) / len(obs.keypoints) if obs.keypoints else math.nan
        self._boxes.append((obs.t, obs.w, obs.h, kpc))

        angle = posture_angle_deg(obs.keypoints, self.cfg.kp_conf_min)
        if angle is not None:
            self._postures.append((obs.t, angle))

        if self._baseline is None and obs.t <= self.first_seen_t + self.cfg.baseline_window_s:
            self._baseline_positions.append((obs.t, obs.cx, obs.cy))
            if angle is not None:
                self._baseline_angles.append(angle)

    # -- windowed reads -----------------------------------------------------
    def positions(self, window_s: float) -> list[tuple[float, float, float]]:
        """(t, cx, cy) samples with t >= last_seen_t - window_s, oldest first."""
        return self._window(self._positions, window_s)

    def boxes(self, window_s: float) -> list[tuple[float, float, float, float]]:
        """(t, w, h, mean-keypoint-conf-or-nan) samples in the window, oldest first.

        Index-parallel with positions() over the same window: both rings receive one
        unconditional append per observation and share maxlen, so the two lists always
        pair up row for row. The event-verifier backend zips them to rebuild the
        training extraction's row schema.
        """
        return self._window(self._boxes, window_s)

    def posture_angles(self, window_s: float) -> list[tuple[float, float]]:
        """(t, degrees-from-vertical) samples in the window, oldest first.

        Frames whose torso keypoints were below cfg.kp_conf_min contribute nothing, so
        this list is shorter than positions() whenever pose confidence dropped out.
        """
        return self._window(self._postures, window_s)

    def movement_variance(self, window_s: float) -> float | None:
        """Population variance of per-step centre displacement over the window.

        None when the window holds fewer than two position samples.
        """
        return displacement_variance(self.positions(window_s))

    def _window(self, ring: deque, window_s: float) -> list:
        if not window_s > 0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        if not ring:
            return []
        cutoff = self.last_seen_t - window_s
        if not self._ordered:
            return [item for item in ring if item[0] >= cutoff]
        out = []
        for item in reversed(ring):
            if item[0] < cutoff:
                break
            out.append(item)
        out.reverse()
        return out

    # -- baseline -----------------------------------------------------------
    def baseline(self) -> BaselineStats | None:
        """This track's own baseline, or None until baseline_window_s of it has elapsed.

        Computed once from the frozen accumulators and cached, so the value is identical
        whether it is first requested the instant the window closes or an hour later, and
        it never drifts as the rings turn over.
        """
        if self._baseline is not None:
            return self._baseline
        if self.n_observations == 0:
            return None
        if self.last_seen_t - self.first_seen_t < self.cfg.baseline_window_s:
            return None

        # Local snapshots: the dashboard thread can reach this method while the pipeline
        # thread is still appending, and len()/sum() over a live list can tear. Each
        # caller computes from its own consistent copy instead.
        points = list(self._baseline_positions)
        angles = list(self._baseline_angles)
        n_samples = len(points)
        variance = displacement_variance(points)
        cy_mean = sum(p[2] for p in points) / n_samples if n_samples else 0.0
        if angles:
            posture_mean = sum(angles) / len(angles)
            posture_std = math.sqrt(sum((a - posture_mean) ** 2 for a in angles) / len(angles))
        else:
            posture_mean = 0.0
            posture_std = 0.0

        stats = BaselineStats(
            movement_var=0.0 if variance is None else variance,
            posture_deg_mean=posture_mean,
            posture_deg_std=posture_std,
            cy_mean=cy_mean,
            n_samples=n_samples,
            complete=(
                n_samples >= self.cfg.baseline_min_samples
                and len(angles) >= self.cfg.baseline_min_samples
            ),
        )
        # Write-once: if another thread won the race, its value stands and this one is
        # discarded, so a compute that snapshotted the already-cleared accumulators can
        # never overwrite a real baseline with an empty one. Assign before clearing.
        if self._baseline is None:
            self._baseline = stats
            # The accumulators have done their job and the result is frozen; drop them so a
            # long-lived track does not carry two copies of its first five minutes forever.
            self._baseline_positions = []
            self._baseline_angles = []
        return self._baseline


class TrackBook:
    """Every live track, keyed by tracker id.

    The book owns creation and reaping; rules only read. A track absent for more than
    cfg.reap_after_s is dropped, and a person who leaves and returns inside that window
    keeps their id, their baseline, their staff tag and their mock chart binding, because
    the same TrackState object is still there. There is no re-identification anywhere in
    this layer: a track that returns after the reap window is a new person as far as this
    code is concerned, and that is what the code reports.

    The automatic window, and what may not touch it. Two windows produce that behaviour
    and neither one is negotiable from outside this class:

      1. The tracker's own re-association buffer. `vividly.models.tracker.BoTSORT` passes
         `lost_track_buffer` straight through to `trackers.BoTSORTTracker`, which re-issues
         the same track id across a gap that short. The library counts that buffer at
         30 fps whatever the stream rate is, so seconds are the value divided by 30: the
         shipped 1800 is 60 s, and the 30 used by the recorded extraction configs is one
         second. Nothing in this file sees that happen.
      2. This book's `cfg.reap_after_s` (PROVISIONAL 30 s), past which a TrackState is
         dropped whatever the tracker does.

    Which of the two closes first depends on the setting, and both orders occur here. At
    the extraction setting of 30 frames the tracker gives up first, so between the two
    edges the tracker hands out a new id while the departed track's TrackState is still
    here, still accumulating nothing, waiting to be reaped. At the shipped setting of 1800
    frames this book gives up first, so a person who returns between 30 s and 60 s is
    handed back their old track id with a new, empty TrackState behind it.

    Inside the narrower of the two windows association is automatic and always has been.
    `reap` is the edge of this book's window, and it is the only event this class
    publishes: the callback registered through `set_reap_watcher` fires once per dropped
    TrackState so a watcher can copy what it needs before the object is gone. That callback
    cannot stop a reap, cannot resurrect a track, and cannot merge two of them.
    Beyond both windows nothing in this repository binds an identity automatically; a
    proposal is raised and a human answers it (`vividly.reid.stage`,
    `vividly.schemas.reassoc`), and `merge_identity` below is where that answer lands.

    Thread story: the dashboard mutates this book from uvicorn's threadpool (staff tags,
    chart binds, confirmed re-associations) while the pipeline thread ingests and reaps, so
    every public method serializes on one RLock. One uncontended acquire per frame on the
    ingest path, which is noise next to a detector call. `on_reap` is deliberately invoked
    with the lock released: a watcher that enrols an embedding takes a lock of its own, and
    running it under this one would make the book's lock the outer half of a two-lock
    ordering that the dashboard could then take in the opposite direction.
    """

    def __init__(self, cfg: TrackConfig) -> None:
        self.cfg = cfg
        self._tracks: dict[int, TrackState] = {}
        self._now: float = math.nan
        self._lock = threading.RLock()
        # No watcher by default, and with none registered every observable behaviour of
        # this class is exactly what it was before re-identification existed. The constructor
        # signature is (self, cfg) and stays that way: tests/test_tracks.py pins it as a
        # cross-module contract, so the watcher arrives through set_reap_watcher instead.
        self._on_reap: Callable[[TrackState], None] | None = None

    def set_reap_watcher(self, on_reap: Callable[[TrackState], None] | None) -> None:
        """Register (or clear, with None) the one callback fired for each reaped track.

        Set it before the first ingest and leave it alone. A watcher swapped mid-run would
        see part of a session, which for the re-identification gallery means a partial
        enrollment and a confidence computed against a gallery no one can reconstruct
        afterwards.

        One watcher, not a list: the callback is a hand-off of a TrackState that is being
        deleted, and two watchers would be two owners of one snapshot with no defined
        order between them. Fan out inside the callback if that is ever wanted.
        """
        if on_reap is not None and not callable(on_reap):
            raise TypeError(f"on_reap must be callable or None, got {type(on_reap).__name__}")
        with self._lock:
            self._on_reap = on_reap

    @property
    def now(self) -> float:
        """Latest observation time ingested, or nan before the first observation."""
        return self._now

    def ingest(self, obs_list: list[TrackObservation]) -> None:
        """Route a frame's observations to their tracks, creating and reaping as needed.

        Creation happens before reaping, which matters for one case and is worth stating:
        a person who leaves and whose replacement track is born on the same frame ends up
        with the newcomer already in the book when the departure is announced. A watcher
        therefore always sees the enrollment side of a return before it is asked about it.
        """
        with self._lock:
            for obs in obs_list:
                state = self._tracks.get(obs.track_id)
                if state is None:
                    state = TrackState(obs.track_id, self.cfg)
                    self._tracks[obs.track_id] = state
                state.update(obs)
                if math.isnan(self._now) or obs.t > self._now:
                    self._now = obs.t
            dropped = [] if math.isnan(self._now) else self._drop_absent(self._now)
        self._announce_reaped(dropped)

    def reap(self, now: float) -> list[int]:
        """Drop tracks absent for more than cfg.reap_after_s. Returns the dropped ids.

        Called by ingest with the newest observation time. Call it directly to advance the
        book's clock across frames that produced no detections at all, which ingest cannot
        do because an empty batch carries no time.
        """
        with self._lock:
            dropped = self._drop_absent(now)
        self._announce_reaped(dropped)
        return [state.track_id for state in dropped]

    def _drop_absent(self, now: float) -> list[TrackState]:
        """Remove the absent tracks and advance the clock. The caller holds the lock.

        Returns the dropped TrackState objects rather than their ids, because a watcher
        needs the object: once this returns, the book is the only thing that was holding
        it and the tag a nurse attached to that person is about to be unreachable.
        """
        if math.isnan(now):
            return []
        expired = [
            track_id
            for track_id, state in self._tracks.items()
            if now - state.last_seen_t > self.cfg.reap_after_s
        ]
        dropped = [self._tracks.pop(track_id) for track_id in expired]
        if math.isnan(self._now) or now > self._now:
            self._now = now
        return dropped

    def _announce_reaped(self, dropped: list[TrackState]) -> None:
        """Tell the watcher, once per dropped track, with the lock released.

        A watcher that raises takes the ingest call down with it, deliberately: silently
        swallowing an enrollment failure would leave the gallery holding a partial view of
        the session while every downstream count still claimed it was complete.
        """
        if self._on_reap is None:
            return
        for state in dropped:
            self._on_reap(state)

    def tracks(self) -> dict[int, TrackState]:
        """Live tracks in first-seen order. The dict is a copy; the states are live."""
        with self._lock:
            return dict(self._tracks)

    def set_staff(self, track_id: int, staff: bool) -> None:
        """Manual v0 staff tag from the dashboard. Raises KeyError for an unknown id."""
        with self._lock:
            self._require(track_id).is_staff = bool(staff)

    def set_discharged(self, track_id: int, discharged: bool) -> None:
        """Manual v0 discharged tag from the dashboard. Raises KeyError for an unknown id.

        Same shape as `set_staff` and the same kind of thing: a reviewer's statement, not
        an inference and not an integration. The egress head reads it at departure time;
        a person tagged discharged who then walks out through an exit zone is, correctly,
        not an event. Tagging after the person has already left is the egress
        monitor's `mark_discharged`, reached from the dashboard's departures panel.
        """
        with self._lock:
            self._require(track_id).is_discharged = bool(discharged)

    def bind_mrn(self, track_id: int, mrn: str) -> None:
        """Bind a mock chart identifier. Raises KeyError for an unknown id.

        This is a demo affordance, not an integration: nothing here reads or writes a
        real record system, and the string is whatever the caller passes.
        """
        if not isinstance(mrn, str) or not mrn.strip():
            raise ValueError(f"mrn must be a non-empty string, got {mrn!r}")
        with self._lock:
            self._require(track_id).mrn_binding = mrn

    def merge_identity(self, track_id: int, prior: TrackIdentity) -> None:
        """Carry a confirmed earlier identity forward onto a live track.

        Same shape as `bind_mrn` on purpose: validate the argument, take the lock, require
        the track, write a field. The difference is who is allowed to call it. `bind_mrn`
        is a reviewer pressing a button on one track; this is a reviewer confirming that
        one new track is the person behind an earlier one, and it is the only path in this
        repository that joins two track ids. No rule, no tracker and no gallery reaches it:
        beyond `cfg.reap_after_s` the system proposes and a human decides
        (`vividly.reid.stage.ReidStage.decide` is the caller, and it writes the label to
        the alert store before calling this, so the answer survives even if the merge
        cannot be applied).

        The track id does not change. Renaming a live track to a dead one's id would break
        every alert already written against the current id, and would make the merge
        invisible to anyone reading the store. The link is recorded instead, forward-only,
        on `TrackState.reassociated_from`.

        Carry-forward rules, and both of them protect a human's own input:

        * `mrn_binding` moves across only when the live track has none. A binding made
          directly on this track is newer evidence than the reaped track's, and silently
          overwriting it would discard an action a reviewer took deliberately.
        * `is_staff` is OR-ed, never assigned. The tag states that this person is staff: a
          confirmed match to a staff track sets it, and a match to an untagged track does
          not clear a tag that a reviewer applied to the live track.
        * `is_discharged` is OR-ed the same way: a person confirmed to be a discharged
          patient who came back is still discharged, so their next exit is still not an
          egress event, and a live-track tag is never cleared by a match.

        Raises KeyError if `track_id` is not live (it left before the answer arrived),
        ValueError for a self-merge, TypeError for anything that is not a TrackIdentity.
        """
        if not isinstance(prior, TrackIdentity):
            raise TypeError(
                f"prior must be a TrackIdentity snapshot, got {type(prior).__name__}. "
                "Pass the record taken when that track was reaped; a live TrackState is "
                "not accepted because the two tracks are never both live."
            )
        if prior.track_id == track_id:
            raise ValueError(
                f"a track cannot be re-associated with itself (track {track_id}); "
                "a self-merge is a bug upstream, not an identity decision"
            )
        with self._lock:
            state = self._require(track_id)
            state.reassociated_from = prior.track_id
            if state.mrn_binding is None and prior.mrn_binding is not None:
                state.mrn_binding = prior.mrn_binding
            if prior.is_staff:
                state.is_staff = True
            if prior.is_discharged:
                state.is_discharged = True

    def _require(self, track_id: int) -> TrackState:
        state = self._tracks.get(track_id)
        if state is None:
            raise KeyError(f"unknown track_id {track_id!r}; it was never seen or has been reaped")
        return state
