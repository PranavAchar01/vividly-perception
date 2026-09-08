"""Windowing and featurization for event-verifier training.

Turns track-observation sequences (real ones from scripts/events/extract_track_sequences.py,
synthetic ones from vividly.verifier.synth, which share one row schema) into fixed-shape
labelled windows a small temporal classifier trains on. The window is deliberately shaped
like the evidence vividly.rules.verify hands a verifier at inference: the trailing window_s
seconds of one track's positions and posture angles, nothing else, so no ground-truth boxes,
no pixels and no identity.

Features. Each window resamples onto n_steps uniform time steps; every motion channel is
box-relative (the burn-in showed absolute-pixel motion gates are scale artefacts):

    vy, vx        centre velocity in box-heights per second
    h_rel         box height / window-median height (a collapse signature)
    aspect        w / h (lying vs upright)
    angle_norm    posture angle / 90 (0 upright, 1 horizontal), 0 where invalid
    angle_valid   1 where the pose supported an angle estimate
    kpc           mean keypoint confidence, 0 where absent
    present       1 where an observation exists near the step (tracker gaps are data)

Labels. label_window() places a window against the video's event intervals (one or
several: multicam_fall scenarios 4 and 23 each stage two falls) under two policies:
  * "contain" (falls): positive if and only if the whole event sits inside the window. The
    fall-then-still rule fires about 30 s after the drop, so the verifier's ring holds the
    fall itself; a window that clips the event is ambiguous (None) and trains nothing.
  * "overlap" (convulsions): positive if the window holds at least overlap_min_s of the
    event, or, for an event shorter than that, the whole of it. The threshold is taken
    against the event's own duration so a brief event is labelable rather than stuck at
    None forever. Zero overlap is negative; in between is None.

Which track fell. The fall datasets label the video, not the track, and the tracker can
fragment across the fall itself. windows_for_video() assigns the event to the track whose
observed span overlaps the event interval most; other tracks present during the event are
excluded (None) rather than called negatives. Single-actor footage (Le2i) makes this
nearly always exact; the caveat travels here so a multi-person source is re-examined
before it trains anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

CHANNELS: tuple[str, ...] = (
    "vy",
    "vx",
    "h_rel",
    "aspect",
    "angle_norm",
    "angle_valid",
    "kpc",
    "present",
)

# Scene-context channels (WindowSpec.scene_channels, off by default so every v1
# model's 8-channel geometry is untouched). Motivation, measured on the WU-SAHZU corpus
# (research/event-verifier-v1-record.md): during real convulsions the tracker
# shatters, 17 to 33 track fragments against 1 to 2 in calm footage, so the density and
# birth rate of tracks around a window are a convulsion signature the per-track
# channels cannot see. Both are computed from every track in the video, not just
# the window's own track.
SCENE_CHANNELS: tuple[str, ...] = (
    "scene_n_tracks",  # tracks whose observed span covers the step, /SCENE_TRACKS_NORM
    "scene_births",  # tracks born within SCENE_BIRTH_WINDOW_S before the step, /SCENE_BIRTHS_NORM
)
SCENE_TRACKS_NORM = 8.0
SCENE_BIRTHS_NORM = 4.0
SCENE_BIRTH_WINDOW_S = 3.0
SCENE_CLIP = 2.0

# Rhythmicity channels (WindowSpec.rhythm_channels, off by default so every v1-v6
# model's geometry is untouched). Motivation: a convulsion is rhythmic motion at
# roughly 1 to 5 Hz, while the normal in-bed movement it must be told apart from,
# rolling over, scratching, adjusting bedding, carries the same amplitude with no
# periodicity. Nothing in CHANNELS can express periodicity, and the step grid cannot
# even represent it: n_steps 64 over the seizure lane's 12 s window samples at
# 5.3 Hz, so its 2.7 Hz Nyquist limit folds most of the clonic band away before the
# model sees anything. These features are therefore computed on the raw samples at
# native capture rate, before the step resampling, and are constant across the
# window because rhythm is a property of the window, not of one step.
RHYTHM_CHANNELS: tuple[str, ...] = (
    "rhythm_strength",  # power at the dominant in-band frequency / total AC power
    "rhythm_freq",  # that frequency, /RHYTHM_FMAX_HZ
    "rhythm_band_ratio",  # total in-band power / total AC power
    "rhythm_valid",  # 1.0 only when the native rate actually resolves the band
)
RHYTHM_FMIN_HZ = 1.0
RHYTHM_FMAX_HZ = 5.0
RHYTHM_MIN_SAMPLES = 16

# Limb-motion channels (WindowSpec.limb_channels, off by default so every v1-v6
# model's geometry is untouched). Motivation, measured 2026-08-28
# (research/event-verifier-v1-record.md, on the seizure lane being starved of
# information): RHYTHM_CHANNELS measure periodicity of the box centre, and a patient
# convulsing in bed barely moves their box, because the limbs thrash inside a nearly
# stationary bounding box. Taken alone on the real test split, 4,880 windows with 285
# positive, the centre-rhythm channels score ROC 0.412 to 0.494 against chance 0.500. These
# channels run the same spectral measurement on wrist and ankle motion relative to the
# torso, which is the thing a convulsion actually is. They require the 10th row field `kp`
# (extraction config `emit_keypoints: true`); a tracks tree written without it can only
# produce limb_valid = 0.
LIMB_CHANNELS: tuple[str, ...] = (
    "limb_rhythm_strength",  # power at the dominant in-band frequency / total AC power
    "limb_rhythm_freq",  # that frequency, /RHYTHM_FMAX_HZ
    "limb_band_ratio",  # total in-band power / total AC power
    "limb_excursion",  # RMS limb displacement about its own mean, /LIMB_EXCURSION_NORM
    "limb_bilateral",  # left-right agreement of limb speed, mapped to [0, 1]
    "limb_valid",  # 1.0 only when joints and capture rate actually support the measure
)
# COCO-17 order. Torso is the anchor the limbs are measured against; a wrist moving
# with the whole body is the patient being repositioned, not a convulsion.
LIMB_TORSO_JOINTS: tuple[int, ...] = (5, 6, 11, 12)  # shoulders, hips
LIMB_PAIRS: tuple[tuple[int, int], ...] = ((9, 10), (15, 16))  # (wrists), (ankles)
# Measured on the 976,659-row wu_sahzu_tracks_v2kp extraction (2026-08-29): pooled over
# every track the wrist/ankle confidence median is 0.305 and only 25.1% of joints clear
# 0.5, so a 0.2 gate admits mostly noise and the spectral channels below then measure the
# spectrum of that noise, which is the measured cause of the 8/8-negative limb-channel result in
# research/seizure-limb-channel-gate.md. Restricting to tracks whose mean kpc >= 0.5 lifts
# the same median to 0.511 (wrists alone 0.579) while keeping 23.7% of in-event rows and
# every one of the 8 seizure recordings, so a stricter gate is affordable. The default
# stays 0.2 for backwards compatibility; WindowSpec.limb_conf_min is the knob.
LIMB_CONF_MIN = 0.2  # below this the estimator is guessing; the joint is not evidence
LIMB_MIN_JOINT_FRAC = 0.5  # a limb needs this fraction of the window's samples resolved
LIMB_EXCURSION_NORM = 0.5  # box-heights; about a full arm's reach from the torso centre

# Keypoint-geometry channels (WindowSpec.kp_channels, off by default so every
# existing model's geometry is untouched: fall push v3 arm F2, fall_kpgeo_v1).
# Unlike the rhythm and limb channels these are per-step: body geometry varies along the
# window, and that variation (torso tipping over, spine extent collapsing, head dropping)
# is the fall signature itself. They require the 10th row field `kp` (extraction config
# `emit_keypoints: true`); a tree without it yields every kp_* channel = 0.
# Every value channel travels with a validity flag: the flag is 0, and the value is
# 0 then and never NaN, at any step where the defining joints are missing, below the
# confidence gate, or the tracker had no observation (`present` = 0). A model must
# never read a kp value while its flag is 0 (same contract as angle_valid).
KP_CHANNELS: tuple[str, ...] = (
    "kp_torso_cos",  # torso axis (shoulder-mid -> hip-mid) unit vertical component
    "kp_torso_sin",  # ... unit horizontal component (image y grows down: upright = cos 1)
    "kp_torso_valid",
    "kp_extent_headhip",  # |head_y - hip_mid_y| / h_ref (spine compression, head-to-hip)
    "kp_extent_headhip_valid",
    "kp_extent_shankle",  # |shoulder_mid_y - ankle_mid_y| / h_ref (shoulder-to-ankle)
    "kp_extent_shankle_valid",
    "kp_head_vy",  # head vertical velocity, torso-lengths per second (scale-free)
    "kp_head_vy_valid",
    "kp_wrist_dx",  # mean wrist offset from hip-mid, x / h_ref
    "kp_wrist_dy",  # ... y / h_ref
    "kp_wrist_valid",
    "kp_ankle_dx",  # mean ankle offset from hip-mid, x / h_ref
    "kp_ankle_dy",  # ... y / h_ref
    "kp_ankle_valid",
)
# The per-joint gate defaults to TrackConfig.kp_conf_min (0.3), the same gate the
# angle channel's posture_angle_deg already applies (>= passes), not LIMB_CONF_MIN:
# these channels read body geometry the way the angle does, not limb spectra.
KP_CONF_MIN = 0.3
KP_HEAD_JOINTS: tuple[int, ...] = (0, 1, 2, 3, 4)  # nose, eyes, ears
KP_SHOULDER_JOINTS: tuple[int, ...] = (5, 6)
KP_HIP_JOINTS: tuple[int, ...] = (11, 12)
KP_WRIST_JOINTS: tuple[int, ...] = (9, 10)
KP_ANKLE_JOINTS: tuple[int, ...] = (15, 16)
KP_VEL_CLIP = 10.0  # torso-lengths/s, same order as the vy clip
KP_EXTENT_CLIP = 3.0  # extents in box-heights
KP_OFFSET_CLIP = 3.0  # hip-centred offsets in box-heights

LABEL_POLICIES: tuple[str, ...] = ("contain", "overlap")


class FeatureError(ValueError):
    """Inputs that cannot become a training window."""


def channel_names(spec: WindowSpec) -> tuple[str, ...]:
    """Every channel `featurize_window` emits for this spec, in column order.

    The single source of truth for what a model's input columns mean. Callers that
    record or check channel layout must use this rather than re-assembling the
    tuples: hand-assembled copies drift, and a sidecar that misnames a column is
    worse than one that omits it.
    """
    names = CHANNELS
    if spec.scene_channels:
        names += SCENE_CHANNELS
    if spec.rhythm_channels:
        names += RHYTHM_CHANNELS
    if spec.limb_channels:
        names += LIMB_CHANNELS
    if spec.kp_channels:
        names += KP_CHANNELS
    return names


@dataclass(frozen=True)
class WindowSpec:
    """Window geometry. window_s must cover the event the policy expects to contain."""

    window_s: float = 16.0
    stride_s: float = 2.0
    n_steps: int = 64
    policy: str = "contain"
    overlap_min_s: float = 4.0  # used by "overlap" only
    min_present: float = 0.35  # windows with less observed coverage are dropped
    contain_margin_s: float = 0.5  # "contain": event must sit this far inside the edge
    scene_channels: bool = False  # append SCENE_CHANNELS (track density + births)
    rhythm_channels: bool = False  # append RHYTHM_CHANNELS (native-rate periodicity)
    limb_channels: bool = False  # append LIMB_CHANNELS (needs rows carrying `kp`)
    limb_conf_min: float = LIMB_CONF_MIN  # per-joint gate; see LIMB_CONF_MIN
    kp_channels: bool = False  # append KP_CHANNELS (per-step kp geometry; needs `kp` rows)
    kp_conf_min: float = KP_CONF_MIN  # per-joint gate for KP_CHANNELS; see KP_CONF_MIN

    def __post_init__(self) -> None:
        if self.policy not in LABEL_POLICIES:
            raise FeatureError(f"policy must be one of {LABEL_POLICIES}, got {self.policy!r}")
        if self.window_s <= 0 or self.stride_s <= 0 or self.n_steps < 8:
            raise FeatureError("window_s/stride_s must be positive and n_steps >= 8")
        if not 0.0 <= self.limb_conf_min <= 1.0:
            raise FeatureError(f"limb_conf_min must be in [0, 1], got {self.limb_conf_min}")
        if not 0.0 <= self.kp_conf_min <= 1.0:
            raise FeatureError(f"kp_conf_min must be in [0, 1], got {self.kp_conf_min}")


@dataclass(frozen=True)
class TrackArrays:
    """One track's time series, time-sorted. angle/kpc are nan where unavailable."""

    track_id: int
    t: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    w: np.ndarray
    h: np.ndarray
    angle: np.ndarray
    kpc: np.ndarray
    # (n_samples, 17, 3) image-space [x, y, conf] in COCO-17 order, or None for the
    # frozen 9-field rows every extraction before v2kp wrote. nan marks an observation
    # the estimator returned nothing for.
    kpts: np.ndarray | None = None

    @property
    def span(self) -> tuple[float, float]:
        return float(self.t[0]), float(self.t[-1])


@dataclass(frozen=True)
class WindowSource:
    """The raw rows `featurize_window` consumed for one window, plus its exact t_end.

    Carried only when a caller asks for it (`with_sources`), and it exists for
    exactly one purpose: an augmentation that has to change the observations rather
    than the finished feature columns (the camera-geometry transforms of fall push
    v3 arm F3) must re-featurize, and re-featurizing needs the same rows the window
    was built from and the same t_end, to the bit. `t_end` here is the unrounded
    value the featurizer was called with; `LabeledWindow.t_end` is rounded for the
    record, and re-featurizing from the rounded one moves a window's sample selection
    at the edges (measured: 9 of 400 le2i windows change).

    `track` holds only the in-window samples, which is what `featurize_window`
    selects anyway, so featurizing this track at this t_end reproduces `x` exactly.
    """

    track: TrackArrays
    t_end: float


@dataclass(frozen=True)
class LabeledWindow:
    """One training example. x is (n_steps, len(CHANNELS)) float32."""

    key: str
    track_id: int
    t_end: float
    label: int
    x: np.ndarray
    # The meta's "location" is the subject (a patient for wu_sahzu, a subject for the
    # fall sets), not the recording. Carried so a train/val split can be made
    # subject-disjoint: two recordings of the same patient share a bed, a room, a
    # camera and a body, so splitting by recording lets the model be validated on
    # someone it has already seen. None when the extraction records no location.
    location: str | None = None
    # The raw rows behind `x`, attached only under `with_sources` (see WindowSource).
    # None on every default path, so nothing about the existing window build changes.
    source: WindowSource | None = None


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def tracks_from_rows(rows: list[dict[str, Any]]) -> list[TrackArrays]:
    """Group rows by track id; each track time-sorted. Rows with h <= 0 are dropped."""
    by_id: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        if float(r["h"]) > 0:
            by_id.setdefault(int(r["track"]), []).append(r)
    out = []
    for tid in sorted(by_id):
        rs = sorted(by_id[tid], key=lambda r: float(r["t"]))
        out.append(
            TrackArrays(
                track_id=tid,
                t=np.array([r["t"] for r in rs], dtype=np.float64),
                cx=np.array([r["cx"] for r in rs], dtype=np.float64),
                cy=np.array([r["cy"] for r in rs], dtype=np.float64),
                w=np.array([r["w"] for r in rs], dtype=np.float64),
                h=np.array([r["h"] for r in rs], dtype=np.float64),
                angle=np.array(
                    [np.nan if r["angle"] is None else r["angle"] for r in rs],
                    dtype=np.float64,
                ),
                kpc=np.array(
                    [np.nan if r["kpc"] is None else r["kpc"] for r in rs], dtype=np.float64
                ),
                kpts=_kpts_from_rows(rs),
            )
        )
    return out


def _kpts_from_rows(rs: list[dict[str, Any]]) -> np.ndarray | None:
    """(n, 17, 3) from the optional `kp` field, or None when no row carries one.

    None and all-nan are different states on purpose: None means the extraction never
    emitted keypoints (a pre-v2kp tree), all-nan means it did and the estimator found
    nothing. Both end at limb_valid = 0, but only the second is a fact about the
    footage rather than about the file format.
    """
    if not any(r.get("kp") for r in rs):
        return None
    out = np.full((len(rs), 17, 3), np.nan, dtype=np.float64)
    for i, r in enumerate(rs):
        kp = r.get("kp")
        if not kp:
            continue
        a = np.asarray(kp, dtype=np.float64)
        if a.shape == (17, 3):
            out[i] = a
    return out


Events = list[tuple[float, float]]


def normalize_events(event: Events | tuple[float, float] | None) -> Events:
    """None, one (start, end) pair, or a list of pairs -> a list of pairs."""
    if event is None:
        return []
    if isinstance(event, tuple) and len(event) == 2 and not isinstance(event[0], tuple):
        return [(float(event[0]), float(event[1]))]
    return [(float(e0), float(e1)) for e0, e1 in event]


def label_window(
    t0: float,
    t_end: float,
    event: Events | tuple[float, float] | None,
    spec: WindowSpec,
) -> int | None:
    """1 positive / 0 negative / None ambiguous-train-nothing. See module docstring.

    With several events: positive if ANY event qualifies under the policy, negative
    only when the window overlaps NO event, ambiguous otherwise.
    """
    events = normalize_events(event)
    if not events:
        return 0
    any_overlap = False
    for e0, e1 in events:
        overlap = max(0.0, min(t_end, e1) - max(t0, e0))
        any_overlap = any_overlap or overlap > 0.0
        if spec.policy == "contain":
            m = spec.contain_margin_s
            if e0 >= t0 + m and e1 <= t_end - m:
                return 1
        elif overlap >= min(spec.overlap_min_s, max(0.1, e1 - e0)):
            return 1
    return None if any_overlap else 0


def rhythm_features(
    t: np.ndarray, cx: np.ndarray, cy: np.ndarray, h_ref: float
) -> tuple[float, float, float, float]:
    """(strength, freq_norm, band_ratio, valid) for one window's raw samples.

    Periodicity of the centre's velocity, measured at the track's own capture rate.
    The two velocity components are spectrally summed rather than reduced to a speed
    magnitude first: |v| of an oscillation peaks twice per cycle, which would report
    every rhythm at double its true frequency.

    `valid` is 0.0, with the other three forced to 0.0, whenever the samples cannot
    support the measurement (too few, degenerate timing, or a capture rate whose
    Nyquist limit sits below the band's own floor). A caller must never read a
    rhythm value while `rhythm_valid` is 0: absence of evidence here is a property
    of the footage, not of the person, and the flag is how the model is told which
    it is looking at (same contract as `angle_valid`).
    """
    dead = (0.0, 0.0, 0.0, 0.0)
    if t.size < RHYTHM_MIN_SAMPLES or not h_ref > 0:
        return dead
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        return dead
    fs = 1.0 / float(np.median(dt))
    # Below this the band is not merely noisy, it is unrepresentable.
    if not np.isfinite(fs) or fs < 2.0 * RHYTHM_FMIN_HZ:
        return dead
    span = float(t[-1] - t[0])
    if not span > 0:
        return dead

    n = int(max(RHYTHM_MIN_SAMPLES, round(span * fs)))
    grid = np.linspace(float(t[0]), float(t[-1]), n)
    # Heights, so the measure is scale-free the way vy/vx are.
    vx = np.gradient(np.interp(grid, t, cx) / h_ref) * fs
    vy = np.gradient(np.interp(grid, t, cy) / h_ref) * fs
    if not (np.all(np.isfinite(vx)) and np.all(np.isfinite(vy))):
        return dead
    vx = vx - float(np.mean(vx))
    vy = vy - float(np.mean(vy))

    taper = np.hanning(n)
    power = np.abs(np.fft.rfft(vx * taper)) ** 2 + np.abs(np.fft.rfft(vy * taper)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power[0] = 0.0  # DC is the drift just removed; it must never dominate the ratio
    total = float(power.sum())
    if not total > 0:
        return dead

    band = (freqs >= RHYTHM_FMIN_HZ) & (freqs <= min(RHYTHM_FMAX_HZ, fs / 2.0))
    if not band.any():
        return dead
    peak = int(np.argmax(np.where(band, power, -np.inf)))
    return (
        float(power[peak]) / total,
        min(float(freqs[peak]) / RHYTHM_FMAX_HZ, 1.0),
        float(power[band].sum()) / total,
        1.0,
    )


def limb_features(
    t: np.ndarray, kpts: np.ndarray | None, h_ref: float, conf_min: float = LIMB_CONF_MIN
) -> tuple[float, float, float, float, float, float]:
    """(strength, freq_norm, band_ratio, excursion, bilateral, valid) for one window.

    The same spectral measurement as `rhythm_features`, but on wrist and ankle motion
    measured against the torso centre instead of on the box centre. Subtracting the
    torso is the whole point: a wrist that travels with the body is a patient being
    repositioned, and only motion *relative to* the torso is limb motion.

    Power is summed across limbs and across x/y rather than reducing to a speed first,
    for the reason `rhythm_features` documents - |v| of an oscillation peaks twice per
    cycle and would report every rhythm at double its true frequency.

    `bilateral` is the left-right agreement of limb speed, mapped from correlation
    [-1, 1] onto [0, 1]. Generalised convulsions drive both sides together; reaching,
    scratching and being attended to are one-sided. It is the channel that separates
    a seizure from a nurse moving a limb.

    `valid` is 0.0 - with the other five forced to 0.0 - whenever the measurement
    cannot be supported: no keypoints in the tree, too few samples, a capture rate
    whose Nyquist limit sits below the band floor, or no limb resolved often enough.
    A caller must never read these while `limb_valid` is 0, the same contract
    `angle_valid` and `rhythm_valid` carry.
    """
    dead = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if kpts is None or t.size < RHYTHM_MIN_SAMPLES or not h_ref > 0:
        return dead
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        return dead
    fs = 1.0 / float(np.median(dt))
    if not np.isfinite(fs) or fs < 2.0 * RHYTHM_FMIN_HZ:
        return dead
    span = float(t[-1] - t[0])
    if not span > 0:
        return dead

    xy = kpts[:, :, :2]
    conf = kpts[:, :, 2]
    ok = np.isfinite(conf) & np.isfinite(xy).all(axis=2) & (conf >= conf_min)

    # Torso anchor: mean of whichever shoulders/hips are resolved, needing at least two
    # so a single stray joint cannot define the body's centre.
    t_ok = ok[:, LIMB_TORSO_JOINTS]
    n_torso = t_ok.sum(axis=1)
    t_xy = np.where(t_ok[..., None], xy[:, LIMB_TORSO_JOINTS, :], 0.0).sum(axis=1)
    torso = np.where((n_torso >= 2)[:, None], t_xy / np.maximum(n_torso, 1)[:, None], np.nan)

    n = int(max(RHYTHM_MIN_SAMPLES, round(span * fs)))
    grid = np.linspace(float(t[0]), float(t[-1]), n)
    taper = np.hanning(n)
    power = np.zeros(n // 2 + 1, dtype=np.float64)
    excursions: list[float] = []
    speeds: dict[int, np.ndarray] = {}

    for joint in (j for pair in LIMB_PAIRS for j in pair):
        good = ok[:, joint] & np.isfinite(torso).all(axis=1)
        if int(good.sum()) < max(RHYTHM_MIN_SAMPLES, LIMB_MIN_JOINT_FRAC * t.size):
            continue
        off = (xy[:, joint, :] - torso) / h_ref  # box-heights from the torso centre
        gx = np.interp(grid, t[good], off[good, 0])
        gy = np.interp(grid, t[good], off[good, 1])
        excursions.append(float(np.hypot(gx - gx.mean(), gy - gy.mean()).mean()))
        vx = np.gradient(gx) * fs
        vy = np.gradient(gy) * fs
        if not (np.all(np.isfinite(vx)) and np.all(np.isfinite(vy))):
            continue
        vx = vx - float(np.mean(vx))
        vy = vy - float(np.mean(vy))
        power += np.abs(np.fft.rfft(vx * taper)) ** 2 + np.abs(np.fft.rfft(vy * taper)) ** 2
        speeds[joint] = np.hypot(vx, vy)

    if not excursions or not speeds:
        return dead
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power[0] = 0.0  # DC is the drift already removed; never let it dominate the ratio
    total = float(power.sum())
    if not total > 0:
        return dead
    band = (freqs >= RHYTHM_FMIN_HZ) & (freqs <= min(RHYTHM_FMAX_HZ, fs / 2.0))
    if not band.any():
        return dead
    peak = int(np.argmax(np.where(band, power, -np.inf)))

    # Left-right agreement, over whichever pairs resolved on both sides.
    corrs = [
        float(np.corrcoef(speeds[left], speeds[right])[0, 1])
        for left, right in LIMB_PAIRS
        if left in speeds and right in speeds
    ]
    corrs = [c for c in corrs if np.isfinite(c)]
    bilateral = (float(np.mean(corrs)) + 1.0) / 2.0 if corrs else 0.5

    return (
        float(power[peak]) / total,
        min(float(freqs[peak]) / RHYTHM_FMAX_HZ, 1.0),
        float(power[band].sum()) / total,
        min(float(np.mean(excursions)) / LIMB_EXCURSION_NORM, 2.0),
        bilateral,
        1.0,
    )


def _kp_pair_mid(
    xy: np.ndarray, ok: np.ndarray, joints: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """((n, 2) mean of the resolved joints, (n,) bool >=1 resolved). nan where none."""
    o = ok[:, joints]
    n_ok = o.sum(axis=1)
    s = np.where(o[..., None], xy[:, joints, :], 0.0).sum(axis=1)
    good = n_ok >= 1
    mid = np.where(good[:, None], s / np.maximum(n_ok, 1)[:, None], np.nan)
    return mid, good


def kp_geom_samples(
    t: np.ndarray, kpts: np.ndarray | None, h_ref: float, conf_min: float = KP_CONF_MIN
) -> np.ndarray:
    """(n_samples, len(KP_CHANNELS)) per-sample keypoint geometry. Never NaN.

    Column order and meaning are KP_CHANNELS. A joint counts only when its
    confidence is finite and >= conf_min (the posture_angle_deg contract: equal
    passes). Each validity flag is 0, with its value columns forced to 0,
    wherever the defining joints are missing or the arithmetic did not produce a
    finite number; the caller multiplies by `present` afterwards so tracker gaps
    zero both value and flag the way angle/angle_valid already work.
    """
    n = len(t)
    out = np.zeros((n, len(KP_CHANNELS)), dtype=np.float64)
    if kpts is None or n == 0 or not h_ref > 0:
        return out
    xy = kpts[:, :, :2]
    conf = kpts[:, :, 2]
    ok = np.isfinite(conf) & np.isfinite(xy).all(axis=2) & (conf >= conf_min)

    head, head_ok = _kp_pair_mid(xy, ok, KP_HEAD_JOINTS)
    sh, sh_ok = _kp_pair_mid(xy, ok, KP_SHOULDER_JOINTS)
    hip, hip_ok = _kp_pair_mid(xy, ok, KP_HIP_JOINTS)
    wrist, wrist_ok = _kp_pair_mid(xy, ok, KP_WRIST_JOINTS)
    ankle, ankle_ok = _kp_pair_mid(xy, ok, KP_ANKLE_JOINTS)

    def _emit(cols: dict[str, np.ndarray], valid_name: str, valid: np.ndarray) -> None:
        """Write value columns + their flag; force values to 0 wherever invalid."""
        v = valid.copy()
        for arr in cols.values():
            v &= np.isfinite(arr)
        for name, arr in cols.items():
            out[:, KP_CHANNELS.index(name)] = np.where(v, arr, 0.0)
        out[:, KP_CHANNELS.index(valid_name)] = v.astype(np.float64)

    # Torso orientation: unit vector shoulder-mid -> hip-mid. Image y grows down,
    # so an upright torso has cos = +1; a horizontal (lying) torso has |sin| = 1.
    d = hip - sh
    norm = np.hypot(d[:, 0], d[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        _emit(
            {"kp_torso_cos": d[:, 1] / norm, "kp_torso_sin": d[:, 0] / norm},
            "kp_torso_valid",
            sh_ok & hip_ok & (np.where(np.isfinite(norm), norm, 0.0) > 1e-6),
        )

    # Spine compression: vertical extents in box-heights. A standing body spans
    # them fully; a collapsed one compresses both toward 0.
    _emit(
        {"kp_extent_headhip": np.clip(np.abs(head[:, 1] - hip[:, 1]) / h_ref, 0.0, KP_EXTENT_CLIP)},
        "kp_extent_headhip_valid",
        head_ok & hip_ok,
    )
    _emit(
        {"kp_extent_shankle": np.clip(np.abs(sh[:, 1] - ankle[:, 1]) / h_ref, 0.0, KP_EXTENT_CLIP)},
        "kp_extent_shankle_valid",
        sh_ok & ankle_ok,
    )

    # Head vertical velocity in torso-lengths per second. The scale is the window-median
    # torso length (robust to per-frame jitter); no valid torso in the window
    # means the unit does not exist and the whole channel is invalid.
    torso_valid_mask = out[:, KP_CHANNELS.index("kp_torso_valid")] > 0
    if torso_valid_mask.any() and n >= 2:
        torso_len = float(np.median(norm[torso_valid_mask]))
        head_y = np.where(head_ok, head[:, 1], np.nan)
        dt = np.gradient(t)
        dt[dt <= 0] = np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            head_vy = np.gradient(head_y) / dt / torso_len
        _emit(
            {"kp_head_vy": np.clip(head_vy, -KP_VEL_CLIP, KP_VEL_CLIP)},
            "kp_head_vy_valid",
            np.isfinite(head_vy),
        )

    # Hip-centred limb placement in box-heights: where the wrists/ankles sit
    # relative to the pelvis (arms flung out, legs level with the hips = lying).
    for mids, mid_ok, names in (
        (wrist, wrist_ok, ("kp_wrist_dx", "kp_wrist_dy", "kp_wrist_valid")),
        (ankle, ankle_ok, ("kp_ankle_dx", "kp_ankle_dy", "kp_ankle_valid")),
    ):
        off = (mids - hip) / h_ref
        _emit(
            {
                names[0]: np.clip(off[:, 0], -KP_OFFSET_CLIP, KP_OFFSET_CLIP),
                names[1]: np.clip(off[:, 1], -KP_OFFSET_CLIP, KP_OFFSET_CLIP),
            },
            names[2],
            mid_ok & hip_ok,
        )
    return out


def featurize_window(
    tr: TrackArrays,
    t_end: float,
    spec: WindowSpec,
    scene: SceneContext | None = None,
) -> np.ndarray | None:
    """(n_steps, n_channels) float32 for the window ending at t_end, or None when the
    track covers too little of it (spec.min_present). n_channels = len(CHANNELS), plus
    len(SCENE_CHANNELS) under spec.scene_channels - which then REQUIRES ``scene``: a
    caller that cannot supply the video's track spans must refuse a scene-channels
    model loudly, never feed it zeros. (The live backend supplies one from the
    TrackBook via OnnxEventVerifier.observe_book; see vividly.models.event_verifier.)"""
    t0 = t_end - spec.window_s
    sel = (tr.t >= t0) & (tr.t <= t_end)
    if int(sel.sum()) < 2:
        return None
    t = tr.t[sel]
    cx, cy, w, h = tr.cx[sel], tr.cy[sel], tr.w[sel], tr.h[sel]
    angle, kpc = tr.angle[sel], tr.kpc[sel]
    h_ref = float(np.median(h))
    if not h_ref > 0:
        return None

    # Velocities on the raw samples (box-heights/second), then step-sampled.
    dt = np.gradient(t)
    dt[dt <= 0] = np.nan
    vy = np.gradient(cy) / dt / h_ref
    vx = np.gradient(cx) / dt / h_ref

    steps = t0 + (np.arange(spec.n_steps, dtype=np.float64) + 0.5) * (spec.window_s / spec.n_steps)
    # Nearest observation per step, within half the typical observation gap (x2 slack).
    idx = np.searchsorted(t, steps)
    idx = np.clip(idx, 0, len(t) - 1)
    left = np.clip(idx - 1, 0, len(t) - 1)
    use_left = np.abs(t[left] - steps) < np.abs(t[idx] - steps)
    idx = np.where(use_left, left, idx)
    gap = max(2.0 * float(np.nanmedian(dt)), 1e-3)
    present = (np.abs(t[idx] - steps) <= gap).astype(np.float64)

    def _ch(values: np.ndarray, *, fill: float = 0.0) -> np.ndarray:
        v = values[idx]
        v = np.where(np.isfinite(v), v, fill)
        return v * present + fill * (1.0 - present)

    cols = [
        np.clip(_ch(vy), -10.0, 10.0),
        np.clip(_ch(vx), -10.0, 10.0),
        np.clip(_ch(h / h_ref, fill=1.0), 0.0, 4.0),
        np.clip(_ch(w / h, fill=1.0), 0.0, 6.0),
        _ch(np.clip(angle, 0.0, 90.0) / 90.0),
        (np.isfinite(angle[idx]).astype(np.float64)) * present,
        _ch(np.nan_to_num(kpc, nan=0.0)),
        present,
    ]
    x = np.stack(cols, axis=1)
    if spec.scene_channels:
        if scene is None:
            raise FeatureError(
                "spec.scene_channels requires a SceneContext (all track spans of the "
                "video); this caller cannot supply one, so it cannot run this model"
            )
        x = np.concatenate([x, scene_features(scene, steps)], axis=1)
    if spec.rhythm_channels:
        # From the raw in-window samples, never the step grid: the resampling above
        # is exactly what destroys the band these channels exist to measure.
        rh = np.asarray(rhythm_features(t, cx, cy, h_ref), dtype=np.float64)
        x = np.concatenate([x, np.tile(rh, (spec.n_steps, 1))], axis=1)
    if spec.limb_channels:
        # Same reason as rhythm: the clonic band lives above the step grid's Nyquist
        # limit, so this reads the raw in-window samples, not the resampled steps.
        lb = np.asarray(
            limb_features(t, None if tr.kpts is None else tr.kpts[sel], h_ref, spec.limb_conf_min),
            dtype=np.float64,
        )
        x = np.concatenate([x, np.tile(lb, (spec.n_steps, 1))], axis=1)
    if spec.kp_channels:
        # Per-sample geometry, then the same nearest-observation step mapping the
        # base channels use; `present` zeroes both value and flag across tracker gaps.
        kg = kp_geom_samples(t, None if tr.kpts is None else tr.kpts[sel], h_ref, spec.kp_conf_min)
        x = np.concatenate([x, kg[idx] * present[:, None]], axis=1)
    x = x.astype(np.float32)
    if float(present.mean()) < spec.min_present:
        return None
    return x


@dataclass(frozen=True)
class SceneContext:
    """Every track's observed span in one video, for the SCENE_CHANNELS."""

    spans: np.ndarray  # (n_tracks, 2) [start_t, end_t]
    starts: np.ndarray  # sorted start_t values

    @classmethod
    def from_tracks(cls, tracks: list[TrackArrays]) -> SceneContext:
        if tracks:
            spans = np.array([tr.span for tr in tracks], dtype=np.float64)
        else:
            spans = np.zeros((0, 2), dtype=np.float64)
        return cls(spans=spans, starts=np.sort(spans[:, 0]) if len(spans) else spans[:, 0])


def scene_features(ctx: SceneContext, steps: np.ndarray) -> np.ndarray:
    """(len(steps), len(SCENE_CHANNELS)): normalized track density and birth rate."""
    if len(ctx.spans) == 0:
        return np.zeros((len(steps), len(SCENE_CHANNELS)), dtype=np.float64)
    active = (ctx.spans[None, :, 0] <= steps[:, None]) & (steps[:, None] <= ctx.spans[None, :, 1])
    n_active = active.sum(axis=1).astype(np.float64)
    births = np.searchsorted(ctx.starts, steps, side="right") - np.searchsorted(
        ctx.starts, steps - SCENE_BIRTH_WINDOW_S, side="right"
    )
    return np.stack(
        [
            np.clip(n_active / SCENE_TRACKS_NORM, 0.0, SCENE_CLIP),
            np.clip(births / SCENE_BIRTHS_NORM, 0.0, SCENE_CLIP),
        ],
        axis=1,
    )


def _event_track(tracks: list[TrackArrays], events: Events) -> int | None:
    """The track whose observed span overlaps the events most; None if nobody does."""
    best, best_ov = None, 0.0
    for tr in tracks:
        s0, s1 = tr.span
        ov = sum(max(0.0, min(s1, e1) - max(s0, e0)) for e0, e1 in events)
        if ov > best_ov:
            best, best_ov = tr.track_id, ov
    return best


def _window_source(tr: TrackArrays, t_end: float, spec: WindowSpec) -> WindowSource:
    """The in-window slice of `tr`, the same selection featurize_window makes."""
    sel = (tr.t >= t_end - spec.window_s) & (tr.t <= t_end)
    return WindowSource(
        track=TrackArrays(
            track_id=tr.track_id,
            t=tr.t[sel],
            cx=tr.cx[sel],
            cy=tr.cy[sel],
            w=tr.w[sel],
            h=tr.h[sel],
            angle=tr.angle[sel],
            kpc=tr.kpc[sel],
            kpts=None if tr.kpts is None else tr.kpts[sel],
        ),
        t_end=float(t_end),
    )


def windows_for_video(
    key: str,
    tracks: list[TrackArrays],
    event: Events | tuple[float, float] | None,
    spec: WindowSpec,
    *,
    with_sources: bool = False,
) -> list[LabeledWindow]:
    """Every labelled window from one video's tracks. Ambiguous windows (None labels,
    non-event tracks overlapping the event, sub-coverage windows) are dropped here.

    ``with_sources`` additionally attaches each window's raw rows (WindowSource) so a
    caller can re-featurize it after transforming the observations. Off by default and
    it changes nothing else: the windows, their order and their contents are identical.
    """
    events = normalize_events(event)
    owner = None if not events else _event_track(tracks, events)
    scene = SceneContext.from_tracks(tracks) if spec.scene_channels else None
    out: list[LabeledWindow] = []
    for tr in tracks:
        s0, s1 = tr.span
        # A track shorter than the coverage floor can never survive featurize's
        # min_present gate; a track shorter than the window still yields one
        # left-padded window at its end, because the held-out UR Fall falls live in clips
        # of 3 to 13 s, and silently excluding them would bias the test set easier.
        if s1 - s0 < spec.window_s * spec.min_present:
            continue
        t_end = min(s0 + spec.window_s, s1)
        while t_end <= s1 + spec.stride_s * 0.5:
            te = min(t_end, s1)
            if events and tr.track_id != owner:
                label = 0 if label_window(te - spec.window_s, te, events, spec) == 0 else None
            else:
                label = label_window(te - spec.window_s, te, events, spec)
            if label is not None:
                x = featurize_window(tr, te, spec, scene=scene)
                if x is not None:
                    out.append(
                        LabeledWindow(
                            key=key,
                            track_id=tr.track_id,
                            t_end=round(te, 4),
                            label=label,
                            x=x,
                            source=_window_source(tr, te, spec) if with_sources else None,
                        )
                    )
            t_end += spec.stride_s
    return out


def windows_from_extraction(
    out_dir: Path,
    spec: WindowSpec,
    *,
    event_keys: tuple[str, str] = ("fall_start_t", "fall_end_t"),
    locations: tuple[str, ...] | None = None,
    no_event_only: bool = False,
    with_sources: bool = False,
) -> list[LabeledWindow]:
    """Walk an extraction directory (sequences/ + meta/) into labelled windows.

    A meta record carrying "events_t" (a list of [start, end] second pairs - written
    for multi-event videos like multicam_fall chute04/chute23) wins outright;
    otherwise event_keys names the two meta fields holding the single interval (the
    extractor writes fall_start_t/fall_end_t, the synthetic generator writes
    convulsion_start_t/convulsion_end_t). A meta with neither is all-negative.

    ``locations`` keeps only metas whose "location" is in the set, which is the
    wu_sahzu_emu patient-level train and test split: train and test configs name
    disjoint patient sets over one extraction, and a meta with no location field is
    refused under a filter rather than silently kept.

    ``with_sources`` attaches each window's raw rows (see WindowSource); off by
    default, and the windows themselves are identical either way.

    ``no_event_only`` keeps only videos that carry NO event: the quiet-clip
    selector for negative diets drawn from mixed extractions (caucafall's ADL
    sequences share their subject locations with its falls, so a location filter
    cannot separate them). Filtering happens on the resolved event - the same
    reading the labels use - never on dataset-specific flags.
    """
    metas = sorted((out_dir / "meta").glob("*.json"))
    if not metas:
        raise FeatureError(f"no meta records under {out_dir}")
    windows: list[LabeledWindow] = []
    for meta_path in metas:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if locations is not None:
            loc = meta.get("location")
            if loc is None:
                raise FeatureError(
                    f"{meta_path.name}: no location field but a locations filter is set"
                )
            if loc not in locations:
                continue
        seq = out_dir / "sequences" / f"{meta['key']}.jsonl"
        if not seq.is_file():
            raise FeatureError(f"meta {meta_path.name} has no sequence file {seq}")
        if meta.get("events_t"):
            event: Events | tuple[float, float] | None = [
                (float(e0), float(e1)) for e0, e1 in meta["events_t"]
            ]
        else:
            e0, e1 = meta.get(event_keys[0]), meta.get(event_keys[1])
            event = None if e0 is None or e1 is None else (float(e0), float(e1))
        if no_event_only and event:
            continue
        tracks = tracks_from_rows(load_rows(seq))
        made = windows_for_video(str(meta["key"]), tracks, event, spec, with_sources=with_sources)
        loc = meta.get("location")
        if loc is not None:
            made = [replace(w, location=str(loc)) for w in made]
        windows.extend(made)
    return windows
