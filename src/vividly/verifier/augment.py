"""Track-space augmentation for the event verifier.

Rule 4 of POLICY.md states that label-preserving geometric and temporal
transforms of already-admitted real recorded event tracks are augmentation, not
synthesis. This module implements exactly the three transforms the seizure push-2
protocol (research/seizure-push-2-protocol.md item 3) fixed in advance, and
nothing else:

  1. ``mirror``: negate the ``vx`` channel only. A convulsion mirrored left-right
     is the same convulsion, and among the base track channels nothing else is chiral
     (``vy`` is gravity-aligned, the box scalars and the validity flags are
     orientation-free); the scene, rhythm and limb columns are orientation-free too.
     The keypoint-geometry columns are the exception: ``kp_torso_sin``, ``kp_wrist_dx``
     and ``kp_ankle_dx`` are horizontal components that change sign under a left-right
     mirror, and ``MIRROR_CHANNELS`` does not touch them, so the mirror is defined for a
     spec without ``kp_channels`` only. No shipped config combines an ``augment:`` block
     with ``kp_channels: true``. Adding the keypoint x columns to ``MIRROR_CHANNELS``
     would be a protocol change, not an implementation detail.
  2. ``time_warp``: a bounded in-window resample, factor r in [0.85, 1.18], about
     the window centre. Nearest-step sampling, so the binary channels (``present``,
     ``angle_valid``, every ``*_valid``) stay binary and r = 1.0 is the exact
     identity. Velocity channels are rescaled by 1/r because a warped window is a
     different time base, and re-clipped to the featurizer's own bounds.
  3. ``dropout``: a contiguous run of steps is marked unobserved (``present`` = 0
     and every present-gated channel set to the value ``featurize_window`` writes
     when the tracker has no observation), with the run length bounded so the
     window's ``present`` fraction never drops below 0.5.

Explicitly not part of those three, per the seizure protocol: track shatter,
positive-only jitter, and any amplitude or noise jitter. Adding a fourth seizure-lane
transform is a protocol change, not an implementation detail.

The fall push v3 protocol (research/fall-push-v3-protocol.md arm F3) declares a
second, separate family, camera geometry, which cannot live in feature space at
all and is implemented in its own section at the bottom of this module:

  4. ``camera geometry``: vertical foreshortening s_y in [0.55, 1.0], a perspective
     trapezoid (keystone), roll +/-12 deg and mirror, applied to the raw keypoint and
     box rows of a window before featurization, with every pose-derived quantity
     (the posture angle above all) recomputed from the transformed keypoints rather
     than carried across. It is a different camera looking at the same real event, so
     the label rides along, which is the augmentation clause of rule 4. It is a
     different family from the three above because a foreshortened body is not any
     function of the finished feature columns: ``angle_norm`` and ``aspect`` are not
     recoverable from each other.

Everything in the first family operates on the featurized window
``(n_steps, n_channels)`` produced by
``vividly.verifier.features.featurize_window``; channel roles are resolved by name
from ``channel_names(spec)``, never by index, so a spec that appends scene, rhythm,
limb or kp columns keeps working, with the one carve-out recorded under ``mirror`` above.
Scene, rhythm and limb columns are video-level or window-level, not per-step observations,
so no transform touches them.

The default is off: ``make_augmenter(None)`` returns ``None`` and the caller's array is
untouched. The augmenter draws from its own ``numpy`` Generator and never the
global numpy or torch random stream, so turning it on does not perturb any other
random stream.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .features import (
    CHANNELS,
    KP_CHANNELS,
    KP_CONF_MIN,
    KP_VEL_CLIP,
    TrackArrays,
    WindowSource,
    WindowSpec,
    featurize_window,
)

# The protocol's bounds. Widening either is a protocol change.
WARP_MIN = 0.85
WARP_MAX = 1.18
# "contiguous observation dropout with present >= 0.5 floor" (protocol item 3).
PRESENT_FLOOR = 0.5

# Channels the mirror negates. Exactly vx: see the module docstring.
MIRROR_CHANNELS: tuple[str, ...] = ("vx",)

# Per-second quantities: a time warp changes the time base, so these scale by 1/r.
# The clip bound is the featurizer's own for that channel, so a warped window can
# never leave the value range the model was trained to expect.
VELOCITY_CLIP: dict[str, float] = {"vy": 10.0, "vx": 10.0, "kp_head_vy": KP_VEL_CLIP}

# What featurize_window writes at a step with present = 0, by channel. Every
# per-track channel is gated by `present` there (`v * present + fill * (1 - present)`
# for the base columns, `kg[idx] * present` for the kp columns); the scene, rhythm
# and limb columns are not, so they are absent from this map and dropout leaves them
# alone.
ABSENT_FILL: dict[str, float] = {name: 0.0 for name in CHANNELS}
ABSENT_FILL["h_rel"] = 1.0
ABSENT_FILL["aspect"] = 1.0
ABSENT_FILL.update({name: 0.0 for name in KP_CHANNELS})


class AugmentError(ValueError):
    """An augmentation config that cannot be honoured."""


@dataclass(frozen=True)
class AugmentSpec:
    """Draw probabilities and bounds. Every field has a protocol-legal default."""

    mirror_p: float = 0.5
    warp_p: float = 0.5
    warp_min: float = WARP_MIN
    warp_max: float = WARP_MAX
    dropout_p: float = 0.5
    dropout_max_frac: float = 0.25  # longest contiguous run, as a fraction of n_steps
    present_floor: float = PRESENT_FLOOR
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("mirror_p", "warp_p", "dropout_p"):
            p = float(getattr(self, name))
            if not 0.0 <= p <= 1.0:
                raise AugmentError(f"{name} must be in [0, 1], got {p}")
        if not WARP_MIN <= self.warp_min <= self.warp_max <= WARP_MAX:
            raise AugmentError(
                f"warp range must sit inside [{WARP_MIN}, {WARP_MAX}] "
                f"(protocol item 3), got [{self.warp_min}, {self.warp_max}]"
            )
        if not 0.0 < self.dropout_max_frac <= 0.5:
            raise AugmentError(f"dropout_max_frac must be in (0, 0.5], got {self.dropout_max_frac}")
        if not 0.0 <= self.present_floor <= 1.0:
            raise AugmentError(f"present_floor must be in [0, 1], got {self.present_floor}")


# --------------------------------------------------------------------- transforms
def mirror_vx(x: np.ndarray, cols: Sequence[int]) -> np.ndarray:
    """Negate the mirror channels. Involutive: applying it twice is the identity."""
    out = x.copy()
    for c in cols:
        out[:, c] = -out[:, c]
    return out


def time_warp(
    x: np.ndarray,
    factor: float,
    vel_cols: Sequence[tuple[int, float]],
) -> np.ndarray:
    """Resample the step axis about the window centre by ``factor``.

    Nearest-step sampling (never interpolation): the binary channels must stay
    binary, and ``factor == 1.0`` must be the exact identity. ``vel_cols`` are
    (column, clip) pairs scaled by 1/factor and re-clipped.
    """
    if factor <= 0:
        raise AugmentError(f"time-warp factor must be positive, got {factor}")
    n = x.shape[0]
    u = (np.arange(n, dtype=np.float64) + 0.5) / n
    s = (u - 0.5) / factor + 0.5
    idx = np.clip(np.rint(s * n - 0.5).astype(np.int64), 0, n - 1)
    out = x[idx].copy()
    if factor != 1.0:
        for c, clip in vel_cols:
            out[:, c] = np.clip(out[:, c] / factor, -clip, clip)
    return out


def observation_dropout(
    x: np.ndarray,
    start: int,
    length: int,
    present_col: int,
    fill_cols: Sequence[tuple[int, float]],
) -> np.ndarray:
    """Mark ``x[start:start+length]`` unobserved, exactly as the featurizer would.

    The caller is responsible for the ``present`` floor - it owns the draw, and the
    budget depends on how many of the dropped steps were observed to begin with.
    """
    out = x.copy()
    if length <= 0:
        return out
    stop = min(start + length, x.shape[0])
    for c, fill in fill_cols:
        out[start:stop, c] = fill
    out[start:stop, present_col] = 0.0
    return out


# --------------------------------------------------------------------- augmenter
class Augmenter:
    """Online per-epoch draws over a stack of featurized train windows."""

    def __init__(self, spec: AugmentSpec, channels: Sequence[str]) -> None:
        self.spec = spec
        self.channels = tuple(channels)
        idx = {name: i for i, name in enumerate(self.channels)}
        missing = [n for n in MIRROR_CHANNELS if n not in idx]
        if missing or "present" not in idx:
            raise AugmentError(
                "augmentation needs the base track channels "
                f"(missing {missing or ['present']}) - got {self.channels}"
            )
        self.mirror_cols = tuple(idx[n] for n in MIRROR_CHANNELS)
        self.present_col = idx["present"]
        self.vel_cols = tuple((idx[n], c) for n, c in VELOCITY_CLIP.items() if n in idx)
        self.fill_cols = tuple((idx[n], f) for n, f in ABSENT_FILL.items() if n in idx)

    # ---- one window
    def augment_window(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """mirror -> time-warp -> dropout, each behind its own draw."""
        out = x
        if self.spec.mirror_p > 0 and rng.random() < self.spec.mirror_p:
            out = mirror_vx(out, self.mirror_cols)
        if self.spec.warp_p > 0 and rng.random() < self.spec.warp_p:
            factor = float(rng.uniform(self.spec.warp_min, self.spec.warp_max))
            out = time_warp(out, factor, self.vel_cols)
        if self.spec.dropout_p > 0 and rng.random() < self.spec.dropout_p:
            n = out.shape[0]
            present = out[:, self.present_col]
            # Budget: dropping k steps can remove at most k observed steps, so
            # k <= present_sum - floor*n keeps the window above the floor no matter
            # which steps the run covers.
            budget = int(np.floor(float(present.sum()) - self.spec.present_floor * n))
            max_run = min(budget, int(np.floor(self.spec.dropout_max_frac * n)))
            if max_run >= 1:
                length = int(rng.integers(1, max_run + 1))
                start = int(rng.integers(0, n - length + 1))
                out = observation_dropout(out, start, length, self.present_col, self.fill_cols)
        return out if out is not x else x.copy()

    # ---- one epoch over the train rows
    def augment_epoch(self, X: np.ndarray, epoch: int) -> np.ndarray:
        """A fresh augmented view of every row. ``X`` is never modified in place.

        Row order and row count are preserved exactly, so the caller's labels and
        group ids stay aligned. The augmenter never sees a label and cannot change
        one, so the transform is label-preserving by construction.
        """
        rng = np.random.default_rng((int(self.spec.seed) & 0xFFFFFFFF, int(epoch)))
        out = np.empty_like(X)
        for i in range(X.shape[0]):
            out[i] = self.augment_window(X[i], rng)
        return out


def make_augmenter(cfg: Any, channels: Sequence[str]) -> Augmenter | None:
    """``None`` (the default-OFF path) unless the config asks for augmentation.

    Accepts the config's ``augment:`` block: absent, ``None``, ``{}`` or
    ``enabled: false`` all mean OFF and the caller must leave its data untouched.
    """
    if not cfg:
        return None
    if isinstance(cfg, dict):
        params = dict(cfg)
        if not params.pop("enabled", True):
            return None
        unknown = set(params) - set(AugmentSpec.__dataclass_fields__)
        if unknown:
            raise AugmentError(f"unknown augment keys: {sorted(unknown)}")
        spec = AugmentSpec(**params)
    elif isinstance(cfg, AugmentSpec):
        spec = cfg
    else:
        raise AugmentError(f"augment config must be a mapping, got {type(cfg).__name__}")
    return Augmenter(spec, channels)


# ============================================================ camera geometry (F3)
# Fall push v3 arm F3 (research/fall-push-v3-protocol.md): "vertical foreshortening
# s_y 0.55 to 1.0, perspective trapezoid, roll +/-12 deg, mirror, with the angle
# recomputed from transformed keypoints. This family is separate from the three above and
# shares none of their machinery, because it cannot be done in feature space: a body
# seen from a steeper camera has a different posture angle, a different box aspect and
# a different height profile, and none of those is a function of the others once the
# featurizer has run. It therefore transforms the raw rows and re-featurizes.
#
# Why the label survives. A fall recorded by a camera with a different pitch,
# roll or lens is the same fall: the transform changes the observer, never the event.
# The draw is constant across the window's samples, one virtual camera per augmented
# view, so no temporal structure is invented, and the time base is untouched (this
# family contains no time transform at all).
GEO_SY_MIN = 0.55  # protocol floor for vertical foreshortening; below is a protocol change
GEO_ROLL_MAX_DEG = 12.0  # protocol bound on roll
GEO_TRAPEZOID_MAX = 0.12  # keystone gain, per box-height of vertical offset (see below)
# The keystone is a real projective map, u,v -> u/(1+g v), v/(1+g v), whose vanishing
# line sits at v = -1/g. This floor keeps the denominator far away from it, so the map
# stays a well-behaved diffeomorphism over the window: at the bound g = 0.12 a point one
# box-height above the anchor divides by 0.88 and one below by 1.12, and the floor is
# never reached by any real row. It is a guard, not a regime.
GEO_DENOM_FLOOR = 0.5

# COCO-17, mirroring vividly.tracks.state (which this module must not import: the
# verifier package is a leaf on the training side and `tracks` pulls the whole live
# track-book in). tests/test_verifier_augment.py asserts the recomputed angle equals
# vividly.tracks.posture_angle_deg on random keypoints, so the copy cannot drift.
GEO_LEFT_SHOULDER, GEO_RIGHT_SHOULDER = 5, 6
GEO_LEFT_HIP, GEO_RIGHT_HIP = 11, 12
GEO_TORSO_JOINTS: tuple[int, ...] = (
    GEO_LEFT_SHOULDER,
    GEO_RIGHT_SHOULDER,
    GEO_LEFT_HIP,
    GEO_RIGHT_HIP,
)
GEO_MIN_TORSO_PX = 1e-9  # vividly.tracks.state.MIN_TORSO_PX

# The extraction's own row quantization (scripts/events/extract_track_sequences.py):
# augmented rows are written on the same numeric grid real rows are, so a warped window
# cannot be told from a recorded one by its decimals.
GEO_ROUND_KP = 1
GEO_ROUND_BOX = 2
GEO_ROUND_ANGLE = 2


@dataclass(frozen=True)
class GeoSpec:
    """Draw ranges for the camera-geometry family. Defaults are the identity.

    Every default is neutral (s_y = 1, no roll, no trapezoid, no mirror) so a config
    that only says `enabled: true` augments nothing and says so loudly in the report,
    rather than silently applying somebody's idea of a reasonable strength.
    """

    sy_min: float = 1.0  # vertical foreshortening, drawn U(sy_min, sy_max)
    sy_max: float = 1.0
    roll_deg: float = 0.0  # image-plane roll, drawn U(-roll_deg, +roll_deg)
    trapezoid: float = 0.0  # keystone gain, drawn U(-trapezoid, +trapezoid)
    mirror_p: float = 0.0  # left-right mirror probability
    k: int = 1  # augmented draws per source window
    kp_conf_min: float = KP_CONF_MIN  # the extraction's posture-angle gate (0.3)
    seed: int = 0

    def __post_init__(self) -> None:
        if not GEO_SY_MIN <= self.sy_min <= self.sy_max <= 1.0:
            raise AugmentError(
                f"s_y range must sit inside [{GEO_SY_MIN}, 1.0] with sy_min <= sy_max "
                f"(protocol arm F3), got [{self.sy_min}, {self.sy_max}]"
            )
        if not 0.0 <= self.roll_deg <= GEO_ROLL_MAX_DEG:
            raise AugmentError(f"roll_deg must be in [0, {GEO_ROLL_MAX_DEG}], got {self.roll_deg}")
        if not 0.0 <= self.trapezoid <= GEO_TRAPEZOID_MAX:
            raise AugmentError(
                f"trapezoid must be in [0, {GEO_TRAPEZOID_MAX}], got {self.trapezoid}"
            )
        if not 0.0 <= self.mirror_p <= 1.0:
            raise AugmentError(f"mirror_p must be in [0, 1], got {self.mirror_p}")
        if int(self.k) < 1:
            raise AugmentError(f"k must be >= 1, got {self.k}")
        if not 0.0 <= self.kp_conf_min <= 1.0:
            raise AugmentError(f"kp_conf_min must be in [0, 1], got {self.kp_conf_min}")


@dataclass(frozen=True)
class GeoDraw:
    """One virtual camera: applied to every sample of one window, unchanged."""

    sy: float = 1.0
    roll_rad: float = 0.0
    gain: float = 0.0
    mirror: bool = False

    @property
    def is_identity(self) -> bool:
        return self.sy == 1.0 and self.roll_rad == 0.0 and self.gain == 0.0 and not self.mirror


def torso_angles_deg(kpts: np.ndarray, conf_min: float) -> np.ndarray:
    """(n,) posture angle in degrees from vertical, nan where undefined.

    A vectorized copy of ``vividly.tracks.posture_angle_deg`` - same torso midpoints,
    same absolute-value convention (a tilt magnitude, not a heading), same gate
    semantics (strictly below conf_min fails; equal passes), same zero-length guard.
    ``nan`` stands where the scalar function returns ``None``, which is what the row
    schema writes as ``angle: null`` and what the featurizer reads as angle_valid = 0.
    """
    conf = kpts[:, GEO_TORSO_JOINTS, 2]
    # `not (c < conf_min)`, not `c >= conf_min`: nan must follow the scalar function's
    # own comparison, which is False for nan and so passes the gate (the nan then
    # travels through the arithmetic and comes out as an undefined angle anyway).
    gated = (conf < conf_min).any(axis=1)
    sh_x = (kpts[:, GEO_LEFT_SHOULDER, 0] + kpts[:, GEO_RIGHT_SHOULDER, 0]) / 2.0
    sh_y = (kpts[:, GEO_LEFT_SHOULDER, 1] + kpts[:, GEO_RIGHT_SHOULDER, 1]) / 2.0
    hip_x = (kpts[:, GEO_LEFT_HIP, 0] + kpts[:, GEO_RIGHT_HIP, 0]) / 2.0
    hip_y = (kpts[:, GEO_LEFT_HIP, 1] + kpts[:, GEO_RIGHT_HIP, 1]) / 2.0
    dx, dy = hip_x - sh_x, hip_y - sh_y
    with np.errstate(invalid="ignore"):
        ang = np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))
        ang = np.where(np.hypot(dx, dy) < GEO_MIN_TORSO_PX, np.nan, ang)
    return np.where(gated, np.nan, ang)


def warp_points(
    x: np.ndarray, y: np.ndarray, draw: GeoDraw, ax: float, ay: float, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one virtual camera to image-space points.

    Anchored at (ax, ay) and measured in units of ``scale`` (the window's median box
    height), so the warp is a property of the body's geometry and not of where in the
    frame the person happens to stand: a fall at the left edge and the same fall at
    the centre get the same family of views.

    Declared order, and it is part of the protocol: mirror -> keystone -> vertical
    foreshortening -> roll. The first three model where the camera is (a mirrored,
    pitched, off-axis view); roll is the image-plane rotation that a real camera
    applies last, after the projection. nan travels through untouched, so a sample the
    pose estimator never resolved stays unresolved.
    """
    u = (x - ax) / scale
    v = (y - ay) / scale
    if draw.mirror:
        u = -u
    if draw.gain != 0.0:
        denom = 1.0 + draw.gain * v
        denom = np.where(denom >= GEO_DENOM_FLOOR, denom, GEO_DENOM_FLOOR)
        u = u / denom
        v = v / denom
    if draw.sy != 1.0:
        v = v * draw.sy
    if draw.roll_rad != 0.0:
        c, s = float(np.cos(draw.roll_rad)), float(np.sin(draw.roll_rad))
        u, v = u * c - v * s, u * s + v * c
    return ax + u * scale, ay + v * scale


def warp_track_window(tr: TrackArrays, draw: GeoDraw, conf_min: float) -> TrackArrays:
    """One window's raw rows as a different camera would have recorded them.

    Keypoints go through ``warp_points``; the BOX goes through the same map (the
    axis-aligned hull of its four warped corners), which is what a detector sees on the
    warped image and keeps box and pose mutually consistent. Deriving the box from the
    warped keypoints alone was considered and rejected: keypoint hulls are
    systematically tighter than detector boxes, so every augmented row would carry an
    out-of-distribution h_rel/aspect and the arm would be measuring that instead of
    camera geometry.

    The posture angle is recomputed from the warped keypoints, never carried across.
    Copying it is the one thing that would make this transform a lie: foreshortening a
    standing body is exactly a change of apparent torso tilt.

    ``kpc`` (mean keypoint confidence) and ``t`` are untouched: confidence is not a
    geometric quantity and this family contains no time transform.
    """
    if draw.is_identity:
        return tr
    if tr.kpts is None:
        raise AugmentError(
            "camera-geometry augmentation needs keypoint rows (`emit_keypoints: true`): "
            "the posture angle has to be recomputed from the warped keypoints, and "
            "carrying the recorded angle across a warp would mislabel the geometry"
        )
    scale = float(np.median(tr.h))
    if not scale > 0:
        raise AugmentError("cannot warp a window whose median box height is not positive")
    ax, ay = float(np.median(tr.cx)), float(np.median(tr.cy))

    kx, ky = warp_points(tr.kpts[:, :, 0], tr.kpts[:, :, 1], draw, ax, ay, scale)
    kpts = tr.kpts.copy()
    kpts[:, :, 0] = np.round(kx, GEO_ROUND_KP)
    kpts[:, :, 1] = np.round(ky, GEO_ROUND_KP)

    half_w, half_h = tr.w / 2.0, tr.h / 2.0
    corners_x = np.stack([tr.cx - half_w, tr.cx + half_w, tr.cx + half_w, tr.cx - half_w], axis=1)
    corners_y = np.stack([tr.cy - half_h, tr.cy - half_h, tr.cy + half_h, tr.cy + half_h], axis=1)
    bx, by = warp_points(corners_x, corners_y, draw, ax, ay, scale)
    x0, x1 = bx.min(axis=1), bx.max(axis=1)
    y0, y1 = by.min(axis=1), by.max(axis=1)
    cx = np.round((x0 + x1) / 2.0, GEO_ROUND_BOX)
    cy = np.round((y0 + y1) / 2.0, GEO_ROUND_BOX)
    w = np.round(x1 - x0, GEO_ROUND_BOX)
    h = np.round(y1 - y0, GEO_ROUND_BOX)

    angle = np.round(torso_angles_deg(kpts, conf_min), GEO_ROUND_ANGLE)
    return TrackArrays(
        track_id=tr.track_id,
        t=tr.t,
        cx=cx,
        cy=cy,
        w=w,
        h=h,
        angle=angle,
        kpc=tr.kpc,
        kpts=kpts,
    )


class GeoAugmenter:
    """Online per-epoch camera-geometry views of a fixed set of source windows.

    The caller hands it the raw rows of the windows it may augment (train rows only,
    selected after the split, which is the scope clause of rule 4) together with their
    already featurized originals, and gets back ``k`` extra rows per source window each
    epoch. Row order is source-major (source 0's k draws, then source 1's, and so on), so
    the caller's label vector for the extra rows is a constant, and since every source is
    a positive window, that constant is 1. The augmenter never sees a label.
    """

    def __init__(
        self,
        spec: GeoSpec,
        window_spec: WindowSpec,
        sources: Sequence[WindowSource],
        base: np.ndarray,
    ) -> None:
        if window_spec.scene_channels:
            raise AugmentError(
                "camera-geometry augmentation cannot re-featurize a scene-channels spec: "
                "SCENE_CHANNELS need every track span of the video, which a single "
                "window's own rows cannot supply"
            )
        if len(sources) != len(base):
            raise AugmentError(f"{len(sources)} sources but {len(base)} base rows")
        missing = sum(1 for s in sources if s.track.kpts is None)
        if missing:
            raise AugmentError(
                f"{missing} of {len(sources)} source windows carry no keypoints; camera "
                "geometry needs `emit_keypoints: true` extractions (the posture angle is "
                "recomputed, never copied)"
            )
        self.spec = spec
        self.window_spec = window_spec
        self.sources = tuple(sources)
        self.base = base
        self.n_fallback = 0  # windows a warp made unfeaturizable; counted, never hidden

    # ---- one draw
    def draw(self, rng: np.random.Generator) -> GeoDraw:
        s = self.spec
        return GeoDraw(
            sy=float(rng.uniform(s.sy_min, s.sy_max)) if s.sy_max > s.sy_min else s.sy_min,
            roll_rad=float(np.radians(rng.uniform(-s.roll_deg, s.roll_deg))) if s.roll_deg else 0.0,
            gain=float(rng.uniform(-s.trapezoid, s.trapezoid)) if s.trapezoid else 0.0,
            mirror=bool(rng.random() < s.mirror_p) if s.mirror_p > 0 else False,
        )

    def augment_one(self, i: int, draw: GeoDraw) -> np.ndarray:
        """Source window ``i`` under one virtual camera, re-featurized."""
        src = self.sources[i]
        warped = warp_track_window(src.track, draw, self.spec.kp_conf_min)
        x = featurize_window(warped, src.t_end, self.window_spec)
        if x is None:  # the warp cost the window its coverage floor: keep the original
            self.n_fallback += 1
            return self.base[i]
        return x

    def verify_identity(self) -> int:
        """How many sources do NOT re-featurize to their recorded row. Must be 0.

        The plumbing guard: it proves the raw rows, the t_end and the window spec held
        here are the very ones that produced `base`, so any difference in an augmented
        row is the camera and nothing else.
        """
        bad = 0
        for i, src in enumerate(self.sources):
            x = featurize_window(src.track, src.t_end, self.window_spec)
            if x is None or not np.array_equal(x, self.base[i]):
                bad += 1
        return bad

    # ---- one epoch
    def augment_epoch(self, epoch: int) -> np.ndarray:
        """``(len(sources) * k, n_steps, n_channels)`` fresh views. Never in place."""
        rng = np.random.default_rng((int(self.spec.seed) & 0xFFFFFFFF, int(epoch)))
        k = int(self.spec.k)
        out = np.empty((len(self.sources) * k, *self.base.shape[1:]), dtype=self.base.dtype)
        row = 0
        for i in range(len(self.sources)):
            for _ in range(k):
                out[row] = self.augment_one(i, self.draw(rng))
                row += 1
        return out


def make_geo_augmenter(
    cfg: Any,
    window_spec: WindowSpec,
    sources: Sequence[WindowSource],
    base: np.ndarray,
) -> GeoAugmenter | None:
    """``None`` (the default-OFF path) unless the config asks for camera geometry.

    Same contract as ``make_augmenter``: absent, ``None``, ``{}`` or ``enabled: false``
    all mean OFF, and an unknown key is refused rather than ignored.
    """
    if not cfg:
        return None
    if isinstance(cfg, dict):
        params = dict(cfg)
        if not params.pop("enabled", True):
            return None
        unknown = set(params) - set(GeoSpec.__dataclass_fields__)
        if unknown:
            raise AugmentError(f"unknown geo_augment keys: {sorted(unknown)}")
        spec = GeoSpec(**params)
    elif isinstance(cfg, GeoSpec):
        spec = cfg
    else:
        raise AugmentError(f"geo_augment config must be a mapping, got {type(cfg).__name__}")
    return GeoAugmenter(spec, window_spec, sources, base)
