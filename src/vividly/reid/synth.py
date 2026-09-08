"""Synthetic fixture for the leave-and-return track-cut re-ID benchmark.

The benchmark this fixture feeds cuts each ground-truth track into an enrollment
segment and a return segment separated by an artificial gap, stratifies by gap length
(30 s, 5 min, 30 min, 2 h), injects other identities' activity into the gap, and asks the
system to re-associate. This module builds that structure out of arithmetic instead of
footage, so the metric implementations, the open-set threshold machinery and the TrackEval
leg are exercised with no clips, no weights, no GPU and no network.

What this is not
================
No number computed on this fixture is evidence about re-identification performance on real
overhead footage, and none of it may be quoted as one. Every constant below (`separability`,
`noise`, the drift curve, the seat-change probability) is an invented modelling assumption
with no empirical basis: this is a synthetic geometry that happens to have the shape of a
re-ID problem. The one public data point on the overhead domain gap (FRIDA, Cokbas et al.,
IEEE Access 2023, roughly 18 points lost by street-trained embeddings) says the real
difficulty is not something a Gaussian can be tuned to reproduce. A passing run here shows
only that the harness computes what it claims to compute, never that re-ID works. The same
calibration discipline that governs rule thresholds (docs/CALIBRATION.md) governs these:
PROVISIONAL until measured on real overhead data.

No part of this module touches, derives from, or embeds any dataset. Nothing here is a
person.

Mechanisms
==========

1. Identity separability is a cone angle, and inter-identity cosine is monotone in it.

   Draw one anchor direction `a` and, per identity, one direction `w_i` in the orthogonal
   complement of `a`, both from the seed and not from `separability`. Place every centroid
   on a cone of half-angle theta = separability * pi/2 around the anchor:

       c_i = cos(theta) * a + sin(theta) * w_i        (unit, since a is orthogonal to w_i)

   Because a is orthogonal to every w_i, the pairwise cosine has a closed form:

       <c_i, c_j> = cos^2(theta) + sin^2(theta) * g_ij
                  = g_ij + cos^2(theta) * (1 - g_ij),      g_ij = <w_i, w_j>

   `cos^2(theta)` is strictly decreasing in `separability` over [0, 1] and `(1 - g_ij)` is
   non-negative, so every pair's cosine is non-increasing in separability, and strictly
   decreasing whenever g_ij < 1 (true for distinct random directions once the complement is
   at least 2-D, which is why `dim >= 3` is enforced). The maximum and the minimum over
   pairs are pointwise max/min of decreasing functions, so both fall too. At separability 0
   every identity collapses onto the anchor (cosine 1); at separability 1 the centroids are
   random directions in the complement (cosine near 0, heterogeneous, so hard pairs exist).

2. Intra-identity spread is dimension-independent.

   A sample is `centroid + (noise / sqrt(dim)) * g`, g ~ N(0, I), renormalised. The
   1/sqrt(dim) is the load-bearing part: without it, a noise of 0.35 would mean an almost
   orthogonal sample at dim 512 and an almost clean one at dim 16, and no default would
   survive a change of embedding size. With it, the perturbation has expected L2 norm
   `noise` whatever the dimension, and the expected sample-to-centroid cosine is about
   1 / sqrt(1 + noise^2) at every dim.

3. Gap length costs something, through a drift curve.

   A gap-stratified benchmark whose tiers are statistically identical measures nothing, so
   the return centroid of identity i after a gap G is the enrollment centroid rotated by
   d = drift_fraction(G) towards a fresh orthogonal direction:

       c' = sqrt(1 - d^2) * c + d * u,  u unit and orthogonal to c  ->  <c, c'> = sqrt(1 - d^2)

   with d = drift_max * G / (G + drift_half_gap_s), a saturating curve whose half-drift
   point is a config field. This curve is invented. It encodes only the qualitative claim
   that longer absences are harder, which is the one thing the literature does agree on.

   The new-identity probes carry the same tier drift, because drift belongs to the moment
   an appearance was observed and not to the person: everything in the return window is
   seen under return-window conditions. Exempting them would leave a never-enrolled
   stranger systematically closer to the enrollment gallery than the returning patient is,
   which pins the false-merge rate at zero by construction and makes the one metric this
   whole benchmark exists to report impossible to exercise.

Layout
======
One MOT sequence per gap tier, each a self-contained replay of the same waiting room:

    [0, enrollment_s)                     enrollment tracklet per enrolled identity
    [enrollment_s, enrollment_s + G)      gap: distractor tracklets, strictly inside
    [enrollment_s + G, ... + return_s)    return tracklet per enrolled identity,
                                          plus the new-identity probes for the open-set leg

Ground-truth ids are the identity (`identity + 1`, 1-based per the MOT convention), so the
cut is invisible to the ground truth and re-associating across it is exactly what HOTA AssA
and IDF1 measure. `Tracklet.track_id` is the other thing: a tracker-side fragment id,
unique across the whole scenario, which is what a system sees before it re-associates.

Box positions are seats on a grid plus a small sway. Position is irrelevant to the
appearance leg, but it is the seat prior, the one non-appearance signal POLICY.md, section 4
allows, and it has to be valid MOT, so the boxes are built through `vividly.eval.mot.MotRow`,
which is the only MOT definition in this repository and refuses a 0-based frame or a
degenerate box.

Determinism
===========
Same config in, byte-identical scenario out, on any platform: seeds are mixed with a
64-bit FNV-1a over the parts (not `hash()`, which is salted per process), every random
stream is keyed by (seed, role, tier, identity), and `scenario_digest` gives a benchmark
runner a provenance handle to record next to its numbers. Arrays stored on a `Tracklet`
are copies with `writeable=False`: a consumer that normalises a fixture in place would
otherwise silently invalidate every number computed after it.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from vividly.eval.mot import GT_CONSIDER, PEDESTRIAN_CLASS_ID, MotRow

#: What a tracklet is for. `enrollment` and `return` are the two halves of one cut track,
#: `distractor` lives inside a gap and is never enrolled or probed, `new_identity` is an
#: open-set probe whose identity appears nowhere in enrollment.
TrackletRole = Literal["enrollment", "return", "distractor", "new_identity"]

#: Stream tags. Every random draw in this module is keyed by one of these, so two roles can
#: never accidentally share a noise pattern.
_STREAM_BANK = 1
_STREAM_SAMPLE = 2
_STREAM_DRIFT = 3
_STREAM_SEAT = 4
_STREAM_ENROLL = 5
_STREAM_RETURN = 6
_STREAM_DISTRACTOR = 7
_STREAM_NEW = 8

# FNV-1a 64-bit. Chosen over `hash()` (salted per process, so it would break determinism
# across runs) and over SeedSequence.spawn (whose child ordering is easy to get wrong when
# streams are built in a loop). Small, stable, and identical on every platform.
_FNV_OFFSET_BASIS = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_UINT64_MASK = 0xFFFFFFFFFFFFFFFF

#: Norms below this mean a direction carries no information; the draw is repeated. Reaching
#: `_MAX_REDRAWS` is a probability-zero event, so it raises rather than degrading quietly.
_MIN_NORM = 1e-12
_MAX_REDRAWS = 16

#: Box coordinates are rounded to this many decimals before they reach a MotRow. The MOT
#: writer formats with %.2f, so anything finer is invisible on disk and only makes digests
#: sensitive to the last bit of a float.
_BOX_DECIMALS = 3


def _mix(*parts: int) -> int:
    """Mix integers into a stable 64-bit seed. Deterministic across processes and machines."""
    digest = _FNV_OFFSET_BASIS
    for part in parts:
        for byte in (int(part) & _UINT64_MASK).to_bytes(8, "little"):
            digest ^= byte
            digest = (digest * _FNV_PRIME) & _UINT64_MASK
    return digest


def _unit_rows(rows: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation in float64. Raises on a zero row rather than dividing by 0."""
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if not np.all(norms > _MIN_NORM):
        raise ValueError("cannot normalise a zero-length row")
    return rows / norms


def _orthogonal_unit_rows(rng: np.random.Generator, base: np.ndarray, count: int) -> np.ndarray:
    """`count` unit directions, each orthogonal to the corresponding row of `base`.

    `base` is either one unit row broadcast against every draw (the identity-bank anchor) or
    one unit row per draw (the per-identity drift directions).
    """
    dim = base.shape[-1]
    rows = rng.standard_normal((count, dim))
    rows -= base * np.sum(rows * base, axis=-1, keepdims=True)
    norms = np.linalg.norm(rows, axis=1)
    for index in np.flatnonzero(norms <= _MIN_NORM):
        anchor = base if base.ndim == 1 else base[index]
        for _ in range(_MAX_REDRAWS):
            candidate = rng.standard_normal(dim)
            candidate -= anchor * float(np.dot(candidate, anchor))
            if float(np.linalg.norm(candidate)) > _MIN_NORM:
                rows[index] = candidate
                break
        else:  # pragma: no cover - probability zero for dim >= 2
            raise RuntimeError(
                f"could not draw a direction orthogonal to row {index} in {_MAX_REDRAWS} "
                "attempts; this is a probability-zero event, not a tuning problem"
            )
    return _unit_rows(rows)


def _frozen_float_array(values: np.ndarray, dtype: type[np.floating]) -> np.ndarray:
    """A contiguous, read-only copy. The copy is deliberate: see the module docstring."""
    out = np.array(values, dtype=dtype, order="C", copy=True)
    out.flags.writeable = False
    return out


# ---------------------------------------------------------------------------
# identity bank and per-identity samples
# ---------------------------------------------------------------------------
def make_identity_bank(n_identities: int, dim: int, separability: float, seed: int) -> np.ndarray:
    """`(n_identities, dim)` float32 unit rows, one appearance centroid per identity.

    `separability` in [0, 1] is a cone half-angle in units of pi/2 around a shared anchor
    direction: 0 collapses every identity onto the anchor (all pairwise cosines 1), 1 spreads
    them over the anchor's orthogonal complement (pairwise cosines near 0). Both the maximum
    and the minimum inter-identity cosine are non-increasing in `separability`, per pair and
    therefore in aggregate; see mechanism 1 in the module docstring for the closed form.

    The random draw depends on `seed` alone, never on `separability`, which is what makes
    the monotonicity a statement about one fixed configuration of identities rather than
    about two unrelated ones.
    """
    if n_identities < 1:
        raise ValueError(f"n_identities must be >= 1, got {n_identities}")
    if dim < 3:
        raise ValueError(
            f"dim must be >= 3, got {dim}. The anchor's orthogonal complement has dim - 1 "
            "dimensions, and a 1-D complement holds only two distinct directions, which "
            "collapses identities onto a line and breaks the separability monotonicity."
        )
    if not 0.0 <= separability <= 1.0:
        raise ValueError(f"separability must be in [0, 1], got {separability}")

    rng = np.random.default_rng(_mix(seed, _STREAM_BANK))
    anchor = _unit_rows(rng.standard_normal((1, dim)))[0]
    directions = _orthogonal_unit_rows(rng, anchor, n_identities)

    theta = separability * (math.pi / 2.0)
    rows = math.cos(theta) * anchor[None, :] + math.sin(theta) * directions
    return _as_float32_rows(_unit_rows(rows))


def _as_float32_rows(rows: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rows, dtype=np.float32)


def sample_embeddings(
    bank: np.ndarray, identity: int, n: int, noise: float, seed: int
) -> np.ndarray:
    """`(n, dim)` L2-normalised float32 samples around `bank[identity]`.

    `noise` is the expected L2 magnitude of the perturbation relative to the unit centroid
    and is dimension-independent by construction (mechanism 2): the Gaussian is scaled by
    `noise / sqrt(dim)`, so the expected sample-to-centroid cosine is about
    `1 / sqrt(1 + noise ** 2)` at any `dim`. `noise = 0` returns `n` copies of the centroid.

    The random stream is keyed by `(seed, identity)`, so two identities sampled with the same
    `seed` do not share a noise pattern, and one identity's samples are reproducible without
    reproducing the call order that produced its neighbours.
    """
    matrix = np.asarray(bank)
    if matrix.ndim != 2:
        raise ValueError(f"bank must be a 2-D (n_identities, dim) array, got {matrix.shape}")
    if not 0 <= identity < matrix.shape[0]:
        raise IndexError(f"identity {identity} is outside a bank of {matrix.shape[0]} rows")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if noise < 0.0:
        raise ValueError(f"noise must be >= 0, got {noise}")

    centroid = np.asarray(matrix[identity], dtype=np.float64)
    dim = centroid.shape[0]
    rows = np.repeat(centroid[None, :], n, axis=0)
    if noise > 0.0:
        rng = np.random.default_rng(_mix(seed, _STREAM_SAMPLE, identity))
        rows = rows + (noise / math.sqrt(dim)) * rng.standard_normal((n, dim))
    return _as_float32_rows(_unit_rows(rows))


def drift_fraction(gap_s: float, drift_max: float, half_gap_s: float) -> float:
    """Sine of the appearance-drift angle after a gap of `gap_s` seconds.

    `drift_max * gap / (gap + half_gap_s)`: 0 at gap 0, `drift_max / 2` at `half_gap_s`,
    asymptotic to `drift_max`. The enrollment-to-return centroid cosine is `sqrt(1 - d^2)`.
    INVENTED (see the module docstring): it encodes "longer absences are harder" and nothing
    more, and it is PROVISIONAL until measured on real overhead data.
    """
    if gap_s < 0.0:
        raise ValueError(f"gap_s must be >= 0, got {gap_s}")
    if not 0.0 <= drift_max <= 1.0:
        raise ValueError(f"drift_max must be in [0, 1], got {drift_max}")
    if half_gap_s <= 0.0:
        raise ValueError(f"half_gap_s must be > 0, got {half_gap_s}")
    return drift_max * gap_s / (gap_s + half_gap_s)


def drifted_bank(bank: np.ndarray, drift: float, seed: int) -> np.ndarray:
    """`bank` with every centroid rotated by `drift` towards a fresh orthogonal direction.

    Row-wise, `<bank[i], out[i]> == sqrt(1 - drift ** 2)` up to float32 rounding, since the
    returned bank is float32 while the rotation is computed in float64. Feed the result to
    `sample_embeddings` to draw a return tracklet whose appearance has moved since
    enrollment.
    """
    matrix = np.asarray(bank, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"bank must be a 2-D (n_identities, dim) array, got {matrix.shape}")
    if not 0.0 <= drift < 1.0:
        raise ValueError(f"drift must be in [0, 1), got {drift}")
    if drift == 0.0:
        return _as_float32_rows(matrix)
    rng = np.random.default_rng(_mix(seed, _STREAM_DRIFT))
    orthogonal = _orthogonal_unit_rows(rng, matrix, matrix.shape[0])
    rotated = math.sqrt(1.0 - drift * drift) * matrix + drift * orthogonal
    return _as_float32_rows(_unit_rows(rotated))


# ---------------------------------------------------------------------------
# tracklets and scenario
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Tracklet:
    """One contiguous run of appearance samples from one identity.

    `identity` is the ground-truth person (a row index into the identity bank, 0-based).
    `track_id` is the tracker-side fragment id (1-based, unique across a scenario): the two
    halves of a cut track share an `identity` and never share a `track_id`, which is the
    whole point of the benchmark.

    `eq=False` on purpose: the generated `__eq__` would compare numpy arrays inside a tuple
    comparison and raise "truth value of an array is ambiguous". Compare scenarios with
    `scenario_digest`, or arrays with `np.array_equal`.

    `gap_s`, `role`, `sequence` and `seat` carry defaults so the five contract fields can
    still be passed positionally.
    """

    identity: int
    track_id: int
    t0: float
    embeddings: np.ndarray
    timestamps: np.ndarray
    #: The gap tier this tracklet belongs to, in seconds. `None` only for a hand-built
    #: tracklet outside a scenario.
    gap_s: float | None = None
    role: TrackletRole = "enrollment"
    #: MOT sequence name, i.e. which tier's replay this tracklet appears in.
    sequence: str = ""
    #: Seat index on the scenario's seat grid, or -1 when unknown. The seat prior is the one
    #: non-appearance identity signal POLICY.md, section 4 allows.
    seat: int = -1

    def __post_init__(self) -> None:
        embeddings = np.asarray(self.embeddings)
        timestamps = np.asarray(self.timestamps)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be (n, dim), got shape {embeddings.shape}")
        if timestamps.ndim != 1:
            raise ValueError(f"timestamps must be 1-D, got shape {timestamps.shape}")
        if embeddings.shape[0] != timestamps.shape[0]:
            raise ValueError(
                f"{embeddings.shape[0]} embeddings against {timestamps.shape[0]} timestamps: "
                "a tracklet carries one embedding per sampled instant"
            )
        if embeddings.shape[0] < 1:
            raise ValueError("a tracklet needs at least one sample")
        if self.identity < 0:
            raise ValueError(f"identity must be >= 0, got {self.identity}")
        if self.track_id < 1:
            raise ValueError(
                f"track_id must be 1-based to stay writable as MOT, got {self.track_id}"
            )
        if timestamps.shape[0] > 1 and not np.all(np.diff(timestamps) > 0.0):
            raise ValueError("timestamps must be strictly increasing")
        object.__setattr__(self, "embeddings", _frozen_float_array(embeddings, np.float32))
        object.__setattr__(self, "timestamps", _frozen_float_array(timestamps, np.float64))

    @property
    def n_samples(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def t_start(self) -> float:
        return float(self.timestamps[0])

    @property
    def t_end(self) -> float:
        return float(self.timestamps[-1])

    @property
    def span_s(self) -> float:
        """Seconds covered by the samples themselves (0 for a single-sample tracklet)."""
        return self.t_end - self.t_start


@dataclass(frozen=True)
class ScenarioConfig:
    """Everything `build_scenario` needs. Every default is PROVISIONAL and invented.

    How the defaults were chosen, and it was not by fitting anything: `separability`,
    `noise` and `drift_max` were swept until a nearest-neighbour baseline over the default
    fixture stopped saturating, i.e. until the four gap tiers produced visibly different
    Rank-1, DIR at FAR 1 percent, and false-merge numbers instead of 1.0, 1.0, 1.0, 1.0.
    A fixture where every tier scores the same measures nothing, and one where the
    false-merge rate is structurally zero cannot exercise the headline patient-safety KPI
    at all. That is the entire criterion. It makes these numbers useful for developing
    metric code and useless as evidence about real re-ID; see the module docstring.
    """

    #: Identities enrolled before the cut. These are the ones a return probe may match.
    n_identities: int = 24
    #: Identities that only ever appear INSIDE a gap. Never enrolled, never probed: they are
    #: the "other identities' activity" this benchmark injects into the gap, and they are
    #: what a system that enrols everything it sees would pollute its gallery with.
    n_distractor_identities: int = 8
    #: Identities that appear only in the return window and were never enrolled. These are
    #: the open-set FAR denominator.
    n_new_identities: int = 8

    #: Embedding width. 512 matches the OSNet-x1_0 export recorded in
    #: final_models/reid/model.json ("embedding_dim": 512).
    dim: int = 512
    #: Cone half-angle in units of pi/2 (mechanism 1). Higher means identities further apart.
    separability: float = 0.36
    #: Expected L2 magnitude of the per-sample perturbation (mechanism 2).
    noise: float = 1.05

    #: The gap stratification: 30 s, 5 min, 30 min, 2 h. One MOT sequence per tier.
    gap_tiers_s: tuple[float, ...] = (30.0, 300.0, 1800.0, 7200.0)
    enrollment_s: float = 60.0
    return_s: float = 30.0
    #: Nominal distractor duration; shortened automatically to fit inside a short gap.
    distractor_s: float = 20.0
    #: Dead time kept clear at both ends of a gap so a distractor can never touch a cut edge.
    gap_margin_s: float = 2.0

    #: Embedding sampling rate. One vector per second is a realistic tracklet subsampling and
    #: keeps the default fixture at tens of MB instead of tens of GB. Set it equal to `fps`
    #: for dense per-frame ground truth.
    sample_hz: float = 1.0
    #: Frame rate the MOT frame numbers are derived at. 15 fps is this repository's overhead
    #: working point.
    fps: float = 15.0

    #: Appearance drift across a gap (mechanism 3). INVENTED.
    drift_max: float = 0.96
    drift_half_gap_s: float = 900.0

    #: Fraction of returns that come back to a DIFFERENT seat. Without this a seat-prior-only
    #: system would score 100 percent and the fixture would be worthless for fusion work.
    return_seat_change_prob: float = 0.25

    #: Frame geometry for the MOT boxes. Square, so the near-square seat grid gets square cells.
    frame_w: int = 1280
    frame_h: int = 1280
    #: Box size as a fraction of its seat cell.
    box_fill: float = 0.6
    #: Amplitude and period of the deterministic sway that keeps boxes from being frozen.
    sway_px: float = 6.0
    sway_period_s: float = 20.0

    seed: int = 20260807

    def __post_init__(self) -> None:
        if self.n_identities < 1:
            raise ValueError(f"n_identities must be >= 1, got {self.n_identities}")
        if self.n_distractor_identities < 0:
            raise ValueError("n_distractor_identities must be >= 0")
        if self.n_new_identities < 0:
            raise ValueError("n_new_identities must be >= 0")
        if self.dim < 3:
            raise ValueError(f"dim must be >= 3, got {self.dim}")
        if not 0.0 <= self.separability <= 1.0:
            raise ValueError(f"separability must be in [0, 1], got {self.separability}")
        if self.noise < 0.0:
            raise ValueError(f"noise must be >= 0, got {self.noise}")
        if not self.gap_tiers_s:
            raise ValueError("gap_tiers_s must list at least one gap tier")
        if any(gap < 1.0 for gap in self.gap_tiers_s):
            raise ValueError(f"every gap tier must be >= 1 s, got {self.gap_tiers_s}")
        names = [sequence_name(gap) for gap in self.gap_tiers_s]
        if len(set(names)) != len(names):
            raise ValueError(
                f"gap tiers {self.gap_tiers_s} collide after rounding to whole seconds "
                f"({names}); each tier is one MOT sequence and needs its own name"
            )
        for name, value in (
            ("enrollment_s", self.enrollment_s),
            ("return_s", self.return_s),
            ("distractor_s", self.distractor_s),
            ("sample_hz", self.sample_hz),
            ("fps", self.fps),
            ("drift_half_gap_s", self.drift_half_gap_s),
            ("sway_period_s", self.sway_period_s),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if self.gap_margin_s < 0.0:
            raise ValueError(f"gap_margin_s must be >= 0, got {self.gap_margin_s}")
        if self.sample_hz > self.fps:
            raise ValueError(
                f"sample_hz ({self.sample_hz}) must be <= fps ({self.fps}): two samples "
                "landing on one frame would put two ground-truth boxes with the same id in "
                "the same frame, which is not legal MOT"
            )
        if not 0.0 <= self.drift_max <= 1.0:
            raise ValueError(f"drift_max must be in [0, 1], got {self.drift_max}")
        if not 0.0 <= self.return_seat_change_prob <= 1.0:
            raise ValueError("return_seat_change_prob must be in [0, 1]")
        if self.frame_w < 1 or self.frame_h < 1:
            raise ValueError("frame_w and frame_h must be >= 1 pixel")
        if not 0.0 < self.box_fill <= 1.0:
            raise ValueError(f"box_fill must be in (0, 1], got {self.box_fill}")
        if self.sway_px < 0.0:
            raise ValueError(f"sway_px must be >= 0, got {self.sway_px}")

    @property
    def n_total_identities(self) -> int:
        """Enrolled plus distractor plus new. All of them share one identity bank."""
        return self.n_identities + self.n_distractor_identities + self.n_new_identities

    @property
    def enrolled_identities(self) -> tuple[int, ...]:
        return tuple(range(self.n_identities))

    @property
    def distractor_identities(self) -> tuple[int, ...]:
        return tuple(range(self.n_identities, self.n_identities + self.n_distractor_identities))

    @property
    def new_identities(self) -> tuple[int, ...]:
        first = self.n_identities + self.n_distractor_identities
        return tuple(range(first, first + self.n_new_identities))


@dataclass(frozen=True)
class TrackCutScenario:
    """A built leave-and-return benchmark: tracklets plus the MOT ground truth beside them.

    The four tracklet groups are flat across gap tiers; filter with `gap_s` (or the
    `*_for_gap` helpers) to score one tier. `gt_mot` maps sequence name to the tier's
    ground-truth rows, ready for `vividly.eval.mot.write_gt_file`.

    Tuples, not lists: the dataclass is frozen and a frozen container of mutable lists is a
    lie. Iteration, indexing, `len` and concatenation all behave the same; call `list()` if
    a caller needs to mutate.
    """

    enrollment: tuple[Tracklet, ...]
    returns: tuple[Tracklet, ...]
    distractors: tuple[Tracklet, ...]
    new_identity_probes: tuple[Tracklet, ...]
    #: Sequence name -> ground-truth rows. Convention: (frame, id, x, y, w, h, 1, 1, 1),
    #: 1-based frames and ids. Treat as read-only; mutating it invalidates `scenario_digest`.
    gt_mot: dict[str, tuple[MotRow, ...]]
    config: ScenarioConfig

    @property
    def sequences(self) -> tuple[str, ...]:
        """Sequence names in gap-tier order (not sorted order)."""
        return tuple(sequence_name(gap) for gap in self.config.gap_tiers_s)

    def enrollment_for_gap(self, gap_s: float) -> tuple[Tracklet, ...]:
        return _for_gap(self.enrollment, gap_s)

    def returns_for_gap(self, gap_s: float) -> tuple[Tracklet, ...]:
        return _for_gap(self.returns, gap_s)

    def distractors_for_gap(self, gap_s: float) -> tuple[Tracklet, ...]:
        return _for_gap(self.distractors, gap_s)

    def new_probes_for_gap(self, gap_s: float) -> tuple[Tracklet, ...]:
        return _for_gap(self.new_identity_probes, gap_s)


def _for_gap(tracklets: Sequence[Tracklet], gap_s: float) -> tuple[Tracklet, ...]:
    return tuple(t for t in tracklets if t.gap_s is not None and math.isclose(t.gap_s, gap_s))


def sequence_name(gap_s: float) -> str:
    """MOT sequence name for a gap tier. Zero-padded so lexical order is numeric order."""
    return f"gap_{round(float(gap_s)):05d}s"


# ---------------------------------------------------------------------------
# geometry: seats and boxes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _SeatCell:
    """One seat: where its box sits and how big it is. Pure geometry, no identity."""

    center_x: float
    center_y: float
    width: float
    height: float


def _seat_grid(n_seats: int, cfg: ScenarioConfig) -> tuple[_SeatCell, ...]:
    """A near-square grid of `n_seats` cells covering the frame, in row-major order."""
    cols = max(1, math.ceil(math.sqrt(n_seats)))
    rows = max(1, math.ceil(n_seats / cols))
    cell_w = cfg.frame_w / cols
    cell_h = cfg.frame_h / rows
    return tuple(
        _SeatCell(
            center_x=(seat % cols + 0.5) * cell_w,
            center_y=(seat // cols + 0.5) * cell_h,
            width=cell_w * cfg.box_fill,
            height=cell_h * cfg.box_fill,
        )
        for seat in range(n_seats)
    )


def _box_at(
    cell: _SeatCell, elapsed_s: float, identity: int, cfg: ScenarioConfig
) -> tuple[float, float, float, float]:
    """(left, top, width, height), swayed, clamped inside the frame, rounded for stability."""
    phase = (identity % 8) * (math.pi / 4.0)
    angle = 2.0 * math.pi * elapsed_s / cfg.sway_period_s + phase
    left = cell.center_x - cell.width / 2.0 + cfg.sway_px * math.sin(angle)
    top = cell.center_y - cell.height / 2.0 + cfg.sway_px * math.cos(angle)
    left = min(max(left, 0.0), max(0.0, cfg.frame_w - cell.width))
    top = min(max(top, 0.0), max(0.0, cfg.frame_h - cell.height))
    return (
        round(left, _BOX_DECIMALS),
        round(top, _BOX_DECIMALS),
        round(cell.width, _BOX_DECIMALS),
        round(cell.height, _BOX_DECIMALS),
    )


def _gt_rows_for(tracklet: Tracklet, cell: _SeatCell, cfg: ScenarioConfig) -> list[MotRow]:
    """MOT ground-truth rows for one tracklet: one box per sampled instant.

    The MOT id is the identity, 1-based. A cut track therefore keeps one ground-truth id
    across its gap, which is what makes HOTA AssA and IDF1 measure re-association.
    """
    rows: list[MotRow] = []
    for timestamp in tracklet.timestamps:
        t = float(timestamp)
        left, top, width, height = _box_at(cell, t - tracklet.t0, tracklet.identity, cfg)
        rows.append(
            MotRow(
                frame=round(t * cfg.fps) + 1,
                track_id=tracklet.identity + 1,
                left=left,
                top=top,
                width=width,
                height=height,
                conf=float(GT_CONSIDER),
                class_id=PEDESTRIAN_CLASS_ID,
                visibility=1.0,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# scenario construction
# ---------------------------------------------------------------------------
def identity_bank(cfg: ScenarioConfig) -> np.ndarray:
    """The exact identity bank `build_scenario(cfg)` uses, in identity order.

    Exposed so an experiment can quote an oracle ceiling. It is not a gallery: scoring
    against the centroids that generated the samples measures the fixture, not a system.
    """
    return make_identity_bank(cfg.n_total_identities, cfg.dim, cfg.separability, cfg.seed)


def _sample_count(duration_s: float, sample_hz: float) -> int:
    return max(1, round(duration_s * sample_hz))


def _timestamps(t0: float, count: int, sample_hz: float) -> np.ndarray:
    return t0 + np.arange(count, dtype=np.float64) / sample_hz


def build_scenario(cfg: ScenarioConfig) -> TrackCutScenario:
    """Build the whole leave-and-return benchmark. Deterministic: same config, same bytes.

    One self-contained MOT sequence per gap tier (enrollment window, gap, return window),
    with distractors strictly inside every gap and new-identity probes sharing the return
    window so the open-set decision cannot be made on timing alone.
    """
    bank = identity_bank(cfg)
    seats = _seat_grid(max(1, cfg.n_total_identities), cfg)
    n_seats = len(seats)

    enrollment: list[Tracklet] = []
    returns: list[Tracklet] = []
    distractors: list[Tracklet] = []
    new_probes: list[Tracklet] = []
    gt_mot: dict[str, tuple[MotRow, ...]] = {}

    # Fragment ids are handed out in one deterministic traversal: tier order, then
    # enrollment, distractors, returns, new probes, each in identity order.
    next_track_id = 1
    n_enroll = _sample_count(cfg.enrollment_s, cfg.sample_hz)
    n_return = _sample_count(cfg.return_s, cfg.sample_hz)

    for tier_index, gap_s in enumerate(cfg.gap_tiers_s):
        sequence = sequence_name(gap_s)
        gap_start = cfg.enrollment_s
        gap_end = gap_start + gap_s
        rows: list[MotRow] = []
        seat_of: dict[int, int] = {}

        # ---- enrollment ---------------------------------------------------
        for identity in cfg.enrolled_identities:
            seat = identity % n_seats
            seat_of[identity] = seat
            tracklet = Tracklet(
                identity=identity,
                track_id=next_track_id,
                t0=0.0,
                embeddings=sample_embeddings(
                    bank, identity, n_enroll, cfg.noise, _mix(cfg.seed, _STREAM_ENROLL, tier_index)
                ),
                timestamps=_timestamps(0.0, n_enroll, cfg.sample_hz),
                gap_s=float(gap_s),
                role="enrollment",
                sequence=sequence,
                seat=seat,
            )
            next_track_id += 1
            enrollment.append(tracklet)
            rows.extend(_gt_rows_for(tracklet, seats[seat], cfg))

        # ---- distractors, strictly inside the gap --------------------------
        for offset, identity in enumerate(cfg.distractor_identities):
            t0, count = _distractor_window(cfg, gap_s, offset, len(cfg.distractor_identities))
            seat = identity % n_seats
            tracklet = Tracklet(
                identity=identity,
                track_id=next_track_id,
                t0=t0,
                embeddings=sample_embeddings(
                    bank,
                    identity,
                    count,
                    cfg.noise,
                    _mix(cfg.seed, _STREAM_DISTRACTOR, tier_index),
                ),
                timestamps=_timestamps(t0, count, cfg.sample_hz),
                gap_s=float(gap_s),
                role="distractor",
                sequence=sequence,
                seat=seat,
            )
            next_track_id += 1
            distractors.append(tracklet)
            rows.extend(_gt_rows_for(tracklet, seats[seat], cfg))

        # ---- returns, after a gap that cost the appearance something -------
        drift = drift_fraction(float(gap_s), cfg.drift_max, cfg.drift_half_gap_s)
        returned_bank = drifted_bank(bank, drift, _mix(cfg.seed, _STREAM_DRIFT, tier_index))
        return_seats = _return_seats(cfg, seat_of, n_seats, tier_index)
        for identity in cfg.enrolled_identities:
            seat = return_seats[identity]
            tracklet = Tracklet(
                identity=identity,
                track_id=next_track_id,
                t0=gap_end,
                embeddings=sample_embeddings(
                    returned_bank,
                    identity,
                    n_return,
                    cfg.noise,
                    _mix(cfg.seed, _STREAM_RETURN, tier_index),
                ),
                timestamps=_timestamps(gap_end, n_return, cfg.sample_hz),
                gap_s=float(gap_s),
                role="return",
                sequence=sequence,
                seat=seat,
            )
            next_track_id += 1
            returns.append(tracklet)
            rows.extend(_gt_rows_for(tracklet, seats[seat], cfg))

        # ---- new identities, in the same window as the returns -------------
        # Drawn from the DRIFTED bank, like the returns. Drift is a property of when an
        # appearance was observed, not of the person: everything in the return window is
        # seen under return-window conditions. Sampling new identities from the undrifted
        # bank instead would make them systematically closer to the enrollment gallery than
        # a returning patient is, which pins the false-merge rate at zero by construction
        # and makes the headline patient-safety KPI untestable.
        for identity in cfg.new_identities:
            seat = identity % n_seats
            tracklet = Tracklet(
                identity=identity,
                track_id=next_track_id,
                t0=gap_end,
                embeddings=sample_embeddings(
                    returned_bank,
                    identity,
                    n_return,
                    cfg.noise,
                    _mix(cfg.seed, _STREAM_NEW, tier_index),
                ),
                timestamps=_timestamps(gap_end, n_return, cfg.sample_hz),
                gap_s=float(gap_s),
                role="new_identity",
                sequence=sequence,
                seat=seat,
            )
            next_track_id += 1
            new_probes.append(tracklet)
            rows.extend(_gt_rows_for(tracklet, seats[seat], cfg))

        gt_mot[sequence] = tuple(sorted(rows, key=lambda row: (row.frame, row.track_id)))

    return TrackCutScenario(
        enrollment=tuple(enrollment),
        returns=tuple(returns),
        distractors=tuple(distractors),
        new_identity_probes=tuple(new_probes),
        gt_mot=gt_mot,
        config=cfg,
    )


def _distractor_window(
    cfg: ScenarioConfig, gap_s: float, index: int, total: int
) -> tuple[float, int]:
    """Start time and sample count for one distractor, strictly inside the gap.

    The nominal duration is shortened to half the usable window on short tiers, and the
    starts are spread evenly, so a 30 s gap and a 2 h gap both hold every distractor without
    either touching a cut edge.
    """
    margin = min(cfg.gap_margin_s, gap_s * 0.1)
    inner_start = cfg.enrollment_s + margin
    inner_width = max(gap_s - 2.0 * margin, 0.0)
    duration = min(cfg.distractor_s, inner_width * 0.5)
    count = _sample_count(duration, cfg.sample_hz)
    covered = (count - 1) / cfg.sample_hz
    slack = max(inner_width - covered, 0.0)
    t0 = inner_start + slack * (index + 1) / (total + 1)
    return t0, count


def _return_seats(
    cfg: ScenarioConfig, seat_of: dict[int, int], n_seats: int, tier_index: int
) -> dict[int, int]:
    """Where each enrolled identity sits after the gap. Mostly the same seat, sometimes not."""
    seats = dict(seat_of)
    if cfg.return_seat_change_prob <= 0.0 or n_seats < 2:
        return seats
    rng = np.random.default_rng(_mix(cfg.seed, _STREAM_SEAT, tier_index))
    draws = rng.random(len(seat_of))
    steps = rng.integers(1, n_seats, size=len(seat_of))
    for index, identity in enumerate(sorted(seat_of)):
        if float(draws[index]) < cfg.return_seat_change_prob:
            seats[identity] = int((seat_of[identity] + int(steps[index])) % n_seats)
    return seats


# ---------------------------------------------------------------------------
# consumption helpers
# ---------------------------------------------------------------------------
def stack_embeddings(
    tracklets: Sequence[Tracklet], label: Literal["identity", "track_id"] = "identity"
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate tracklets into `(n_rows, dim)` float32 embeddings and `(n_rows,)` int64 ids.

    Row order follows `tracklets`, and within a tracklet its own sample order. `label`
    chooses what the ids mean: `identity` builds the gallery or probe labelling every metric
    in `vividly.reid.metrics` expects, `track_id` keeps the fragments separable for
    tracklet-level aggregation. An empty input returns empty arrays, which those metrics
    reject on purpose: an empty gallery is a construction bug, not a score of zero.
    """
    if not tracklets:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    dims = {t.dim for t in tracklets}
    if len(dims) != 1:
        raise ValueError(f"tracklets carry mixed embedding dims {sorted(dims)}")
    embeddings = np.concatenate([t.embeddings for t in tracklets], axis=0).astype(
        np.float32, copy=False
    )
    ids = np.concatenate(
        [
            np.full(t.n_samples, t.identity if label == "identity" else t.track_id, dtype=np.int64)
            for t in tracklets
        ]
    )
    return embeddings, ids


def scenario_digest(scenario: TrackCutScenario) -> str:
    """sha256 over every byte a scenario carries. Two builds agree iff nothing moved.

    Record it next to any number computed from a scenario: it is the provenance handle that
    says which fixture produced the row, the POLICY.md, rule 8 discipline applied to a
    fixture.
    """
    digest = hashlib.sha256()
    groups = (
        ("enrollment", scenario.enrollment),
        ("returns", scenario.returns),
        ("distractors", scenario.distractors),
        ("new_identity_probes", scenario.new_identity_probes),
    )
    for name, tracklets in groups:
        digest.update(f"[{name}:{len(tracklets)}]".encode())
        for tracklet in tracklets:
            digest.update(
                (
                    f"{tracklet.identity}|{tracklet.track_id}|{tracklet.t0!r}|{tracklet.role}|"
                    f"{tracklet.sequence}|{tracklet.gap_s!r}|{tracklet.seat}|"
                    f"{tracklet.embeddings.shape}|"
                ).encode()
            )
            digest.update(tracklet.embeddings.tobytes())
            digest.update(tracklet.timestamps.tobytes())
    for sequence in sorted(scenario.gt_mot):
        rows = scenario.gt_mot[sequence]
        digest.update(f"[{sequence}:{len(rows)}]".encode())
        for row in rows:
            digest.update(
                (
                    f"{row.frame},{row.track_id},{row.left:.4f},{row.top:.4f},"
                    f"{row.width:.4f},{row.height:.4f},{row.conf:g},{row.class_id:d},"
                    f"{row.visibility:g};"
                ).encode()
            )
    return digest.hexdigest()
