"""Open-set re-identification metrics for the leave-and-return benchmark.

Pure numpy. No sklearn, no torch, no cv2, no model objects: every function here takes two
float matrices plus identity labels, so the same code scores synthetic fixtures on a laptop
today and real OSNet embeddings on a GPU box later without a line changing.

Why this module exists, and what it leads with. The headline patient-safety measure of
every re-ID stage is the false-merge rate: a false merge joins two patients' identities, and
in a chart-bound deployment that is a patient-safety event, not an accuracy point. So
`false_merge_rate` is a first-class function with its own definition rather than something
squeezed out of a confusion matrix at the end of a table. Rank-1 is the closed-set sanity
number. DIR and FAR are the only pair that answer the question the product actually asks,
which is whether this person is one of the people already seen this visit or somebody new.

Definitions (open-set identification, following Liao et al.):

  * A gallery is (n_embeddings, dim) with one identity label per row, not per identity.
    Several rows per identity is the normal case, not an exception: an enrollment tracklet
    contributes many views of the same person.
  * A probe's score against an identity is the maximum cosine over that identity's rows,
    and the probe's prediction is the identity with the highest such score.
  * DIR(tau) = (return probes whose top-1 identity is correct and scores >= tau) / (all
    return probes).
  * FAR(tau) = (new-identity probes whose top-1 score is >= tau) / (all new-identity
    probes). A new-identity probe has no mate in the gallery, so any acceptance at all is a
    false alarm.
  * false-merge rate(tau) = (return probes whose top-1 identity is wrong and scores >= tau)
    / (all return probes).

DIR and the false-merge rate are not complements: a probe can be rejected (top-1 below tau)
and count in neither. That gap is the whole point of an open-set threshold, and it is why
the two are reported side by side.

Why maximum aggregation rather than mean. A gallery holds multiple views of one person:
different frames, postures, and lighting inside a single enrollment tracklet. The deployment
decision rule asks whether this probe matches any enrolled view of this person, so the metric
must ask the same thing. Mean punishes an identity for its
worst views, and punishes it more the more views it has, which is exactly backwards: a
longer enrollment should help. The fixture in tests/test_reid_metrics.py makes the
difference concrete. A probe that is a near-exact match to one of identity 10's two views
scores a maximum of 0.96 (correct) but a mean of (0.6 + 0.96) / 2 = 0.78, which loses to
identity 20's single view at 0.80 and produces a false merge. The maximum is also the
standard single-shot-to-multi-shot re-ID rule, so numbers stay comparable with the
literature.

No thresholds live here. Whatever a benchmark treats as a pass or a fail belongs in that
benchmark's own config, not in a metric function: this module computes, and the config
decides. No re-identification benchmark config exists in this repository yet, and any
criteria written into one would be PROVISIONAL under the standing threshold discipline
(docs/CALIBRATION.md describes the protocol that would replace such placeholders).

Precision. Similarities are computed in float64 even when the embeddings arrive as float32.
Metric values are compared against thresholds and against each other, and the cost of the
wider accumulation on benchmark-sized matrices is irrelevant next to a tie flipping under a
different BLAS build. `match_latency` is the one exception: it times whatever dtype it is
handed, because the point of that function is to report the production path's cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Guards the division in row normalisation. A zero row stays zero instead of becoming NaN:
#: a NaN score poisons every comparison it touches, a zero one simply matches nothing.
#: Same value and same reasoning as `vividly.models.reid_backend.l2_normalise`, deliberately
#: duplicated rather than imported: that function is float32 and lives in the inference
#: tree, and this module depends on nothing but numpy.
_L2_EPS = 1e-12


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _as_matrix(x: Any, name: str, *, cast: bool = True) -> np.ndarray:
    """Validate an (n, dim) embedding matrix and return it as an array.

    `cast=False` preserves the caller's dtype, which only `match_latency` wants.
    """
    arr = np.asarray(x, dtype=np.float64) if cast else np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D (n_embeddings, dim) array, got shape {arr.shape}. "
            "A single embedding is a (1, dim) matrix, not a (dim,) vector."
        )
    if arr.shape[0] == 0:
        raise ValueError(
            f"{name} is empty (0 rows). Nothing here is defined over an empty {name}: an "
            "empty probe set has no ratio to report and an empty gallery has nothing to "
            "match against. That is a benchmark construction bug, not a score of 0."
        )
    if arr.shape[1] == 0:
        raise ValueError(f"{name} has zero columns: embeddings need at least one dimension")
    if not np.isfinite(arr).all():
        raise ValueError(
            f"{name} contains NaN or inf. A non-finite embedding does not fail loudly "
            "downstream: it silently loses every comparison it takes part in."
        )
    return arr


def _as_ids(x: Any, name: str, n_rows: int) -> np.ndarray:
    """Validate a 1-D integer identity-label array aligned with `n_rows` rows."""
    ids = np.asarray(x)
    if ids.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array of identity labels, got shape {ids.shape}")
    if ids.dtype.kind not in "iu":
        raise ValueError(
            f"{name} must have an integer dtype, got {ids.dtype}. Identity labels are "
            "compared for exact equality; float labels make that comparison a coin flip."
        )
    if ids.shape[0] != n_rows:
        raise ValueError(
            f"{name} has {ids.shape[0]} labels for {n_rows} rows. A gallery is labelled "
            "per ROW, not per identity: an enrollment tracklet contributes many rows "
            "carrying the same label."
        )
    return ids


def _check_same_dim(a: np.ndarray, b: np.ndarray, a_name: str, b_name: str) -> None:
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"{a_name} has dim {a.shape[1]} but {b_name} has dim {b.shape[1]}. "
            "Both must come from the same embedder."
        )


def _check_enrolled(probe_ids: np.ndarray, identities: np.ndarray) -> None:
    """Every probe identity must appear in the gallery.

    Rank-1, DIR and the false-merge rate are all defined over RETURN probes, whose whole
    premise is that the person was enrolled before the gap. A probe whose identity is
    missing from the gallery can never be scored correct, so allowing it would deflate
    every number silently. New-identity probes have their own argument (`new_probes` in
    `dir_far_curve`) and their own metric (FAR); they are never mixed in here.
    """
    missing = np.setdiff1d(probe_ids, identities)
    if missing.size:
        raise ValueError(
            f"probe identities {missing.tolist()} do not appear in gallery_ids. These "
            "metrics are defined over return probes (enrolled before the gap). Score "
            "never-enrolled probes with dir_far_curve's `new_probes` argument, which "
            "reports them as FAR."
        )


# ---------------------------------------------------------------------------
# similarity and aggregation
# ---------------------------------------------------------------------------
def _unit_rows(m: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. A zero row is returned unchanged (still zero)."""
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), _L2_EPS)


def cosine_similarity(a: Any, b: Any) -> np.ndarray:
    """(n_a, n_b) float64 cosine similarities between the rows of `a` and the rows of `b`.

    Rows are re-normalised here rather than assumed to be unit length. The synthetic
    fixtures and the ONNX embedder both emit L2-normalised rows, in which case this is a
    no-op to within float error, but a caller who passes raw features gets a true cosine
    instead of an unbounded dot product. A zero row scores 0 against everything.
    """
    left = _unit_rows(_as_matrix(a, "a"))
    right = _unit_rows(_as_matrix(b, "b"))
    _check_same_dim(left, right, "a", "b")
    return left @ right.T


def identity_scores(gallery: Any, gallery_ids: Any, probes: Any) -> tuple[np.ndarray, np.ndarray]:
    """Per-identity probe scores. Returns (identities, scores).

    `identities` is the ascending unique gallery labels, (n_identities,). `scores` is
    (n_probes, n_identities), where scores[i, k] is the MAXIMUM cosine between probe i and
    any gallery row labelled identities[k]. See the module docstring for why max.

    Gallery row order does not matter: rows are grouped by label, not by position.
    """
    probe_matrix = _as_matrix(probes, "probes")
    gallery_matrix = _as_matrix(gallery, "gallery")
    _check_same_dim(probe_matrix, gallery_matrix, "probes", "gallery")
    ids = _as_ids(gallery_ids, "gallery_ids", gallery_matrix.shape[0])

    sims = cosine_similarity(probe_matrix, gallery_matrix)
    identities = np.unique(ids)
    scores = np.empty((probe_matrix.shape[0], identities.shape[0]), dtype=np.float64)
    for k, identity in enumerate(identities):
        scores[:, k] = sims[:, ids == identity].max(axis=1)
    return identities, scores


def _top1(gallery: Any, gallery_ids: Any, probes: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(identities, predicted_ids, top1_scores). The one implementation of the decision rule."""
    identities, scores = identity_scores(gallery, gallery_ids, probes)
    best = scores.argmax(axis=1)
    rows = np.arange(scores.shape[0])
    return identities, identities[best], scores[rows, best]


def top1_match(gallery: Any, gallery_ids: Any, probes: Any) -> tuple[np.ndarray, np.ndarray]:
    """The decision rule every metric in this module shares. Returns (predicted_ids, scores).

    `predicted_ids[i]` is the highest-scoring identity for probe i and `scores[i]` is that
    identity's score. This is the single place the top-1 rule is implemented, so DIR, the
    false-merge rate and Rank-1 can never disagree about what the system would have done.

    TIES. `np.argmax` returns the first maximum, and `identity_scores` orders identities
    ascending, so an exact tie resolves to the SMALLER identity label. The rule is
    deterministic and cannot see the probe's true label, so it neither flatters nor
    penalises the metric; it just has to be written down, because a tie is common with
    synthetic embeddings and rare-to-never with real ones.
    """
    _, predicted, scores = _top1(gallery, gallery_ids, probes)
    return predicted, scores


# ---------------------------------------------------------------------------
# closed set
# ---------------------------------------------------------------------------
def rank1(gallery: Any, gallery_ids: Any, probes: Any, probe_ids: Any) -> float:
    """Closed-set Rank-1 accuracy: the fraction of probes whose top-1 identity is correct.

    Closed-set means no threshold and no reject option: every probe is assigned to its
    best identity, so this is the ceiling that DIR approaches as tau falls. It answers
    "can the embedding tell these people apart at all", never "should the system have
    bound this track to a chart" - that question needs `dir_far_curve` and
    `false_merge_rate`.
    """
    identities, predicted, _ = _top1(gallery, gallery_ids, probes)
    truth = _as_ids(probe_ids, "probe_ids", predicted.shape[0])
    _check_enrolled(truth, identities)
    return float(np.mean(predicted == truth))


# ---------------------------------------------------------------------------
# open set
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class DirFarCurve:
    """An exact open-set operating curve: three aligned arrays, one row per threshold.

    `taus` is strictly increasing. `dirs` and `fars` are non-increasing in tau (raising a
    threshold can only reject probes, never accept new ones), so the curve runs from
    (FAR high, DIR high) down to (0, 0).

    eq=False because the fields are arrays: the dataclass-generated `__eq__` would compare
    them elementwise and raise on the ambiguous truth value. Compare the arrays.
    """

    taus: np.ndarray
    dirs: np.ndarray
    fars: np.ndarray
    n_probes: int
    n_new_probes: int

    def __post_init__(self) -> None:
        if not (self.taus.shape == self.dirs.shape == self.fars.shape):
            raise ValueError(
                f"curve arrays must align: taus {self.taus.shape}, dirs {self.dirs.shape}, "
                f"fars {self.fars.shape}"
            )
        if self.n_probes < 1 or self.n_new_probes < 1:
            raise ValueError("a DIR/FAR curve needs at least one probe and one new-identity probe")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, for the benchmark's results file."""
        return {
            "taus": self.taus.tolist(),
            "dirs": self.dirs.tolist(),
            "fars": self.fars.tolist(),
            "n_probes": self.n_probes,
            "n_new_probes": self.n_new_probes,
        }


@dataclass(frozen=True)
class OperatingPoint:
    """One chosen threshold and what it costs: `far` is what was ACHIEVED, not requested.

    Achieved FAR is almost always strictly below the target, because the curve is a step
    function over observed scores and there is no interpolation. `far_target` is carried
    along so a results file records the request next to the outcome.
    """

    tau: float
    dir: float
    far: float
    far_target: float = float("nan")

    def to_dict(self) -> dict[str, float]:
        return {"tau": self.tau, "dir": self.dir, "far": self.far, "far_target": self.far_target}


def dir_far_curve(
    gallery: Any,
    gallery_ids: Any,
    probes: Any,
    probe_ids: Any,
    new_probes: Any,
) -> DirFarCurve:
    """Sweep tau over every observed score and return the exact DIR/FAR step function.

    `probes`/`probe_ids` are RETURN probes: people enrolled in the gallery before the gap.
    `new_probes` are probes whose identity is not in the gallery at all, so any acceptance
    of one is a false alarm and they need no labels.

    THE SWEEP. Candidate thresholds are the union of the observed decision scores: every
    return probe's top-1 score and every new probe's top-1 score. Every distinct achievable
    (FAR, DIR) pair therefore appears exactly once, and nothing is interpolated - the arrays
    ARE the step function, so reading an operating point off them cannot invent a threshold
    the data never justified. One extra threshold is appended above the largest observed
    score (`np.nextafter(max, inf)`, the smallest float strictly greater than it), which is
    the only way the (FAR=0, DIR=0) endpoint exists at all; without it `dir_at_far(curve, 0)`
    would be undefined whenever a new-identity probe holds the highest score in the run.
    That appended row is the single documented departure from "taus are observed scores".

    TIES. Acceptance is `score >= tau`, not `>`. Since every observed score is itself a
    candidate tau, at tau equal to some probe's score that probe is still accepted, and the
    row for that tau is exactly the operating point a deployment running `>= tau` would
    have. `false_merge_rate` uses the same comparison, so a false-merge number quoted at an
    operating point from this curve is the number that operating point actually produces.
    """
    identities, predicted, scores = _top1(gallery, gallery_ids, probes)
    truth = _as_ids(probe_ids, "probe_ids", predicted.shape[0])
    _check_enrolled(truth, identities)

    new_matrix = _as_matrix(new_probes, "new_probes")
    _, _, new_scores = _top1(gallery, gallery_ids, new_matrix)

    correct_scores = np.sort(scores[predicted == truth])
    new_sorted = np.sort(new_scores)

    observed = np.unique(np.concatenate([scores, new_scores]))
    taus = np.concatenate([observed, [np.nextafter(observed[-1], np.inf)]])

    # count(score >= tau) = n - searchsorted(sorted_scores, tau, "left"), which is exact
    # integer arithmetic: no tolerance, no interpolation, no float comparison of ratios.
    n_probes = int(scores.shape[0])
    n_new = int(new_scores.shape[0])
    dirs = (correct_scores.shape[0] - np.searchsorted(correct_scores, taus, side="left")) / n_probes
    fars = (n_new - np.searchsorted(new_sorted, taus, side="left")) / n_new
    return DirFarCurve(taus=taus, dirs=dirs, fars=fars, n_probes=n_probes, n_new_probes=n_new)


def _index_at_far(curve: DirFarCurve, far: float) -> int:
    """Row index of the operating point that meets a FAR budget with the highest DIR."""
    target = float(far)
    if not 0.0 <= target <= 1.0:
        raise ValueError(f"far target must be in [0, 1], got {far}")
    eligible = curve.fars <= target
    if not eligible.any():  # pragma: no cover - the appended tau always has FAR 0
        raise ValueError(f"no threshold on this curve reaches FAR <= {target}")
    return int(np.argmax(eligible))


def dir_at_far(curve: DirFarCurve, far: float) -> tuple[float, float]:
    """DIR at the FAR budget, and the threshold that buys it. Returns (dir, tau).

    Both DIR and FAR fall as tau rises, so the whole family of thresholds meeting a budget
    is an upper interval [tau*, inf), and DIR is maximised at its lower end. This returns
    tau*: the SMALLEST tau whose FAR is within budget, which is the largest DIR available
    at that budget. Taking the largest eligible tau instead would always select the
    appended top row, where DIR is 0 by construction; what this implements is the DIR@FAR
    operating point of Liao et al., cited in the module docstring.

    The budget is a hard ceiling with no epsilon slack: FAR is a ratio of exact integer
    counts, and this is the number a patient-safety threshold gets set against. Achieved
    FAR is usually strictly below the target because the curve is a step function; use
    `operating_point` when the results file needs to record what was actually achieved.
    """
    i = _index_at_far(curve, far)
    return float(curve.dirs[i]), float(curve.taus[i])


def operating_point(curve: DirFarCurve, far_target: float) -> OperatingPoint:
    """`dir_at_far` plus the FAR actually achieved, as one record for a results file."""
    i = _index_at_far(curve, far_target)
    return OperatingPoint(
        tau=float(curve.taus[i]),
        dir=float(curve.dirs[i]),
        far=float(curve.fars[i]),
        far_target=float(far_target),
    )


def false_merge_rate(
    gallery: Any,
    gallery_ids: Any,
    probes: Any,
    probe_ids: Any,
    tau: float,
) -> float:
    """THE patient-safety KPI: return probes bound to the WRONG enrolled identity at tau.

    Numerator: probes whose top-1 identity is not their own AND scores >= tau, which is
    exactly the set the system would have merged into another patient's chart. Denominator:
    all return probes, so the number is directly comparable across gap tiers and is not
    inflated by how many probes happened to clear the threshold.

    A wrong identity scoring above tau is not a merge when the correct identity outranks
    it: the system binds to its top-1, so the runner-up never touches a chart. This
    mirrors the deployment decision rule rather than a set-membership rule, because the
    KPI has to count events that could actually happen.

    Comparison is `>= tau`, matching `dir_far_curve`, so quoting this at a tau taken from
    a curve gives the rate that curve's operating point implies.
    """
    identities, predicted, scores = _top1(gallery, gallery_ids, probes)
    truth = _as_ids(probe_ids, "probe_ids", predicted.shape[0])
    _check_enrolled(truth, identities)
    merged = (scores >= float(tau)) & (predicted != truth)
    return float(np.mean(merged))


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------
def match_latency(gallery: Any, probes: Any, repeats: int = 3) -> float:
    """Seconds per probe to match against this gallery. MACHINE-RELATIVE, not a spec number.

    What is timed: the dense cosine product plus the top-1 search over gallery rows, which
    is the whole arithmetic cost of matching and the part that grows with gallery size.
    Row normalisation is done once BEFORE the timed region because it is an embed-time
    cost paid on the way in, not per match. Per-identity max aggregation is left out
    because it adds no arithmetic: it is a reduction over the matrix already computed.

    Dtype is preserved rather than promoted to float64 like the rest of this module, so
    passing float32 embeddings times the production path. Non-float input is timed as
    float64.

    `repeats` runs the measurement several times and the MINIMUM is reported, after one
    untimed warm-up call: the minimum is the least contaminated estimator here, since every
    source of noise on a shared machine (scheduling, thermal, BLAS thread spin-up) only
    ever adds time.

    This number is a property of the machine, the BLAS build and the array shapes on the
    day it was measured. Record it next to the accuracy numbers (every eval in this repo
    logs latency beside accuracy) and never quote it as a platform figure. POLICY.md, rule 9
    exists because such a number is otherwise repeated as an Orin figure.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    gallery_matrix = _as_matrix(gallery, "gallery", cast=False)
    probe_matrix = _as_matrix(probes, "probes", cast=False)
    _check_same_dim(probe_matrix, gallery_matrix, "probes", "gallery")

    dtype = np.promote_types(np.promote_types(gallery_matrix.dtype, probe_matrix.dtype), np.float32)
    if dtype.kind != "f" or dtype.itemsize > 8:
        dtype = np.dtype(np.float64)
    left = _unit_rows(probe_matrix.astype(dtype, copy=False)).astype(dtype, copy=False)
    right = _unit_rows(gallery_matrix.astype(dtype, copy=False)).astype(dtype, copy=False)

    (left @ right.T).argmax(axis=1)  # warm-up: BLAS thread pool, first-touch page faults
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        (left @ right.T).argmax(axis=1)
        best = min(best, time.perf_counter() - t0)
    return best / float(probe_matrix.shape[0])
