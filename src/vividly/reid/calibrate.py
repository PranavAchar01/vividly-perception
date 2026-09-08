"""Choosing tau for the session gallery: an operating point at a stated FAR budget.

`vividly.reid.gallery.GalleryConfig.tau` is the threshold below which no candidate identity
is even shown to a human. It ships as an invented placeholder, and this module is how it
stops being one: pick tau where the open-set false-accept rate meets a budget, then record
what that budget actually bought (DIR, achieved FAR, and the false-merge rate at that tau)
in the same file as the number itself, so nobody can quote the threshold without the
evidence travelling beside it.

Nothing here makes a threshold non-provisional. The default source is the synthetic
track-cut fixture, which is arithmetic and not footage: a tau chosen from it is a tau for a
made-up geometry, useful for wiring the pipeline and useless as evidence about people. A tau
becomes real when it is chosen from real recorded footage under a re-identification
calibration protocol, which docs/CALIBRATION.md does not yet contain, with the checkpoint
hash that produced the embeddings recorded next to it, which is the same rule that binds
every accuracy number in this repository to its checkpoint.
Every artifact this module writes says so at the top, and `provisional: true` is a field in
the emitted YAML rather than a comment, so a reader who only parses the file still sees it.

What it reuses, and the one thing it deliberately does not
==========================================================

The curve and the operating point come from `vividly.reid.metrics` (`dir_far_curve`,
`operating_point`, `false_merge_rate`), and the gallery comes from `build_gallery` below:
every enrollment row, plus gap distractors by default, because a waiting-room system enrols
everyone it sees. Acceptance is compared the one way everything here compares it, `>= tau`.
Building the gallery that way keeps tau on the gallery a deployment enrols into.

The probe is where this tool parts from a pooled measurement on purpose. A pooled probe is
one vector per return tracklet, the L2-normalised mean of its embeddings, which is what a
measurement about a whole returning track needs. But the only consumer of tau,
`SessionGallery.propose` (and `ReidStage._probe` feeding it), scores a single per-frame
`(dim,)` row. A tau chosen against pooled means would be a tau for an easier task than the
one deployed: a mean of thirty views is closer to its own gallery than any single view is,
so the pooled evidence for detection and identification rate, false-accept rate and
false-merge rate would flatter the number. So calibration defaults to `probe: single`,
scoring one per-frame row exactly as the gallery does, and the shipped config ships
`single`. `pooled` stays available (`ProbeMode`) for anyone reproducing a pooled operating
point, but it is not what the gallery runs and it is never the default.

Per tier, and pooled. A deployment runs one tau against every gap length at once, so the
recommended number comes from the pooled curve over all gap tiers' decisions
(`pooled_dir_far_curve` below, the only sweep that can express one threshold across four
independent galleries). The per-tier rows are reported next to it because they are what
shows the shape of the problem: a pooled tau that looks fine can be carried entirely by the
30-second tier while the 2-hour tier contributes nothing.

Captured arrays. `calibrate_tau` also takes a `ScoreArrays` bundle straight from a `.npz`,
which is how a tau gets chosen from real embeddings later without this module ever learning
to read a dataset. Producing that file is a crop-extraction step that the dataset registry
gates; this module reads five arrays and does arithmetic on them.
"""

from __future__ import annotations

import contextlib
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from vividly.reid import metrics as M
from vividly.reid.synth import (
    ScenarioConfig,
    TrackCutScenario,
    Tracklet,
    build_scenario,
    scenario_digest,
    sequence_name,
    stack_embeddings,
)

#: How a probe is formed for scoring. `single` is one probe per per-frame embedding row, which
#: is exactly what `vividly.reid.gallery.SessionGallery.propose` and `ReidStage._probe` score in
#: the deployed pipeline. `pooled` is one L2-normalised tracklet-mean probe per return/new
#: tracklet, a pooled construction and an easier task than the one the gallery runs.
#: Calibration defaults to `single` so the recommended tau matches deployment.
ProbeMode = Literal["single", "pooled"]

#: Goes at the top of every report and into the emitted YAML. It is stated here rather than
#: inherited, because a calibration file travels further than a report: it is meant to be
#: referenced by a pipeline config.
PROVISIONAL_NOTE = (
    "PROVISIONAL. This tau is an engineering placeholder, not a calibrated operating point. "
    "It becomes real only when it is chosen from real recorded footage under a "
    "re-identification calibration protocol, which docs/CALIBRATION.md does not yet "
    "contain, with the checkpoint hash that produced the embeddings recorded beside it. Do "
    "not quote it, do not put it in a presentation, and do not read any rate measured "
    "against it as a product number."
)

#: Emitted at the top of the generated YAML so a pipeline config referencing it is versioned.
CALIBRATION_SCHEMA_VERSION = 1


class CalibrationConfigError(ValueError):
    """The calibration config is internally inconsistent or names something unusable.

    It is a `ValueError`, so a caller that catches `ValueError` catches every config
    failure this module raises, including the ones raised while building the scenario.
    """


def _check_keys(raw: Mapping[str, Any], allowed: set[str], where: str) -> None:
    """Reject unknown keys. A silently ignored typo in a threshold is the worst failure mode."""
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise CalibrationConfigError(
            f"{where}: unknown key(s) {unknown}. Known keys are {sorted(allowed)}. A key that "
            "is ignored rather than refused turns a typo into a silently different threshold."
        )


@contextlib.contextmanager
def _as_config_error(where: str) -> Iterator[None]:
    """Re-raise a dataclass's own ValueError or TypeError as a `CalibrationConfigError`.

    Two things reach this module as the wrong exception type. `ScenarioConfig` lives in
    `vividly.reid.synth`, which knows nothing about this module and raises plain ValueError
    for every range violation; and a YAML value of the wrong type fails inside a comparison
    as a TypeError before the config dataclass can raise its own error. Both are config
    errors, and letting either escape would turn a typo in a threshold into a stack trace.

    A `CalibrationConfigError` (itself a ValueError) passes through unwrapped, so a message
    that already names its key is never prefixed twice.
    """
    try:
        yield
    except CalibrationConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise CalibrationConfigError(f"{where}: {exc}") from exc


def scenario_from_mapping(raw: Mapping[str, Any]) -> ScenarioConfig:
    """Build a `ScenarioConfig` from a config mapping, refusing unknown keys."""
    allowed = {f.name for f in fields(ScenarioConfig)}
    _check_keys(raw, allowed, "scenario")
    kwargs = dict(raw)
    with _as_config_error("scenario"):
        if "gap_tiers_s" in kwargs:
            kwargs["gap_tiers_s"] = tuple(float(gap) for gap in kwargs["gap_tiers_s"])
        return ScenarioConfig(**kwargs)


# ---------------------------------------------------------------------------
# the gallery and the pooled sweep
# ---------------------------------------------------------------------------
def build_gallery(
    scenario: TrackCutScenario, gap_s: float, *, include_distractors: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """The tier's gallery: every enrollment row, plus every gap-distractor row by default.

    Rows, not pooled tracklets: the gallery is where multi-view enrollment pays off, and
    `vividly.reid.metrics` takes the max over an identity's rows precisely so a longer
    enrollment can only help. Distractors are enrolled because a waiting room enrols
    everyone it sees, so a return probe that binds to one of them is a false merge like
    any other.
    """
    tracklets = list(scenario.enrollment_for_gap(gap_s))
    if include_distractors:
        tracklets += list(scenario.distractors_for_gap(gap_s))
    if not tracklets:
        raise CalibrationConfigError(f"gap tier {gap_s} has no enrollment tracklets")
    embeddings = np.concatenate(
        [np.asarray(t.embeddings, dtype=np.float64) for t in tracklets], axis=0
    )
    ids = np.concatenate(
        [np.full(t.n_samples, t.identity, dtype=np.int64) for t in tracklets], axis=0
    )
    return embeddings, ids


def pool_tracklet(tracklet: Tracklet) -> np.ndarray:
    """One `(dim,)` float64 probe vector for a tracklet: the L2-normalised mean of its rows.

    Mean-then-normalise, not normalise-then-mean: the rows are already unit length, so the
    mean is the direction the tracklet as a whole points in, and its shrunken norm carries
    only how much the samples disagreed. Cosine ignores norm, so the renormalisation is
    bookkeeping rather than a modelling choice. A tracklet whose samples cancel exactly is a
    degenerate fixture, not a score of zero, so it raises.
    """
    mean = np.asarray(tracklet.embeddings, dtype=np.float64).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= 1e-12:
        raise CalibrationConfigError(
            f"tracklet {tracklet.track_id} (identity {tracklet.identity}) pools to a "
            "zero-length vector: its samples cancel exactly, which no embedder produces. "
            "That is a fixture construction bug, not a probe that matches nothing."
        )
    return mean / norm


def pool_tracklets(tracklets: Sequence[Tracklet]) -> tuple[np.ndarray, np.ndarray]:
    """`(n_tracklets, dim)` pooled probes and their `(n_tracklets,)` int64 identity labels."""
    if not tracklets:
        raise CalibrationConfigError("cannot pool an empty tracklet sequence")
    matrix = np.stack([pool_tracklet(t) for t in tracklets], axis=0)
    ids = np.asarray([t.identity for t in tracklets], dtype=np.int64)
    return matrix, ids


def pooled_dir_far_curve(
    correct: np.ndarray, scores: np.ndarray, new_scores: np.ndarray
) -> M.DirFarCurve:
    """A `DirFarCurve` over already-taken decisions, for pooling tiers into one curve.

    `metrics.dir_far_curve` sweeps matrices, which cannot express "one tau across four
    independent galleries". The sweep itself is identical, down to the appended
    `nextafter` threshold that provides the (FAR=0, DIR=0) endpoint and the `>= tau`
    acceptance rule, so on a single tier the two curves agree.
    """
    scores = np.asarray(scores, dtype=np.float64)
    new_scores = np.asarray(new_scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    if scores.ndim != 1 or new_scores.ndim != 1:
        raise ValueError("scores and new_scores must be 1-D arrays of top-1 decision scores")
    if correct.shape != scores.shape:
        raise ValueError(
            f"correct has shape {correct.shape} for {scores.shape} scores; one flag per probe"
        )
    if scores.size == 0 or new_scores.size == 0:
        raise ValueError("a DIR/FAR curve needs at least one probe and one new-identity probe")

    correct_sorted = np.sort(scores[correct])
    new_sorted = np.sort(new_scores)
    observed = np.unique(np.concatenate([scores, new_scores]))
    taus = np.concatenate([observed, [np.nextafter(observed[-1], np.inf)]])
    n_probes = int(scores.shape[0])
    n_new = int(new_scores.shape[0])
    dirs = (correct_sorted.shape[0] - np.searchsorted(correct_sorted, taus, side="left")) / n_probes
    fars = (n_new - np.searchsorted(new_sorted, taus, side="left")) / n_new
    return M.DirFarCurve(taus=taus, dirs=dirs, fars=fars, n_probes=n_probes, n_new_probes=n_new)


def pooled_false_merge_rate(correct: np.ndarray, scores: np.ndarray, tau: float) -> float:
    """`metrics.false_merge_rate` over already-taken decisions. Same `>= tau` comparison."""
    correct = np.asarray(correct, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    return float(np.mean((scores >= float(tau)) & ~correct))


# ---------------------------------------------------------------------------
# the input bundle
# ---------------------------------------------------------------------------
#: The five arrays every open-set score needs, and the names they carry in a `.npz`.
SCORE_ARRAY_KEYS = ("gallery", "gallery_ids", "probes", "probe_ids", "new_probes")


@dataclass(frozen=True, eq=False)
class ScoreArrays:
    """One open-set scoring problem: a labelled gallery, labelled return probes, new probes.

    Exactly the five arguments `vividly.reid.metrics.dir_far_curve` takes, bundled so a
    caller cannot get their order wrong and so the same bundle can come from a synthetic
    scenario, from a `.npz` of captured real embeddings, or from a test fixture built by
    hand. Deep validation is left to the metrics functions, which already refuse empty
    matrices, mismatched dims, float identity labels and never-enrolled probe identities
    with messages that say what to do about it.

    `eq=False` because the fields are arrays: the generated `__eq__` would compare them
    elementwise and raise on the ambiguous truth value.
    """

    gallery: np.ndarray
    gallery_ids: np.ndarray
    probes: np.ndarray
    probe_ids: np.ndarray
    new_probes: np.ndarray

    @property
    def n_probes(self) -> int:
        return int(np.asarray(self.probes).shape[0])

    @property
    def n_new_probes(self) -> int:
        return int(np.asarray(self.new_probes).shape[0])

    @property
    def n_gallery_rows(self) -> int:
        return int(np.asarray(self.gallery).shape[0])


def score_arrays_from_npz(path: Path | str) -> ScoreArrays:
    """Load a `ScoreArrays` bundle from a `.npz` holding the five `SCORE_ARRAY_KEYS`.

    This is the seam for calibrating against captured real embeddings without this module
    ever reading a dataset: whatever produced the file went through the registry, and by the
    time it lands here it is five numeric arrays. A missing key raises rather than defaulting
    to something empty, because a calibration silently run without its new-identity probes
    would report FAR 0 at every threshold and recommend an unusably tight tau.
    """
    file = Path(path)
    if not file.is_file():
        raise CalibrationConfigError(f"score array file not found: {file}")
    # Every way a bad file can fail to parse is funnelled into CalibrationConfigError, a
    # ValueError, so a caller that maps ValueError onto a config-error exit code gets one for
    # every malformed score file. Without this, a raw .npy loads as a bare ndarray that has no
    # context-manager protocol (TypeError), and a truncated .npz raises zipfile.BadZipFile:
    # neither is a ValueError, so both would escape as an unhandled stack trace.
    try:
        loaded = np.load(file, allow_pickle=False)
    except (zipfile.BadZipFile, OSError, ValueError, EOFError) as exc:
        raise CalibrationConfigError(
            f"{file}: not a readable .npz score archive ({type(exc).__name__}: {exc}). It must "
            f"be an .npz holding {list(SCORE_ARRAY_KEYS)}; a truncated file is not one."
        ) from exc
    if isinstance(loaded, np.ndarray):
        # A .npy holds one anonymous array, not the five named ones a score bundle needs.
        raise CalibrationConfigError(
            f"{file}: expected an .npz archive of {list(SCORE_ARRAY_KEYS)}, got a single .npy "
            f"array of shape {loaded.shape}. Save the five arrays together with numpy.savez."
        )
    try:
        with loaded as data:
            missing = [key for key in SCORE_ARRAY_KEYS if key not in data]
            if missing:
                raise CalibrationConfigError(
                    f"{file}: missing array(s) {missing}; a score bundle needs all of "
                    f"{list(SCORE_ARRAY_KEYS)}"
                )
            return ScoreArrays(*(np.asarray(data[key]) for key in SCORE_ARRAY_KEYS))
    except CalibrationConfigError:
        raise
    except (zipfile.BadZipFile, OSError, ValueError, EOFError) as exc:
        # A member that only fails to decompress when it is read, not when the archive opens.
        raise CalibrationConfigError(
            f"{file}: unreadable .npz score archive ({type(exc).__name__}: {exc})"
        ) from exc


def scenario_score_arrays(
    scenario: TrackCutScenario,
    gap_s: float,
    *,
    include_distractors: bool = True,
    probe: ProbeMode = "single",
) -> ScoreArrays:
    """One gap tier of a synthetic scenario, as the arrays the metrics functions score.

    The gallery is the same either way (`build_gallery`: every enrollment row, plus gap
    distractors by default), so a tau chosen here is chosen against the gallery a deployment
    enrols into. The probes differ by `probe`, and the default is the one the pipeline
    actually runs:

      * ``single`` (default) scores one probe per per-frame embedding row, which is exactly
        what `SessionGallery.propose` and `ReidStage._probe` score: a single ``(dim,)`` vector,
        never a pooled mean. Calibrating any other way recommends a tau for an easier task than
        the deployed one, and the DIR, FAR and false-merge evidence written beside it would
        describe that easier task too.
      * ``pooled`` scores one L2-normalised tracklet-mean probe per return or new tracklet.
        Use it only to reproduce an operating point taken over pooled probes; it is not what
        the gallery scores.
    """
    if probe not in ("single", "pooled"):
        raise CalibrationConfigError(f"probe must be 'single' or 'pooled', got {probe!r}")
    gallery, gallery_ids = build_gallery(
        scenario, float(gap_s), include_distractors=include_distractors
    )
    returns = scenario.returns_for_gap(float(gap_s))
    news = scenario.new_probes_for_gap(float(gap_s))
    if not returns:
        raise CalibrationConfigError(f"gap tier {gap_s} has no return tracklets to score")
    if not news:
        raise CalibrationConfigError(
            f"gap tier {gap_s} has no new-identity probes, so there is no open-set FAR "
            "denominator and no threshold to quote a false-merge rate at"
        )
    if probe == "pooled":
        probes, probe_ids = pool_tracklets(returns)
        new_probes, _ = pool_tracklets(news)
    else:
        probes, probe_ids = stack_embeddings(returns, label="identity")
        new_probes, _ = stack_embeddings(news, label="identity")
    return ScoreArrays(
        gallery=gallery,
        gallery_ids=gallery_ids,
        probes=probes,
        probe_ids=probe_ids,
        new_probes=new_probes,
    )


# ---------------------------------------------------------------------------
# one operating point
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TauCalibration:
    """One chosen tau and everything needed to judge it. Never just the number.

    `far` is what the threshold ACHIEVED, which on a step curve is almost always strictly
    below `far_target`; both are carried so a reader sees the request next to the outcome.
    `false_merge_rate` is the headline patient-safety measure at this tau and leads every
    rendering of this record for the same reason it leads every re-identification table in
    this repository: a false merge joins two patients' identities.
    """

    #: What was scored: a gap tier's sequence name, "pooled", or "arrays" for a `.npz`.
    label: str
    tau: float
    dir: float
    far: float
    far_target: float
    false_merge_rate: float
    n_probes: int
    n_new_probes: int
    #: The gap tier in seconds when this row is one tier, None for pooled or captured arrays.
    gap_s: float | None = None
    #: True when the FAR budget was met only by the curve's appended reject-everything
    #: sentinel, i.e. no ACCEPTING threshold reached the budget on the observed grid (the
    #: highest-scoring probe in the run is a never-enrolled stranger). See
    #: `_operating_point_in_bounds`: in that case tau is the (DIR=0, FAR=0) endpoint, clipped
    #: to <= 1.0 so `GalleryConfig` can hold it.
    far_target_unreachable: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON/YAML-serialisable form. Native floats and ints only, never numpy scalars."""
        return {
            "label": self.label,
            "gap_s": self.gap_s,
            "tau": float(self.tau),
            "false_merge_rate": float(self.false_merge_rate),
            "dir": float(self.dir),
            "far_achieved": float(self.far),
            "far_target": float(self.far_target),
            "far_target_unreachable": bool(self.far_target_unreachable),
            "n_probes": int(self.n_probes),
            "n_new_probes": int(self.n_new_probes),
        }


def calibrate_tau(
    source: ScoreArrays | TrackCutScenario,
    far_target: float,
    *,
    gap_s: float | None = None,
    include_distractors: bool = True,
    label: str | None = None,
    probe: ProbeMode = "single",
) -> TauCalibration:
    """Pick tau at a FAR budget from a synthetic scenario or from captured score arrays.

    `far_target` is a hard ceiling on the fraction of never-enrolled probes whose top-1
    identity would be shown to a human. It is a budget on how often the system suggests a
    stranger is somebody it already knows; the tau it selects is the SMALLEST threshold
    meeting that ceiling, which is the one with the highest DIR (see
    `vividly.reid.metrics.dir_at_far` for why the whole eligible family is an upper interval).

    A `TrackCutScenario` needs `gap_s` unless it has exactly one tier, because each tier is
    a self-contained replay with its own gallery and averaging them into one answer here
    would hide which gap lengths the threshold actually serves. `calibrate_scenario` is the
    function that answers the multi-tier question properly, with a pooled curve.

    `probe` (default `single`) applies to the scenario path only, and `single` scores the same
    per-frame row the deployed gallery scores; see `scenario_score_arrays`. It is ignored for a
    `ScoreArrays` source, whose probes were already formed upstream.
    """
    target = float(far_target)
    if not 0.0 <= target <= 1.0:
        raise CalibrationConfigError(f"far_target must be in [0, 1], got {far_target}")

    if isinstance(source, TrackCutScenario):
        tiers = tuple(source.config.gap_tiers_s)
        if gap_s is None:
            if len(tiers) != 1:
                raise CalibrationConfigError(
                    f"this scenario has {len(tiers)} gap tiers {list(tiers)}; pass gap_s to "
                    "calibrate one of them, or call calibrate_scenario for the pooled tau a "
                    "deployment actually runs (one threshold across every gap length)."
                )
            gap_s = float(tiers[0])
        arrays = scenario_score_arrays(
            source, float(gap_s), include_distractors=include_distractors, probe=probe
        )
        row_label = label if label is not None else _tier_label(float(gap_s))
        return _calibrate_arrays(arrays, target, row_label, float(gap_s))

    if not isinstance(source, ScoreArrays):
        raise CalibrationConfigError(
            f"source must be a ScoreArrays bundle or a TrackCutScenario, got "
            f"{type(source).__name__}"
        )
    return _calibrate_arrays(arrays=source, far_target=target, label=label or "arrays", gap_s=gap_s)


def _operating_point_in_bounds(
    curve: M.DirFarCurve, far_target: float
) -> tuple[M.OperatingPoint, bool]:
    """The operating point at a FAR budget, but never a tau `GalleryConfig` cannot hold.

    `M.operating_point` can select the curve's appended sentinel threshold,
    `np.nextafter(max_score, inf)`, which sits strictly above every observed score so the
    (FAR=0, DIR=0) 'reject everything' endpoint exists. Landing there means NO accepting
    threshold met the budget: the highest-scoring probe in the run is a never-enrolled
    stranger, so the FAR budget is unreachable on the observed grid. That is reported.

    When the largest observed score is itself 1.0 the sentinel is > 1.0, and
    `GalleryConfig.__post_init__` refuses it (a cosine threshold must lie in [-1, 1]),
    yielding an unusable generated threshold. In that one case tau is clipped to 1.0, the
    tightest threshold the gallery can represent, and DIR and FAR are re-read off the same
    curve at tau = 1.0 so the number and the evidence beside it stay consistent (accepting a
    probe that scores exactly 1.0 changes both, and the sentinel's 0.0 would misreport them).
    """
    point = M.operating_point(curve, far_target)
    unreachable = float(point.tau) == float(curve.taus[-1])
    if point.tau <= 1.0:
        return point, unreachable
    idx = int(np.searchsorted(curve.taus, 1.0, side="left"))
    clipped = M.OperatingPoint(
        tau=1.0,
        dir=float(curve.dirs[idx]),
        far=float(curve.fars[idx]),
        far_target=float(far_target),
    )
    return clipped, unreachable


def _calibrate_arrays(
    arrays: ScoreArrays, far_target: float, label: str, gap_s: float | None
) -> TauCalibration:
    curve = M.dir_far_curve(
        arrays.gallery, arrays.gallery_ids, arrays.probes, arrays.probe_ids, arrays.new_probes
    )
    point, unreachable = _operating_point_in_bounds(curve, far_target)
    merges = M.false_merge_rate(
        arrays.gallery, arrays.gallery_ids, arrays.probes, arrays.probe_ids, point.tau
    )
    return TauCalibration(
        label=label,
        tau=point.tau,
        dir=point.dir,
        far=point.far,
        far_target=point.far_target,
        false_merge_rate=merges,
        n_probes=curve.n_probes,
        n_new_probes=curve.n_new_probes,
        gap_s=gap_s,
        far_target_unreachable=unreachable,
    )


def _tier_label(gap_s: float) -> str:
    return sequence_name(gap_s)


# ---------------------------------------------------------------------------
# the whole run
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CalibrationConfig:
    """What `calibrate_scenario` needs. `source` and `out` are the caller's, not this.

    A hand-written calibration config may carry them, and `config_from_mapping` accepts and
    ignores them; no field here holds them.
    """

    scenario: ScenarioConfig

    # PROVISIONAL (the re-identification scope, POLICY.md; docs/CALIBRATION.md): the
    # false-accept budget tau is chosen at. 0.01 is a default with no fitted basis; 1
    # percent is the reporting convention the literature review records for detection and
    # identification rate at a false-accept rate (research/reid-scale-literature.md), and
    # it is written here so a run records which budget it was scored against. The real
    # ceiling is a patient-safety threshold counsel sets, not an engineering preference.
    far_target: float = 0.01

    #: Whether people who only appeared during the gap sit in the gallery when the patient
    #: returns. True by default because a waiting-room system enrols everyone it sees, so a
    #: return probe binding to one of them is a false merge like any other. Setting it false
    #: calibrates a hypothetical system that somehow knew which tracks not to enrol.
    include_distractors_in_gallery: bool = True

    #: How a probe is formed (see `ProbeMode`). Not a threshold, but it materially moves the
    #: recommended tau, so it is a first-class field recorded in the emitted YAML. `single`
    #: (default) scores one per-frame row exactly as `SessionGallery.propose` and
    #: `ReidStage._probe` do in the deployed pipeline; `pooled` scores tracklet means, an
    #: easier task than the one the gallery runs. Ship `single`.
    probe: str = "single"

    def __post_init__(self) -> None:
        if not 0.0 <= self.far_target <= 1.0:
            raise CalibrationConfigError(f"far_target must be in [0, 1], got {self.far_target}")
        if self.probe not in ("single", "pooled"):
            raise CalibrationConfigError(
                f"probe must be 'single' or 'pooled', got {self.probe!r}. `single` scores one "
                "per-frame row like the deployed gallery; `pooled` scores tracklet means."
            )


def config_from_mapping(raw: Mapping[str, Any]) -> CalibrationConfig:
    """Build a `CalibrationConfig` from a parsed YAML document, refusing unknown keys.

    `schema_version`, `source` and `out` are accepted and ignored here: they belong to the
    caller, which is the module API (`calibrate_scenario` followed by `write_calibration`).
    """
    _check_keys(
        raw,
        {
            "schema_version",
            "source",
            "out",
            "scenario",
            "far_target",
            "include_distractors_in_gallery",
            "probe",
        },
        "config",
    )
    scenario = scenario_from_mapping(raw.get("scenario") or {})
    try:
        far_target = float(raw.get("far_target", 0.01))
    except (TypeError, ValueError) as exc:
        raise CalibrationConfigError(f"far_target: {exc}") from exc
    return CalibrationConfig(
        scenario=scenario,
        far_target=far_target,
        include_distractors_in_gallery=bool(raw.get("include_distractors_in_gallery", True)),
        probe=str(raw.get("probe", "single")),
    )


@dataclass(frozen=True)
class CalibrationReport:
    """Per-tier operating points, the pooled one a deployment would run, and provenance."""

    #: The recommended row. Pooled across gap tiers for a scenario; the only row for arrays.
    recommended: TauCalibration
    #: One row per gap tier, in tier order. Empty for the captured-arrays path.
    tiers: tuple[TauCalibration, ...]
    far_target: float
    source_kind: str
    source_note: str
    #: sha256 over every byte the fixture carries, or None when the source is not synthetic.
    scenario_digest: str | None = None
    #: Whether gap distractors sat in the gallery. A bool for the synthetic path, and None on
    #: the captured-arrays path, where gallery composition is fixed by the capture and the flag
    #: does not apply. None is rendered as `null` with the reason in `source_note`.
    include_distractors_in_gallery: bool | None = True
    #: How the probes were formed: `single` or `pooled` for the synthetic path (see
    #: `ProbeMode`), `captured` for a `.npz` whose probes were already formed upstream.
    probe: str = "single"

    @property
    def tau(self) -> float:
        """The number a pipeline config would take. Provisional, like everything here."""
        return self.recommended.tau

    def to_dict(self) -> dict[str, Any]:
        return {
            "provisional": True,
            "note": PROVISIONAL_NOTE,
            "far_target": float(self.far_target),
            "probe": self.probe,
            "source_kind": self.source_kind,
            "source_note": self.source_note,
            "scenario_digest": self.scenario_digest,
            "include_distractors_in_gallery": (
                None
                if self.include_distractors_in_gallery is None
                else bool(self.include_distractors_in_gallery)
            ),
            "recommended": self.recommended.to_dict(),
            "gap_tiers": [tier.to_dict() for tier in self.tiers],
        }


def calibrate_scenario(cfg: CalibrationConfig) -> CalibrationReport:
    """Build the synthetic scenario, calibrate every gap tier, and pool them into one tau.

    The pooled row is not a fifth tier. It is the operating curve a deployment faces, where
    one tau has to serve every gap length at once, built by concatenating the per-tier
    decisions (`pooled_dir_far_curve`). It is the recommended number
    for exactly that reason, and the per-tier rows sit beside it so a pooled tau carried
    entirely by the short-gap tier is visible rather than implied.
    """
    scenario = build_scenario(cfg.scenario)
    tier_arrays = [
        (
            float(gap_s),
            sequence_name(float(gap_s)),
            scenario_score_arrays(
                scenario,
                float(gap_s),
                include_distractors=cfg.include_distractors_in_gallery,
                probe=_probe_mode(cfg.probe),
            ),
        )
        for gap_s in cfg.scenario.gap_tiers_s
    ]
    tiers = tuple(
        _calibrate_arrays(arrays, cfg.far_target, sequence, gap)
        for gap, sequence, arrays in tier_arrays
    )
    pooled = _pool_arrays([arrays for _, _, arrays in tier_arrays], cfg.far_target)
    return CalibrationReport(
        recommended=pooled,
        tiers=tiers,
        far_target=cfg.far_target,
        source_kind="synthetic",
        source_note=(
            "vividly.reid.synth track-cut fixture: identity centroids on a unit hypersphere, "
            "no footage, no weights, no GPU, no network."
        ),
        scenario_digest=scenario_digest(scenario),
        include_distractors_in_gallery=cfg.include_distractors_in_gallery,
        probe=cfg.probe,
    )


def _probe_mode(probe: str) -> ProbeMode:
    """Narrow a config string to a `ProbeMode`. `CalibrationConfig` already validated it."""
    if probe not in ("single", "pooled"):  # pragma: no cover - CalibrationConfig refuses first
        raise CalibrationConfigError(f"probe must be 'single' or 'pooled', got {probe!r}")
    return probe  # type: ignore[return-value]


def calibrate_arrays_report(
    arrays: ScoreArrays, far_target: float, *, source_note: str = ""
) -> CalibrationReport:
    """One captured score bundle, as a report the same writer can render."""
    row = _calibrate_arrays(arrays, float(far_target), "arrays", None)
    return CalibrationReport(
        recommended=row,
        tiers=(),
        far_target=float(far_target),
        source_kind="arrays",
        source_note=source_note
        or (
            "captured score arrays. Whatever produced them is what carries the checkpoint "
            "hash and the dataset provenance; this file records only the arithmetic. The "
            "gallery composition is fixed by the capture, so include_distractors_in_gallery "
            "is not applicable and is written null."
        ),
        include_distractors_in_gallery=None,
        probe="captured",
    )


def _pool_arrays(tiers: Sequence[ScoreArrays], far_target: float) -> TauCalibration:
    """Pool per-tier decisions into the one curve a deployment runs: one tau, every gap length.

    The per-tier top-1 decisions are recomputed from each tier's own gallery and probes (the
    same `probe` granularity `calibrate_scenario` built them at) and concatenated, then swept
    by `pooled_dir_far_curve`, which is the only construction that can express one threshold
    across four independent galleries. On a single tier it is the same curve
    `metrics.dir_far_curve` produces.
    """
    if not tiers:
        raise CalibrationConfigError("no gap tiers to pool: the scenario produced no matches")
    correct_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    new_parts: list[np.ndarray] = []
    for arrays in tiers:
        predicted, scores = M.top1_match(arrays.gallery, arrays.gallery_ids, arrays.probes)
        _, new_scores = M.top1_match(arrays.gallery, arrays.gallery_ids, arrays.new_probes)
        correct_parts.append(predicted == np.asarray(arrays.probe_ids))
        score_parts.append(scores)
        new_parts.append(new_scores)
    correct = np.concatenate(correct_parts)
    scores = np.concatenate(score_parts)
    new_scores = np.concatenate(new_parts)
    curve = pooled_dir_far_curve(correct, scores, new_scores)
    point, unreachable = _operating_point_in_bounds(curve, float(far_target))
    return TauCalibration(
        label="pooled",
        tau=point.tau,
        dir=point.dir,
        far=point.far,
        far_target=point.far_target,
        false_merge_rate=pooled_false_merge_rate(correct, scores, point.tau),
        n_probes=curve.n_probes,
        n_new_probes=curve.n_new_probes,
        gap_s=None,
        far_target_unreachable=unreachable,
    )


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def summarise(report: CalibrationReport) -> str:
    """The PROVISIONAL banner and the operating-point table, for a terminal."""
    lines = [
        PROVISIONAL_NOTE,
        f"  source: {report.source_kind}. {report.source_note}",
    ]
    if report.scenario_digest:
        lines.append(f"  scenario digest {report.scenario_digest}")
    lines += [
        "",
        f"{'tier':>12}  {'tau':>8}  {'false-merge':>11}  {'DIR':>7}  {'FAR':>7}",
        "-" * 54,
    ]
    for row in (*report.tiers, report.recommended):
        lines.append(
            f"{row.label:>12}  {row.tau:8.4f}  {row.false_merge_rate:11.4f}  "
            f"{row.dir:7.4f}  {row.far:7.4f}"
        )
    lines += [
        "",
        f"  recommended tau {report.tau!r} at FAR target {report.far_target:g} "
        f"(achieved {report.recommended.far:g})",
        "  PROVISIONAL: wire it through a config, never by hand into code.",
    ]
    if report.recommended.far_target_unreachable:
        lines.append(
            f"  NOTE: no accepting threshold met FAR <= {report.far_target:g} on this grid, so "
            "the recommended row is the reject-everything endpoint (tau clipped to <= 1.0). "
            "far_target_unreachable: true."
        )
    return "\n".join(lines)


def render_yaml(report: CalibrationReport) -> str:
    """The generated document: a `reid:` block a pipeline config can reference, plus evidence.

    No timestamp and no hostname anywhere, on purpose: the file is a deterministic function
    of its config, so re-running the calibration produces a byte-identical file and a diff
    means the numbers moved rather than that the clock did.

    tau is written unrounded. A rounded threshold does not reproduce the FAR reported next
    to it, and this file exists precisely so those two travel together.
    """
    header = [
        "# GENERATED by vividly.reid.calibrate. Do not edit by hand: re-run the calibration.",
        "#",
        f"# {PROVISIONAL_NOTE}",
        "#",
        "# tau is the accept threshold of vividly.reid.gallery.SessionGallery. Below it no",
        "# candidate identity is shown to a human at all. It does not make the system bind",
        "# anything: beyond the tracker's own re-association window nothing is ever bound",
        "# automatically, a proposal is confirmed or rejected by a person, and that answer is",
        "# what becomes a label. See POLICY.md, re-identification scope, and CQ-16.",
        "#",
        "# It is written under reid.gallery.tau, which is a GalleryConfig field: that is the",
        "# exact shape vividly.reid.stage.load_reid_config accepts, so a pipeline config can",
        "# reference this file's reid block directly without the loader refusing an unknown key.",
        "#",
        f"# source: {report.source_kind}. {report.source_note}",
    ]
    if report.scenario_digest:
        header.append(f"# scenario digest: {report.scenario_digest}")
    body = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "reid": {"gallery": {"tau": float(report.tau)}},
        "calibration": report.to_dict(),
    }
    dumped = yaml.safe_dump(body, sort_keys=False, default_flow_style=False, width=100)
    return "\n".join(header) + "\n" + dumped


def write_calibration(report: CalibrationReport, path: Path | str) -> Path:
    """Write the generated YAML, creating parent directories. Returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_yaml(report), encoding="utf-8")
    return out
