#!/usr/bin/env python3
"""Replay-mode pipeline runner (structure recorded 2026-08-07).

    uv run python scripts/run_pipeline.py --config configs/pipeline/synthetic_demo.yaml --headless
    uv run python scripts/run_pipeline.py --config configs/pipeline/replay_demo.yaml

Wires the replay pipeline end to end (docs/PIPELINE.md):

    source -> detector + pose -> tracker -> TrackObservations -> TrackBook -> RuleEngine
           -> EventStore -> dashboard (uvicorn) or a headless run that writes the same
              artifacts to disk

Replay only. Every frame this script can reach comes out of a file somebody else licensed,
or out of a generated fixture. There is no capture path in this runner: the live-camera
seam is `vividly.sources.RTSPSource`, and the separate bring-up script
scripts/live_camera.py drives its working implementation against a network video recorder
(docs/LIVE_CAMERA.md).

Two flows, one code path.

  * Zero-footage. `detections_from: <jsonl>` in the config replaces the entire detect,
    pose and track stack with `SyntheticDetectionSource`, which replays generated tracker
    observations. No weights, no cv2, no network, no images. This is the flow that runs in
    CI and the one to demo on a laptop that has downloaded nothing.
  * Footage. `source: {path, rate}` (a clip) or `source: {images, fps, pattern, rate}` (a
    directory of numbered frames) plus `detector`, `pose` and `tracker` blocks build real
    backends through `vividly.models.factory`, which is the only place a backend is named.
    Weights are a config swap by construction; this script never imports a backend module.

Stage order, and why it is not the order in the diagram. Detection runs, then the tracker
assigns ids, then pose runs on the tracked boxes. Posing before tracking would mean
matching poses back to track ids afterwards, because a tracker returns its own boxes in its
own order, and that is a second matching problem with its own failure modes for no gain.
The six stage names are fixed by the `STAGES` tuple of `vividly.dashboard.app` (source,
detect, pose, track, rules, render) and the tables below always print them in that
canonical order whatever the execution order is.

Latency. Every stage of every frame is timed with the same pattern as
`TimedModel._timed` in src/vividly/models/base.py: a stage name, a millisecond duration, an
item count and a backend string. `Timing` is re-declared here rather than imported because
vividly.models.base imports numpy at module level and this module has to import on a runner
with no numpy installed (the same reason vividly/tracks/state.py re-declares the COCO-17
torso indices). Per-frame rows go to latency.jsonl; p50 and p95 per stage are printed on
exit. A stage that did not run in this mode reports null, not zero: a zero-footage run has
no detector, and printing 0.00 ms for it would read as a very fast detector.

Re-identification is optional and it never binds anything. A config may carry a `reid:`
block (or point at configs/pipeline/reid.yaml with `reid_config:`). With neither, or with
`enabled: false`, this script builds no gallery, loads no embedder and behaves exactly as
it did before re-ID existed, which tests/test_reid_pipeline.py pins against a recorded
alert stream. With it on, tracks that the TrackBook reaps, that is people who left for
longer than `reap_after_s`, are enrolled into a session-scoped gallery, tracks born
afterwards are probed against it, and a match at or above tau becomes a PENDING row in the
alert store's `reassoc_proposals` table. That is the entire effect. Nothing is merged, no
track is renamed and no chart binding moves until a human confirms on the dashboard; the
confirmation and the rejection are both written down, and both are labels. See
src/vividly/reid/stage.py.

The re-ID stage is deliberately not one of the six timed stages. Those six names are the
`STAGES` tuple the dashboard reads; folding embedding work into `rules`
would make the rules row describe work the rules did not do. Re-ID reports its own wall cost
in the exit summary instead.

Run identity. Every run derives a `run_id` and hands it to the re-ID stage, so every
proposal row it writes carries the run that raised it and no run can answer another run's
question. One store file is reachable by more than one run (its path is
`<output.dir>/<run_name>/alerts.sqlite`, and two different clips can be run under one
name), and the store's per-run identity indexes are over `(run_id, ...)`: a NULL run_id
makes every row distinct in a UNIQUE index, so an unscoped run is exactly the case the
database cannot protect. This runner therefore always supplies one. See `derive_run_id`
for why it is a digest and not a uuid.

Session scope. The gallery holds appearance embeddings for one visit and lives in memory
only. `Pipeline.close()` empties it on every exit path, including an interrupted run, and
the summary printed afterwards says so with the retained count next to it (CQ-16 in
docs/COUNSEL_QUESTIONS.md). That ordering is why close runs before the summary is printed:
the line is a post-condition, not an intention.

Shutdown order. When the dashboard is served, the pipeline runs in a background thread and
uvicorn owns the main thread. `serve` stops that thread and joins it before main's finally
block closes the pipeline, so the session-end deletion never races a tick that is still
enrolling somebody into the gallery it is about to empty.

Thresholds. Every rule threshold this script loads is PROVISIONAL until pilot shadow data
exists, every verifier head's confirm threshold in the `verify:` block is provisional for
the same reason, and so is every re-ID threshold. No firing count from any run of this
script is a false-alarm rate, no proposal count is a false-merge rate, and no number here
belongs in a deck. See configs/pipeline/rules.yaml, configs/pipeline/reid.yaml,
configs/pipeline/replay_demo.yaml and docs/CALIBRATION.md.

Exit codes:
    0  ran
    2  usage error (argparse)
    3  config error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import uvicorn
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vividly.dashboard.app import STAGES, PipelineState, create_app, encode_jpeg_b64  # noqa: E402
from vividly.events import DuplicateAlertError, EventStore  # noqa: E402
from vividly.models.checkpoints import is_lfs_pointer  # noqa: E402
from vividly.rules import (  # noqa: E402
    EgressMonitor,
    EgressZones,
    RuleConfig,
    RuleEngine,
    RuleFiring,
    build_default_rules,
    load_rule_config,
)
from vividly.schemas.events import AlertRecord, make_alert_id, utc_now_iso  # noqa: E402
from vividly.sources import SyntheticDetectionSource  # noqa: E402
from vividly.tracks import TrackBook, TrackConfig, TrackObservation  # noqa: E402

EXIT_OK = 0
EXIT_CONFIG = 3

#: Everything a run writes lands under <output.dir>/<run_name>/.
ALERT_DB_NAME = "alerts.sqlite"
LATENCY_LOG_NAME = "latency.jsonl"

#: How many pending re-association proposals the exit summary counts. Display sizing, not
#: a threshold: a session-scoped gallery cannot produce more questions than one visit's
#: worth of returns, and a summary line that had to page would not be a summary.
PENDING_SUMMARY_LIMIT = 10_000

#: Hex characters of the run digest kept in a run_id. Sizing, not a threshold and not a
#: security parameter: 16 hex characters is 64 bits, which is far past the point where two
#: distinct configs in one output directory collide, and short enough to read in a sqlite
#: row. Nothing in the pipeline's behaviour changes with this number; a longer id would be
#: the same id.
RUN_ID_DIGEST_CHARS = 16

#: Seconds `serve` waits for the pipeline thread to finish its tick after being asked to
#: stop. A shutdown bound, not a threshold: no rule, no score and no count depends on it,
#: and docs/CALIBRATION.md has nothing to say about it. It is generous because a tick is a
#: frame of work (embed, rules, one SQLite insert) and stingy because a reviewer closing the
#: page should get their prompt back. What happens if it expires is spelled out in `serve`.
WORKER_JOIN_TIMEOUT_S = 10.0

#: Sentinel for "the source is exhausted". A dedicated object, because None and an empty
#: observation batch are both legitimate values a source can yield.
_EXHAUSTED = object()


class ConfigError(ValueError):
    """The pipeline config is missing something, or asks for two modes at once."""


def _plural(count: int, singular: str, plural: str) -> str:
    """ "1 identity", "2 identities". A summary a human reads should read like one."""
    return f"{count} {singular if count == 1 else plural}"


# --------------------------------------------------------------------- latency
@dataclass(frozen=True)
class Timing:
    """One timed stage of one frame.

    Same shape as vividly.models.base.Timing, deliberately re-declared: importing that
    module pulls numpy in at import time and this file must import without it.
    """

    stage: str
    ms: float
    n_items: int
    backend: str

    @property
    def ms_per_item(self) -> float:
        return self.ms / self.n_items if self.n_items else float("nan")


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile, matching numpy.percentile's default method.

    Stdlib arithmetic, not numpy: this module is imported by tests that Linux CI collects
    with no numpy installed. The method is pinned to numpy's default so a latency number
    from this script and one from src/vividly/eval/runner.py mean the same thing.

    Returns None for an empty sample, which is how a stage that never ran is reported.
    """
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (q / 100.0)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


class StageTimer:
    """Per-stage latency for the whole run, and for the frame in flight.

    `run` is the `TimedModel._timed` pattern from src/vividly/models/base.py: it wraps a
    call, records a Timing, and returns whatever the call returned, so a stage cannot be
    executed without being measured. Stages are keyed by the canonical STAGES names; a
    stage that is never called keeps an empty sample list and reports None everywhere,
    which is what a zero-footage run's detect/pose/track rows show.
    """

    def __init__(self, backend: str, stages: Sequence[str] = STAGES) -> None:
        self.backend = backend
        self.stages: tuple[str, ...] = tuple(stages)
        self.timings: list[Timing] = []
        self._samples: dict[str, list[float]] = {stage: [] for stage in self.stages}
        self._frame: dict[str, float] = {}

    def start_frame(self) -> None:
        """Begin a new per-frame row. Call once per tick, before the source stage."""
        self._frame = {}

    def record(self, stage: str, ms: float, n_items: int) -> None:
        if stage not in self._samples:
            raise KeyError(f"unknown stage {stage!r}; the contract fixes them to {self.stages}")
        self.timings.append(Timing(stage=stage, ms=ms, n_items=n_items, backend=self.backend))
        self._samples[stage].append(ms)
        self._frame[stage] = self._frame.get(stage, 0.0) + ms

    def run(self, stage: str, n_items: int, fn: Callable[..., Any], *args: Any, **kw: Any) -> Any:
        """Time one call and record it. Same contract as TimedModel._timed."""
        t0 = time.perf_counter()
        out = fn(*args, **kw)
        self.record(stage, (time.perf_counter() - t0) * 1000.0, n_items)
        return out

    # -- reporting ---------------------------------------------------------
    def frame_ms(self) -> dict[str, float | None]:
        """The frame in flight, all six stages, None for the ones that did not run."""
        return {stage: self._frame.get(stage) for stage in self.stages}

    def n_samples(self) -> dict[str, int]:
        return {stage: len(self._samples[stage]) for stage in self.stages}

    def quantile_ms(self, q: float) -> dict[str, float | None]:
        return {stage: percentile(self._samples[stage], q) for stage in self.stages}

    def table(self) -> str:
        """The exit summary: one row per stage, p50 and p95 in ms.

        Stages that did not run print a dash. That is the whole point of the dash: a
        zero-footage run must not be readable as a run where a detector was fast.
        """
        p50 = self.quantile_ms(50.0)
        p95 = self.quantile_ms(95.0)
        counts = self.n_samples()
        lines = [
            f"{'stage':<10}{'n':>8}{'p50 ms':>12}{'p95 ms':>12}",
            f"{'-' * 10}{'-' * 8:>8}{'-' * 12:>12}{'-' * 12:>12}",
        ]
        for stage in self.stages:
            fifty = p50[stage]
            ninety_five = p95[stage]
            lines.append(
                f"{stage:<10}{counts[stage]:>8}"
                f"{'-' if fifty is None else format(fifty, '.3f'):>12}"
                f"{'-' if ninety_five is None else format(ninety_five, '.3f'):>12}"
            )
        skipped = [stage for stage in self.stages if not counts[stage]]
        if skipped:
            lines.append(f"stages that did not run in this mode: {', '.join(skipped)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------- config
def load_config(path: str | Path) -> dict[str, Any]:
    """Read a pipeline config. The only place this script learns a path or a threshold."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        try:
            data = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            # Unparseable YAML is a config error like any other, and the operator wants the
            # scanner's line and column, not a traceback through the yaml package.
            raise ConfigError(f"{config_path} is not valid YAML: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigError(f"{config_path} must contain a mapping, got {type(data).__name__}")
    return dict(data)


def resolve(path: str | Path, repo_root: Path) -> Path:
    """Config paths are relative to the repo root, absolute paths are left alone."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else repo_root / candidate


def build_rule_config(cfg: Mapping[str, Any], repo_root: Path) -> RuleConfig:
    """Thresholds come from an inline `rules:` block or from a `rules_config:` file.

    Never both: two sources of truth for a threshold is precisely the failure mode
    docs/CALIBRATION.md exists to prevent, and a silent merge would hide which number
    actually ran.
    """
    inline = cfg.get("rules")
    pointer = cfg.get("rules_config")
    if inline is not None and pointer is not None:
        raise ConfigError(
            "config sets both `rules` and `rules_config`; pick one, because a run has to "
            "be able to say which file its thresholds came from"
        )
    if pointer is not None:
        return load_rule_config(resolve(pointer, repo_root))
    if inline is not None:
        if not isinstance(inline, Mapping):
            raise ConfigError(f"`rules` must be a mapping, got {type(inline).__name__}")
        return RuleConfig.from_mapping(inline)
    return RuleConfig()


def _canonical(value: Any) -> Any:
    """A JSON-dumpable copy of a config with a total, stable ordering of every mapping.

    Two things a raw `json.dumps(cfg, sort_keys=True)` cannot promise over a YAML document.
    A config can hold non-string mapping keys (`identity_by_track: {1: 0}` does), and
    `sort_keys` raises on a dict whose keys are of mixed types; and it can hold a value json
    has no encoder for. Keys become strings and are ordered as strings, everything json
    cannot encode becomes its `str()`, so the result is always dumpable and always the same
    for the same config.
    """
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    if value is None or isinstance(value, str | bool | int | float):
        return value
    return str(value)


def derive_run_id(run_name: str, source_path: Path, cfg: Mapping[str, Any]) -> str:
    """This run's identity: `<run_name>-<digest>` over the name, the source and the config.

    Why a digest and not a uuid, which is the choice this function exists to record. A
    run_id scopes the proposal rows in `reassoc_proposals`: a run's pending queue lists only
    its own questions and its decisions land only on its own rows (the dashboard threads
    this id through `PipelineState.run_id`; file-wide proposal history stays visible by
    design), and the store's per-run identity indexes are over `(run_id, ...)`. A
    fresh uuid per process would make every replay of one clip a brand new run, so a second
    pass over the same footage would re-ask every question a human already answered, and the
    answers from the first pass would sit in the file forever as another run's history. A
    digest of what actually defines the run makes a re-run the same run: it resumes its own
    rows, skips the pairs somebody has answered, and keeps its identity constraints against
    the rows it wrote last time.

    What is in the digest, and why each part.

      * `run_name`, because it is what the operator called this run and it is what picks the
        store file (`<output.dir>/<run_name>/alerts.sqlite`).
      * the resolved source path, because two clips run under one name share that one store
        file, and they are not the same run. This is the collision the whole mechanism is
        for.
      * the config, because a threshold change makes a run ask different questions, and the
        answers to those questions should not be silently reconciled with a different run's.
        The output directory is part of the config and so part of the identity; a run moved
        to another directory writes a different store file anyway.

    What it is not. It is an opaque run label. It is not derived from a clip's contents, it
    is not derived from anything about a person, and it cannot be mapped back to a patient
    or a visit (`vividly.schemas.reassoc` states the same contract for the stored column).
    blake2b rather than `hash()`, which is salted per process and would make "the same run
    twice" false across restarts, which is the one property this has to have.
    """
    payload = json.dumps(
        {"run_name": run_name, "source": str(source_path), "config": _canonical(cfg)},
        sort_keys=True,
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()
    return f"{run_name}-{digest[:RUN_ID_DIGEST_CHARS]}"


def build_track_config(cfg: Mapping[str, Any]) -> TrackConfig:
    block = cfg.get("tracks", {})
    if not isinstance(block, Mapping):
        raise ConfigError(f"`tracks` must be a mapping, got {type(block).__name__}")
    return TrackConfig.from_mapping(block)


def build_egress_zones(cfg: Mapping[str, Any]) -> EgressZones:
    """Parse the optional `egress:` block. Absent or `enabled: false` means no head at all.

    The zones are per-site geometry (source-frame pixels) and belong to the pipeline
    config; the three egress thresholds belong to the `rules`/`rules_config` block with
    every other threshold, so a run can still say which file its numbers came from.
    """
    block = cfg.get("egress", {})
    if block is None:
        block = {}
    if not isinstance(block, Mapping):
        raise ConfigError(f"`egress` must be a mapping, got {type(block).__name__}")
    try:
        return EgressZones.from_mapping(block)
    except ValueError as exc:
        raise ConfigError(f"`egress`: {exc}") from exc


def build_reid_config(cfg: Mapping[str, Any], repo_root: Path) -> Any | None:
    """Parse the optional `reid:` block, or None when this config has no re-ID at all.

    None is the common answer and the quiet one: with no re-ID block, `vividly.reid.stage`
    is never even imported, which is what keeps this script importable on a runner with no
    numpy (tests/test_dashboard.py checks that by importing it in a subprocess and looking
    at sys.modules). The import is inside this function for exactly that reason.

    `reid` and `reid_config` are mutually exclusive for the same reason `rules` and
    `rules_config` are: two sources of truth for a threshold means a run cannot say which
    file its numbers came from.
    """
    inline = cfg.get("reid")
    pointer = cfg.get("reid_config")
    if inline is not None and pointer is not None:
        raise ConfigError(
            "config sets both `reid` and `reid_config`; pick one, because a run has to be "
            "able to say which file its re-ID thresholds came from"
        )
    if inline is None and pointer is None:
        return None

    from vividly.reid.stage import ReidConfig, ReidConfigError, load_reid_config

    try:
        if pointer is not None:
            return load_reid_config(resolve(pointer, repo_root))
        if not isinstance(inline, Mapping):
            raise ConfigError(f"`reid` must be a mapping, got {type(inline).__name__}")
        return ReidConfig.from_mapping(inline)
    except ReidConfigError as exc:
        # ReidConfigError is a ValueError and main() already maps those to exit 3, but
        # restating it as a ConfigError keeps every "your config is wrong" path in this
        # script one type, and keeps the message pointing at the config rather than at a
        # dataclass field.
        raise ConfigError(f"`reid`: {exc}") from exc


# -------------------------------------------------------------------- pipeline
class Pipeline:
    """One run: a source, the perception stack it needs, and everything downstream of it.

    Constructing one validates the config, creates the output directory and opens the alert
    store, but decodes no frame and loads no detector, pose model or tracker. That split is
    what lets a test, a config check and a dashboard launch all build a Pipeline cheaply,
    and what makes a bad config an error before a browser is attached rather than on the
    first frame.

    The one model this constructor does build is the optional appearance embedder, and only
    when a config turned re-ID on. Two reasons, both about failing in the right place: the
    session gallery has to exist before the dashboard is served, because the page shows the
    proposal queue from the first request; and a wrong weights path should be an exit-3
    config error, not a traceback forty minutes into a replay.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any],
        repo_root: Path = REPO_ROOT,
    ) -> None:
        # Validation first, side effects afterwards. A config error must not leave an open
        # SQLite file and a half-written latency log behind, and it must surface before a
        # dashboard is served rather than on the first frame with a browser attached.
        self.cfg = dict(cfg)
        self.repo_root = repo_root
        self.run_name = str(self.cfg.get("run_name", "pipeline"))

        detections_from = self.cfg.get("detections_from")
        self.zero_footage = detections_from is not None
        self.mode = "synthetic" if self.zero_footage else "replay"

        self.track_cfg = build_track_config(self.cfg)
        self.rule_cfg = build_rule_config(self.cfg, repo_root)
        self.reid_cfg = build_reid_config(self.cfg, repo_root)
        self.egress_zones = build_egress_zones(self.cfg)

        if self.zero_footage:
            self.source_path = resolve(str(detections_from), repo_root)
            backend = "synthetic-fixture"
        else:
            self.source_path = self._footage_source_path(self.cfg.get("source"), repo_root)
            detector_cfg = self.cfg.get("detector")
            backend = (
                str(detector_cfg.get("backend", "unknown"))
                if isinstance(detector_cfg, Mapping)
                else "unknown"
            )

        # Derived here, before anything is opened, because the re-ID stage below is built
        # with it and because a run has to be able to say what it is called in the one file
        # more than one run can reach. Never None: see derive_run_id.
        self.run_id = derive_run_id(self.run_name, self.source_path, self.cfg)

        output = self.cfg.get("output", {})
        if not isinstance(output, Mapping):
            raise ConfigError(f"`output` must be a mapping, got {type(output).__name__}")
        dashboard = self.cfg.get("dashboard", {})
        if not isinstance(dashboard, Mapping):
            raise ConfigError(f"`dashboard` must be a mapping, got {type(dashboard).__name__}")
        self.dashboard_cfg = dict(dashboard)
        self.jpeg_quality = int(self.dashboard_cfg.get("jpeg_quality", 70))

        self.book = TrackBook(self.track_cfg)
        # The egress head is an engine attachment, not a rule: its subject is a track that
        # has already been reaped. None (the shipped default) leaves the engine exactly as
        # it was before the head existed.
        self.egress = (
            EgressMonitor(self.rule_cfg, self.egress_zones) if self.egress_zones.enabled else None
        )
        # The verify seam, which is what makes a candidate an alert: no `verify:` block
        # means no verifier, which leaves the engine exactly as it was before the trained
        # backend existed. With heads configured, the ONNX event verifier, an interim
        # prototype weight under rule 4 of POLICY.md, scores every firing; the import is
        # deferred so verifier-less runs never touch onnxruntime.
        verify_cfg = self.cfg.get("verify", {})
        if not isinstance(verify_cfg, Mapping):
            raise ConfigError(f"`verify` must be a mapping, got {type(verify_cfg).__name__}")
        verifier = None
        verify_policy = str(verify_cfg.get("policy", "annotate"))
        heads_cfg = verify_cfg.get("heads", {})
        if heads_cfg:
            from vividly.models.event_verifier import OnnxEventVerifier

            verifier = OnnxEventVerifier.from_config(heads_cfg, repo_root)
        self.engine = RuleEngine(
            build_default_rules(self.rule_cfg),
            self.rule_cfg,
            egress=self.egress,
            verifier=verifier,
            verify_policy=verify_policy,
        )
        self.timer = StageTimer(backend)

        self._source: Any = (
            SyntheticDetectionSource(self.source_path) if self.zero_footage else None
        )
        self._detector: Any = None
        self._pose: Any = None
        self._tracker: Any = None

        self.out_dir = resolve(output.get("dir", "outputs/pipeline"), repo_root) / self.run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.store = EventStore(self.out_dir / ALERT_DB_NAME)
        self.latency_path = self.out_dir / LATENCY_LOG_NAME
        self._latency_fh = self.latency_path.open("w", encoding="utf-8")

        # None whenever re-ID is off, which is the default and every config that shipped
        # before it existed. Built after the store because a proposal is a row in it, and
        # after the book because the stage registers the book's reap watcher.
        self.reid = self._build_reid()
        self._reid_stats: Any = None
        self._reid_pending = 0

        self.state = PipelineState(
            book=self.book,
            store=self.store,
            run_name=self.run_name,
            mode=self.mode,
            # The scoping identity, not the display name: this is what keeps the proposal
            # panel's pending queue and the no-stage decide fallback inside this run when
            # two runs share one store file.
            run_id=self.run_id,
            source_label=str(self.source_path),
            stream_hz=float(self.dashboard_cfg.get("stream_hz", 4.0)),
            alert_limit=int(self.dashboard_cfg.get("alert_limit", 25)),
            # The stage itself, not a copy of its numbers, and None when re-ID is off. This
            # is what closes the loop: without it the page would still list the questions
            # out of the store, but a Confirm would take the store-only branch and the
            # answer would never reach TrackBook.merge_identity, so a reviewer would be
            # told the label was stored while the live track kept its blank slate.
            reid=self.reid,
            # The one directory the crops route is allowed to serve from. Nothing writes
            # crops today, so every card renders its placeholder tile; the route still has
            # to be pinned to this run's directory rather than left unset.
            run_dir=self.out_dir,
            # The egress monitor, or None when the head is off: the departures panel and
            # its "mark discharged" button reach it through here.
            egress=self.egress,
        )

        self.n_frames = 0
        self.n_alerts = 0
        self.n_proposals = 0
        self._closed = False
        # Set by request_stop (and by close), read at the top of every tick. This is how a
        # served run stops its worker thread before anything closes the store or empties the
        # gallery underneath it; a daemon thread left running through close is what made the
        # CQ-16 deletion a race with an enrollment.
        self._stop = threading.Event()
        # Cached so run() can be called again and continue rather than restart. A source is
        # consumed once; re-entering _ticks() would re-open the clip and replay it from the
        # top, which is never what a second call means.
        self._tick_iter: Iterator[tuple[int, float, list[TrackObservation], Any]] | None = None

    @staticmethod
    def _footage_source_path(source_cfg: Any, repo_root: Path) -> Path:
        """Validate the `source` block of a footage run and return what it points at.

        Two shapes, never both: `{path: <clip>}` plays a video through ReplaySource;
        `{images: <dir>, fps: <Hz>}` plays a directory of numbered frames through
        ImageSequenceSource. `fps` is mandatory for a directory because no frame directory
        records its own rate and t_source is index / fps: a default here would put every
        rule window on a guessed clock, and no number in the alert path is guessed rather
        than measured. An optional `pattern` (default
        `*.jpg`) selects the frame files; `rate` paces playback in either shape.
        """
        usage = (
            "a footage run needs `source: {path: <clip>}` or `source: {images: <dir>, "
            "fps: <Hz>}`, or `detections_from: <jsonl>` for the zero-footage flow"
        )
        if not isinstance(source_cfg, Mapping):
            raise ConfigError(usage)
        has_path, has_images = "path" in source_cfg, "images" in source_cfg
        if has_path == has_images:
            raise ConfigError(
                usage if not has_path else "`source` takes `path` OR `images`, not both"
            )
        if has_path:
            return resolve(str(source_cfg["path"]), repo_root)
        if "fps" not in source_cfg:
            raise ConfigError(
                "`source.images` needs `source.fps`: no frame directory records its own rate, "
                "so write the dataset's (configs/datasets/*.yaml notes say where it came from)"
            )
        try:
            fps = float(source_cfg["fps"])
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"`source.fps` must be a number, got {source_cfg['fps']!r}") from exc
        if not math.isfinite(fps) or fps <= 0.0:
            raise ConfigError(f"`source.fps` must be a finite positive frame rate, got {fps!r}")
        pattern = source_cfg.get("pattern", "*.jpg")
        if not isinstance(pattern, str) or not pattern:
            raise ConfigError(f"`source.pattern` must be a non-empty glob string, got {pattern!r}")
        return resolve(str(source_cfg["images"]), repo_root)

    def _footage_source(self) -> Any:
        """The FrameSource for a footage run, built and not yet opened."""
        source_cfg = dict(self.cfg["source"])
        rate = source_cfg.get("rate")
        rate = None if rate is None else float(rate)
        if "images" in source_cfg:
            from vividly.sources.image_sequence import ImageSequenceSource

            return ImageSequenceSource(
                self.source_path,
                float(source_cfg["fps"]),
                pattern=str(source_cfg.get("pattern", "*.jpg")),
                rate=rate,
            )
        from vividly.sources.replay import ReplaySource

        return ReplaySource(self.source_path, rate)

    # -- lifecycle ---------------------------------------------------------
    def _build_reid(self) -> Any:
        """The optional re-ID stage, or None. Imports nothing when re-ID is off."""
        if self.reid_cfg is None or not self.reid_cfg.enabled:
            return None
        from vividly.reid.stage import ReidConfigError, build_reid_stage

        try:
            return build_reid_stage(
                self.reid_cfg,
                store=self.store,
                book=self.book,
                reap_after_s=self.track_cfg.reap_after_s,
                repo_root=self.repo_root,
                # Always a real id, never None. An unscoped stage writes NULL into the
                # column the store's two partial-unique identity indexes are keyed on, and a
                # NULL makes every row distinct in a UNIQUE index, so the database could not
                # refuse a second confirm onto one departed identity. The runner is the
                # caller that shares store files across runs, so it is the caller that must
                # not be the degenerate case.
                run_id=self.run_id,
            )
        except (ReidConfigError, FileNotFoundError) as exc:
            # The store and the latency file are already open at this point, so a failure
            # here has to close them itself: a bad embedder path must not leave a
            # half-written run directory behind, which is the same promise the config
            # validation above makes.
            self._latency_fh.close()
            self.store.close()
            raise ConfigError(f"`reid`: {exc}") from exc

    def request_stop(self) -> None:
        """Ask a running `run()` to return after the tick in flight. Safe from any thread.

        Idempotent and non-blocking: it sets a flag and returns, it does not wait for the
        worker and it does not close anything. `serve` sets it and then joins, which is the
        ordering that keeps the session-end deletion out of a race with an enrollment.
        """
        self._stop.set()

    @property
    def stop_requested(self) -> bool:
        """True once `request_stop` (or `close`) has asked the loop to end."""
        return self._stop.is_set()

    def close(self) -> None:
        """Write the latency summary, close the store and release the source. Idempotent.

        The summary is written here rather than at the end of run() so that an interrupted
        run still leaves its percentiles on disk. A run that was cut short is exactly the
        run somebody wants the numbers from.

        Asks the loop to stop first. A caller that closes while a worker thread is still
        ticking (a hard interrupt, a test, a caller that forgot to join) is not the path this
        is designed around, but it must not corrupt anything: the flag ends the loop at the
        next tick boundary, and a tick already inside this one is handled by `run`, which
        treats a failure raised while the pipeline is closing as the shutdown it is rather
        than as an error. What that combination cannot produce, whatever the interleaving,
        is an enrollment that survives the deletion: `ReidStage` re-checks its own cleared
        flag inside the lock that guards both (CQ-16).
        """
        self._stop.set()
        if self._closed:
            return
        self._closed = True
        self._shutdown_reid()
        self._write_summary()
        self._latency_fh.close()
        self.store.close()
        close = getattr(self._source, "close", None)
        if callable(close):
            close()

    def _shutdown_reid(self) -> None:
        """Snapshot the re-ID counters, then empty the gallery. Session-scoped retention.

        The snapshot has to happen first because clearing resets the counters along with
        the embeddings, and the summary still has to be able to say how many identities the
        session held. After this returns, the appearance embeddings of everyone in that
        waiting room are gone from the process; what is left on disk is the alert store,
        which holds no embedding, no crop and no appearance vector of any kind.

        This runs before the latency summary and before anything is printed, so the line
        the summary prints about deletion is a statement about what already happened.
        CQ-16 in docs/COUNSEL_QUESTIONS.md is the counsel question it answers in code.
        """
        if self.reid is None:
            return
        stats = self.reid.stats()
        # Read while the store is still open: close() shuts it immediately after this.
        self._reid_pending = len(self.reid.pending(limit=PENDING_SUMMARY_LIMIT))
        self.reid.close()
        # The counts the session held, with the deletion flag flipped to what is now true.
        # Re-reading stats() after the clear would report zero identities, which reads as
        # "nobody was ever enrolled" rather than "everybody enrolled has been forgotten",
        # and those are different claims about the same run.
        self._reid_stats = replace(stats, cleared=True)

    def __enter__(self) -> Pipeline:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- perception --------------------------------------------------------
    def _build_perception(self) -> None:
        """Build the real backends. Only ever called on the footage path.

        The factory is the single registry of backends (it refuses an unknown key rather
        than importing an arbitrary module path), so this script names no backend module
        and swapping weights stays a config change.
        """
        from vividly.models.factory import build_detector, build_pose, build_tracker

        for key in ("detector", "pose", "tracker"):
            if not isinstance(self.cfg.get(key), Mapping):
                raise ConfigError(f"a footage run needs a `{key}` block naming a backend")
        self._detector = build_detector(self._with_local_graph(dict(self.cfg["detector"])))
        pose_spec = self._with_local_graph(dict(self.cfg["pose"]))
        # `pose.color_order` is a runner concern, not a backend kwarg (same contract and
        # wording as scripts/pose/eval_pose_ladder.py): rtmlib never converts colour and
        # every RTMPose graph is RGB-trained, so `rgb` swaps the image handed to the pose
        # stage only and the detector keeps its own convention. **The default stays "bgr".**
        # The live demo subclasses this class (DemoPipeline), and its behaviour may only
        # change through its own config. tests/test_pipeline_color_order.py fences that:
        # configs/pipeline/ and src/vividly/demo/ must never set the key.
        self._pose_color_order = str(pose_spec.pop("color_order", "bgr")).lower()
        if self._pose_color_order not in ("bgr", "rgb"):
            raise ConfigError(
                f"pose.color_order must be bgr or rgb, got {self._pose_color_order!r}"
            )
        self._pose = build_pose(pose_spec)
        self._tracker = build_tracker(dict(self.cfg["tracker"]))

    def _with_local_graph(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Resolve a repository-relative `onnx` path against the repository root.

        The committed graphs live under final_models/<lane>/model.onnx and a config names
        them relative to the repository, so the config reads the same from any working
        directory. A relative path that does not exist under the root is handed through
        untouched: the rtmlib backend treats a non-existent path as a zoo asset name, and
        the RT-DETRv4 backend reports the missing file itself. A path that resolves to a
        Git LFS pointer is refused here by name, because neither backend would recognise
        the pointer for what it is.
        """
        onnx = spec.get("onnx")
        if isinstance(onnx, str) and not Path(onnx).is_absolute():
            candidate = self.repo_root / onnx
            if candidate.is_file():
                if is_lfs_pointer(candidate):
                    raise ConfigError(
                        f"{onnx} is a Git LFS pointer, not the graph; run `git lfs pull` "
                        "to fetch it"
                    )
                spec["onnx"] = str(candidate)
        return spec

    def _observations_from(self, tracked: Any, poses: Any, t: float) -> list[TrackObservation]:
        """Tracked boxes plus their poses, as the track layer's own record type.

        Poses are index-aligned with the tracked boxes because pose ran on those boxes.
        Keypoint confidences carry through untouched: TrackConfig.kp_conf_min is the only
        place a confidence floor is applied, and applying one here as well would make the
        configured value a lie.
        """
        out: list[TrackObservation] = []
        track_ids = getattr(tracked, "track_ids", None)
        if track_ids is None:
            return out
        for i in range(len(tracked)):
            x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
            keypoints = None
            if poses is not None and i < len(poses):
                keypoints = tuple(
                    (float(x), float(y), float(conf))
                    for (x, y), conf in zip(poses.keypoints[i], poses.scores[i], strict=True)
                )
            out.append(
                TrackObservation(
                    track_id=int(track_ids[i]),
                    t=t,
                    cx=(x1 + x2) / 2.0,
                    cy=(y1 + y2) / 2.0,
                    w=x2 - x1,
                    h=y2 - y1,
                    keypoints=keypoints,
                )
            )
        return out

    # -- rules and persistence --------------------------------------------
    def _rules_step(self, observations: list[TrackObservation], t: float) -> list[RuleFiring]:
        """Ingest one frame's observations, advance the clock, evaluate, persist.

        `reap` is called with the frame's own time and not just left to `ingest`, because
        an empty frame carries no timestamp: without this a room that emptied would freeze
        the book's clock and every rule window with it.

        Persisting the firings happens inside this stage. There is no seventh stage name in
        the contract, and a SQLite insert on the same thread is where the write cost lands.
        """
        self.book.ingest(observations)
        self.book.reap(t)
        firings = self.engine.tick(self.book, t)
        for firing in firings:
            self._persist(firing)
        return firings

    def _persist(self, firing: RuleFiring) -> None:
        """Write one firing as an AlertRecord. A duplicate id is a no-op, not an error.

        Alert ids are deterministic (rule, track, onset millisecond), so re-running the
        same clip re-derives the same ids: the second run updates nothing and doubles
        nothing, which is what makes a replay run safe to repeat.
        """
        track = self.book.tracks().get(firing.track_id)
        if track is not None:
            mrn_binding = track.mrn_binding
        else:
            # An egress firing names a track the book has already reaped; the monitor
            # carries the mock binding it had when it left in the firing's own details.
            carried = firing.details.get("mrn_binding")
            mrn_binding = carried if isinstance(carried, str) else None
        record = AlertRecord(
            alert_id=make_alert_id(firing.rule_id, firing.track_id, firing.onset_t),
            track_id=firing.track_id,
            rule_id=firing.rule_id,
            rule_version=firing.rule_version,
            onset_ts=firing.onset_t,
            detected_ts=firing.detected_t,
            created_utc=utc_now_iso(),
            baseline=None if firing.baseline is None else asdict(firing.baseline),
            details=dict(firing.details),
            mrn_binding=mrn_binding,
        )
        try:
            self.store.insert_alert(record)
        except DuplicateAlertError:
            return
        self.n_alerts += 1

    # -- publication -------------------------------------------------------
    def _render(self, image: Any) -> str | None:
        """JPEG-encode one frame for the page. This is the whole `render` stage.

        It runs in headless mode too. Encoding is a real cost of showing a frame to a
        reviewer, and a headless latency table that quietly excluded it would not describe
        the run the dashboard actually performs. In zero-footage mode there is no image and
        the stage is skipped entirely, so it reports null rather than a flattering zero.
        """
        return encode_jpeg_b64(image, self.jpeg_quality)

    def _publish(self, frame_index: int, t: float, frame_b64: str | None) -> None:
        """Hand the finished payload to the dashboard state.

        Deliberately outside the render timing: this is a dict copy under a lock, not work
        the pipeline does on a frame, and folding it into a stage would inflate the only
        number on the page that describes rendering.
        """
        self.state.publish(
            frame_b64=frame_b64,
            frame_index=frame_index,
            t_source=t,
            latency_ms=self.timer.frame_ms(),
            latency_p50_ms=self.timer.quantile_ms(50.0),
            latency_p95_ms=self.timer.quantile_ms(95.0),
        )

    def _log_latency(self, frame_index: int, t: float, n_tracks: int, n_alerts: int) -> None:
        """One line per processed frame. `ms` always carries all six stage keys."""
        row = {
            "frame_index": frame_index,
            "t_source": t,
            "n_tracks": n_tracks,
            "n_alerts": n_alerts,
            "ms": self.timer.frame_ms(),
        }
        self._latency_fh.write(json.dumps(row, sort_keys=True) + "\n")

    def _write_summary(self) -> None:
        """The last line of latency.jsonl: the same p50 and p95 the exit table prints.

        One artifact rather than two, and tagged with `summary: true` so a reader walking
        per-frame rows can skip it on a key rather than on a line number.
        """
        row: dict[str, Any] = {
            "summary": True,
            "run_name": self.run_name,
            "mode": self.mode,
            "n_frames": self.n_frames,
            "n_samples": self.timer.n_samples(),
            "p50_ms": self.timer.quantile_ms(50.0),
            "p95_ms": self.timer.quantile_ms(95.0),
            "threshold_status": "PROVISIONAL (see docs/CALIBRATION.md)",
        }
        if self._reid_stats is not None:
            # Only present when re-ID ran. An absent key means the stage was off, which is
            # not the same claim as a row of zeros and must not read like one.
            row["reid"] = self._reid_stats.to_dict()
        if self.egress is not None:
            # Same rule: present only when the egress head ran.
            row["egress"] = self.egress.stats()
        self._latency_fh.write(json.dumps(row, sort_keys=True) + "\n")

    # -- the loop ----------------------------------------------------------
    def _ticks(self) -> Iterator[tuple[int, float, list[TrackObservation], Any]]:
        """Yield (frame_index, t_source, observations, image), one per pipeline tick.

        The two modes differ only in what the source stage returns and in whether the
        detect, pose and track stages run at all. Everything after this generator is
        identical, which is the point: the zero-footage flow is not a mock of the pipeline,
        it is the pipeline with the perception stack lifted out.
        """
        return self._synthetic_ticks() if self.zero_footage else self._replay_ticks()

    def _pull(self, stream: Iterator[Any]) -> Any:
        """Timed `next`, returning _EXHAUSTED at the end of the source.

        Not StageTimer.run, because the final call produces no frame: recording it would
        put one more sample in the source row than there are frames, and would average the
        cost of discovering end-of-file in with the cost of decoding.
        """
        t0 = time.perf_counter()
        item = next(stream, _EXHAUSTED)
        if item is _EXHAUSTED:
            return item
        self.timer.record("source", (time.perf_counter() - t0) * 1000.0, 1)
        return item

    def _synthetic_ticks(self) -> Iterator[tuple[int, float, list[TrackObservation], Any]]:
        """Zero-footage: each fixture line is one tick, and the image is always None.

        frame_index counts fixture lines rather than video frames, because there are no
        video frames. An empty line (an empty waiting room) carries no timestamp of its
        own, so it reuses the previous tick's clock instead of resetting it to zero.
        """
        stream = self._source.observations()
        index = 0
        last_t = 0.0
        while True:
            self.timer.start_frame()
            batch = self._pull(stream)
            if batch is _EXHAUSTED:
                return
            t = batch[0].t if batch else last_t
            last_t = t
            yield index, t, list(batch), None
            index += 1

    def _replay_ticks(self) -> Iterator[tuple[int, float, list[TrackObservation], Any]]:
        """Footage: decode a frame, detect, track, pose the tracked boxes, emit.

        Backends are built here rather than in __init__ so that constructing a Pipeline
        never loads a model: a config error, a missing clip or a dry run must not pay for
        an ONNX session, and the zero-footage flow must never touch this path at all.
        """
        self._build_perception()
        self._source = self._footage_source()
        info = self._source.open()
        # Warm up before the first measured frame: the first ONNX call carries allocator
        # and graph-init cost that would otherwise land in p95 and describe nothing.
        for model in (self._detector, self._pose):
            model.warmup(shape=(info.height, info.width))
        stream = self._source.frames()
        convert = None
        if self._pose_color_order == "rgb":
            import cv2  # lazy: only the rgb pose path needs cv2; the replay path stays cv2-free

            def convert(img: Any) -> Any:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        while True:
            self.timer.start_frame()
            frame = self._pull(stream)
            if frame is _EXHAUSTED:
                return
            detections = self.timer.run("detect", 1, self._detector.detect_one, frame.image)
            tracked = self.timer.run(
                "track", max(len(detections), 1), self._tracker.update, detections, frame.index
            )
            pose_img = frame.image if convert is None else convert(frame.image)
            poses = self.timer.run(
                "pose", max(len(tracked), 1), self._pose.estimate, pose_img, tracked.xyxy
            )
            observations = self._observations_from(tracked, poses, frame.t_source)
            yield frame.index, frame.t_source, observations, frame.image

    def _reid_step(self, observations: list[TrackObservation], t: float, image: Any) -> None:
        """Enrol whoever the book just reaped, probe whoever is new. A no-op when off.

        Runs after the rules stage, because the rules stage is what ingests and reaps: by
        the time this is called the tick's departures have already been handed to the stage
        and the tick's arrivals are already in the book.

        Outside `StageTimer.run` on purpose. The six stage names are the `STAGES` tuple
        the dashboard reads, so there is no seventh row to put this in,
        and putting it in `rules` would make the rules number describe work the rules did
        not do. The stage keeps its own wall clock and the summary prints it.
        """
        if self.reid is None:
            return
        result = self.reid.step(observations, t, image)
        self.n_proposals += len(result.proposals)

    def run(self, max_frames: int | None = None) -> int:
        """Process frames until the source ends or max_frames is reached.

        Returns the cumulative number of frames processed, and `max_frames` is a total
        budget rather than a per-call one: calling run again continues the same source
        instead of restarting it, so a caller can stop mid-clip, do something to the book
        (bind a chart id, answer a proposal) and carry on. A source is consumed once.

        Safe to call from a background thread while the dashboard serves: TrackBook
        mutation and the socket read both happen under the locks their owners provide.

        Returns early once `request_stop` has been called, at the next tick boundary.
        """
        if max_frames is not None and max_frames < 0:
            raise ValueError(f"max_frames must be >= 0 or None, got {max_frames}")
        if max_frames == 0:
            # Not the same as None. Zero frames means the source is never opened, which is
            # the only way to ask this script to validate a config and do nothing else.
            return 0
        if self._tick_iter is None:
            self._tick_iter = self._ticks()
        for frame_index, t, observations, image in self._tick_iter:
            if self._stop.is_set():
                break
            try:
                firings = self.timer.run(
                    "rules", max(len(observations), 1), self._rules_step, observations, t
                )
                self._reid_step(observations, t, image)
                frame_b64 = (
                    None if image is None else self.timer.run("render", 1, self._render, image)
                )
                self._publish(frame_index, t, frame_b64)
                self._log_latency(frame_index, t, len(self.book.tracks()), len(firings))
            except Exception:
                # Only a shutdown that landed mid-tick is swallowed, and only into ending
                # the loop. `close()` shuts the store, closes the latency file and clears
                # the gallery; a tick already inside this block when that happens raises
                # from whichever of the three it reaches first, and that is the shutdown
                # working, not a failure of the run. The condition is what keeps this from
                # being a bare except that hides a real bug: outside a close, every
                # exception propagates exactly as it did before.
                if self._closed or self._stop.is_set():
                    break
                raise
            self.n_frames += 1
            if max_frames is not None and self.n_frames >= max_frames:
                break
        if not self._closed:
            self._latency_fh.flush()
        return self.n_frames

    # -- reporting ---------------------------------------------------------
    def summary(self) -> str:
        """What the run did and where it put it. Printed on every exit path, after close.

        Printed after `close()` rather than before it so the re-ID block below describes a
        gallery that has already been emptied. "Cleared at session end" as a promise is
        worth nothing; as a post-condition with the retained count next to it, it is the
        thing CQ-16 asks about.
        """
        return (
            f"run      : {self.run_name} ({self.mode})\n"
            f"source   : {self.source_path}\n"
            f"frames   : {self.n_frames}\n"
            f"tracks   : {len(self.book.tracks())} live at exit\n"
            f"alerts   : {self.n_alerts} newly written to {self.out_dir / ALERT_DB_NAME}\n"
            f"{self._egress_summary()}"
            f"latency  : {self.latency_path}\n"
            f"{self._reid_summary()}"
            "\n"
            f"{self.timer.table()}\n"
            "\n"
            "Every threshold in this run is PROVISIONAL: no count above is a "
            "false-alarm rate. See docs/CALIBRATION.md."
        )

    def _egress_summary(self) -> str:
        """One line when the egress head ran; nothing when it was off (absent, not zero)."""
        if self.egress is None:
            return ""
        s = self.egress.stats()
        return (
            f"egress   : {s['n_fired']} candidate(s) fired, {s['n_pending']} pending, "
            f"{s['n_departures']} departure(s) seen (staff {s['n_suppressed_staff']}, "
            f"discharged {s['n_suppressed_discharged'] + s['n_discharged_after_departure']}, "
            f"outside a zone {s['n_left_outside_zone']}, short {s['n_short_presence']}, "
            f"returned {s['n_returned']}); zones {s['zones']}\n"
        )

    def _reid_summary(self) -> str:
        """The re-ID block, or nothing at all when the stage was off.

        Absent rather than zeroed when re-ID did not run, for the same reason the latency
        table prints a dash for a stage that did not run: a row of zeros reads as "it ran
        and found nothing", which is a different and much stronger claim.
        """
        if self.reid is None:
            return ""
        stats = self._reid_stats
        if stats is None:  # pragma: no cover - summary before close, not a shipped path
            stats = self.reid.stats()
        retained = self.reid.gallery.size()
        identities = _plural(stats.n_identities, "identity", "identities")
        probes = _plural(stats.n_probes, "probe", "probes")
        return (
            f"re-ID    : {stats.source} embeddings, tau {stats.tau:g} (PROVISIONAL); "
            f"{identities} enrolled, {probes}\n"
            f"proposals: {stats.n_proposals} written, NONE auto-bound; "
            f"{self._reid_pending} awaiting a human on the dashboard\n"
            f"gallery  : CLEARED at session end, {retained} embeddings retained "
            "(CQ-16 session-scoped retention)\n"
        )


# ------------------------------------------------------------------- dashboard
def serve(pipeline: Pipeline, max_frames: int | None = None) -> None:
    """Run the pipeline in a background thread and serve the review page over loopback.

    uvicorn owns the main thread because it installs signal handlers there. The pipeline
    thread is a daemon so that a worker which will not stop cannot keep the process alive.

    Stopping is part of this function, and it is why the join is in a finally. When uvicorn
    returns (the operator pressed ctrl-c, or the server stopped for any other reason) this
    asks the pipeline to stop and waits for the tick in flight, before returning to `main`,
    whose finally block calls `Pipeline.close()`. Close is what empties the session gallery
    (CQ-16) and shuts the store; leaving a thread mid-tick against both of those is a race
    with an enrollment on one side and a patient identity on the other, and "the thread is a
    daemon so the interpreter will not wait for it" is not an answer to that.

    If the join times out (`WORKER_JOIN_TIMEOUT_S`), which means one tick has been stuck far
    longer than a frame of work: this says so on stderr and returns anyway. `close()` then
    runs against a thread that is still going, which is exactly the pre-existing race and is
    the best available answer, because blocking a shutdown forever on a wedged decode is
    worse. The two things that stay true in that case: the store translates every failure
    the doomed tick can hit into an `EventStoreError` rather than a corrupt row, and the
    re-ID stage re-checks its cleared flag inside the lock that guards the gallery, so a
    tick that is mid-enrollment when the deletion lands still cannot leave an appearance
    vector behind it.
    """
    app = create_app(pipeline.state)
    worker = threading.Thread(
        target=pipeline.run,
        kwargs={"max_frames": max_frames},
        name="vividly-pipeline",
        daemon=True,
    )
    worker.start()
    host = str(pipeline.dashboard_cfg.get("host", "127.0.0.1"))
    port = int(pipeline.dashboard_cfg.get("port", 8600))
    print(f"dashboard: http://{host}:{port}  (replay only, thresholds PROVISIONAL)")
    try:
        uvicorn.run(
            app, host=host, port=port, log_level=str(pipeline.dashboard_cfg.get("log", "warning"))
        )
    finally:
        pipeline.request_stop()
        worker.join(WORKER_JOIN_TIMEOUT_S)
        if worker.is_alive():
            print(
                f"pipeline thread did not stop within {WORKER_JOIN_TIMEOUT_S:g} s; closing "
                "anyway. The gallery is still emptied at session end and no proposal can be "
                "written after it, but this run's last tick may be incomplete.",
                file=sys.stderr,
            )


# ------------------------------------------------------------------------ main
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path, help="configs/pipeline/<cfg>.yaml")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="no dashboard: write alerts.sqlite and latency.jsonl, print the table, exit",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="stop after N frames (smoke runs and CI)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
        pipeline = Pipeline(cfg, REPO_ROOT)
    except ValueError as exc:
        # ConfigError is a ValueError, and so is every other way this block can fail: the
        # thresholds validate themselves in RuleConfig.__post_init__ and TrackConfig, and a
        # non-numeric value raises out of an int() or float() cast. All of them are the
        # operator's config being wrong, so all of them exit 3 rather than printing a
        # traceback that buries the one line saying which number was rejected.
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    headless = bool(args.headless or cfg.get("headless", False))
    try:
        if headless:
            pipeline.run(max_frames=args.max_frames)
        else:
            serve(pipeline, max_frames=args.max_frames)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
    finally:
        # close() before summary(), not after. Closing is what empties the session gallery
        # (CQ-16), and the summary's re-ID block reports the retained count, so printing
        # first would turn a post-condition into a promise. close() writes the latency
        # percentiles as its first act, so an interrupted run still leaves them on disk.
        pipeline.close()
        print(pipeline.summary())
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
