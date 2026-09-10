"""ClipEmitter: the perception side of the Clip Library boundary (design P3, 2026-09-09).

Hooked after `Pipeline._persist` commits an AlertRecord. For the two allowlisted rule
heads only, it watches the TrackBook through the post-roll and then writes one JSON file
per firing into the spool directory the Clip Library reads:

  * fall_then_still_v0: the window opens `pre_roll_s` before the onset and closes at the
    first contact plus `contact_tail_s`, or at `post_roll_max_s` after the firing, whichever
    is first. First contact is measured here: the first moment another track's box stays
    within `contact_px` of the down person's box for `contact_min_s`. Its source is
    `staff_boundary` only when that track's own history crossed the configured staff
    boundary polygon, otherwise `another_person`; nothing about appearance or uniform is
    ever inferred.
  * egress_without_discharge_v0: the window ends at the departure (last sighting plus
    `egress_tail_s`) and is written at once; the emitter then keeps watching for a return
    through `no_return_s` and writes a stage-two update record (`absent_s`, `returned`).

What the record carries is reduced before writing: every `details` key outside the
allowlist below is dropped, `baseline` keeps only `n_samples` and `complete`, the mock
chart binding is dropped, and the basis names only the fired rule's own thresholds and its
own file hash. The file name is `sha256(alert_id)[:16]` and carries no rule id. A firing
of any other rule head produces no file and no side effect (tests/test_clip_emitter.py).

Import rule: standard library plus vividly.tracks and vividly.schemas only. No numpy, no
cv2, no model runtime: the module imports on a runner with none of them installed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vividly.schemas.events import AlertRecord
from vividly.tracks import TrackBook, TrackState

SCHEMA_VERSION = 1
SPOOL_ID_CHARS = 16

#: The only rule heads whose firings may become a Clip (design D-C). Mirrors
#: vividly_clip.vocabulary.CLIP_TRIGGER_RULES; the two repositories share no import.
CLIP_TRIGGER_RULES: frozenset[str] = frozenset(
    {"fall_then_still_v0", "egress_without_discharge_v0"}
)

#: `details` keys that may travel, per rule: the measured geometry the Clip Library maps
#: into its observation card (DETAILS_ALLOWLIST there) plus the threshold and internal
#: estimate keys its basis panel shows as PROVISIONAL (BASIS_DETAIL_KEYS there). Everything
#: else, including the mock chart binding and the last box position, is dropped (CL-12).
DETAILS_ALLOWLIST: Mapping[str, frozenset[str]] = {
    "fall_then_still_v0": frozenset(
        {
            "fall_s",
            "fall_drop_px",
            "down_s",
            "n_down_samples",
            "down_stale_s",
            "upright_deg",
            "down_deg_newest",
            "down_mean_step_px",
            "threshold_down_move_max_px",
            "threshold_down_still_min_s",
            "threshold_fall_drop_min_px",
            "threshold_fall_max_s",
            "threshold_status",
        }
    ),
    "egress_without_discharge_v0": frozenset(
        {
            "exit_zone",
            "absent_s",
            "presence_s",
            "returned",
            "threshold_egress_absent_min_s",
            "threshold_egress_exit_window_s",
            "threshold_egress_min_presence_s",
            "threshold_status",
        }
    ),
}

#: The only baseline keys that travel (L24).
BASELINE_KEYS: tuple[str, ...] = ("n_samples", "complete")

SOURCE_KINDS = ("live", "replay_file", "synthetic_jsonl")


def _expand_path(value: Any) -> str:
    expanded = os.path.expandvars(str(value))
    if "${" in expanded or expanded.startswith("$"):
        raise ValueError(f"path {value!r} names an environment variable that is not set")
    return expanded


def spool_id_for(alert_id: str) -> str:
    return hashlib.sha256(alert_id.encode("utf-8")).hexdigest()[:SPOOL_ID_CHARS]


def utc_iso(epoch_s: float) -> str:
    return (
        datetime.fromtimestamp(epoch_s, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


# --------------------------------------------------------------------- config
@dataclass(frozen=True)
class ClipsConfig:
    """The `clips:` block of a pipeline profile. Its presence is what enables the emitter."""

    spool_dir: Path
    camera_id: str
    source_kind: str
    zone_id: str | None = None
    exit_zone_id: str | None = None
    demo_mode: bool = False
    source_media_path: str | None = None
    pre_roll_s: float = 30.0
    post_roll_max_s: float = 120.0
    contact_tail_s: float = 15.0
    floor_cap_s: float = 240.0
    pre_roll_egress_s: float = 60.0
    egress_tail_s: float = 15.0
    egress_cap_s: float = 90.0
    no_return_s: float = 1200.0
    contact_px: float = 150.0
    contact_min_s: float = 3.0
    obstruction_gap_s: float = 1.0
    staff_boundary: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if self.source_kind not in SOURCE_KINDS:
            raise ValueError(
                f"clips.source_kind must be one of {SOURCE_KINDS}, got {self.source_kind!r}"
            )
        if self.source_kind != "live" and not self.source_media_path:
            raise ValueError("clips.source_media_path is required unless source_kind is live")
        if not self.camera_id:
            raise ValueError("clips.camera_id is required")
        for name in (
            "pre_roll_s",
            "post_roll_max_s",
            "contact_tail_s",
            "pre_roll_egress_s",
            "egress_tail_s",
            "no_return_s",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"clips.{name} must be >= 0")
        for name in ("contact_px", "contact_min_s", "floor_cap_s", "egress_cap_s"):
            if not getattr(self, name) > 0:
                raise ValueError(f"clips.{name} must be > 0")
        if self.staff_boundary is not None and len(self.staff_boundary) < 3:
            raise ValueError("clips.staff_boundary needs at least three points")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, default_source_kind: str) -> ClipsConfig:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown clips keys: {unknown}; known keys are {sorted(known)}")
        kwargs: dict[str, Any] = dict(data)
        kwargs["spool_dir"] = Path(_expand_path(data["spool_dir"])).expanduser()
        kwargs.setdefault("source_kind", default_source_kind)
        if kwargs.get("source_media_path") is not None:
            kwargs["source_media_path"] = _expand_path(kwargs["source_media_path"])
        if kwargs.get("staff_boundary") is not None:
            kwargs["staff_boundary"] = tuple(
                (float(p[0]), float(p[1])) for p in kwargs["staff_boundary"]
            )
        for name in (
            "pre_roll_s",
            "post_roll_max_s",
            "contact_tail_s",
            "floor_cap_s",
            "pre_roll_egress_s",
            "egress_tail_s",
            "egress_cap_s",
            "no_return_s",
            "contact_px",
            "contact_min_s",
            "obstruction_gap_s",
        ):
            if name in kwargs:
                kwargs[name] = float(kwargs[name])
        if "demo_mode" in kwargs:
            kwargs["demo_mode"] = bool(kwargs["demo_mode"])
        return cls(**kwargs)


@dataclass(frozen=True)
class BasisFacts:
    """What the run knows about itself; copied into every record's `basis`."""

    rule_versions: Mapping[str, str]
    rule_thresholds: Mapping[str, Mapping[str, Any]]
    rule_file_sha256: Mapping[str, str]
    pipeline_config_sha256: str
    model_ids: tuple[str, ...] = ()
    onnx_sha256: tuple[str, ...] = ()
    checkpoint_sha256: tuple[str, ...] = ()
    tracker_backend: str | None = None
    tracker_version: str | None = None
    software_git_sha: str | None = None


class SourceClock:
    """Wall clock for source time: `wall_at(t) = epoch + t`. The pipeline re-bases the epoch
    from every live frame's own `wall_utc` stamp; replay and fixture runs keep the epoch
    taken when the run opened."""

    def __init__(self, epoch_s: float) -> None:
        self.epoch_s = epoch_s

    @property
    def epoch_utc(self) -> str:
        return utc_iso(self.epoch_s)

    def rebase(self, *, t_source: float, wall_s: float) -> None:
        self.epoch_s = wall_s - t_source

    def wall_at(self, t: float) -> str:
        return utc_iso(self.epoch_s + t)


# ------------------------------------------------------------------ geometry
def _box_gap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Distance between two (cx, cy, w, h) boxes' edges; 0 when they overlap."""
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return math.hypot(dx, dy)


def _inside(polygon: Sequence[tuple[float, float]], x: float, y: float) -> bool:
    inside = False
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            cross = (x1 - x0) * (y - y0) / (y1 - y0) + x0
            if x < cross:
                inside = not inside
    return inside


def _last_box(track: TrackState) -> tuple[float, float, float, float] | None:
    obs = track.last_observation
    if obs is None:
        return None
    return obs.cx, obs.cy, obs.w, obs.h


def track_boxes(
    track: TrackState, start_t: float, end_t: float, now: float
) -> list[dict[str, float]]:
    """`[{t, x, y, w, h}]` of one track between start_t and end_t, in source pixels."""
    lookback = max(now - start_t, 0.0) + 1.0
    positions = track.positions(lookback)
    boxes = track.boxes(lookback)
    out: list[dict[str, float]] = []
    for (t, cx, cy), (_, w, h, _) in zip(positions, boxes, strict=False):
        if start_t <= t <= end_t:
            out.append(
                {
                    "t": round(t, 3),
                    "x": round(cx - w / 2, 1),
                    "y": round(cy - h / 2, 1),
                    "w": round(w, 1),
                    "h": round(h, 1),
                }
            )
    return out


# ------------------------------------------------------------------ windows
@dataclass
class _FloorWindow:
    record: AlertRecord
    start_t: float
    deadline_t: float
    cap_t: float
    contact_t: float | None = None
    contact_source: str = "none"
    dwell_since: dict[int, float] = field(default_factory=dict)
    last_other_near_t: float | None = None
    end_t: float | None = None
    boxes: list[dict[str, float]] = field(default_factory=list)
    last_seen_t: float | None = None
    obstructed_s: float = 0.0
    latency_line: int = 0


@dataclass
class _ReturnWatch:
    record: AlertRecord
    last_seen_t: float
    deadline_t: float


@dataclass
class _ReapSnapshot:
    boxes: list[dict[str, float]]
    reaped_t: float


class ClipEmitter:
    """Writes `<spool_dir>/<spool_id>.json` for allowlisted firings, after the post-roll."""

    def __init__(
        self,
        cfg: ClipsConfig,
        *,
        book: TrackBook,
        run_id: str,
        basis: BasisFacts,
        clock: SourceClock,
        latency_line: Callable[[], int] = lambda: 0,
    ) -> None:
        self.cfg = cfg
        self.book = book
        self.run_id = run_id
        self.basis = basis
        self.clock = clock
        self.latency_line = latency_line
        self.spool_dir = cfg.spool_dir
        self.spool_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._floor: dict[str, _FloorWindow] = {}
        self._returns: dict[str, _ReturnWatch] = {}
        self._reaped: dict[int, _ReapSnapshot] = {}
        self._now: float = math.nan
        self.n_written = 0
        self.n_updates = 0
        self.n_dropped = 0
        self.written: list[Path] = []

    # -- hooks -------------------------------------------------------------------
    def on_reap(self, track: TrackState) -> None:
        """TrackBook reap watcher: keep the departed track's recent boxes for the egress
        record, which is written long after the track is gone."""
        now = track.last_seen_t
        start = now - self.cfg.pre_roll_egress_s - 1.0
        self._reaped[track.track_id] = _ReapSnapshot(
            boxes=track_boxes(track, start, now + 1.0, now), reaped_t=now
        )
        self._prune_reaped(now)

    def on_alert(self, record: AlertRecord) -> None:
        """Called after the store commits. Non-allowlisted rules produce nothing at all."""
        if record.rule_id not in CLIP_TRIGGER_RULES:
            self.n_dropped += 1
            return
        if record.rule_id == "fall_then_still_v0":
            self._open_floor(record)
        else:
            self._write_exit(record)

    def tick(self, now: float) -> None:
        """Once per frame after the rules ran: contacts, window closes, return watches."""
        if math.isnan(now):
            return
        self._now = now
        live = self.book.tracks()
        for spool_id, window in list(self._floor.items()):
            self._advance_floor(window, live, now)
            end = window.end_t if window.end_t is not None else min(window.deadline_t, window.cap_t)
            if now >= end:
                window.end_t = end
                self._close_floor(spool_id, window, live, now)
        for spool_id, watch in list(self._returns.items()):
            returned = any(
                track.reassociated_from == watch.record.track_id for track in live.values()
            )
            if returned or now >= watch.deadline_t:
                absent = (now if returned else watch.deadline_t) - watch.last_seen_t
                self._write_update(spool_id, absent_s=absent, returned=returned)
                del self._returns[spool_id]
        self._prune_reaped(now)

    def flush(self) -> None:
        """At close: every open floor window closes at the last tick. Return watches are
        left as they are; a run that ends before `no_return_s` has nothing to report."""
        if math.isnan(self._now):
            return
        live = self.book.tracks()
        for spool_id, window in list(self._floor.items()):
            window.end_t = min(self._now, window.cap_t)
            self._close_floor(spool_id, window, live, self._now)

    # -- floor windows -------------------------------------------------------------
    def _open_floor(self, record: AlertRecord) -> None:
        start = record.onset_ts - self.cfg.pre_roll_s
        self._floor[spool_id_for(record.alert_id)] = _FloorWindow(
            record=record,
            start_t=start,
            deadline_t=record.detected_ts + self.cfg.post_roll_max_s,
            cap_t=start + self.cfg.floor_cap_s,
            latency_line=self.latency_line(),
        )

    def _advance_floor(
        self, window: _FloorWindow, live: Mapping[int, TrackState], now: float
    ) -> None:
        down = live.get(window.record.track_id)
        if down is None:
            snap = self._reaped.get(window.record.track_id)
            if snap is not None and not window.boxes:
                window.boxes = snap.boxes
            return
        # boxes of the down person, refreshed while the track lives
        window.boxes = track_boxes(down, window.start_t, now, now)
        if window.last_seen_t is not None and down.last_seen_t > window.last_seen_t:
            gap = down.last_seen_t - window.last_seen_t
            if gap > self.cfg.obstruction_gap_s and window.record.onset_ts <= down.last_seen_t:
                window.obstructed_s += gap
        window.last_seen_t = down.last_seen_t
        if window.contact_t is not None:
            return
        down_box = _last_box(down)
        if down_box is None:
            return
        near_now: set[int] = set()
        for track_id, other in live.items():
            if track_id == window.record.track_id or other.last_seen_t < now - 1.0:
                continue
            box = _last_box(other)
            if box is None:
                continue
            gap = _box_gap(down_box, box)
            if gap <= 3.0 * self.cfg.contact_px and now <= window.record.onset_ts:
                window.last_other_near_t = now
            if gap <= self.cfg.contact_px:
                near_now.add(track_id)
                since = window.dwell_since.setdefault(track_id, now)
                if now - since >= self.cfg.contact_min_s:
                    window.contact_t = since
                    window.contact_source = (
                        "staff_boundary" if self._crossed_boundary(other, now) else "another_person"
                    )
                    window.end_t = min(
                        window.contact_t + self.cfg.contact_tail_s, window.deadline_t, window.cap_t
                    )
                    return
        for track_id in list(window.dwell_since):
            if track_id not in near_now:
                del window.dwell_since[track_id]

    def _crossed_boundary(self, track: TrackState, now: float) -> bool:
        polygon = self.cfg.staff_boundary
        if polygon is None:
            return False
        for _, cx, cy in track.positions(max(now - track.first_seen_t, 1.0) + 1.0):
            if _inside(polygon, cx, cy):
                return True
        return False

    def _close_floor(
        self, spool_id: str, window: _FloorWindow, live: Mapping[int, TrackState], now: float
    ) -> None:
        del self._floor[spool_id]
        record = window.record
        end_t = window.end_t if window.end_t is not None else now
        down = live.get(record.track_id)
        boxes = track_boxes(down, window.start_t, end_t, now) if down is not None else window.boxes
        if window.contact_t is not None:
            staff_absent = window.contact_t - (
                window.last_other_near_t if window.last_other_near_t is not None else window.start_t
            )
        else:
            staff_absent = end_t - (
                window.last_other_near_t if window.last_other_near_t is not None else window.start_t
            )
        payload = self._record(
            record,
            zone_id=self.cfg.zone_id,
            start_t=window.start_t,
            end_t=end_t,
            pre_s=self.cfg.pre_roll_s,
            post_max_s=self.cfg.post_roll_max_s,
            contact_t=window.contact_t,
            contact_source=window.contact_source,
            view_obstructed_s=round(window.obstructed_s, 3),
            staff_absent_s=round(max(staff_absent, 0.0), 3),
            boxes=[b for b in boxes if window.start_t <= b["t"] <= end_t],
            latency_line=window.latency_line,
        )
        self._write(spool_id, payload)

    # -- exits -------------------------------------------------------------------
    def _write_exit(self, record: AlertRecord) -> None:
        last_seen = record.onset_ts
        start = last_seen - self.cfg.pre_roll_egress_s
        end = min(last_seen + self.cfg.egress_tail_s, start + self.cfg.egress_cap_s)
        snap = self._reaped.get(record.track_id)
        boxes = [b for b in (snap.boxes if snap else []) if start <= b["t"] <= end]
        spool_id = spool_id_for(record.alert_id)
        payload = self._record(
            record,
            zone_id=self.cfg.exit_zone_id,
            start_t=start,
            end_t=end,
            pre_s=self.cfg.pre_roll_egress_s,
            post_max_s=self.cfg.egress_tail_s,
            contact_t=None,
            contact_source="none",
            view_obstructed_s=0.0,
            staff_absent_s=None,
            boxes=boxes,
            latency_line=self.latency_line(),
        )
        self._write(spool_id, payload)
        self._returns[spool_id] = _ReturnWatch(
            record=record, last_seen_t=last_seen, deadline_t=last_seen + self.cfg.no_return_s
        )

    def _write_update(self, spool_id: str, *, absent_s: float, returned: bool) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": "update",
            "spool_id": spool_id,
            "absent_s": round(max(absent_s, 0.0), 3),
            "returned": bool(returned),
            "updated_utc": self.clock.wall_at(self._now),
        }
        self._atomic_write(self.spool_dir / f"{spool_id}.update.json", payload)
        self.n_updates += 1

    # -- the record ----------------------------------------------------------------
    def _reduced_alert(self, record: AlertRecord) -> dict[str, Any]:
        allowed = DETAILS_ALLOWLIST[record.rule_id]
        baseline = None
        if record.baseline:
            baseline = {k: record.baseline[k] for k in BASELINE_KEYS if k in record.baseline}
        return {
            "schema_version": record.schema_version,
            "alert_id": record.alert_id,
            "track_id": record.track_id,
            "rule_id": record.rule_id,
            "rule_version": record.rule_version,
            "onset_ts": record.onset_ts,
            "detected_ts": record.detected_ts,
            "created_utc": record.created_utc,
            "baseline": baseline,
            "details": {k: v for k, v in record.details.items() if k in allowed},
            "feedback": None,
            "mrn_binding": None,
        }

    def _basis(self, record: AlertRecord) -> dict[str, Any]:
        b = self.basis
        baseline = {"n_samples": 0, "complete": False}
        if record.baseline:
            baseline = {
                "n_samples": int(record.baseline.get("n_samples", 0)),
                "complete": bool(record.baseline.get("complete", False)),
            }
        return {
            "rule_id": record.rule_id,
            "rule_version": record.rule_version,
            "rule_file_sha256": b.rule_file_sha256.get(record.rule_id, b.pipeline_config_sha256),
            "thresholds": dict(b.rule_thresholds.get(record.rule_id, {})),
            "model_ids": list(b.model_ids),
            "onnx_sha256": list(b.onnx_sha256),
            "checkpoint_sha256": list(b.checkpoint_sha256),
            "tracker_backend": b.tracker_backend,
            "tracker_version": b.tracker_version,
            "pipeline_config_sha256": b.pipeline_config_sha256,
            "software_git_sha": b.software_git_sha,
            "trace_ref": f"latency.jsonl:{self.latency_line()}",
            "baseline": baseline,
        }

    def _record(
        self,
        record: AlertRecord,
        *,
        zone_id: str | None,
        start_t: float,
        end_t: float,
        pre_s: float,
        post_max_s: float,
        contact_t: float | None,
        contact_source: str,
        view_obstructed_s: float,
        staff_absent_s: float | None,
        boxes: list[dict[str, float]],
        latency_line: int,
    ) -> dict[str, Any]:
        wall = self.clock.wall_at
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": "clip",
            "spool_id": spool_id_for(record.alert_id),
            "alert": self._reduced_alert(record),
            "run_id": self.run_id,
            "camera_id": self.cfg.camera_id,
            "zone_id": zone_id,
            "zone_version": None,
            "source_epoch_utc": self.clock.epoch_utc,
            "onset_utc": wall(record.onset_ts),
            "detected_utc": wall(record.detected_ts),
            "contact_utc": None if contact_t is None else wall(contact_t),
            "window": {
                "start_utc": wall(start_t),
                "end_utc": wall(end_t),
                "start_t": round(start_t, 3),
                "end_t": round(end_t, 3),
                "pre_s": pre_s,
                "post_max_s": post_max_s,
            },
            "first_staff_contact_ts": None if contact_t is None else round(contact_t, 3),
            "first_staff_contact_source": contact_source,
            "view_obstructed_s": view_obstructed_s,
            "staff_absent_s": staff_absent_s,
            "basis": self._basis(record),
            "demo_mode": self.cfg.demo_mode,
            "source_kind": self.cfg.source_kind,
            "source_media_path": self.cfg.source_media_path,
            "fps": None,
            "subject_track": boxes,
        }
        payload["basis"]["trace_ref"] = f"latency.jsonl:{latency_line}"
        return payload

    def _write(self, spool_id: str, payload: dict[str, Any]) -> None:
        path = self.spool_dir / f"{spool_id}.json"
        self._atomic_write(path, payload)
        self.n_written += 1
        self.written.append(path)

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
        tmp = path.with_name(f".{path.name}.tmp")
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)

    def _prune_reaped(self, now: float) -> None:
        keep_s = self.cfg.no_return_s + self.cfg.pre_roll_egress_s + 60.0
        for track_id in [t for t, snap in self._reaped.items() if now - snap.reaped_t > keep_s]:
            del self._reaped[track_id]

    # -- reporting -------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {
            "n_written": self.n_written,
            "n_updates": self.n_updates,
            "n_dropped": self.n_dropped,
            "n_open_windows": len(self._floor),
            "n_return_watches": len(self._returns),
            "spool_dir": str(self.spool_dir),
        }
