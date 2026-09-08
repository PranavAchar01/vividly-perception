"""egress_without_discharge_v0: a patient who left through an exit zone and did not come back.

The third head of the acute-event product, carried in the EVENT_MODEL_NAMES table of
src/vividly/demo/jobs.py as "Egress without discharge", and the one head that is a state
machine over identity rather than a motion classifier. A tracked person whose last
sighting was inside a configured exit zone, who has then been absent for
egress_absent_min_s with no human confirming that a later track is the same person, and
whose status is not discharged, is an egress candidate. The discharge gate is the primary
false-positive guard: a discharged patient crossing the exit is invisible to this head.

Why this is not a `Rule`. The two motion rules are pure functions of one live TrackState.
An egress candidate is, by definition, a track that is no longer live: the TrackBook reaps
it reap_after_s after its last sighting, and "absent for ten minutes" is a claim about a
track that was reaped nine and a half minutes ago. So this head keeps state, namely the
small set of departures still waiting on the clock, and the package's rule that only the
engine is stateful is kept by making it an attachment of the engine
(RuleEngine(..., egress=...)), dispatched through the same debounce and verifier path as
every other firing. Its firings carry rule_id "egress_without_discharge_v0" and look like
any other RuleFiring to the store and the dashboard.

How a departure is noticed, without touching the TrackBook's single reap watcher (which the
re-ID stage owns when re-ID is on): on every evaluation the monitor snapshots each live
track's last position, its exit-zone membership over the last egress_exit_window_s of its
own history, its presence so far and its human-applied tags. A track id that was in the
snapshot and is no longer in the book has been reaped (the book's only removal path), and
the snapshot is what it looked like when it left. The snapshot is at most one evaluation
interval stale, which is exact for a track that stopped being observed (nothing about it
changed after its last sighting) provided the engine evaluates at least once between a
track's last sighting and its reap, that is RuleConfig.eval_every_s <
TrackConfig.reap_after_s, which every shipped config satisfies by more than an order of
magnitude.

A departure becomes a pending candidate only if all of these hold at reap time:
  * the track's last egress_exit_window_s of positions touched an exit zone (a person the
    tracker lost in the middle of the room is an occlusion or a tracker failure, not an
    exit; it is counted as such and never alerted);
  * the person was present for at least egress_min_presence_s (a walk-through is not a
    patient who left);
  * the track carries no staff tag and no discharge tag.

A pending candidate is retired without an alert when
  * a human marks it discharged after the fact (EgressMonitor.mark_discharged, reached from
    the dashboard's departures panel), or
  * a human confirms that a live track is the same person (TrackBook.merge_identity sets the
    live track's reassociated_from to the departed id; the monitor sees that on its next
    evaluation). An unanswered re-ID proposal does not retire it: this head, like the
    gallery, never treats an unconfirmed match as a return;
and it fires once, when now - last_seen_t >= egress_absent_min_s, after which it is gone:
one departure, one candidate, and the verifier seam decides whether it becomes an alert.
The firing's onset_t is the last sighting, not the moment the clock ran out, so the record
says when the person left.

Zones are per-site geometry in source-frame pixels and live in the pipeline config's
`egress:` block (EgressZones.from_mapping), never in rules.yaml. The three thresholds live
in RuleConfig / configs/pipeline/rules.yaml with every other threshold, and like every
other threshold they are PROVISIONAL (docs/CALIBRATION.md): no count this head produces is
a false-alarm rate.

Import rule: standard library plus vividly.tracks and vividly.rules.base only, same as the
rest of the package. The lock is here because the dashboard's "mark discharged" button
reaches `mark_discharged` from uvicorn's threadpool while the pipeline thread is ticking.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vividly.rules.base import RuleConfig, RuleFiring
from vividly.tracks import TrackBook, TrackState

EGRESS_RULE_ID = "egress_without_discharge_v0"
EGRESS_RULE_VERSION = "0.1.0"

Point = tuple[float, float]


def _finite_point(value: Any, where: str) -> Point:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 2:
        raise ValueError(f"{where}: a polygon vertex must be an [x, y] pair, got {value!r}")
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: vertex {value!r} is not numeric") from exc
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"{where}: vertex {value!r} is not finite")
    return (x, y)


@dataclass(frozen=True)
class ExitZone:
    """One exit region: a named polygon in source-frame pixels.

    Per-site geometry, never a threshold: a door is where it is. Membership is the classic
    even-odd ray test, which handles concave outlines; a point exactly on an edge is not
    guaranteed either way, and nothing downstream should depend on it.
    """

    name: str
    polygon: tuple[Point, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"an exit zone needs a non-empty name, got {self.name!r}")
        if isinstance(self.polygon, str | bytes) or not isinstance(self.polygon, Sequence):
            raise ValueError(f"exit zone {self.name!r}: polygon must be a sequence of [x, y]")
        vertices = tuple(_finite_point(p, f"exit zone {self.name!r}") for p in self.polygon)
        if len(vertices) < 3:
            raise ValueError(
                f"exit zone {self.name!r}: a polygon needs at least 3 vertices, got {len(vertices)}"
            )
        object.__setattr__(self, "polygon", vertices)

    def contains(self, x: float, y: float) -> bool:
        inside = False
        j = len(self.polygon) - 1
        for i, (xi, yi) in enumerate(self.polygon):
            xj, yj = self.polygon[j]
            if (yi > y) != (yj > y):
                x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
                if x < x_cross:
                    inside = not inside
            j = i
        return inside


@dataclass(frozen=True)
class EgressZones:
    """The `egress:` block of a pipeline config: the switch and the zones.

    Disabled is the shipped default and means no monitor is built at all (the pipeline is
    byte-for-byte what it was before this head existed). Enabled with no zones is refused:
    a head that is on and can never fire is worse than one that is off.
    """

    enabled: bool = False
    zones: tuple[ExitZone, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(f"egress.enabled must be a boolean, got {self.enabled!r}")
        names = [zone.name for zone in self.zones]
        if len(set(names)) != len(names):
            raise ValueError(f"exit zone names must be unique, got {names}")
        if self.enabled and not self.zones:
            raise ValueError("egress is enabled but no exit zones are configured")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> EgressZones:
        """Parse `{enabled: bool, zones: [{name: str, polygon: [[x, y], ...]}, ...]}`.

        Unknown keys raise, at both levels, for the same reason they do in every other
        config block here: a typo that silently disabled a zone would look like a run with
        egress on and be a run with a door nobody was watching.
        """
        if not isinstance(data, Mapping):
            raise ValueError(f"`egress` must be a mapping, got {type(data).__name__}")
        unknown = sorted(set(data) - {"enabled", "zones"})
        if unknown:
            raise ValueError(
                f"unknown `egress` keys: {unknown}; known keys are ['enabled', 'zones']"
            )
        enabled = data.get("enabled", False)
        raw_zones = data.get("zones", [])
        if raw_zones is None:
            raw_zones = []
        if isinstance(raw_zones, str | bytes | Mapping) or not isinstance(raw_zones, Sequence):
            raise ValueError("`egress.zones` must be a list of {name, polygon} entries")
        zones: list[ExitZone] = []
        for i, entry in enumerate(raw_zones):
            if not isinstance(entry, Mapping):
                raise ValueError(f"egress.zones[{i}] must be a mapping with name and polygon")
            extra = sorted(set(entry) - {"name", "polygon"})
            if extra:
                raise ValueError(f"egress.zones[{i}]: unknown keys {extra}")
            if "name" not in entry or "polygon" not in entry:
                raise ValueError(f"egress.zones[{i}] needs both `name` and `polygon`")
            zones.append(ExitZone(name=str(entry["name"]), polygon=tuple(entry["polygon"])))
        return cls(enabled=enabled, zones=tuple(zones))

    def zone_at(self, x: float, y: float) -> str | None:
        """The first zone (in config order) containing the point, or None."""
        for zone in self.zones:
            if zone.contains(x, y):
                return zone.name
        return None

    def zone_along(self, points: Sequence[tuple[float, float, float]]) -> str | None:
        """The zone of the newest (t, cx, cy) sample that lies inside any zone, or None."""
        for _t, cx, cy in reversed(points):
            name = self.zone_at(cx, cy)
            if name is not None:
                return name
        return None


@dataclass(frozen=True)
class Departure:
    """One departure still waiting on the egress clock. What the person looked like as they left."""

    track_id: int
    zone: str
    first_seen_t: float
    last_seen_t: float
    last_cx: float
    last_cy: float
    mrn_binding: str | None = None

    @property
    def presence_s(self) -> float:
        return self.last_seen_t - self.first_seen_t

    def absent_s(self, now: float) -> float:
        return now - self.last_seen_t


@dataclass(frozen=True)
class _Snapshot:
    """What the monitor remembers about a live track, refreshed every evaluation."""

    first_seen_t: float
    last_seen_t: float
    cx: float
    cy: float
    zone: str | None
    is_staff: bool
    is_discharged: bool
    mrn_binding: str | None


class EgressMonitor:
    """The egress head. Stateful by necessity; attached to a RuleEngine, never a Rule.

    Thread story: `tick` runs on the pipeline thread; `mark_discharged`, `pending` and
    `stats` may be called from the dashboard's threadpool. One RLock covers the pending
    table and the counters; `tick` does its TrackBook reads before taking it and holds it
    only for the bookkeeping.
    """

    rule_id = EGRESS_RULE_ID
    rule_version = EGRESS_RULE_VERSION

    def __init__(self, cfg: RuleConfig | None = None, zones: EgressZones | None = None) -> None:
        if zones is None or not zones.enabled:
            raise ValueError(
                "EgressMonitor needs an ENABLED EgressZones with at least one zone; a disabled "
                "egress block means no monitor at all, so build none rather than a silent one"
            )
        self.cfg = RuleConfig() if cfg is None else cfg
        self.zones = zones
        self._lock = threading.RLock()
        self._snapshots: dict[int, _Snapshot] = {}
        self._pending: dict[int, Departure] = {}
        self.n_departures: int = 0
        self.n_fired: int = 0
        self.n_suppressed_staff: int = 0
        self.n_suppressed_discharged: int = 0
        self.n_left_outside_zone: int = 0
        self.n_short_presence: int = 0
        self.n_returned: int = 0
        self.n_discharged_after_departure: int = 0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        names = ", ".join(zone.name for zone in self.zones.zones)
        return (
            f"EgressMonitor(zones=[{names}], absent_min_s={self.cfg.egress_absent_min_s}, "
            f"pending={len(self._pending)})"
        )

    # -- the tick -----------------------------------------------------------
    def tick(self, book: TrackBook, now: float) -> list[RuleFiring]:
        """Notice departures, watch for confirmed returns, fire the candidates that are due.

        Returned in ascending track_id order so the engine's output stays byte-identical
        for identical input. A nan clock is a no-op, as it is for the engine.
        """
        if math.isnan(now):
            return []
        live = book.tracks()
        fresh = {track_id: self._snapshot(track) for track_id, track in live.items()}
        returned = {
            track.reassociated_from
            for track in live.values()
            if track.reassociated_from is not None
        }
        firings: list[RuleFiring] = []
        with self._lock:
            for track_id in sorted(set(self._snapshots) - set(live)):
                self._classify_departure(track_id, self._snapshots.pop(track_id))
            self._snapshots.update(fresh)
            for prior in sorted(returned):
                if prior in self._pending:
                    del self._pending[prior]
                    self.n_returned += 1
            for track_id in sorted(self._pending):
                departure = self._pending[track_id]
                if departure.absent_s(now) >= self.cfg.egress_absent_min_s:
                    del self._pending[track_id]
                    self.n_fired += 1
                    firings.append(self._firing(departure, now))
        return firings

    def _snapshot(self, track: TrackState) -> _Snapshot:
        observation = track.last_observation
        if observation is None:  # pragma: no cover - the book never hands out an empty state
            raise ValueError(f"track {track.track_id} has no observation to snapshot")
        # Anchored at the track's own last sighting, on purpose: this is "where was this
        # person during their last few seconds", whatever the pipeline clock says now.
        points = track.positions(self.cfg.egress_exit_window_s)
        return _Snapshot(
            first_seen_t=track.first_seen_t,
            last_seen_t=track.last_seen_t,
            cx=observation.cx,
            cy=observation.cy,
            zone=self.zones.zone_along(points),
            is_staff=bool(track.is_staff),
            is_discharged=bool(track.is_discharged),
            mrn_binding=track.mrn_binding,
        )

    def _classify_departure(self, track_id: int, snap: _Snapshot) -> None:
        """The caller holds the lock. Counts every departure exactly once, by its first reason."""
        self.n_departures += 1
        if snap.is_staff:
            self.n_suppressed_staff += 1
            return
        if snap.is_discharged:
            self.n_suppressed_discharged += 1
            return
        if snap.zone is None:
            self.n_left_outside_zone += 1
            return
        if snap.last_seen_t - snap.first_seen_t < self.cfg.egress_min_presence_s:
            self.n_short_presence += 1
            return
        self._pending[track_id] = Departure(
            track_id=track_id,
            zone=snap.zone,
            first_seen_t=snap.first_seen_t,
            last_seen_t=snap.last_seen_t,
            last_cx=snap.cx,
            last_cy=snap.cy,
            mrn_binding=snap.mrn_binding,
        )

    def _firing(self, departure: Departure, now: float) -> RuleFiring:
        cfg = self.cfg
        details: dict[str, float | str] = {
            "exit_zone": departure.zone,
            "absent_s": float(departure.absent_s(now)),
            "presence_s": float(departure.presence_s),
            "last_seen_t": float(departure.last_seen_t),
            "first_seen_t": float(departure.first_seen_t),
            "last_cx": float(departure.last_cx),
            "last_cy": float(departure.last_cy),
            "threshold_egress_absent_min_s": float(cfg.egress_absent_min_s),
            "threshold_egress_min_presence_s": float(cfg.egress_min_presence_s),
            "threshold_egress_exit_window_s": float(cfg.egress_exit_window_s),
            "threshold_status": "PROVISIONAL (acute-event rules; see docs/CALIBRATION.md)",
        }
        if departure.mrn_binding is not None:
            # The live track is gone, so the runner cannot read the binding off the book;
            # carrying it here is what lets the alert record still name the MOCK chart.
            details["mrn_binding"] = departure.mrn_binding
        return RuleFiring(
            rule_id=self.rule_id,
            rule_version=self.rule_version,
            track_id=departure.track_id,
            onset_t=departure.last_seen_t,
            detected_t=now,
            baseline=None,
            details=details,
        )

    # -- the human's levers ---------------------------------------------------
    def mark_discharged(self, track_id: int) -> bool:
        """Retire a pending departure when a reviewer records that person as discharged.

        True if a candidate was retired, False if no departure by that id is pending (it
        already fired, was already retired, or never existed). For a track that is still
        live, the lever is TrackBook.set_discharged instead.
        """
        with self._lock:
            if track_id not in self._pending:
                return False
            del self._pending[track_id]
            self.n_discharged_after_departure += 1
            return True

    def pending(self) -> list[Departure]:
        """Departures still on the clock, oldest departure first."""
        with self._lock:
            return sorted(self._pending.values(), key=lambda d: (d.last_seen_t, d.track_id))

    def pending_view(self, now: float) -> list[dict[str, Any]]:
        """The departures panel payload: plain dicts, finite numbers, oldest first."""
        rows: list[dict[str, Any]] = []
        for departure in self.pending():
            absent = None if math.isnan(now) else departure.absent_s(now)
            rows.append(
                {
                    "track_id": departure.track_id,
                    "zone": departure.zone,
                    "first_seen_t": departure.first_seen_t,
                    "last_seen_t": departure.last_seen_t,
                    "presence_s": departure.presence_s,
                    "absent_s": absent,
                    "fires_at": departure.last_seen_t + self.cfg.egress_absent_min_s,
                    "mrn_binding": departure.mrn_binding,
                }
            )
        return rows

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "rule_id": self.rule_id,
                "rule_version": self.rule_version,
                "zones": [zone.name for zone in self.zones.zones],
                "n_pending": len(self._pending),
                "n_departures": self.n_departures,
                "n_fired": self.n_fired,
                "n_suppressed_staff": self.n_suppressed_staff,
                "n_suppressed_discharged": self.n_suppressed_discharged,
                "n_left_outside_zone": self.n_left_outside_zone,
                "n_short_presence": self.n_short_presence,
                "n_returned": self.n_returned,
                "n_discharged_after_departure": self.n_discharged_after_departure,
                "threshold_status": "PROVISIONAL (acute-event rules; see docs/CALIBRATION.md)",
            }
