"""egress_without_discharge_v0 tests: the third acute-event head (vividly.rules.egress).

Same four properties as tests/test_rules.py: deterministic, network-free, model-free;
the false-positive controls are the centre (a mid-room loss, a short visit, a staff
member, a discharged patient, a confirmed return and an ordinary hour must all produce
nothing); the public shapes are pinned; and the shipped demo config is exercised through
the real runner so the thresholds that fire here are the thresholds that ship (shrunk
where the config says so, and only there).

POLICY.md, rule 4 is not in play here: a person walking out of a room is not a medical event,
and the fixtures are abstract points on a plane in any case.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from vividly.events import EventStore
from vividly.rules import (
    DEFAULT_RULE_IDS,
    EGRESS_RULE_ID,
    EGRESS_RULE_VERSION,
    AbstainingVerifier,
    Departure,
    EgressMonitor,
    EgressZones,
    ExitZone,
    RuleConfig,
    RuleEngine,
    RuleFiring,
    VerifierVerdict,
    build_default_rules,
    load_rule_config,
)
from vividly.rules.fixtures import (
    EXIT_POINT,
    EXIT_ZONE_NAME,
    EXIT_ZONE_POLYGON,
    PATIENT_TRACK_ID,
    SEAT,
    STAFF_TRACK_ID,
    depart_through_exit,
    leave_and_return,
    normal_fidget,
)
from vividly.sources import SyntheticDetectionSource, write_observations_jsonl
from vividly.tracks import TrackBook, TrackConfig, TrackIdentity, TrackObservation, TrackState

REPO = Path(__file__).resolve().parents[1]
RUN_PIPELINE = REPO / "scripts" / "run_pipeline.py"
EGRESS_DEMO_CONFIG = REPO / "configs" / "pipeline" / "egress_demo.yaml"
SCENE_PATH = REPO / "tests" / "fixtures" / "pipeline_synth" / "egress_scene.jsonl"
RULES_YAML = REPO / "configs" / "pipeline" / "rules.yaml"

TRACK_CFG = TrackConfig(baseline_window_s=300.0, reap_after_s=30.0, ring_size=20_000)
ZONES = EgressZones(enabled=True, zones=(ExitZone(EXIT_ZONE_NAME, EXIT_ZONE_POLYGON),))
RETURN_TRACK_ID = 7

# depart_through_exit defaults: 3 min seated, 10 s walk -> last sighting at t = 190.
LEAVE_T = 190.0


# ------------------------------------------------------------------ helpers
class FixedVerifier:
    verifier_id = "fixed_v0"
    verifier_version = "0.0.0"

    def __init__(self, verdict: str, confidence: float | None = None) -> None:
        self._verdict = verdict
        self._confidence = confidence
        self.calls: list[tuple[RuleFiring, TrackState | None]] = []

    def verify(self, firing: RuleFiring, track: TrackState | None) -> VerifierVerdict:
        self.calls.append((firing, track))
        return VerifierVerdict(
            verdict=self._verdict,
            verifier_id=self.verifier_id,
            verifier_version=self.verifier_version,
            confidence=self._confidence,
        )


def run_egress_scene(
    batches: list[list[TrackObservation]],
    *,
    cfg: RuleConfig | None = None,
    zones: EgressZones = ZONES,
    track_cfg: TrackConfig = TRACK_CFG,
    staff_ids: tuple[int, ...] = (STAFF_TRACK_ID,),
    verifier: Any = None,
    verify_policy: str = "annotate",
    on_tick: Any = None,
) -> tuple[TrackBook, RuleEngine, EgressMonitor, list[RuleFiring]]:
    """Ingest every batch, tag staff as it appears, tick the engine at the batch's own time.

    `on_tick(book, monitor, t)` runs after each ingest and before the tick, so a test can
    press a dashboard lever (discharge, chart-bind, confirm a return) at a chosen instant.
    """
    config = RuleConfig() if cfg is None else cfg
    book = TrackBook(track_cfg)
    monitor = EgressMonitor(config, zones)
    engine = RuleEngine(
        build_default_rules(config),
        config,
        verifier=verifier,
        verify_policy=verify_policy,
        egress=monitor,
    )
    fired: list[RuleFiring] = []
    tagged: set[int] = set()
    last_t = 0.0
    for batch in batches:
        book.ingest(batch)
        t = batch[0].t if batch else last_t
        last_t = t
        book.reap(t)
        for staff_id in staff_ids:
            if staff_id not in tagged and staff_id in book.tracks():
                book.set_staff(staff_id, True)
                tagged.add(staff_id)
        if on_tick is not None:
            on_tick(book, monitor, t)
        fired.extend(engine.tick(book, t))
    return book, engine, monitor, fired


def egress_firings(fired: list[RuleFiring]) -> list[RuleFiring]:
    return [f for f in fired if f.rule_id == EGRESS_RULE_ID]


# --------------------------------------------------------------- geometry
def test_exit_zone_membership_and_validation() -> None:
    zone = ExitZone("door", EXIT_ZONE_POLYGON)
    assert zone.contains(*EXIT_POINT)
    assert zone.contains(1641.0, 381.0)
    assert not zone.contains(*SEAT)
    assert not zone.contains(1639.0, 540.0)
    assert not zone.contains(1780.0, 701.0)
    # A concave outline (a U): the notch is outside, the arms are inside.
    u = ExitZone("u", ((0, 0), (30, 0), (30, 30), (20, 30), (20, 10), (10, 10), (10, 30), (0, 30)))
    assert u.contains(5, 20) and u.contains(25, 20) and not u.contains(15, 20)
    # Vertices are normalised to floats and kept in order.
    assert u.polygon[1] == (30.0, 0.0)
    with pytest.raises(ValueError, match="at least 3"):
        ExitZone("line", ((0, 0), (1, 1)))
    with pytest.raises(ValueError, match="not finite"):
        ExitZone("nan", ((0, 0), (1, math.nan), (2, 2)))
    with pytest.raises(ValueError, match="non-empty name"):
        ExitZone("  ", EXIT_ZONE_POLYGON)
    with pytest.raises(ValueError, match="pair"):
        ExitZone("bad", ((0, 0), (1,), (2, 2)))  # type: ignore[arg-type]


def test_egress_zones_from_mapping_parses_the_config_shape_and_refuses_junk() -> None:
    zones = EgressZones.from_mapping(
        {
            "enabled": True,
            "zones": [
                {
                    "name": "main door",
                    "polygon": [[1640, 380], [1920, 380], [1920, 700], [1640, 700]],
                },
                {"name": "side door", "polygon": [[0, 0], [10, 0], [10, 10]]},
            ],
        }
    )
    assert zones.enabled and [z.name for z in zones.zones] == ["main door", "side door"]
    assert zones.zone_at(*EXIT_POINT) == "main door"
    assert zones.zone_at(5.0, 2.0) == "side door"
    assert zones.zone_at(*SEAT) is None
    # Off is the shipped default: an empty block builds a disabled, zone-less object.
    off = EgressZones.from_mapping({})
    assert off.enabled is False and off.zones == ()
    with pytest.raises(ValueError, match="unknown `egress` keys"):
        EgressZones.from_mapping({"enabled": True, "zone": []})
    with pytest.raises(ValueError, match="no exit zones"):
        EgressZones.from_mapping({"enabled": True})
    with pytest.raises(ValueError, match="unique"):
        EgressZones.from_mapping(
            {
                "enabled": True,
                "zones": [
                    {"name": "d", "polygon": [[0, 0], [1, 0], [1, 1]]},
                    {"name": "d", "polygon": [[0, 0], [1, 0], [1, 1]]},
                ],
            }
        )
    with pytest.raises(ValueError, match="unknown keys"):
        EgressZones.from_mapping(
            {"enabled": True, "zones": [{"name": "d", "polygon": [[0, 0], [1, 0], [1, 1]], "x": 1}]}
        )
    with pytest.raises(ValueError, match="needs both"):
        EgressZones.from_mapping({"enabled": True, "zones": [{"name": "d"}]})
    with pytest.raises(ValueError, match="must be a boolean"):
        EgressZones.from_mapping({"enabled": "yes"})
    with pytest.raises(ValueError, match="must be a list"):
        EgressZones.from_mapping({"enabled": True, "zones": {"name": "d"}})


def test_zone_along_takes_the_newest_sample_inside_any_zone() -> None:
    points = [(0.0, *SEAT), (1.0, *EXIT_POINT), (2.0, *SEAT)]
    # The newest sample is outside, the one before it inside: the exit window is what
    # absorbs a last frame of tracker noise at the door.
    assert ZONES.zone_along(points) == EXIT_ZONE_NAME
    assert ZONES.zone_along([(0.0, *SEAT)]) is None
    assert ZONES.zone_along([]) is None


# ------------------------------------------------------------- the positive
def test_a_patient_who_leaves_through_the_exit_and_stays_away_fires_once() -> None:
    """The scene: sit three minutes, walk out, never come back. One candidate, onset at
    the last sighting, detection when the egress clock runs out, nothing else at all."""
    book, engine, monitor, fired = run_egress_scene(depart_through_exit())
    assert [(f.rule_id, f.track_id) for f in fired] == [(EGRESS_RULE_ID, PATIENT_TRACK_ID)]
    firing = fired[0]
    assert firing.rule_version == EGRESS_RULE_VERSION
    assert firing.onset_t == pytest.approx(LEAVE_T, abs=1.0)
    assert firing.detected_t == pytest.approx(LEAVE_T + RuleConfig().egress_absent_min_s, abs=1.5)
    assert firing.baseline is None
    assert firing.details["exit_zone"] == EXIT_ZONE_NAME
    assert firing.details["absent_s"] >= RuleConfig().egress_absent_min_s
    assert firing.details["presence_s"] == pytest.approx(LEAVE_T, abs=1.0)
    assert firing.details["last_cx"] == pytest.approx(EXIT_POINT[0], abs=60.0)
    assert firing.details["threshold_status"].startswith("PROVISIONAL")
    assert "mrn_binding" not in firing.details
    # The person was reaped long before the clock ran out: the firing names a track the
    # book no longer holds, which is the whole reason this head is not a Rule.
    assert PATIENT_TRACK_ID not in book.tracks()
    assert STAFF_TRACK_ID in book.tracks()
    assert monitor.stats()["n_fired"] == 1 and monitor.stats()["n_pending"] == 0
    assert monitor.pending() == []
    # The debounce table forgets the departed id on the next tick and the monitor has
    # retired the departure, so nothing re-fires: one departure, one candidate.
    assert engine.n_suppressed == 0


def test_the_candidate_is_pending_between_the_reap_and_the_clock() -> None:
    seen: dict[str, Any] = {}

    def on_tick(book: TrackBook, monitor: EgressMonitor, t: float) -> None:
        if LEAVE_T + 40.0 <= t < LEAVE_T + 41.0:
            seen["pending"] = monitor.pending()
            seen["view"] = monitor.pending_view(t)
            seen["in_book"] = PATIENT_TRACK_ID in book.tracks()

    run_egress_scene(depart_through_exit(), on_tick=on_tick)
    (departure,) = seen["pending"]
    assert isinstance(departure, Departure)
    assert departure.track_id == PATIENT_TRACK_ID and departure.zone == EXIT_ZONE_NAME
    assert departure.last_seen_t == pytest.approx(LEAVE_T, abs=1.0)
    assert departure.presence_s == pytest.approx(LEAVE_T, abs=1.0)
    assert seen["in_book"] is False
    (row,) = seen["view"]
    assert row["track_id"] == PATIENT_TRACK_ID
    assert row["fires_at"] == pytest.approx(
        departure.last_seen_t + RuleConfig().egress_absent_min_s
    )
    assert 39.0 <= row["absent_s"] <= 42.0


def test_the_mock_chart_binding_rides_along_in_the_firing() -> None:
    def on_tick(book: TrackBook, monitor: EgressMonitor, t: float) -> None:
        if 60.0 <= t < 61.0:
            book.bind_mrn(PATIENT_TRACK_ID, "MOCK-0001")

    _, _, _, fired = run_egress_scene(depart_through_exit(), on_tick=on_tick)
    assert egress_firings(fired)[0].details["mrn_binding"] == "MOCK-0001"


# --------------------------- the false-positive controls (the acute-event priority)
def test_a_track_lost_in_the_middle_of_the_room_is_not_an_exit() -> None:
    """The occlusion / tracker-failure case: the person vanishes from the seat, nowhere
    near a door. Counted, never alerted."""
    _, _, monitor, fired = run_egress_scene(depart_through_exit(walk_s=0.0))
    assert fired == []
    assert monitor.stats()["n_departures"] == 1
    assert monitor.stats()["n_left_outside_zone"] == 1
    assert monitor.pending() == []


def test_a_short_visit_through_the_exit_is_not_a_patient_leaving() -> None:
    _, _, monitor, fired = run_egress_scene(depart_through_exit(seated_minutes=1.0))
    assert fired == []
    assert monitor.stats()["n_short_presence"] == 1


def test_staff_leaving_through_the_exit_never_fires() -> None:
    _, _, monitor, fired = run_egress_scene(
        depart_through_exit(), staff_ids=(STAFF_TRACK_ID, PATIENT_TRACK_ID)
    )
    assert fired == []
    assert monitor.stats()["n_suppressed_staff"] == 1


def test_a_discharged_patient_leaving_is_invisible() -> None:
    """The primary false-positive killer: tagged discharged while still in the room, then
    walks out through the door. Nothing fires, and the reason is counted."""

    def on_tick(book: TrackBook, monitor: EgressMonitor, t: float) -> None:
        if 100.0 <= t < 101.0:
            book.set_discharged(PATIENT_TRACK_ID, True)
            assert book.tracks()[PATIENT_TRACK_ID].identity().is_discharged is True

    _, _, monitor, fired = run_egress_scene(depart_through_exit(), on_tick=on_tick)
    assert fired == []
    assert monitor.stats()["n_suppressed_discharged"] == 1
    assert monitor.pending() == []


def test_marking_discharged_after_the_departure_retires_the_candidate() -> None:
    outcomes: list[bool] = []

    def on_tick(book: TrackBook, monitor: EgressMonitor, t: float) -> None:
        if LEAVE_T + 60.0 <= t < LEAVE_T + 61.0:
            outcomes.append(monitor.mark_discharged(PATIENT_TRACK_ID))
            outcomes.append(monitor.mark_discharged(PATIENT_TRACK_ID))  # already retired
            outcomes.append(monitor.mark_discharged(4242))  # never existed

    _, _, monitor, fired = run_egress_scene(depart_through_exit(), on_tick=on_tick)
    assert fired == []
    assert outcomes == [True, False, False]
    assert monitor.stats()["n_discharged_after_departure"] == 1


def test_a_confirmed_return_under_a_new_id_retires_the_candidate() -> None:
    """The person comes back after 2 minutes under a new tracker id. On its own that is a
    stranger (no merge, still fires); once a human confirms the re-association the
    candidate is retired and nothing fires."""
    scene = depart_through_exit(return_after_s=120.0, return_track_id=RETURN_TRACK_ID)

    _, _, monitor_unconfirmed, fired_unconfirmed = run_egress_scene(scene)
    assert [f.track_id for f in egress_firings(fired_unconfirmed)] == [PATIENT_TRACK_ID]
    assert monitor_unconfirmed.stats()["n_returned"] == 0

    def confirm(book: TrackBook, monitor: EgressMonitor, t: float) -> None:
        if (
            RETURN_TRACK_ID in book.tracks()
            and book.tracks()[RETURN_TRACK_ID].reassociated_from is None
        ):
            book.merge_identity(RETURN_TRACK_ID, TrackIdentity(track_id=PATIENT_TRACK_ID))

    _, _, monitor, fired = run_egress_scene(scene, on_tick=confirm)
    assert fired == []
    assert monitor.stats()["n_returned"] == 1
    assert monitor.pending() == []


def test_a_return_inside_the_reap_window_is_never_a_departure() -> None:
    """Same id, short gap: the book never reaps the track, so there is nothing to notice.
    (The patient is lost at the seat here; this is the track layer's automatic window.)"""
    _, _, monitor, fired = run_egress_scene(leave_and_return(20.0, gap_s=20.0))
    assert fired == []
    assert monitor.stats()["n_departures"] == 0


def test_an_ordinary_hour_with_the_head_attached_fires_nothing() -> None:
    _, engine, monitor, fired = run_egress_scene(normal_fidget(60.0))
    assert fired == []
    assert monitor.stats()["n_departures"] == 0
    assert engine.n_evaluations > 3000


def test_the_walk_to_the_door_wakes_neither_event_rule() -> None:
    """The exit walk is upright and directionally coherent: no fall, no convulsion, at any
    evaluation opportunity (the default rules are in the engine of every scene above)."""
    _, _, _, fired = run_egress_scene(depart_through_exit(hz=5.0))
    assert {f.rule_id for f in fired} == {EGRESS_RULE_ID}


# ------------------------------------------------------------- engine seam
def test_egress_firings_take_the_verifier_path() -> None:
    rejecting = FixedVerifier("rejected")
    _, engine, _, fired = run_egress_scene(depart_through_exit(), verifier=rejecting)
    assert fired == [] and engine.n_rejected == 1
    # The verifier is handed None for the track: the person has already been reaped.
    (call,) = rejecting.calls
    assert call[0].rule_id == EGRESS_RULE_ID and call[1] is None

    _, engine, _, fired = run_egress_scene(
        depart_through_exit(), verifier=AbstainingVerifier(), verify_policy="require"
    )
    assert fired == [] and engine.n_unconfirmed == 1

    _, _, _, fired = run_egress_scene(
        depart_through_exit(), verifier=FixedVerifier("confirmed", 0.9)
    )
    (firing,) = fired
    assert firing.details["verifier_verdict"] == "confirmed"
    assert firing.details["verifier_confidence"] == pytest.approx(0.9)


def test_engine_and_monitor_refuse_the_wrong_things() -> None:
    with pytest.raises(TypeError, match="EgressMonitor"):
        RuleEngine(build_default_rules(), RuleConfig(), egress=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ENABLED"):
        EgressMonitor(RuleConfig(), EgressZones())
    with pytest.raises(ValueError, match="ENABLED"):
        EgressMonitor(RuleConfig(), None)
    # The default rule list is untouched: the head is an attachment, not a rule.
    assert DEFAULT_RULE_IDS == ("fall_then_still_v0", "seizure_candidate_v0")
    assert EGRESS_RULE_ID not in DEFAULT_RULE_IDS
    assert "egress=off" in repr(RuleEngine(build_default_rules(), RuleConfig()))


def test_the_head_is_deterministic() -> None:
    a = run_egress_scene(depart_through_exit(hz=2.0))[3]
    b = run_egress_scene(depart_through_exit(hz=2.0))[3]
    assert a == b and len(a) == 1


def test_a_nan_clock_is_a_no_op() -> None:
    monitor = EgressMonitor(RuleConfig(), ZONES)
    assert monitor.tick(TrackBook(TRACK_CFG), math.nan) == []


# ------------------------------------------------------------- thresholds
@pytest.mark.parametrize(
    "field", ["egress_absent_min_s", "egress_min_presence_s", "egress_exit_window_s"]
)
def test_rule_config_validates_the_egress_thresholds(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        RuleConfig(**{field: 0.0})
    with pytest.raises(ValueError, match=field):
        RuleConfig(**{field: -1.0})


def test_rules_yaml_carries_the_egress_thresholds() -> None:
    cfg = load_rule_config(RULES_YAML)
    assert cfg == RuleConfig()
    names = {f.name for f in dataclasses.fields(RuleConfig)}
    assert {"egress_absent_min_s", "egress_min_presence_s", "egress_exit_window_s"} <= names


# ------------------------------------------------------- track-layer tags
def test_discharge_tag_is_snapshotted_and_carried_forward_or_ed() -> None:
    book = TrackBook(TRACK_CFG)
    book.ingest([TrackObservation(track_id=1, t=0.0, cx=1.0, cy=1.0, w=5.0, h=5.0)])
    book.ingest([TrackObservation(track_id=2, t=0.5, cx=1.0, cy=1.0, w=5.0, h=5.0)])
    assert book.tracks()[1].is_discharged is False
    book.set_discharged(1, True)
    snapshot = book.tracks()[1].identity()
    assert snapshot.is_discharged is True and snapshot.track_id == 1
    with pytest.raises(KeyError):
        book.set_discharged(99, True)
    # Carry-forward is a logical or: a match to a discharged identity discharges the live track;
    # a match to an undischarged identity does not clear a tag a reviewer applied.
    book.merge_identity(2, snapshot)
    assert book.tracks()[2].is_discharged is True
    book.ingest([TrackObservation(track_id=3, t=1.0, cx=1.0, cy=1.0, w=5.0, h=5.0)])
    book.set_discharged(3, True)
    book.merge_identity(3, TrackIdentity(track_id=1, is_discharged=False))
    assert book.tracks()[3].is_discharged is True
    # Default stays False so every existing TrackIdentity(...) construction is unchanged.
    assert TrackIdentity(track_id=5).is_discharged is False


# ------------------------------------------------------- the committed demo
SCENE_HZ = 2.0
SCENE_DECIMALS = 3
SCENE_SEATED_MINUTES = 1.0
SCENE_WALK_S = 10.0
SCENE_ABSENT_MINUTES = 2.5
SCENE_LEAVE_T = SCENE_SEATED_MINUTES * 60.0 + SCENE_WALK_S  # 70 s
SCENE_SECONDS = SCENE_LEAVE_T + SCENE_ABSENT_MINUTES * 60.0  # 220 s


def _rounded(obs: TrackObservation) -> TrackObservation:
    return TrackObservation(
        track_id=obs.track_id,
        t=round(obs.t, SCENE_DECIMALS),
        cx=round(obs.cx, SCENE_DECIMALS),
        cy=round(obs.cy, SCENE_DECIMALS),
        w=round(obs.w, SCENE_DECIMALS),
        h=round(obs.h, SCENE_DECIMALS),
        keypoints=(
            None
            if obs.keypoints is None
            else tuple(
                (round(x, SCENE_DECIMALS), round(y, SCENE_DECIMALS), round(c, SCENE_DECIMALS))
                for x, y, c in obs.keypoints
            )
        ),
    )


def build_egress_scene() -> list[list[TrackObservation]]:
    """The committed fixture, in closed form: sit 60 s, walk out over 10 s, stay away.

    Track 1 is last seen at t = 70 inside the exit zone; track 99 patrols throughout and
    never enters it. The demo config's demo-shrunk thresholds (absent 90 s, presence 45 s,
    reap 8 s) make the one candidate fire at t = 160 inside a 220 s scene.
    """
    batches = depart_through_exit(
        seated_minutes=SCENE_SEATED_MINUTES,
        walk_s=SCENE_WALK_S,
        absent_minutes=SCENE_ABSENT_MINUTES,
        hz=SCENE_HZ,
    )
    return [[_rounded(obs) for obs in batch] for batch in batches]


def test_committed_egress_fixture_is_deterministic(tmp_path: Path) -> None:
    regenerated = write_observations_jsonl(tmp_path / "egress_scene.jsonl", build_egress_scene())
    assert regenerated.read_bytes() == SCENE_PATH.read_bytes()


def test_egress_fixture_holds_a_departure_and_a_patrol() -> None:
    batches = list(SyntheticDetectionSource(SCENE_PATH).observations())
    last_seen: dict[int, float] = {}
    last_pos: dict[int, tuple[float, float]] = {}
    for batch in batches:
        for obs in batch:
            last_seen[obs.track_id] = obs.t
            last_pos[obs.track_id] = (obs.cx, obs.cy)
    assert sorted(last_seen) == [PATIENT_TRACK_ID, STAFF_TRACK_ID]
    assert last_seen[PATIENT_TRACK_ID] == pytest.approx(SCENE_LEAVE_T, abs=0.6)
    assert last_seen[STAFF_TRACK_ID] == pytest.approx(SCENE_SECONDS, abs=0.6)
    assert ZONES.zone_at(*last_pos[PATIENT_TRACK_ID]) == EXIT_ZONE_NAME
    assert SCENE_PATH.stat().st_size < 512 * 1024


def test_egress_demo_config_declares_the_fixture_the_zone_and_its_shrunk_thresholds() -> None:
    cfg = yaml.safe_load(EGRESS_DEMO_CONFIG.read_text(encoding="utf-8"))
    assert (REPO / cfg["detections_from"]).resolve() == SCENE_PATH.resolve()
    assert cfg["headless"] is True
    zones = EgressZones.from_mapping(cfg["egress"])
    assert zones.enabled and [z.name for z in zones.zones] == [EXIT_ZONE_NAME]
    assert zones.zones[0].polygon == EXIT_ZONE_POLYGON
    rules = RuleConfig.from_mapping(cfg["rules"])
    assert rules.egress_absent_min_s < RuleConfig().egress_absent_min_s
    assert rules.egress_min_presence_s <= SCENE_LEAVE_T
    # Shrunk far enough to fire inside the scene, and only the two egress knobs are touched.
    assert set(cfg["rules"]) == {"egress_absent_min_s", "egress_min_presence_s"}
    assert SCENE_LEAVE_T + rules.egress_absent_min_s < SCENE_SECONDS
    track_cfg = TrackConfig.from_mapping(cfg["tracks"])
    assert track_cfg.reap_after_s < rules.egress_absent_min_s
    text = EGRESS_DEMO_CONFIG.read_text(encoding="utf-8")
    assert "PROVISIONAL" in text and "CALIBRATION.md" in text and "DEMO-SHRUNK" in text


def _load_runner() -> Any:
    name = "vividly_run_pipeline_script_egress"
    spec = importlib.util.spec_from_file_location(name, RUN_PIPELINE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _demo_config(tmp_path: Path, **overrides: Any) -> Path:
    cfg = yaml.safe_load(EGRESS_DEMO_CONFIG.read_text(encoding="utf-8"))
    cfg["run_name"] = "test_egress"
    cfg["output"] = {"dir": str(tmp_path / "outputs")}
    cfg.update(overrides)
    path = tmp_path / "egress.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_the_headless_egress_demo_writes_exactly_one_egress_alert(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = _load_runner()
    assert runner.main(["--config", str(_demo_config(tmp_path)), "--headless"]) == 0
    out_dir = tmp_path / "outputs" / "test_egress"
    store = EventStore(out_dir / "alerts.sqlite")
    try:
        alerts = store.alerts(limit=100)
    finally:
        store.close()
    assert [(a.rule_id, a.track_id) for a in alerts] == [(EGRESS_RULE_ID, PATIENT_TRACK_ID)]
    alert = alerts[0]
    assert alert.onset_ts == pytest.approx(SCENE_LEAVE_T, abs=0.6)
    assert alert.detected_ts == pytest.approx(SCENE_LEAVE_T + 90.0, abs=1.5)
    assert alert.details["exit_zone"] == EXIT_ZONE_NAME
    assert alert.baseline is None
    # The summary line and the latency summary row both carry the head's counts.
    printed = capsys.readouterr().out
    assert "egress   : 1 candidate(s) fired" in printed
    rows = [json.loads(line) for line in (out_dir / "latency.jsonl").read_text().splitlines()]
    summary = [row for row in rows if row.get("summary")][-1]
    assert summary["egress"]["n_fired"] == 1 and summary["egress"]["zones"] == [EXIT_ZONE_NAME]


def test_a_run_with_no_egress_block_builds_no_monitor_and_writes_no_egress_key(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    cfg_path = _demo_config(tmp_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    del cfg["egress"]
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    pipeline = runner.Pipeline(yaml.safe_load(cfg_path.read_text(encoding="utf-8")), REPO)
    try:
        assert pipeline.egress is None and pipeline.engine.egress is None
        pipeline.run()
    finally:
        pipeline.close()
    store = EventStore(tmp_path / "outputs" / "test_egress" / "alerts.sqlite")
    try:
        assert store.alerts(limit=100) == []
    finally:
        store.close()
    rows = [
        json.loads(line)
        for line in (tmp_path / "outputs" / "test_egress" / "latency.jsonl")
        .read_text()
        .splitlines()
    ]
    assert "egress" not in [row for row in rows if row.get("summary")][-1]


def test_the_runner_reports_a_bad_egress_block_as_a_config_error(tmp_path: Path) -> None:
    runner = _load_runner()
    bad = _demo_config(tmp_path, egress={"enabled": True})
    assert runner.main(["--config", str(bad), "--headless"]) == runner.EXIT_CONFIG
    worse = _demo_config(
        tmp_path, egress={"enabled": True, "zones": [{"name": "d", "polygon": [[0, 0]]}]}
    )
    assert runner.main(["--config", str(worse), "--headless"]) == runner.EXIT_CONFIG


def test_discharging_the_departed_person_in_a_run_silences_the_alert(tmp_path: Path) -> None:
    """The demo's own claim, through the runner: run until the departure is pending, press
    the dashboard's lever (the monitor's mark_discharged), finish, and nothing is written."""
    runner = _load_runner()
    pipeline = runner.Pipeline(yaml.safe_load(_demo_config(tmp_path).read_text()), REPO)
    try:
        # 2 Hz: 100 s in is frame 200; the person left at 70 s and was reaped at 78 s.
        pipeline.run(max_frames=200)
        assert PATIENT_TRACK_ID not in pipeline.book.tracks()
        assert [d.track_id for d in pipeline.egress.pending()] == [PATIENT_TRACK_ID]
        assert pipeline.state.departures()["pending"][0]["track_id"] == PATIENT_TRACK_ID
        assert pipeline.egress.mark_discharged(PATIENT_TRACK_ID) is True
        pipeline.run()
        assert pipeline.n_alerts == 0
        assert pipeline.egress.stats()["n_discharged_after_departure"] == 1
    finally:
        pipeline.close()
