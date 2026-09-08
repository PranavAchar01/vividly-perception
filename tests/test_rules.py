"""Rule layer tests: the acute-event cascade.

Four properties hold for the whole file and are load-bearing:

  * Deterministic, network-free, model-free. No RNG, no clock, no weights, no video. Every
    scenario is a closed-form trajectory from src/vividly/rules/fixtures.py evaluated on a
    fixed grid, so a failure here is a real failure and never a flake. Anything that would
    need real detector weights is not a test, it is an acceptance run.
  * The false-positive controls are the centre of the file, because avoiding false alarms
    is the stated acute-event priority: a fall followed by getting up fires nothing, a full
    simulated hour of ordinary fidgeting (with a second person walking laps through it)
    fires nothing, and the verifier seam can veto anything that does fire before it
    becomes an alert.
  * The public shapes are pinned mechanically (field names and order, rule ids, method
    signatures), because sibling modules are built against the same interface contract, so a
    silent rename breaks them at integration.
  * configs/pipeline/rules.yaml is pinned equal to the RuleConfig defaults, and every key
    in it is required to carry a PROVISIONAL comment naming the acute-event rule set and
    docs/CALIBRATION.md.
    That is a test because it is the one claim in this layer that matters outside it.

Timing: every scene below runs in tens of milliseconds to a couple of seconds. Fixtures
are pure loops over simulated time and nothing sleeps.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import math
from itertools import pairwise
from pathlib import Path

import pytest

from vividly.rules import (
    DEFAULT_RULE_IDS,
    AbstainingVerifier,
    BookRule,
    ConvulsiveMotionRule,
    EventVerifier,
    FallThenStillRule,
    Rule,
    RuleConfig,
    RuleEngine,
    RuleFiring,
    VerifierVerdict,
    annotate_firing,
    build_default_rules,
    load_rule_config,
    window_positions,
    window_postures,
)
from vividly.rules import base as base_mod
from vividly.rules.fixtures import (
    FALL_END_DEG,
    FALL_START_DEG,
    PATIENT_TRACK_ID,
    SEAT,
    STAFF_TRACK_ID,
    collapse_and_stay_down,
    collapse_then_recover,
    convulsive_episode,
    leave_and_return,
    normal_fidget,
    torso_keypoints,
)
from vividly.tracks import TrackBook, TrackConfig, TrackObservation, TrackState, posture_angle_deg

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_YAML = REPO_ROOT / "configs" / "pipeline" / "rules.yaml"
RULES_PKG = REPO_ROOT / "src" / "vividly" / "rules"

# Track sizing used by every scene: a 5 minute baseline, a 30 second reap window, and a
# ring big enough to hold an hour-long scene at 5 Hz without evicting anything.
TRACK_CFG = TrackConfig(baseline_window_s=300.0, reap_after_s=30.0, ring_size=20_000)


# ------------------------------------------------------------------ helpers
def run_scene(
    batches: list[list[TrackObservation]],
    *,
    cfg: RuleConfig | None = None,
    track_cfg: TrackConfig = TRACK_CFG,
    tick_every: int = 1,
    staff_ids: tuple[int, ...] = (STAFF_TRACK_ID,),
) -> tuple[TrackBook, RuleEngine, list[RuleFiring]]:
    """Ingest every batch, tag staff as it appears, tick the engine every `tick_every`.

    The default ticks on every batch, because under the acute-event rule set the strongest
    claim a scene can make is that nothing fired at any evaluation opportunity; the
    engine's own
    eval_every_s cadence still applies underneath. Onset accuracy never depends on the
    tick rate: every onset is recovered from stored history by the rule.
    """
    config = RuleConfig() if cfg is None else cfg
    book = TrackBook(track_cfg)
    engine = RuleEngine(build_default_rules(config), config)
    fired: list[RuleFiring] = []
    tagged: set[int] = set()
    for i, batch in enumerate(batches):
        book.ingest(batch)
        for staff_id in staff_ids:
            if staff_id not in tagged and staff_id in book.tracks():
                book.set_staff(staff_id, True)
                tagged.add(staff_id)
        if i % tick_every == 0:
            fired.extend(engine.tick(book, book.now))
    return book, engine, fired


def make_track(
    samples: list[tuple[float, float, float, float | None]],
    *,
    track_id: int = PATIENT_TRACK_ID,
    cfg: TrackConfig = TRACK_CFG,
) -> TrackState:
    """Build a TrackState from (t, cx, cy, angle_deg or None) tuples."""
    state = TrackState(track_id, cfg)
    for t, cx, cy, angle in samples:
        state.update(
            TrackObservation(
                track_id=track_id,
                t=t,
                cx=cx,
                cy=cy,
                w=80.0,
                h=110.0,
                keypoints=None if angle is None else torso_keypoints(cx, cy, angle),
            )
        )
    return state


def fallen_track(
    *,
    upright_s: float = 60.0,
    down_s: float = 40.0,
    hz: float = 5.0,
    blackout: tuple[float, float] | None = None,
    down_step_px: float = 0.1,
) -> tuple[TrackState, float]:
    """A hand-built fall: upright at 5 deg, a 1 s drop, then down at 78 deg.

    Returns the track and its final sample time. `blackout` removes posture (not
    position) inside the given (start, end) offsets measured from the start of the down
    phase. `down_step_px` is the per-sample movement while down.
    """
    dt = 1.0 / hz
    samples: list[tuple[float, float, float, float | None]] = []
    t = 0.0
    while t < upright_s:
        samples.append((t, SEAT[0], SEAT[1], 5.0))
        t += dt
    fall_t0 = t
    while t < fall_t0 + 1.0:
        progress = (t - fall_t0) / 1.0
        samples.append((t, SEAT[0], SEAT[1] + 120.0 * progress, 5.0 + 73.0 * progress))
        t += dt
    down_t0 = t
    i = 0
    while t < down_t0 + down_s:
        offset = t - down_t0
        angle: float | None = 78.0
        if blackout is not None and blackout[0] <= offset < blackout[1]:
            angle = None
        samples.append((t, SEAT[0] + (i % 2) * down_step_px, SEAT[1] + 120.0, angle))
        t += dt
        i += 1
    track = make_track(samples)
    return track, samples[-1][0]


class AlwaysRule:
    """Test stub: fires on every evaluation. Not a production rule, and not pure."""

    rule_id = "always_v0"
    rule_version = "0.0.0"

    def __init__(self, rule_id: str = "always_v0") -> None:
        self.rule_id = rule_id
        self.calls = 0

    def evaluate(self, track: TrackState, now: float) -> RuleFiring | None:
        self.calls += 1
        return RuleFiring(
            rule_id=self.rule_id,
            rule_version=self.rule_version,
            track_id=track.track_id,
            onset_t=track.first_seen_t,
            detected_t=now,
            baseline=None,
            details={},
        )


class BookAwareRule(AlwaysRule):
    """Test stub proving the engine still dispatches evaluate_book (BookRule seam)."""

    def __init__(self) -> None:
        super().__init__(rule_id="book_aware_v0")
        self.books_seen: list[TrackBook] = []

    def evaluate_book(self, book: TrackBook, track: TrackState, now: float) -> RuleFiring | None:
        self.books_seen.append(book)
        return self.evaluate(track, now)


class FixedVerifier:
    """Test stub verifier answering one fixed verdict."""

    verifier_id = "fixed_v0"
    verifier_version = "0.0.0"

    def __init__(self, verdict: str, confidence: float | None = None) -> None:
        self._verdict = verdict
        self._confidence = confidence
        self.calls = 0

    def verify(self, firing: RuleFiring, track: TrackState) -> VerifierVerdict:
        self.calls += 1
        return VerifierVerdict(
            verdict=self._verdict,
            verifier_id=self.verifier_id,
            verifier_version=self.verifier_version,
            confidence=self._confidence,
        )


def obs_at(track_id: int, t: float, cx: float = 0.0, cy: float = 0.0) -> TrackObservation:
    return TrackObservation(track_id=track_id, t=t, cx=cx, cy=cy, w=10.0, h=10.0)


# ------------------------------------------------------- contract shape pins
def test_rule_ids_and_versions_are_fixed_by_the_contract():
    rules = build_default_rules()
    assert [rule.rule_id for rule in rules] == list(DEFAULT_RULE_IDS)
    assert DEFAULT_RULE_IDS == (
        "fall_then_still_v0",
        "seizure_candidate_v0",
    )
    for rule in rules:
        assert isinstance(rule.rule_version, str) and rule.rule_version


def test_rule_firing_field_names_and_order_match_the_contract():
    names = [f.name for f in dataclasses.fields(RuleFiring)]
    assert names == [
        "rule_id",
        "rule_version",
        "track_id",
        "onset_t",
        "detected_t",
        "baseline",
        "details",
    ]
    # Frozen: a firing handed to the event store cannot be edited behind its back.
    firing = RuleFiring("r", "v", 1, 0.0, 1.0, None, {})
    with pytest.raises(dataclasses.FrozenInstanceError):
        firing.track_id = 2  # type: ignore[misc]


def test_every_rule_satisfies_the_rule_protocol_with_the_contract_signature():
    for rule in build_default_rules():
        assert isinstance(rule, Rule)
        params = list(inspect.signature(rule.evaluate).parameters)
        assert params == ["track", "now"]


def test_no_default_rule_carries_the_evaluate_book_hook_but_the_seam_still_works():
    """The last BookRule is gone; the engine keeps the dispatch seam for a future one."""
    with_hook = [r.rule_id for r in build_default_rules() if hasattr(r, "evaluate_book")]
    assert with_hook == []
    stub = BookAwareRule()
    assert isinstance(stub, BookRule)
    engine = RuleEngine([stub], RuleConfig())
    book = TrackBook(TRACK_CFG)
    book.ingest([obs_at(1, 0.0)])
    fired = engine.tick(book, 0.0)
    assert [f.rule_id for f in fired] == ["book_aware_v0"]
    assert len(stub.books_seen) == 1 and stub.books_seen[0] is book


def test_rule_engine_tick_signature_matches_the_contract():
    engine = RuleEngine(build_default_rules(), RuleConfig())
    assert list(inspect.signature(engine.tick).parameters) == ["book", "now"]


def test_rules_hold_no_mutable_state():
    """A rule instance carries its frozen config and nothing else, and repeats itself."""
    book, _, _ = run_scene(collapse_and_stay_down(hz=5.0), tick_every=25)
    track = book.tracks()[PATIENT_TRACK_ID]
    for rule in build_default_rules():
        assert set(vars(rule)) == {"cfg"}
        assert dataclasses.is_dataclass(rule.cfg)
        first = rule.evaluate(track, book.now)
        second = rule.evaluate(track, book.now)
        assert first == second


def test_no_heavy_imports_anywhere_in_the_rules_package():
    """This package is collected with no numpy and no cv2 installed.

    The default groups mark numpy, opencv and rtmlib as darwin-only, so on a Linux
    environment synced without the harness group an import of any of them here, at module
    level or inside a function, breaks collection. tests/test_tracks.py pins the same rule
    for the layer below.
    """
    banned = {"numpy", "cv2", "torch", "torchvision", "rtmlib", "onnxruntime", "trackers"}
    offenders: list[str] = []
    files = sorted(RULES_PKG.glob("*.py"))
    assert [p.name for p in files] == [
        "__init__.py",
        "base.py",
        "collapse.py",
        "convulsion.py",
        "egress.py",
        "engine.py",
        "fixtures.py",
        "verify.py",
    ]
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            offenders.extend(f"{path.name}: {name}" for name in names if name in banned)
    assert offenders == []


# --------------------------------------------------------------- config file
def test_rules_yaml_matches_the_rule_config_defaults():
    assert load_rule_config(RULES_YAML) == RuleConfig()


def test_rules_yaml_covers_every_threshold_exactly_once():
    field_names = {f.name for f in dataclasses.fields(RuleConfig)}
    keys = _yaml_rule_keys()
    assert sorted(keys) == sorted(field_names)
    assert len(keys) == len(set(keys))


def test_every_rules_yaml_threshold_is_commented_provisional():
    """Each key carries a comment block naming the acute-event rule set and pointing at
    docs/CALIBRATION.md.

    This is the one claim in this layer that has consequences outside the repository: an
    uncommented threshold is a threshold someone eventually quotes as a product number.
    """
    lines = RULES_YAML.read_text(encoding="utf-8").splitlines()
    checked = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not line.startswith("  ") or stripped.startswith("#") or ":" not in stripped:
            continue
        block: list[str] = []
        j = i - 1
        while j >= 0 and lines[j].strip().startswith("#"):
            block.append(lines[j])
            j -= 1
        key = stripped.split(":", 1)[0]
        joined = "\n".join(block)
        assert "PROVISIONAL" in joined, f"{key} has no PROVISIONAL comment"
        assert "acute-event" in joined, f"{key} does not name the acute-event rule set"
        assert "CALIBRATION.md" in joined, f"{key} does not point at docs/CALIBRATION.md"
        checked += 1
    assert checked == len(dataclasses.fields(RuleConfig))


def _yaml_rule_keys() -> list[str]:
    keys: list[str] = []
    for line in RULES_YAML.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if line.startswith("  ") and not stripped.startswith("#") and ":" in stripped:
            keys.append(stripped.split(":", 1)[0])
    return keys


def test_rule_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown RuleConfig keys"):
        RuleConfig.from_mapping({"fall_max_s": 3.0, "fal_max_s": 3.0})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fall_max_s", 0.0),
        ("fall_drop_min_px", -1.0),
        ("down_still_min_s", 0.0),
        ("down_move_max_px", 0.0),
        ("down_min_samples", 1),
        ("collapse_search_s", 0.0),
        ("convulse_reversal_frac", 1.5),
        ("convulse_min_samples", 1),
        ("convulse_amp_min_px", 0.0),
        ("eval_every_s", -1.0),
    ],
)
def test_rule_config_validates_its_thresholds(field: str, value: float):
    with pytest.raises(ValueError, match=field):
        RuleConfig(**{field: value})


def test_rule_config_rejects_a_search_window_too_short_for_the_cascade():
    with pytest.raises(ValueError, match="collapse_search_s"):
        RuleConfig(collapse_search_s=20.0, down_still_min_s=30.0)


def test_rule_config_rejects_overlapping_upright_and_down_bands():
    with pytest.raises(ValueError, match="fall_down_min_deg"):
        RuleConfig(fall_upright_max_deg=60.0, fall_down_min_deg=45.0)


def test_rule_config_rejects_a_convulse_window_shorter_than_the_min_span():
    with pytest.raises(ValueError, match="convulse_window_s"):
        RuleConfig(convulse_window_s=5.0, convulse_min_s=8.0)


def test_load_rule_config_accepts_an_inline_block(tmp_path: Path):
    path = tmp_path / "inline.yaml"
    path.write_text("down_still_min_s: 45.0\nrefire_after_s: 60.0\n", encoding="utf-8")
    cfg = load_rule_config(path)
    assert cfg.down_still_min_s == 45.0
    assert cfg.refire_after_s == 60.0
    assert cfg.convulse_window_s == RuleConfig().convulse_window_s


# -------------------------------------------------------------- window helpers
def test_window_helpers_are_anchored_on_now_not_on_last_seen():
    """A track that went quiet 10 s ago gets a window ending now, not one ending then."""
    track = make_track([(float(t), 0.0, 0.0, 5.0) for t in range(101)])
    assert track.last_seen_t == 100.0
    # TrackState anchors on its own last_seen_t: [80, 100].
    assert track.positions(20.0)[0][0] == 80.0
    # The rule layer anchors on now: [90, 100] when now is 110.
    assert window_positions(track, 110.0, 20.0)[0][0] == 90.0
    assert window_positions(track, 110.0, 20.0)[-1][0] == 100.0
    assert window_postures(track, 110.0, 20.0)[0][0] == 90.0
    # Far enough past and the window is empty rather than stale.
    assert window_positions(track, 200.0, 20.0) == []


def test_window_helpers_never_show_a_rule_the_future():
    """Asked about t=50 on a track that runs to t=100, a rule sees [30, 50] and no more."""
    track = make_track([(float(t), 0.0, float(t), 5.0) for t in range(101)])
    window = window_positions(track, 50.0, 20.0)
    assert [p[0] for p in window] == [float(t) for t in range(30, 51)]
    assert [p[0] for p in window_postures(track, 50.0, 20.0)] == [float(t) for t in range(30, 51)]


def test_window_helpers_reject_a_nonpositive_window_and_survive_an_empty_track():
    track = make_track([(0.0, 1.0, 2.0, 5.0)])
    with pytest.raises(ValueError, match="window_s"):
        window_positions(track, 1.0, 0.0)
    with pytest.raises(ValueError, match="window_s"):
        window_postures(track, 1.0, -1.0)
    empty = TrackState(7, TRACK_CFG)
    assert window_positions(empty, 10.0, 5.0) == []
    assert window_postures(empty, 10.0, 5.0) == []
    assert window_positions(track, math.nan, 5.0) == []


# --------------------------------------- the cascade: fall_then_still_v0 fires
def test_fall_then_still_fires_once_with_onset_at_the_fall():
    """The one positive scene: fall at t=360, down and still after it, one firing.

    Onset is the last upright sample the rule can anchor on, which at 5 Hz sits within
    half a second of the scripted fall start. Detection lands once the down span reaches
    down_still_min_s: the confirmation stage is the latency, by design.
    """
    lead_in_s = 360.0
    _, _, fired = run_scene(collapse_and_stay_down(hz=5.0))
    falls = [f for f in fired if f.rule_id == "fall_then_still_v0"]
    assert len(falls) == 1
    firing = falls[0]
    assert firing.track_id == PATIENT_TRACK_ID
    assert firing.onset_t == pytest.approx(lead_in_s, abs=0.6)
    expected_detect = lead_in_s + 1.5 + RuleConfig().down_still_min_s
    assert firing.detected_t == pytest.approx(expected_detect, abs=2.0)
    assert firing.details["down_s"] >= RuleConfig().down_still_min_s
    assert firing.details["fall_s"] <= RuleConfig().fall_max_s
    assert firing.details["fall_drop_px"] >= RuleConfig().fall_drop_min_px
    # The scene fires nothing else: no seizure on a still person, nothing on the patrol.
    assert [(f.rule_id, f.track_id) for f in fired] == [("fall_then_still_v0", PATIENT_TRACK_ID)]


def test_fall_then_still_stays_silent_on_a_track_with_no_pose():
    track = make_track([(float(t) / 5.0, SEAT[0], SEAT[1] + float(t), None) for t in range(600)])
    assert FallThenStillRule().evaluate(track, track.last_seen_t) is None


# --------------------- the false-positive controls (the acute-event priority)
def test_a_fall_followed_by_getting_up_fires_nothing():
    """The control. Stage 1 completes, stage 2 never does: trip, kneel, pick something
    up; the person is down for 8 s and gets up. Zero firings at every evaluation."""
    _, _, fired = run_scene(collapse_then_recover(hz=5.0))
    assert fired == []


def test_a_down_span_shorter_than_the_confirmation_window_fires_nothing():
    _, _, fired = run_scene(collapse_and_stay_down(hz=5.0, down_minutes=0.25))
    assert fired == []


def test_a_person_who_entered_the_frame_already_down_fires_nothing():
    """No upright phase inside the search window means no fall transition: abstain."""
    track = make_track([(float(t) / 5.0, SEAT[0], SEAT[1] + 120.0, 78.0) for t in range(900)])
    assert FallThenStillRule().evaluate(track, track.last_seen_t) is None


def test_a_slow_slump_to_the_ground_fires_nothing():
    """The transition takes 60 s: not a fall (fall_max_s), whatever the drift rule of
    old would have said. The removed prevention paradigm stays removed."""
    samples: list[tuple[float, float, float, float | None]] = []
    t = 0.0
    while t < 100.0:
        samples.append((t, SEAT[0], SEAT[1], 5.0))
        t += 0.2
    slump_t0 = t
    while t < slump_t0 + 60.0:
        progress = (t - slump_t0) / 60.0
        samples.append((t, SEAT[0], SEAT[1] + 120.0 * progress, 5.0 + 73.0 * progress))
        t += 0.2
    down_t0 = t
    while t < down_t0 + 40.0:
        samples.append((t, SEAT[0], SEAT[1] + 120.0, 78.0))
        t += 0.2
    track = make_track(samples)
    assert FallThenStillRule().evaluate(track, track.last_seen_t) is None


def test_a_posture_blackout_breaks_the_down_span():
    """Occlusion (a crowd, a responder) inside the down phase breaks the evidence chain.
    The same scene without the blackout fires; with it, silence."""
    clear, clear_end = fallen_track(down_s=40.0)
    assert FallThenStillRule().evaluate(clear, clear_end) is not None
    occluded, occluded_end = fallen_track(down_s=40.0, blackout=(15.0, 25.0))
    assert FallThenStillRule().evaluate(occluded, occluded_end) is None


def test_movement_on_the_ground_is_not_down_and_still():
    """A person moving on the ground fails the stillness gate (and if the motion is
    rhythmic, seizure_candidate_v0 owns it; see the seizure tests)."""
    moving, moving_end = fallen_track(down_s=40.0, down_step_px=15.0)
    assert FallThenStillRule().evaluate(moving, moving_end) is None


def test_a_small_drop_with_an_angle_flip_fires_nothing():
    """Pose-noise protection: the angle transition alone, without the centre actually
    dropping, is not a fall."""
    _, _, fired = run_scene(collapse_and_stay_down(hz=5.0, drop_px=10.0))
    assert fired == []


def test_normal_fidget_fires_no_rule_at_all_for_a_full_simulated_hour():
    """The whole-layer negative control, ticked at every batch: an hour of ordinary
    seated fidgeting with a second person walking laps through the room. One firing here
    is a false positive by construction."""
    _, engine, fired = run_scene(normal_fidget(60.0))
    assert fired == []
    assert engine.n_evaluations > 3000


def test_leave_and_return_keeps_id_and_baseline_and_fires_nothing():
    book, _, fired = run_scene(leave_and_return(30.0))
    assert fired == []
    track = book.tracks()[PATIENT_TRACK_ID]
    baseline = track.baseline()
    assert baseline is not None and baseline.complete
    assert track.n_observations > 0


def test_a_gap_longer_than_the_reap_window_is_a_new_person_and_says_so():
    book, _, fired = run_scene(leave_and_return(20.0, leave_at_minutes=8.0, gap_s=45.0))
    assert fired == []
    assert PATIENT_TRACK_ID in book.tracks()
    track = book.tracks()[PATIENT_TRACK_ID]
    # The reap happened: the "returned" track was born after the gap, not before it.
    assert track.first_seen_t >= 8.0 * 60.0 + 45.0


# ------------------------------------------- seizure_candidate_v0
def test_convulsive_episode_fires_the_seizure_rule_and_only_it():
    lead_in_s = 360.0
    _, _, fired = run_scene(convulsive_episode(hz=5.0, episode_s=20.0))
    seizures = [f for f in fired if f.rule_id == "seizure_candidate_v0"]
    assert len(seizures) == 1
    firing = seizures[0]
    assert firing.track_id == PATIENT_TRACK_ID
    # The onset is the start of the evidence window, so it can lead the true episode
    # start by up to convulse_window_s and must not trail it.
    assert firing.onset_t >= lead_in_s - RuleConfig().convulse_window_s
    assert firing.onset_t <= lead_in_s + RuleConfig().convulse_window_s
    assert firing.details["reversal_frac"] >= RuleConfig().convulse_reversal_frac
    assert firing.details["mean_step_px"] >= RuleConfig().convulse_amp_min_px
    # Nothing else fires: the thrash never reaches the down band, so no fall cascade.
    assert [(f.rule_id, f.track_id) for f in fired] == [("seizure_candidate_v0", PATIENT_TRACK_ID)]


def test_a_brief_shudder_fires_nothing():
    """Four seconds of oscillation is under convulse_min_s: dropped bag, chair scrape."""
    _, _, fired = run_scene(convulsive_episode(hz=5.0, episode_s=4.0))
    assert fired == []


def test_low_amplitude_oscillation_fires_nothing():
    """Sub-threshold alternation is jitter, not convulsion, whatever its rhythm."""
    _, _, fired = run_scene(convulsive_episode(hz=5.0, amp_x_px=0.5, amp_y_px=0.25))
    assert fired == []


def test_seizure_rule_demands_fresh_evidence():
    """The same oscillating track, evaluated 5 s after its last sample: abstain."""
    batches = convulsive_episode(hz=5.0, episode_s=20.0, staff_track_id=None)
    book = TrackBook(TRACK_CFG)
    for batch in batches:
        book.ingest(batch)
    track = book.tracks()[PATIENT_TRACK_ID]
    rule = ConvulsiveMotionRule()
    # At an evaluation inside the episode the rule fires...
    assert rule.evaluate(track, 370.0) is not None
    # ...and once the evidence is stale it abstains rather than re-reporting the past.
    assert rule.evaluate(track, track.last_seen_t + 5.0) is None


# ------------------------------------------------------------------ the engine
def test_engine_debounces_a_rule_and_track_pair():
    rule = AlwaysRule()
    engine = RuleEngine([rule], RuleConfig(refire_after_s=100.0, eval_every_s=0.0))
    book = TrackBook(TRACK_CFG)
    book.ingest([obs_at(1, 0.0)])
    assert len(engine.tick(book, 0.0)) == 1
    book.ingest([obs_at(1, 10.0)])
    assert engine.tick(book, 10.0) == []
    assert engine.n_suppressed == 1
    book.ingest([obs_at(1, 101.0)])
    assert len(engine.tick(book, 101.0)) == 1


def test_debounce_is_keyed_on_the_rule_and_track_pair_not_on_either_alone():
    rule_a, rule_b = AlwaysRule("a_v0"), AlwaysRule("b_v0")
    engine = RuleEngine([rule_a, rule_b], RuleConfig(refire_after_s=1000.0, eval_every_s=0.0))
    book = TrackBook(TRACK_CFG)
    book.ingest([obs_at(1, 0.0), obs_at(2, 0.0)])
    fired = engine.tick(book, 0.0)
    assert {(f.rule_id, f.track_id) for f in fired} == {
        ("a_v0", 1),
        ("a_v0", 2),
        ("b_v0", 1),
        ("b_v0", 2),
    }
    book.ingest([obs_at(1, 1.0), obs_at(2, 1.0)])
    assert engine.tick(book, 1.0) == []
    assert engine.n_suppressed == 4


def test_engine_skips_evaluation_passes_inside_eval_every_s():
    rule = AlwaysRule()
    engine = RuleEngine([rule], RuleConfig(refire_after_s=0.001, eval_every_s=5.0))
    book = TrackBook(TRACK_CFG)
    book.ingest([obs_at(1, 0.0)])
    assert len(engine.tick(book, 0.0)) == 1
    assert engine.tick(book, 2.0) == []
    assert rule.calls == 1
    assert engine.n_ticks == 2 and engine.n_evaluations == 1
    assert len(engine.tick(book, 5.0)) == 1


def test_engine_is_a_no_op_before_the_first_observation():
    engine = RuleEngine([AlwaysRule()], RuleConfig())
    book = TrackBook(TRACK_CFG)
    assert engine.tick(book, book.now) == []
    assert engine.n_ticks == 1 and engine.n_evaluations == 0


def test_engine_forgets_debounce_state_for_a_reaped_track():
    rule = AlwaysRule()
    engine = RuleEngine([rule], RuleConfig(refire_after_s=10_000.0, eval_every_s=0.0))
    book = TrackBook(TrackConfig(reap_after_s=5.0, ring_size=100))
    book.ingest([obs_at(1, 0.0)])
    assert len(engine.tick(book, 0.0)) == 1
    assert engine.last_fired_t(rule.rule_id, 1) == 0.0
    # The track disappears; the book reaps it; the next tick clears its debounce entry,
    # so a recycled id starts clean rather than inheriting a stranger's suppression.
    book.reap(100.0)
    assert engine.tick(book, 100.0) == []
    assert engine.last_fired_t(rule.rule_id, 1) is None
    book.ingest([obs_at(1, 101.0)])
    assert len(engine.tick(book, 101.0)) == 1


def test_engine_forget_drops_one_track_by_hand():
    engine = RuleEngine([AlwaysRule()], RuleConfig(eval_every_s=0.0))
    book = TrackBook(TRACK_CFG)
    book.ingest([obs_at(1, 0.0), obs_at(2, 0.0)])
    engine.tick(book, 0.0)
    assert engine.last_fired_t("always_v0", 1) is not None
    engine.forget(1)
    assert engine.last_fired_t("always_v0", 1) is None
    assert engine.last_fired_t("always_v0", 2) is not None


# ------------------------------------------- the acute-event verifier seam
def test_a_rejecting_verifier_stops_the_firing_and_the_refire_window_still_starts():
    verifier = FixedVerifier("rejected")
    engine = RuleEngine(
        [AlwaysRule()], RuleConfig(refire_after_s=100.0, eval_every_s=0.0), verifier=verifier
    )
    book = TrackBook(TRACK_CFG)
    book.ingest([obs_at(1, 0.0)])
    assert engine.tick(book, 0.0) == []
    assert engine.n_rejected == 1
    # Debounce recorded before the verifier: the persisting condition costs one verifier
    # call per refire window, not one per tick.
    book.ingest([obs_at(1, 1.0)])
    assert engine.tick(book, 1.0) == []
    assert verifier.calls == 1 and engine.n_suppressed == 1


def test_a_confirming_verifier_stamps_the_firing():
    engine = RuleEngine(
        [AlwaysRule()],
        RuleConfig(eval_every_s=0.0),
        verifier=FixedVerifier("confirmed", confidence=0.93),
    )
    book = TrackBook(TRACK_CFG)
    book.ingest([obs_at(1, 0.0)])
    (firing,) = engine.tick(book, 0.0)
    assert firing.details["verifier_verdict"] == "confirmed"
    assert firing.details["verifier_id"] == "fixed_v0"
    assert firing.details["verifier_confidence"] == pytest.approx(0.93)


def test_the_abstaining_verifier_annotates_by_default_and_blocks_under_require():
    """Both shipped behaviours of the no-verifier-configured state: visible pass-through, or
    fail-closed. The fail-closed mode is the maximum false-positive suppression a
    deployment can choose before a trained verifier head is wired into its config."""
    assert isinstance(AbstainingVerifier(), EventVerifier)

    annotating = RuleEngine(
        [AlwaysRule()], RuleConfig(eval_every_s=0.0), verifier=AbstainingVerifier()
    )
    book = TrackBook(TRACK_CFG)
    book.ingest([obs_at(1, 0.0)])
    (firing,) = annotating.tick(book, 0.0)
    assert firing.details["verifier_verdict"] == "unavailable"
    assert "verifier_confidence" not in firing.details

    requiring = RuleEngine(
        [AlwaysRule()],
        RuleConfig(eval_every_s=0.0),
        verifier=AbstainingVerifier(),
        verify_policy="require",
    )
    other = TrackBook(TRACK_CFG)
    other.ingest([obs_at(1, 0.0)])
    assert requiring.tick(other, 0.0) == []
    assert requiring.n_unconfirmed == 1


def test_a_scene_aware_verifier_is_handed_the_book_before_rules_run():
    """The observe_book hook (scene-channels heads, seizure final_v2 onward): optional
    via getattr, so a verifier that exposes it sees the book on every evaluation tick,
    one that does not is dispatched exactly as before."""

    class SceneVerifier(FixedVerifier):
        def __init__(self) -> None:
            super().__init__("confirmed", confidence=0.9)
            self.observed: list[tuple[TrackBook, float]] = []

        def observe_book(self, book: TrackBook, now: float) -> None:
            self.observed.append((book, now))

    verifier = SceneVerifier()
    engine = RuleEngine([AlwaysRule()], RuleConfig(eval_every_s=0.0), verifier=verifier)
    book = TrackBook(TRACK_CFG)
    book.ingest([obs_at(1, 0.0)])
    (firing,) = engine.tick(book, 0.0)
    assert firing.details["verifier_verdict"] == "confirmed"
    assert verifier.observed == [(book, 0.0)]

    plain = FixedVerifier("confirmed")
    engine2 = RuleEngine([AlwaysRule()], RuleConfig(eval_every_s=0.0), verifier=plain)
    book2 = TrackBook(TRACK_CFG)
    book2.ingest([obs_at(1, 0.0)])
    assert len(engine2.tick(book2, 0.0)) == 1 and plain.calls == 1


def test_engine_rejects_a_bad_policy_and_a_non_verifier():
    with pytest.raises(ValueError, match="verify_policy"):
        RuleEngine([AlwaysRule()], RuleConfig(), verify_policy="maybe")
    with pytest.raises(TypeError, match="EventVerifier"):
        RuleEngine([AlwaysRule()], RuleConfig(), verifier=object())  # type: ignore[arg-type]


def test_verifier_verdict_validates_itself():
    with pytest.raises(ValueError, match="verdict"):
        VerifierVerdict(verdict="perhaps", verifier_id="x", verifier_version="1")
    with pytest.raises(ValueError, match="confidence"):
        VerifierVerdict(verdict="confirmed", verifier_id="x", verifier_version="1", confidence=1.5)
    with pytest.raises(ValueError, match="verifier_id"):
        VerifierVerdict(verdict="confirmed", verifier_id="", verifier_version="1")


def test_annotate_firing_returns_a_copy_and_never_mutates():
    firing = RuleFiring("r", "v", 1, 0.0, 1.0, None, {"kept": 1.0})
    verdict = VerifierVerdict(
        verdict="confirmed", verifier_id="x", verifier_version="1", confidence=0.5
    )
    stamped = annotate_firing(firing, verdict)
    assert stamped is not firing
    assert firing.details == {"kept": 1.0}
    assert stamped.details["kept"] == 1.0
    assert stamped.details["verifier_verdict"] == "confirmed"


# ------------------------------------------------------------------- fixtures
def test_fixture_generators_are_deterministic():
    for generator in (
        lambda: collapse_and_stay_down(hz=5.0),
        lambda: collapse_then_recover(hz=5.0),
        lambda: convulsive_episode(hz=5.0),
        lambda: normal_fidget(2.0),
        lambda: leave_and_return(10.0),
    ):
        assert generator() == generator()


def test_fixture_batches_are_sorted_by_track_id_on_the_grid():
    batches = collapse_and_stay_down(hz=5.0, lead_in_minutes=1.0, down_minutes=1.0)
    assert batches[0][0].t == 0.0
    for batch in batches:
        ids = [obs.track_id for obs in batch]
        assert ids == sorted(ids)
    dt = [b[0].t - a[0].t for a, b in pairwise(batches)]
    assert all(step == pytest.approx(0.2) for step in dt)


def test_fixture_keypoints_round_trip_through_the_posture_formula():
    for angle in (0.0, 6.0, 30.0, 60.0, 78.0, 90.0):
        keypoints = torso_keypoints(100.0, 200.0, angle)
        assert posture_angle_deg(keypoints, 0.3) == pytest.approx(angle)


def test_fixture_generators_reject_nonsense_durations():
    with pytest.raises(ValueError, match="minutes"):
        normal_fidget(0.0)
    with pytest.raises(ValueError, match="fall_s"):
        collapse_and_stay_down(fall_s=0.0)
    with pytest.raises(ValueError, match="episode_s"):
        convulsive_episode(episode_s=0.0)
    with pytest.raises(ValueError, match="gap_s"):
        leave_and_return(gap_s=0.0)


def test_a_one_person_scene_has_no_staff_track():
    batches = collapse_and_stay_down(hz=5.0, lead_in_minutes=1.0, staff_track_id=None)
    assert {obs.track_id for batch in batches for obs in batch} == {PATIENT_TRACK_ID}


def test_the_collapse_fixture_traces_the_cascade_shape():
    """Upright before, down after, and the drop lands where the constants say."""
    lead_in_s = 60.0
    batches = collapse_and_stay_down(
        hz=5.0, lead_in_minutes=1.0, down_minutes=1.0, staff_track_id=None
    )
    by_t = {batch[0].t: batch[0] for batch in batches}
    before = by_t[lead_in_s - 1.0]
    after = by_t[lead_in_s + 5.0]
    angle_before = posture_angle_deg(before.keypoints, 0.3)
    angle_after = posture_angle_deg(after.keypoints, 0.3)
    assert angle_before is not None and angle_before < 15.0
    assert angle_after is not None and angle_after == pytest.approx(FALL_END_DEG, abs=1.0)
    assert after.cy - before.cy == pytest.approx(120.0, abs=6.0)
    assert FALL_START_DEG < 15.0 < FALL_END_DEG


def test_the_convulsive_fixture_reverses_direction_every_step():
    batches = convulsive_episode(hz=5.0, lead_in_minutes=1.0, episode_s=20.0, staff_track_id=None)
    inside = [b[0] for b in batches if 62.0 <= b[0].t <= 78.0]
    steps = [(b.cx - a.cx, b.cy - a.cy) for a, b in pairwise(inside)]
    for a, b in pairwise(steps):
        assert a[0] * b[0] + a[1] * b[1] < 0.0


def test_the_staff_patrol_moves_at_a_constant_speed_by_construction():
    batches = normal_fidget(10.0)
    staff = [obs for batch in batches for obs in batch if obs.track_id == STAFF_TRACK_ID]
    steps = [math.hypot(b.cx - a.cx, b.cy - a.cy) for a, b in pairwise(staff)]
    assert max(steps) - min(steps) < 1e-6


# ------------------------------------------------------------------ the surface
def test_build_default_rules_shares_one_config_object():
    cfg = RuleConfig(refire_after_s=123.0)
    rules = build_default_rules(cfg)
    assert all(rule.cfg is cfg for rule in rules)


def test_base_module_exposes_the_contract_surface():
    for name in (
        "SECONDS_PER_MINUTE",
        "BookRule",
        "Rule",
        "RuleConfig",
        "RuleFiring",
        "load_rule_config",
        "window_positions",
        "window_postures",
    ):
        assert hasattr(base_mod, name), name
