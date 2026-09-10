"""The operating-point application: the config, and the two engine seams.

research/operating-points.md measured PROVISIONAL operating points for the event
verifier heads and named two decision-layer behaviours the live engine could not express
(section 5): multi-window persistence/hysteresis, and a camera-level episode merge. This
file pins what the shipped configuration applies.

The claim that matters most is the negative one. Both seams are off by default, and off
means byte-identical, not merely similar: an unconfigured head calls the graph exactly
once with the same array and returns the same verdict and the same confidence float it
returned before the seam existed, and the shipped configs/pipeline/rules.yaml leaves the
camera debounce at 0.0, where its map is never even written. A default-on decision knob
would silently re-point every number ever measured for these heads.

No graphs are loaded here. Heads take anything with session.run(), so scripted fakes
stand in, and the run directories the applied config names are the committed lane
directories final_models/seizure and final_models/fall. The config assertions below read
YAML, never weights.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from vividly.models.event_verifier import (
    DEFAULT_CONFIRM_THRESHOLD,
    DEFAULT_PERSIST_K,
    DEFAULT_PRE_PAD_S,
    HEAD_CONFIG_KEYS,
    OnnxEventVerifier,
    VerifierBackendError,
    VerifierHead,
    validate_decision_policy,
)
from vividly.rules.base import RuleConfig, RuleFiring, load_rule_config
from vividly.rules.engine import RuleEngine
from vividly.tracks import (
    NUM_KEYPOINTS,
    TrackBook,
    TrackConfig,
    TrackObservation,
    TrackState,
)
from vividly.verifier.features import WindowSpec, featurize_window, tracks_from_rows

REPO = Path(__file__).resolve().parents[1]
PIPELINE_CONFIGS = REPO / "configs" / "pipeline"
REPLAY_CONFIG = PIPELINE_CONFIGS / "replay_demo.yaml"
RULES_YAML = PIPELINE_CONFIGS / "rules.yaml"
CALIBRATION_DOC = REPO / "docs" / "CALIBRATION.md"
#: The measurement report the two engine seams come from. The zero-footage pipeline
#: configs, the rule thresholds file and docs/CALIBRATION.md all cite this path, and this
#: constant is what keeps them in step.
REPORT_REL = "research/operating-points.md"
REPORT_DOC = REPO / REPORT_REL

KP_CONF_MIN = 0.3
FPS = 10.0
SPEC = WindowSpec(window_s=8.0, stride_s=1.5, n_steps=64, policy="contain")


# ------------------------------------------------------------------ fixtures
def kps(angle_deg: float, conf: float = 0.9) -> tuple[tuple[float, float, float], ...]:
    """COCO-17 keypoints whose torso tilts angle_deg from vertical."""
    out = [(0.0, 0.0, conf)] * NUM_KEYPOINTS
    dx = 40.0 * math.sin(math.radians(angle_deg))
    dy = 40.0 * math.cos(math.radians(angle_deg))
    out[5] = (-10.0, 0.0, conf)
    out[6] = (10.0, 0.0, conf)
    out[11] = (-10.0 + dx, dy, conf)
    out[12] = (10.0 + dx, dy, conf)
    return tuple(out)


def fall_track(track_id: int = 3) -> TrackState:
    """A collapse at t=10 then 30 s down and still (same shape as the backend tests)."""
    cfg = TrackConfig(kp_conf_min=KP_CONF_MIN, ring_size=2000)
    st = TrackState(track_id, cfg)
    t = 0.0
    while t < 45.0:
        if t < 10.0:
            cy, angle, h = 100.0 + 0.3337 * t, 5.0, 80.111
        elif t <= 11.0:
            frac = t - 10.0
            cy, angle, h = 100.0 + 160.0 * frac, 5.0 + 80.0 * frac, 80.111 - 20.0 * frac
        else:
            cy, angle, h = 260.0, 85.0, 60.111
        st.update(
            TrackObservation(
                track_id=track_id,
                t=round(t, 4),
                cx=160.123456,
                cy=cy,
                w=40.667,
                h=h,
                keypoints=kps(angle),
            )
        )
        t += 1.0 / FPS
    return st


class ScriptedSession:
    """ort.InferenceSession stand-in: one logit per call, in order, feeds recorded."""

    def __init__(self, *logits: float) -> None:
        self.logits = list(logits)
        self.feeds: list[np.ndarray] = []

    def run(self, outputs, feeds):
        assert outputs is None
        x = feeds["windows"]
        self.feeds.append(x)
        logit = self.logits[len(self.feeds) - 1] if len(self.feeds) <= len(self.logits) else 0.0
        return [np.full((x.shape[0], 1), logit, dtype=np.float32)]


def head(session, **overrides) -> VerifierHead:
    kwargs = {
        "session": session,
        "spec": SPEC,
        "task": "fall",
        "weight_class": "prototype",
        "sha8": "deadbeef",
    }
    kwargs.update(overrides)
    return VerifierHead(**kwargs)


def firing(rule_id: str = "fall_then_still_v0", onset_t: float = 10.0) -> RuleFiring:
    return RuleFiring(
        rule_id=rule_id,
        rule_version="0.1.0",
        track_id=3,
        onset_t=onset_t,
        detected_t=onset_t + 30.0,
        baseline=None,
        details={},
    )


def reference_window(track: TrackState, t_end: float, spec: WindowSpec = SPEC) -> np.ndarray:
    """What the TRAINING path would featurize for this window, rows rounded as the
    extraction writer rounds them. The backend must feed exactly this."""
    rows = []
    for (t, cx, cy), (_, w, h, kpc) in zip(track.positions(1e6), track.boxes(1e6), strict=True):
        angle = dict(track.posture_angles(1e6)).get(t)
        rows.append(
            {
                "t": round(t, 4),
                "track": track.track_id,
                "cx": round(cx, 2),
                "cy": round(cy, 2),
                "w": round(w, 2),
                "h": round(h, 2),
                "angle": None if angle is None else round(angle, 2),
                "kpc": None if math.isnan(kpc) else round(kpc, 4),
            }
        )
    out = featurize_window(tracks_from_rows(rows)[0], t_end, spec)
    assert out is not None
    return out


def sigmoid(logit: float, temperature: float = 1.0) -> float:
    """The score the backend computes for a graph that emitted `logit`.

    The float32 round-trip is not decoration: the graph's output dtype is float32, so
    float(np.float32(0.4)) != 0.4, and these tests compare confidences with `==` on
    purpose: "the same float, not approximately the same float" is the byte-identity
    claim.
    """
    return float(1.0 / (1.0 + math.exp(-(float(np.float32(logit)) / temperature))))


class AlwaysRule:
    """Test stub: fires on every evaluation, so the engine's admission logic is the
    only thing under test."""

    rule_version = "0.0.0"

    def __init__(self, rule_id: str = "always_v0") -> None:
        self.rule_id = rule_id

    def evaluate(self, track: TrackState, now: float) -> RuleFiring | None:
        return RuleFiring(
            rule_id=self.rule_id,
            rule_version=self.rule_version,
            track_id=track.track_id,
            onset_t=track.first_seen_t,
            detected_t=now,
            baseline=None,
            details={},
        )


def obs_at(track_id: int, t: float) -> TrackObservation:
    return TrackObservation(track_id=track_id, t=t, cx=0.0, cy=0.0, w=10.0, h=10.0)


# ==================================================================== part 1:
# the applied config
def test_replay_demo_carries_the_recommended_operating_points() -> None:
    """The operating points the two lane pages record, as applied."""
    cfg = yaml.safe_load(REPLAY_CONFIG.read_text(encoding="utf-8"))
    verify = cfg["verify"]
    assert verify["policy"] == "annotate"
    heads = verify["heads"]
    assert set(heads) == {"seizure_candidate_v0", "fall_then_still_v0"}
    # Seizure: convulsion_v2_real, graph 07c399be. The lane page's RECOMMENDED point is 0.8;
    # the applied point is 0.5, set on 2026-09-09 at the founder's direction for live
    # demonstration. That is a deviation, so the guard is not "it equals the recommended
    # number" but the stricter thing POLICY.md rule 10 actually asks for: the applied number
    # is one the lane page measured, and the file says out loud what it costs.
    assert heads["seizure_candidate_v0"]["run_dir"] == "final_models/seizure"
    assert heads["seizure_candidate_v0"]["confirm_threshold"] == 0.5
    # Comments wrap, so compare against the prose with the wrapping taken out rather than
    # against a line, which would make this guard break on a reflow instead of on a change.
    prose = " ".join(
        line.strip().lstrip("#").strip() for line in REPLAY_CONFIG.read_text(encoding="utf-8").splitlines()
    )
    assert "0.5 catches 12 of 13 events with 72.4 percent of quiet footage in alarm" in prose
    assert "0.8 catches 8 of 13 at 33.2 percent" in prose
    assert "demonstration setting and not a deployment one" in prose
    # Fall: fall_v3, graph 67954c04, at the 0.95 final_models/fall/README.md records. The
    # interim arm fall_v2pose (sha8 b66c3a0a) is a recorded arm only.
    assert heads["fall_then_still_v0"]["run_dir"] == "final_models/fall"
    assert heads["fall_then_still_v0"]["confirm_threshold"] == 0.95
    for rule_id, block in heads.items():
        unknown = set(block) - HEAD_CONFIG_KEYS
        assert not unknown, f"{rule_id} carries keys the backend would refuse: {unknown}"
        # Neither applied head turns on the persistence seam: the measurements say it
        # loses catches on both surfaces (section 5.1).
        assert "persist_k" not in block and "hyst_hi" not in block


def test_every_applied_threshold_travels_with_its_provenance() -> None:
    """POLICY.md, rule 10, in the file a threshold is read from: no number without the sha8
    it was measured against, the surface it was fitted on, and the page it came from."""
    text = REPLAY_CONFIG.read_text(encoding="utf-8")
    assert "final_models/seizure/README.md" in text and "final_models/fall/README.md" in text
    assert "07c399be" in text and "67954c04" in text
    assert "PROVISIONAL" in text and "CALIBRATION.md" in text
    # The fitting surface is the caveat that must not get lost.
    assert "fitted on the surfaces" in text


def test_the_pointers_into_the_report_exist_where_a_reader_looks() -> None:
    assert REPORT_DOC.is_file()
    rules_text = RULES_YAML.read_text(encoding="utf-8")
    assert "confirm_threshold" in rules_text and REPORT_REL in rules_text
    calibration = CALIBRATION_DOC.read_text(encoding="utf-8")
    assert REPORT_REL in calibration
    assert "fitted on exam negatives" in calibration


def test_replay_demo_is_the_only_config_that_wires_heads() -> None:
    """The zero-footage demos stay verifier-less on purpose (abstract points are not
    motion a trained head can read), and each says so where the next editor will look."""
    for config in sorted(PIPELINE_CONFIGS.glob("*.yaml")):
        text = config.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        has_heads = isinstance(data, dict) and bool((data.get("verify") or {}).get("heads"))
        if config.name == "replay_demo.yaml":
            assert has_heads
        else:
            assert not has_heads, f"{config.name} gained a verify: block unnoticed"
        if config.name in {"synthetic_demo.yaml", "egress_demo.yaml"}:
            assert "`verify:` block, on purpose" in text.lower()
            assert REPORT_REL in text


# ==================================================================== part 2:
# seam A (persistence/hysteresis) is off by default, and off is byte-identical
def test_default_head_scores_exactly_one_window_at_t_end() -> None:
    session = ScriptedSession(2.0)
    v = OnnxEventVerifier({"fall_then_still_v0": head(session)})
    track = fall_track()
    verdict = v.verify(firing(onset_t=10.0), track)

    assert len(session.feeds) == 1, "an unconfigured head must call the graph once"
    expected = reference_window(track, 10.0 - DEFAULT_PRE_PAD_S + SPEC.window_s)
    np.testing.assert_array_equal(session.feeds[0][0], expected.astype(np.float32))
    assert verdict.verdict == "confirmed"
    assert verdict.confidence == sigmoid(2.0)


@pytest.mark.parametrize("logit", [-3.0, -0.5, 0.0, 0.4, 2.0])
@pytest.mark.parametrize("threshold", [0.0, 0.5, 0.8, 0.9, 1.0])
def test_default_head_reproduces_the_single_threshold_decision_exactly(
    logit: float, threshold: float
) -> None:
    """score >= threshold -> confirmed, below -> rejected, confidence is the score:
    the same float, not an approximation of it, at every applied threshold."""
    v = OnnxEventVerifier(
        {"fall_then_still_v0": head(ScriptedSession(logit), confirm_threshold=threshold)}
    )
    verdict = v.verify(firing(), fall_track())
    score = sigmoid(logit)
    assert verdict.confidence == score
    assert verdict.verdict == ("confirmed" if score >= threshold else "rejected")


def test_the_seam_defaults_are_the_off_position() -> None:
    default = head(ScriptedSession(0.0))
    assert default.persist_k == DEFAULT_PERSIST_K == 1
    assert default.hyst_hi is None
    assert default.confirm_threshold == DEFAULT_CONFIRM_THRESHOLD


# ==================================================================== part 3:
# persistence requires k consecutive windows
def test_persist_k_scores_k_stride_offset_windows() -> None:
    session = ScriptedSession(2.0, 2.0, 2.0)
    v = OnnxEventVerifier({"fall_then_still_v0": head(session, persist_k=3)})
    track = fall_track()
    v.verify(firing(onset_t=10.0), track)

    assert len(session.feeds) == 3
    t_end = 10.0 - DEFAULT_PRE_PAD_S + SPEC.window_s
    for i in range(3):
        expected = reference_window(track, t_end + i * SPEC.stride_s)
        np.testing.assert_array_equal(session.feeds[i][0], expected.astype(np.float32))


def test_persist_k_confirms_only_when_every_window_clears_the_threshold() -> None:
    track, thr = fall_track(), 0.8
    high, low = 2.0, 0.5  # sigmoid: 0.881 (>= 0.8) and 0.622 (< 0.8)
    assert sigmoid(high) >= thr > sigmoid(low)

    both = OnnxEventVerifier(
        {
            "fall_then_still_v0": head(
                ScriptedSession(high, high), persist_k=2, confirm_threshold=thr
            )
        }
    )
    verdict = both.verify(firing(), track)
    assert verdict.verdict == "confirmed"
    # The binding score of an all-k confirmation is the weakest window.
    assert verdict.confidence == sigmoid(high)

    for logits in ((high, low), (low, high)):
        broken = OnnxEventVerifier(
            {
                "fall_then_still_v0": head(
                    ScriptedSession(*logits), persist_k=2, confirm_threshold=thr
                )
            }
        )
        verdict = broken.verify(firing(), track)
        assert verdict.verdict == "rejected", f"{logits} is not k consecutive"
        # ...and the reviewer still sees how close it got: the peak of the k.
        assert verdict.confidence == sigmoid(high)


def test_a_single_window_that_would_confirm_alone_is_not_enough_under_persistence() -> None:
    """The whole point of persist:k@thr against single:thr, in one comparison."""
    logits = (2.0, -2.0)
    single = OnnxEventVerifier(
        {"fall_then_still_v0": head(ScriptedSession(*logits), confirm_threshold=0.8)}
    )
    persistent = OnnxEventVerifier(
        {"fall_then_still_v0": head(ScriptedSession(*logits), persist_k=2, confirm_threshold=0.8)}
    )
    assert single.verify(firing(), fall_track()).verdict == "confirmed"
    assert persistent.verify(firing(), fall_track()).verdict == "rejected"


def test_hysteresis_confirms_immediately_on_one_high_window() -> None:
    """hyst:hi|k@lo means immediate at >= hi, or k consecutive >= lo (the report's
    section 3 row hyst:0.9|2@0.65, the one recorded combination that would pay)."""
    hi, lo = 0.9, 0.65
    strong, weak = 3.0, -2.0  # 0.953 (>= hi) and 0.119 (< lo)
    assert sigmoid(strong) >= hi and sigmoid(weak) < lo

    def build(*logits):
        return OnnxEventVerifier(
            {
                "fall_then_still_v0": head(
                    ScriptedSession(*logits), persist_k=2, confirm_threshold=lo, hyst_hi=hi
                )
            }
        )

    # One window over hi is enough even though the pair is not sustained.
    verdict = build(strong, weak).verify(firing(), fall_track())
    assert verdict.verdict == "confirmed"
    assert verdict.confidence == sigmoid(strong)
    # The hi leg reads any of the k, not just the first.
    assert build(weak, strong).verify(firing(), fall_track()).verdict == "confirmed"
    # Sustained-but-never-high still confirms through the lo leg.
    sustained = 1.0  # 0.731: >= lo, < hi
    assert lo <= sigmoid(sustained) < hi
    assert build(sustained, sustained).verify(firing(), fall_track()).verdict == "confirmed"
    # Neither leg: rejected.
    assert build(sustained, weak).verify(firing(), fall_track()).verdict == "rejected"


def test_persistence_abstains_when_a_later_window_has_no_evidence() -> None:
    """An incomplete pattern is missing evidence, not evidence of absence: unavailable,
    never rejected. (The track here ends at t=45; the third window starts at t=48.)"""
    spec = WindowSpec(window_s=8.0, stride_s=20.0, n_steps=64, policy="contain")
    session = ScriptedSession(2.0, 2.0, 2.0)
    v = OnnxEventVerifier({"fall_then_still_v0": head(session, spec=spec, persist_k=3)})
    verdict = v.verify(firing(onset_t=10.0), fall_track())
    assert verdict.verdict == "unavailable"
    assert verdict.confidence is None
    assert len(session.feeds) < 3, "the run should stop once the evidence runs out"


# ==================================================================== part 4:
# seam B (camera-level episode merge) is off by default
def test_shipped_config_leaves_the_camera_debounce_off() -> None:
    assert RuleConfig().camera_refire_after_s == 0.0
    assert load_rule_config(RULES_YAML).camera_refire_after_s == 0.0


def test_camera_debounce_off_admits_every_track_and_never_writes_its_map() -> None:
    """Byte-identity for the engine seam: under the shipped thresholds, every track that
    clears its own debounce is admitted and the camera map is never even written,
    exactly as before the seam existed."""
    engine = RuleEngine([AlwaysRule()], load_rule_config(RULES_YAML))
    book = TrackBook(TrackConfig(reap_after_s=30.0, ring_size=1000))
    book.ingest([obs_at(1, 0.0), obs_at(2, 0.0), obs_at(3, 0.0)])
    fired = engine.tick(book, 0.0)
    assert [f.track_id for f in fired] == [1, 2, 3]
    assert engine.n_suppressed == 0
    assert engine.n_camera_suppressed == 0
    assert engine.last_fired_camera_t("always_v0") is None
    # Two tracks firing 15 minutes later still both come through: nothing about one
    # person's alert reaches another's, which is what "off" has to mean.
    book.ingest([obs_at(1, 1000.0), obs_at(2, 1000.0)])
    assert [f.track_id for f in engine.tick(book, 1000.0)] == [1, 2]
    assert engine.n_camera_suppressed == 0


# ==================================================================== part 5:
# camera debounce behaviour when it is configured
def camera_engine(window_s: float) -> tuple[RuleEngine, TrackBook]:
    cfg = RuleConfig(refire_after_s=900.0, eval_every_s=0.0, camera_refire_after_s=window_s)
    return (
        RuleEngine([AlwaysRule()], cfg),
        TrackBook(TrackConfig(reap_after_s=30.0, ring_size=1000)),
    )


def test_camera_debounce_suppresses_a_second_track_inside_the_window() -> None:
    engine, book = camera_engine(60.0)
    book.ingest([obs_at(1, 0.0), obs_at(2, 0.0)])
    fired = engine.tick(book, 0.0)

    assert [f.track_id for f in fired] == [1], "one camera, one episode"
    assert engine.n_camera_suppressed == 1
    assert engine.n_suppressed == 1
    assert engine.last_fired_camera_t("always_v0") == 0.0
    # The suppressed track must not have started its own 15-minute refire window: it
    # never fired, and muting that person on the strength of somebody else's alert is
    # a different (and much worse) behaviour than merging an episode.
    assert engine.last_fired_t("always_v0", 2) is None
    assert engine.last_fired_t("always_v0", 1) == 0.0


def test_camera_debounce_admits_the_second_track_after_the_window() -> None:
    engine, book = camera_engine(60.0)
    book.ingest([obs_at(1, 0.0), obs_at(2, 0.0)])
    engine.tick(book, 0.0)

    # Still inside the window: nothing at all, from either track.
    book.ingest([obs_at(1, 30.0), obs_at(2, 30.0)])
    assert engine.tick(book, 30.0) == []
    assert engine.n_camera_suppressed == 2

    # Past it: track 2 is admitted (track 1 is still held by its own per-track debounce,
    # which is 900 s and is checked first).
    book.ingest([obs_at(1, 61.0), obs_at(2, 61.0)])
    fired = engine.tick(book, 61.0)
    assert [f.track_id for f in fired] == [2]
    assert engine.last_fired_camera_t("always_v0") == 61.0


def test_camera_debounce_is_scoped_per_rule_not_across_rules() -> None:
    cfg = RuleConfig(refire_after_s=900.0, eval_every_s=0.0, camera_refire_after_s=60.0)
    engine = RuleEngine([AlwaysRule("a_v0"), AlwaysRule("b_v0")], cfg)
    book = TrackBook(TrackConfig(reap_after_s=30.0, ring_size=1000))
    book.ingest([obs_at(1, 0.0), obs_at(2, 0.0)])
    fired = engine.tick(book, 0.0)
    # Each rule keeps its own camera clock: one alert per rule, not one alert total.
    assert {(f.rule_id, f.track_id) for f in fired} == {("a_v0", 1), ("b_v0", 1)}


def test_camera_debounce_survives_a_reaped_track() -> None:
    """Its subject is the camera, not the person, so the debounce cleanup that drops a
    reaped track's per-track entries must leave it alone."""
    engine, book = camera_engine(600.0)
    book.ingest([obs_at(1, 0.0)])
    assert len(engine.tick(book, 0.0)) == 1
    book.reap(100.0)
    assert engine.tick(book, 100.0) == []
    assert engine.last_fired_t("always_v0", 1) is None, "per-track entry is cleaned up"
    assert engine.last_fired_camera_t("always_v0") == 0.0, "camera entry is not"
    # A brand-new person inside the window does not re-open the episode.
    book.ingest([obs_at(9, 101.0)])
    assert engine.tick(book, 101.0) == []
    assert engine.n_camera_suppressed == 1
    engine.forget(9)
    assert engine.last_fired_camera_t("always_v0") == 0.0


# ==================================================================== part 6:
# config parsing refuses nonsense
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"confirm_threshold": 1.5}, "confirm_threshold"),
        ({"confirm_threshold": -0.1}, "confirm_threshold"),
        ({"persist_k": 0}, "persist_k must be >= 1"),
        ({"persist_k": -2}, "persist_k must be >= 1"),
        ({"persist_k": 2.5}, "persist_k must be an int"),
        ({"persist_k": True}, "persist_k must be an int"),
        ({"persist_k": 2, "hyst_hi": 1.5}, "hyst_hi must be in"),
        ({"hyst_hi": 0.9}, "hyst_hi needs persist_k >= 2"),
        ({"persist_k": 2, "hyst_hi": 0.4, "confirm_threshold": 0.8}, "must be >="),
    ],
)
def test_a_head_refuses_a_policy_that_cannot_mean_what_it_says(kwargs, match) -> None:
    with pytest.raises(VerifierBackendError, match=match):
        head(ScriptedSession(0.0), **kwargs)


def test_validate_decision_policy_accepts_the_shapes_the_report_measured() -> None:
    validate_decision_policy(0.8, 1, None)  # single:0.8 (both applied heads)
    validate_decision_policy(0.9, 1, None)  # single:0.9
    validate_decision_policy(0.8, 2, None)  # persist:2@0.8
    validate_decision_policy(0.65, 2, 0.9)  # hyst:0.9|2@0.65
    validate_decision_policy(0.9, 2, 0.9)  # hi == lo is degenerate but not nonsense


def test_from_config_refuses_an_unknown_head_key(tmp_path: Path) -> None:
    """A typo'd threshold that silently kept the 0.5 default would make an uncalibrated
    run look like a calibrated one. Refused before the filesystem is touched."""
    cfg = {"fall_then_still_v0": {"run_dir": "outputs/verifier/whatever", "confirm_treshold": 0.9}}
    with pytest.raises(VerifierBackendError, match="unknown keys"):
        OnnxEventVerifier.from_config(cfg, tmp_path)


def test_from_config_refuses_a_head_with_no_run_dir(tmp_path: Path) -> None:
    with pytest.raises(VerifierBackendError, match="no run_dir"):
        OnnxEventVerifier.from_config({"fall_then_still_v0": {"confirm_threshold": 0.9}}, tmp_path)


@pytest.mark.parametrize(
    ("block", "match"),
    [
        ({"persist_k": "two"}, r"persist_k must be a whole number"),
        ({"persist_k": 2.5}, r"persist_k must be a whole number"),
        ({"persist_k": True}, r"persist_k must be a whole number"),
        ({"confirm_threshold": "high"}, r"confirm_threshold must be a number"),
        ({"hyst_hi": []}, r"hyst_hi must be a number"),
        ({"pre_pad_s": {}}, r"pre_pad_s must be a number"),
    ],
)
def test_from_config_refuses_a_value_of_the_wrong_shape(block, match, tmp_path: Path) -> None:
    """These are caught while the arguments are built, so the message names the rule id
    and the key rather than surfacing as a bare ValueError from float()/int()."""
    cfg = {"fall_then_still_v0": {"run_dir": str(tmp_path / "nope"), **block}}
    with pytest.raises(VerifierBackendError, match=match):
        OnnxEventVerifier.from_config(cfg, tmp_path)


def test_rule_config_refuses_a_negative_camera_window() -> None:
    with pytest.raises(ValueError, match="camera_refire_after_s must be >= 0"):
        RuleConfig(camera_refire_after_s=-1.0)
    with pytest.raises(ValueError, match="camera_refire_after_s"):
        RuleConfig.from_mapping({"camera_refire_after_s": -0.5})


def test_rule_config_still_refuses_an_unknown_key() -> None:
    """The camera window is a real field now; a plausible-looking neighbour is not."""
    with pytest.raises(ValueError, match="unknown RuleConfig keys"):
        RuleConfig.from_mapping({"camera_debounce_s": 60.0})
