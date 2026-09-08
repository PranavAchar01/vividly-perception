"""Track state layer tests (replay mode).

Every rule in src/vividly/rules/ trusts this module, so the coverage here is deliberately
exhaustive: ring eviction, window boundaries, the posture formula against hand-computed
goldens, baseline completion and stability, staff tagging, mock chart binding, reaping,
and leave-and-return.

Three properties hold for the whole file and are load-bearing:

  * No numpy, no cv2, no model, no network, no RNG, no clock. Every number below was
    worked out by hand and is written in the assertion with its arithmetic in a comment.
  * The public shapes are pinned mechanically (field names and order, method signatures),
    because the field names and signatures are a public contract other modules depend on.
  * `test_state_module_has_no_top_level_heavy_imports` pins the Linux-CI import rule the
    same way tests/test_mot.py does for the eval adapter.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import math
from collections import deque
from pathlib import Path

import pytest

from vividly import tracks as tracks_pkg
from vividly.tracks import (
    LEFT_HIP,
    LEFT_SHOULDER,
    NUM_KEYPOINTS,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    BaselineStats,
    TrackBook,
    TrackConfig,
    TrackObservation,
    TrackState,
    displacement_variance,
    posture_angle_deg,
)
from vividly.tracks import state as state_mod


# ------------------------------------------------------------------ helpers
def make_keypoints(
    shoulder_l: tuple[float, float] = (-10.0, 0.0),
    shoulder_r: tuple[float, float] = (10.0, 0.0),
    hip_l: tuple[float, float] = (-10.0, 40.0),
    hip_r: tuple[float, float] = (10.0, 40.0),
    conf: float = 0.9,
    torso_confs: tuple[float, float, float, float] | None = None,
) -> tuple[tuple[float, float, float], ...]:
    """COCO-17 keypoints with only the four torso points set to anything meaningful.

    Defaults describe an upright torso: shoulders midpoint (0, 0), hips midpoint (0, 40),
    so the axis is straight down the image y axis and the angle from vertical is 0.
    `torso_confs` overrides confidence per torso keypoint in (sl, sr, hl, hr) order.
    """
    kps = [(0.0, 0.0, conf) for _ in range(NUM_KEYPOINTS)]
    confs = (conf, conf, conf, conf) if torso_confs is None else torso_confs
    kps[LEFT_SHOULDER] = (shoulder_l[0], shoulder_l[1], confs[0])
    kps[RIGHT_SHOULDER] = (shoulder_r[0], shoulder_r[1], confs[1])
    kps[LEFT_HIP] = (hip_l[0], hip_l[1], confs[2])
    kps[RIGHT_HIP] = (hip_r[0], hip_r[1], confs[3])
    return tuple(kps)


VERTICAL = make_keypoints()
HORIZONTAL = make_keypoints(hip_l=(40.0, -10.0), hip_r=(40.0, 10.0))


def obs(
    track_id: int,
    t: float,
    cx: float = 0.0,
    cy: float = 0.0,
    keypoints: tuple[tuple[float, float, float], ...] | None = None,
) -> TrackObservation:
    return TrackObservation(
        track_id=track_id, t=t, cx=cx, cy=cy, w=40.0, h=80.0, keypoints=keypoints
    )


def small_cfg(**overrides: float | int) -> TrackConfig:
    """Test-sized config. Real defaults are minutes long and would need 4500 samples."""
    base: dict[str, float | int] = {
        "baseline_window_s": 4.0,
        "reap_after_s": 5.0,
        "kp_conf_min": 0.5,
        "ring_size": 100,
        "baseline_min_samples": 2,
    }
    base.update(overrides)
    return TrackConfig(**base)  # type: ignore[arg-type]


# ------------------------------------------------------- contract shape pins
def test_track_observation_fields_match_contract_exactly() -> None:
    assert [f.name for f in dataclasses.fields(TrackObservation)] == [
        "track_id",
        "t",
        "cx",
        "cy",
        "w",
        "h",
        "keypoints",
    ]


def test_baseline_stats_fields_match_contract_exactly() -> None:
    assert [f.name for f in dataclasses.fields(BaselineStats)] == [
        "movement_var",
        "posture_deg_mean",
        "posture_deg_std",
        "cy_mean",
        "n_samples",
        "complete",
    ]


def test_observation_and_baseline_are_frozen() -> None:
    o = obs(1, 0.0)
    b = BaselineStats(1.0, 2.0, 3.0, 4.0, 5, True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.cx = 5.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.complete = False  # type: ignore[misc]


def test_public_method_signatures_match_contract() -> None:
    """These names are a public contract other modules depend on; a rename here is a
    silent integration break, so the parameter names are asserted, not just the methods."""
    sigs = {
        TrackState.update: ["self", "obs"],
        TrackState.positions: ["self", "window_s"],
        TrackState.boxes: ["self", "window_s"],
        TrackState.posture_angles: ["self", "window_s"],
        TrackState.movement_variance: ["self", "window_s"],
        TrackState.baseline: ["self"],
        TrackBook.__init__: ["self", "cfg"],
        TrackBook.ingest: ["self", "obs_list"],
        TrackBook.tracks: ["self"],
        TrackBook.set_staff: ["self", "track_id", "staff"],
        TrackBook.bind_mrn: ["self", "track_id", "mrn"],
    }
    for fn, names in sigs.items():
        assert list(inspect.signature(fn).parameters) == names, fn


def test_track_state_exposes_contract_attributes() -> None:
    st = TrackState(7, small_cfg())
    assert st.track_id == 7
    assert math.isnan(st.first_seen_t)
    assert math.isnan(st.last_seen_t)
    assert st.is_staff is False
    assert st.mrn_binding is None


def test_state_module_has_no_top_level_heavy_imports() -> None:
    """numpy and cv2 are darwin-marked in the default `mac` group, so an environment that
    syncs only the default groups on Linux has neither, and it still collects this file. A
    module-level import of either (or of a model runtime) would break that collection. AST,
    not grep: the docstring mentions numpy on purpose."""
    banned = {"numpy", "cv2", "torch", "rtmlib", "trackers", "supervision", "onnxruntime"}
    for module in (state_mod, tracks_pkg):
        tree = ast.parse(Path(module.__file__).read_text())
        offenders: list[str] = []
        for node in tree.body:  # module level only; function bodies are not walked
            if isinstance(node, ast.Import):
                offenders += [a.name for a in node.names if a.name.split(".")[0] in banned]
            elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in banned:
                offenders.append(node.module or "")
        assert offenders == [], f"{module.__name__} imports {offenders} at module level"


# --------------------------------------------------------------- TrackConfig
def test_track_config_defaults_are_sane() -> None:
    cfg = TrackConfig()
    assert cfg.baseline_window_s == 300.0
    assert cfg.reap_after_s == 30.0
    assert cfg.kp_conf_min == 0.3
    # 54000 = 3600 s at 15 fps: generous headroom over the longest rule window
    # (collapse_search_s, 120 s), so a config that widens a window later still has evidence.
    assert cfg.ring_size == 54_000
    assert cfg.baseline_min_samples == 30


@pytest.mark.parametrize(
    "bad",
    [
        {"baseline_window_s": 0.0},
        {"baseline_window_s": -1.0},
        {"reap_after_s": 0.0},
        {"kp_conf_min": -0.01},
        {"kp_conf_min": 1.01},
        {"ring_size": 1},
        {"ring_size": 0},
        {"baseline_min_samples": 1},
    ],
)
def test_track_config_rejects_nonsense(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        TrackConfig(**bad)  # type: ignore[arg-type]


def test_track_config_from_mapping_empty_gives_defaults() -> None:
    assert TrackConfig.from_mapping({}) == TrackConfig()


def test_track_config_from_mapping_reads_every_field() -> None:
    cfg = TrackConfig.from_mapping(
        {
            "baseline_window_s": 120,
            "reap_after_s": 20,
            "kp_conf_min": 0.4,
            "ring_size": 900,
            "baseline_min_samples": 10,
        }
    )
    # yaml hands over ints for `120`; the config stores floats where the field is a float.
    assert isinstance(cfg.baseline_window_s, float)
    assert cfg == TrackConfig(120.0, 20.0, 0.4, 900, 10)


def test_track_config_from_mapping_derives_ring_size_from_fps() -> None:
    cfg = TrackConfig.from_mapping({"fps": 15, "ring_window_s": 60})
    assert cfg.ring_size == 900  # 15 * 60


def test_track_config_ring_size_derivation_rounds_up_and_floors_at_two() -> None:
    assert TrackConfig.from_mapping({"fps": 15, "ring_window_s": 0.5}).ring_size == 8  # 7.5
    assert TrackConfig.from_mapping({"fps": 1, "ring_window_s": 0.5}).ring_size == 2  # 0.5


def test_track_config_explicit_ring_size_wins_over_derivation() -> None:
    cfg = TrackConfig.from_mapping({"fps": 15, "ring_window_s": 60, "ring_size": 10})
    assert cfg.ring_size == 10


def test_track_config_from_mapping_needs_both_derivation_keys() -> None:
    with pytest.raises(ValueError, match="together"):
        TrackConfig.from_mapping({"fps": 15})
    with pytest.raises(ValueError, match="together"):
        TrackConfig.from_mapping({"ring_window_s": 60})


def test_track_config_from_mapping_rejects_nonpositive_derivation() -> None:
    with pytest.raises(ValueError):
        TrackConfig.from_mapping({"fps": 0, "ring_window_s": 60})


def test_track_config_from_mapping_rejects_unknown_keys() -> None:
    """A typo that silently kept a default is the failure mode this project cannot afford."""
    with pytest.raises(ValueError, match="unknown TrackConfig keys"):
        TrackConfig.from_mapping({"baseline_window_sec": 120})


# ------------------------------------------------------- posture angle golds
def test_posture_vertical_torso_is_zero_degrees() -> None:
    # shoulders midpoint (0, 0), hips midpoint (0, 40): straight down the y axis.
    assert posture_angle_deg(VERTICAL, 0.5) == 0.0


def test_posture_horizontal_torso_is_ninety_degrees() -> None:
    # shoulders midpoint (0, 0), hips midpoint (40, 0): straight along the x axis.
    assert posture_angle_deg(HORIZONTAL, 0.5) == pytest.approx(90.0)


def test_posture_forty_five_degrees() -> None:
    kps = make_keypoints(hip_l=(30.0, 20.0), hip_r=(30.0, 40.0))  # hips mid (30, 30)
    assert posture_angle_deg(kps, 0.5) == pytest.approx(45.0)  # atan2(30, 30)


def test_posture_thirty_degrees() -> None:
    y = 30.0 * math.sqrt(3.0)
    kps = make_keypoints(hip_l=(30.0, y), hip_r=(30.0, y))  # dx 30, dy 30*sqrt(3)
    assert posture_angle_deg(kps, 0.5) == pytest.approx(30.0)


def test_posture_uses_midpoints_not_single_keypoints() -> None:
    # shoulders (0,0) and (20,0) -> mid (10, 0); hips (10,10) and (30,10) -> mid (20, 10).
    kps = make_keypoints(
        shoulder_l=(0.0, 0.0), shoulder_r=(20.0, 0.0), hip_l=(10.0, 10.0), hip_r=(30.0, 10.0)
    )
    assert posture_angle_deg(kps, 0.5) == pytest.approx(45.0)  # dx 10, dy 10


def test_posture_is_a_tilt_magnitude_not_a_heading() -> None:
    """Leaning left, leaning right, and hanging upside down are the same tilt. The angle
    is measured from vertical with both components in absolute value, so it is always in
    [0, 90] and never encodes which way the person faces."""
    left = make_keypoints(hip_l=(-30.0, 30.0), hip_r=(-30.0, 30.0))
    right = make_keypoints(hip_l=(30.0, 30.0), hip_r=(30.0, 30.0))
    inverted = make_keypoints(hip_l=(-10.0, -40.0), hip_r=(10.0, -40.0))
    assert posture_angle_deg(left, 0.5) == pytest.approx(45.0)
    assert posture_angle_deg(right, 0.5) == pytest.approx(45.0)
    assert posture_angle_deg(inverted, 0.5) == 0.0


def test_posture_none_keypoints_gives_none() -> None:
    assert posture_angle_deg(None, 0.5) is None


def test_posture_low_confidence_on_any_torso_keypoint_gives_none() -> None:
    for i in range(4):
        confs = [0.9, 0.9, 0.9, 0.9]
        confs[i] = 0.49
        kps = make_keypoints(torso_confs=(confs[0], confs[1], confs[2], confs[3]))
        assert posture_angle_deg(kps, 0.5) is None, f"torso keypoint {i}"


def test_posture_confidence_gate_is_strictly_below() -> None:
    """Contract wording: None if conf < kp_conf_min. Exactly at the floor is usable."""
    kps = make_keypoints(torso_confs=(0.5, 0.5, 0.5, 0.5))
    assert posture_angle_deg(kps, 0.5) == 0.0


def test_posture_ignores_confidence_of_non_torso_keypoints() -> None:
    kps = list(make_keypoints())
    for i in range(NUM_KEYPOINTS):
        if i not in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP):
            kps[i] = (0.0, 0.0, 0.0)  # nose, eyes, ears, limbs all dropped out
    assert posture_angle_deg(tuple(kps), 0.5) == 0.0


def test_posture_zero_length_torso_gives_none() -> None:
    """Shoulders and hips on top of each other carry no direction; atan2(0, 0) would
    silently report a confident 0 degrees."""
    kps = make_keypoints(hip_l=(-10.0, 0.0), hip_r=(10.0, 0.0))
    assert posture_angle_deg(kps, 0.5) is None


def test_posture_rejects_non_coco17_keypoints() -> None:
    with pytest.raises(ValueError, match="COCO-17"):
        posture_angle_deg(tuple((0.0, 0.0, 1.0) for _ in range(16)), 0.5)


# ------------------------------------------------------- observation guards
def test_observation_accepts_none_keypoints() -> None:
    assert obs(1, 0.0).keypoints is None


def test_observation_rejects_wrong_keypoint_count() -> None:
    with pytest.raises(ValueError, match="COCO-17"):
        obs(1, 0.0, keypoints=tuple((0.0, 0.0, 1.0) for _ in range(12)))


def test_observation_rejects_malformed_keypoint_triples() -> None:
    kps = list(make_keypoints())
    kps[3] = (1.0, 2.0)  # type: ignore[assignment]
    with pytest.raises(ValueError, match="keypoint 3"):
        obs(1, 0.0, keypoints=tuple(kps))


# ------------------------------------------------------ displacement variance
def test_displacement_variance_golden() -> None:
    # steps 1, 2, 3 -> mean 2 -> ((1-2)^2 + 0 + (3-2)^2) / 3 = 2/3
    points = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 3.0, 0.0), (3.0, 6.0, 0.0)]
    assert displacement_variance(points) == pytest.approx(2.0 / 3.0)


def test_displacement_variance_is_population_not_sample() -> None:
    # steps 1 and 3: population variance = ((1-2)^2 + (3-2)^2)/2 = 1.0, sample would be 2.0
    points = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 4.0, 0.0)]
    assert displacement_variance(points) == pytest.approx(1.0)


def test_displacement_variance_uses_both_axes() -> None:
    # a 3-4-5 step then a 3-4-5 step: identical magnitudes, so variance is 0
    points = [(0.0, 0.0, 0.0), (1.0, 3.0, 4.0), (2.0, 6.0, 8.0)]
    assert displacement_variance(points) == pytest.approx(0.0)


def test_displacement_variance_needs_two_samples() -> None:
    assert displacement_variance([]) is None
    assert displacement_variance([(0.0, 0.0, 0.0)]) is None
    assert displacement_variance([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]) == 0.0


def test_displacement_variance_accepts_a_deque() -> None:
    """positions() hands out a list, but the rings are deques and a deque cannot be
    sliced; the helper must not depend on slicing."""
    ring = deque([(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 3.0, 0.0)])
    assert displacement_variance(ring) == pytest.approx(0.25)  # steps 1, 2


# ------------------------------------------------------------- TrackState IO
def test_update_sets_first_and_last_seen() -> None:
    st = TrackState(1, small_cfg())
    st.update(obs(1, 10.0))
    st.update(obs(1, 12.0))
    assert st.first_seen_t == 10.0
    assert st.last_seen_t == 12.0
    assert st.n_observations == 2


def test_update_rejects_a_foreign_observation() -> None:
    st = TrackState(1, small_cfg())
    with pytest.raises(ValueError, match="track 2 handed to TrackState 1"):
        st.update(obs(2, 0.0))


def test_track_state_defaults_to_default_config() -> None:
    st = TrackState(1)
    assert st.cfg == TrackConfig()


def test_ring_evicts_oldest_at_maxlen() -> None:
    st = TrackState(1, small_cfg(ring_size=3))
    for i in range(5):
        st.update(obs(1, float(i), cx=float(i), keypoints=VERTICAL))
    kept = st.positions(1000.0)
    assert [p[0] for p in kept] == [2.0, 3.0, 4.0]
    assert [p[0] for p in st.posture_angles(1000.0)] == [2.0, 3.0, 4.0]
    assert st.n_observations == 5  # the counter survives eviction


def test_rings_are_deques_sized_from_config() -> None:
    st = TrackState(1, small_cfg(ring_size=7))
    assert isinstance(st._positions, deque)
    assert st._positions.maxlen == 7
    assert st._boxes.maxlen == 7
    assert st._postures.maxlen == 7


def test_posture_ring_skips_frames_without_a_usable_torso() -> None:
    st = TrackState(1, small_cfg())
    st.update(obs(1, 0.0, keypoints=VERTICAL))
    st.update(obs(1, 1.0, keypoints=None))
    st.update(obs(1, 2.0, keypoints=make_keypoints(torso_confs=(0.1, 0.9, 0.9, 0.9))))
    st.update(obs(1, 3.0, keypoints=HORIZONTAL))
    assert len(st.positions(1000.0)) == 4
    assert st.posture_angles(1000.0) == [(0.0, 0.0), (3.0, pytest.approx(90.0))]


def test_boxes_ring_is_index_parallel_with_positions() -> None:
    """One unconditional append per observation, keypoints or not: the event-verifier
    backend zips boxes() with positions() row for row, so the pairing is a contract."""
    st = TrackState(1, small_cfg(ring_size=3))
    st.update(obs(1, 0.0, keypoints=VERTICAL))
    st.update(obs(1, 1.0, keypoints=None))  # no keypoints still appends a box
    for i in range(2, 5):
        st.update(obs(1, float(i), keypoints=VERTICAL))
    kept = st.boxes(1000.0)
    assert [b[0] for b in kept] == [p[0] for p in st.positions(1000.0)] == [2.0, 3.0, 4.0]
    assert [(b[1], b[2]) for b in kept] == [(40.0, 80.0)] * 3  # w, h from the obs helper


def test_boxes_carry_mean_keypoint_confidence_or_nan() -> None:
    st = TrackState(1, small_cfg())
    kps = make_keypoints(torso_confs=(0.5, 0.7, 0.9, 0.9))
    st.update(obs(1, 0.0, keypoints=kps))
    st.update(obs(1, 1.0, keypoints=None))
    boxes = st.boxes(1000.0)
    # mean over all 17 confidences, exactly what the training extraction records
    assert boxes[0][3] == pytest.approx(sum(kp[2] for kp in kps) / len(kps))
    assert math.isnan(boxes[1][3])


def test_boxes_window_matches_positions_window() -> None:
    st = TrackState(1, small_cfg())
    for i in range(5):
        st.update(obs(1, float(i)))
    assert [b[0] for b in st.boxes(2.0)] == [2.0, 3.0, 4.0]


def test_positions_window_is_anchored_at_last_seen_and_inclusive() -> None:
    st = TrackState(1, small_cfg())
    for i in range(5):
        st.update(obs(1, float(i), cx=float(i)))
    # last_seen_t 4, window 2 -> cutoff exactly 2.0, and t == cutoff is kept
    assert [p[0] for p in st.positions(2.0)] == [2.0, 3.0, 4.0]
    assert [p[0] for p in st.positions(0.5)] == [4.0]


def test_posture_angles_window_matches_positions_window() -> None:
    st = TrackState(1, small_cfg())
    for i in range(5):
        st.update(obs(1, float(i), keypoints=VERTICAL))
    assert [a[0] for a in st.posture_angles(2.0)] == [2.0, 3.0, 4.0]


def test_windows_reject_nonpositive_widths() -> None:
    st = TrackState(1, small_cfg())
    st.update(obs(1, 0.0))
    for width in (0.0, -1.0):
        with pytest.raises(ValueError, match="window_s must be > 0"):
            st.positions(width)
        with pytest.raises(ValueError, match="window_s must be > 0"):
            st.boxes(width)
        with pytest.raises(ValueError, match="window_s must be > 0"):
            st.posture_angles(width)
        with pytest.raises(ValueError, match="window_s must be > 0"):
            st.movement_variance(width)


def test_empty_state_reads_are_empty_not_crashes() -> None:
    st = TrackState(1, small_cfg())
    assert st.positions(10.0) == []
    assert st.boxes(10.0) == []
    assert st.posture_angles(10.0) == []
    assert st.movement_variance(10.0) is None
    assert st.baseline() is None


def test_movement_variance_golden_over_full_history() -> None:
    st = TrackState(1, small_cfg())
    for t, cx in ((0.0, 0.0), (1.0, 1.0), (2.0, 3.0), (3.0, 6.0)):
        st.update(obs(1, t, cx=cx))
    assert st.movement_variance(100.0) == pytest.approx(2.0 / 3.0)  # steps 1,2,3


def test_movement_variance_window_does_not_reach_across_its_edge() -> None:
    st = TrackState(1, small_cfg())
    for t, cx in ((0.0, 0.0), (1.0, 1.0), (2.0, 3.0), (3.0, 6.0)):
        st.update(obs(1, t, cx=cx))
    # window 2 -> samples t=1,2,3 -> steps 2 and 3 -> mean 2.5 -> var 0.25
    assert st.movement_variance(2.0) == pytest.approx(0.25)


def test_movement_variance_none_when_window_holds_one_sample() -> None:
    st = TrackState(1, small_cfg())
    st.update(obs(1, 0.0))
    st.update(obs(1, 10.0))
    assert st.movement_variance(1.0) is None


def test_perfectly_still_track_has_zero_movement_variance() -> None:
    st = TrackState(1, small_cfg())
    for i in range(10):
        st.update(obs(1, float(i), cx=100.0, cy=200.0))
    assert st.movement_variance(100.0) == 0.0


def test_out_of_order_sample_keeps_windows_correct() -> None:
    """A late arrival must not silently truncate a window read. The fast path scans the
    ring backwards and stops early, which is only valid while samples are ordered, so the
    state falls back to a full scan the moment one is not."""
    st = TrackState(1, small_cfg())
    for t in (0.0, 1.0, 2.0):
        st.update(obs(1, t, cx=t))
    st.update(obs(1, 0.5, cx=0.5))  # late
    assert st.last_seen_t == 2.0  # never walks backwards
    assert [p[0] for p in st.positions(10.0)] == [0.0, 1.0, 2.0, 0.5]
    assert sorted(p[0] for p in st.positions(1.0)) == [1.0, 2.0]  # cutoff 1.0


# ----------------------------------------------------------------- baseline
def baseline_track(cfg: TrackConfig | None = None) -> TrackState:
    """Five samples at 1 Hz over a 4 s baseline window, all hand-computable:

    cx 0, 1, 3, 6, 10 with cy fixed at 30 -> steps 1, 2, 3, 4 -> mean 2.5 ->
    variance ((1.5^2 + 0.5^2 + 0.5^2 + 1.5^2) / 4) = 1.25
    angles 0, 0, 90, 0, 0 -> mean 18 -> variance (4*18^2 + 72^2)/5 = 1296 -> std 36
    """
    st = TrackState(1, cfg if cfg is not None else small_cfg())
    for t, cx in ((0.0, 0.0), (1.0, 1.0), (2.0, 3.0), (3.0, 6.0), (4.0, 10.0)):
        kps = HORIZONTAL if t == 2.0 else VERTICAL
        st.update(obs(1, t, cx=cx, cy=30.0, keypoints=kps))
    return st


def test_baseline_is_none_before_the_window_elapses() -> None:
    st = TrackState(1, small_cfg())  # baseline_window_s 4
    for t in (0.0, 1.0, 2.0, 3.0):
        st.update(obs(1, t, keypoints=VERTICAL))
    assert st.last_seen_t - st.first_seen_t == 3.0
    assert st.baseline() is None


def test_baseline_completes_exactly_when_the_window_has_elapsed() -> None:
    st = TrackState(1, small_cfg())
    for t in (0.0, 1.0, 2.0, 3.0):
        st.update(obs(1, t, keypoints=VERTICAL))
    assert st.baseline() is None
    st.update(obs(1, 4.0, keypoints=VERTICAL))  # elapsed == window, which counts
    stats = st.baseline()
    assert stats is not None
    assert stats.complete is True


def test_baseline_golden_values() -> None:
    stats = baseline_track().baseline()
    assert stats is not None
    assert stats.movement_var == pytest.approx(1.25)
    assert stats.posture_deg_mean == pytest.approx(18.0)
    assert stats.posture_deg_std == pytest.approx(36.0)
    assert stats.cy_mean == pytest.approx(30.0)
    assert stats.n_samples == 5
    assert stats.complete is True


def test_baseline_is_stable_once_complete() -> None:
    """The whole point of a per-person baseline is that it stops moving. Later behaviour,
    however wild, must not change what normal looked like."""
    st = baseline_track()
    first = st.baseline()
    for i in range(20):
        st.update(obs(1, 5.0 + i, cx=1000.0 * i, cy=900.0, keypoints=HORIZONTAL))
    second = st.baseline()
    assert second == first
    assert second is first  # cached, not recomputed


def test_baseline_ignores_samples_after_its_own_window() -> None:
    st = baseline_track()
    st.update(obs(1, 4.001, cx=10_000.0, cy=10_000.0, keypoints=HORIZONTAL))
    stats = st.baseline()
    assert stats is not None
    assert stats.n_samples == 5
    assert stats.cy_mean == pytest.approx(30.0)


def test_baseline_is_identical_whether_read_early_or_late() -> None:
    early = baseline_track()
    early_stats = early.baseline()
    late = baseline_track()
    for i in range(20):
        late.update(obs(1, 5.0 + i, cx=1000.0 * i, cy=900.0, keypoints=HORIZONTAL))
    assert late.baseline() == early_stats


def test_baseline_survives_ring_eviction() -> None:
    """The baseline accumulators are deliberately not rings. With a ring of 2 the position
    history is long gone, and the baseline must be exactly the same numbers anyway."""
    small_ring = baseline_track(small_cfg(ring_size=2))
    assert len(small_ring.positions(1000.0)) == 2
    assert small_ring.baseline() == baseline_track().baseline()


def test_baseline_incomplete_when_the_track_was_barely_present() -> None:
    cfg = small_cfg(baseline_min_samples=4)
    st = TrackState(1, cfg)
    st.update(obs(1, 0.0, keypoints=VERTICAL))
    st.update(obs(1, 1.0, keypoints=VERTICAL))
    st.update(obs(1, 9.0, keypoints=VERTICAL))  # outside the 4 s baseline window
    stats = st.baseline()
    assert stats is not None
    assert stats.n_samples == 2
    assert stats.complete is False


def test_baseline_incomplete_when_pose_never_resolved() -> None:
    """Positions alone are not a baseline: the fall rule reasons about posture."""
    st = TrackState(1, small_cfg())
    for t in (0.0, 1.0, 2.0, 3.0, 4.0):
        st.update(obs(1, t, cx=t, cy=30.0, keypoints=None))
    stats = st.baseline()
    assert stats is not None
    assert stats.complete is False
    assert stats.posture_deg_mean == 0.0
    assert stats.posture_deg_std == 0.0
    assert stats.n_samples == 5


def test_baseline_single_sample_window_reports_zero_variance_and_incomplete() -> None:
    st = TrackState(1, small_cfg())
    st.update(obs(1, 0.0, cx=5.0, cy=7.0, keypoints=VERTICAL))
    st.update(obs(1, 100.0, cx=5.0, cy=7.0, keypoints=VERTICAL))
    stats = st.baseline()
    assert stats is not None
    assert stats.n_samples == 1
    assert stats.movement_var == 0.0  # no step exists; the field cannot be None
    assert stats.cy_mean == pytest.approx(7.0)
    assert stats.complete is False


# ------------------------------------------------------------------ TrackBook
def test_book_creates_one_state_per_track_id() -> None:
    book = TrackBook(small_cfg())
    book.ingest([obs(1, 0.0, cx=10.0), obs(2, 0.0, cx=20.0)])
    book.ingest([obs(1, 1.0, cx=11.0), obs(2, 1.0, cx=21.0)])
    tracks = book.tracks()
    assert sorted(tracks) == [1, 2]
    assert tracks[1].positions(10.0)[-1][1] == 11.0
    assert tracks[2].positions(10.0)[-1][1] == 21.0
    assert tracks[1].first_seen_t == 0.0


def test_book_tracks_returns_a_copy_of_the_mapping() -> None:
    book = TrackBook(small_cfg())
    book.ingest([obs(1, 0.0)])
    handed_out = book.tracks()
    handed_out.clear()
    assert list(book.tracks()) == [1]


def test_book_tracks_hands_out_live_states() -> None:
    """Rules read through this mapping every tick; copying the states would freeze them."""
    book = TrackBook(small_cfg())
    book.ingest([obs(1, 0.0)])
    before = book.tracks()[1]
    book.ingest([obs(1, 1.0)])
    assert book.tracks()[1] is before
    assert before.last_seen_t == 1.0


def test_book_empty_ingest_is_a_no_op() -> None:
    book = TrackBook(small_cfg())
    book.ingest([])
    assert book.tracks() == {}
    assert math.isnan(book.now)
    book.ingest([obs(1, 3.0)])
    book.ingest([])
    assert book.now == 3.0
    assert list(book.tracks()) == [1]


def test_book_clock_follows_the_newest_observation() -> None:
    book = TrackBook(small_cfg())
    book.ingest([obs(1, 2.0), obs(2, 1.0)])
    assert book.now == 2.0


def test_set_staff_tags_and_untags() -> None:
    book = TrackBook(small_cfg())
    book.ingest([obs(1, 0.0)])
    book.set_staff(1, True)
    assert book.tracks()[1].is_staff is True
    book.ingest([obs(1, 1.0)])
    assert book.tracks()[1].is_staff is True  # survives further observations
    book.set_staff(1, False)
    assert book.tracks()[1].is_staff is False


def test_bind_mrn_stores_the_mock_binding() -> None:
    book = TrackBook(small_cfg())
    book.ingest([obs(1, 0.0)])
    book.bind_mrn(1, "MRN-0001")
    assert book.tracks()[1].mrn_binding == "MRN-0001"


def test_bind_mrn_rejects_an_empty_binding() -> None:
    book = TrackBook(small_cfg())
    book.ingest([obs(1, 0.0)])
    for bad in ("", "   "):
        with pytest.raises(ValueError, match="non-empty string"):
            book.bind_mrn(1, bad)


def test_staff_and_mrn_on_an_unknown_track_raise_key_error() -> None:
    """The dashboard turns this into a 404 rather than inventing a track."""
    book = TrackBook(small_cfg())
    with pytest.raises(KeyError):
        book.set_staff(99, True)
    with pytest.raises(KeyError):
        book.bind_mrn(99, "MRN-0099")


# --------------------------------------------------------------------- reaping
def test_track_is_reaped_after_the_gap_exceeds_reap_after_s() -> None:
    book = TrackBook(small_cfg(reap_after_s=5.0))
    book.ingest([obs(1, 0.0), obs(2, 0.0)])
    book.ingest([obs(2, 5.001)])
    assert list(book.tracks()) == [2]


def test_track_survives_a_gap_exactly_equal_to_reap_after_s() -> None:
    book = TrackBook(small_cfg(reap_after_s=5.0))
    book.ingest([obs(1, 0.0), obs(2, 0.0)])
    book.ingest([obs(2, 5.0)])  # dropped only when strictly greater
    assert sorted(book.tracks()) == [1, 2]


def test_explicit_reap_advances_the_clock_across_empty_frames() -> None:
    """ingest cannot reap on a frame with no detections, because an empty batch carries
    no time. The pipeline calls reap directly for that."""
    book = TrackBook(small_cfg(reap_after_s=5.0))
    book.ingest([obs(1, 0.0)])
    assert book.reap(3.0) == []
    assert book.now == 3.0
    assert book.reap(9.0) == [1]
    assert book.tracks() == {}


def test_reap_before_any_observation_is_harmless() -> None:
    book = TrackBook(small_cfg())
    assert book.reap(math.nan) == []  # nan carries no time and must not set the clock
    assert math.isnan(book.now)
    assert book.reap(10.0) == []
    assert book.now == 10.0


def test_leave_and_return_inside_the_reap_window_keeps_everything() -> None:
    """Leave and return inside the reap window: occlusion must not cost a person their
    identity, their baseline, their staff tag or their mock chart binding."""
    cfg = small_cfg(baseline_window_s=4.0, reap_after_s=5.0)
    book = TrackBook(cfg)
    for t, cx in ((0.0, 0.0), (1.0, 1.0), (2.0, 3.0), (3.0, 6.0), (4.0, 10.0)):
        kps = HORIZONTAL if t == 2.0 else VERTICAL
        book.ingest([obs(1, t, cx=cx, cy=30.0, keypoints=kps), obs(2, t, cx=500.0)])
    book.set_staff(1, True)
    book.bind_mrn(1, "MRN-0001")
    original = book.tracks()[1]
    baseline_before = original.baseline()
    assert baseline_before is not None

    for t in (5.0, 6.0, 7.0, 8.0, 9.0):  # track 1 absent, gap reaches exactly 5 s
        book.ingest([obs(2, t, cx=500.0)])
    assert 1 in book.tracks()

    book.ingest([obs(1, 9.0, cx=11.0, cy=30.0, keypoints=VERTICAL), obs(2, 9.0, cx=500.0)])
    returned = book.tracks()[1]
    assert returned is original
    assert returned.first_seen_t == 0.0
    assert returned.baseline() == baseline_before
    assert returned.is_staff is True
    assert returned.mrn_binding == "MRN-0001"


def test_return_after_the_reap_window_is_a_new_track() -> None:
    """No re-identification anywhere in this layer, and pretending otherwise would be the
    dishonest option. A returning id starts clean."""
    cfg = small_cfg(baseline_window_s=4.0, reap_after_s=5.0)
    book = TrackBook(cfg)
    for t in (0.0, 1.0, 2.0, 3.0, 4.0):
        book.ingest([obs(1, t, cx=t, cy=30.0, keypoints=VERTICAL)])
    book.set_staff(1, True)
    book.bind_mrn(1, "MRN-0001")
    original = book.tracks()[1]
    assert original.baseline() is not None

    book.ingest([obs(2, 20.0)])  # gap of 16 s reaps track 1
    assert 1 not in book.tracks()

    book.ingest([obs(1, 21.0, cx=1.0)])
    fresh = book.tracks()[1]
    assert fresh is not original
    assert fresh.first_seen_t == 21.0
    assert fresh.baseline() is None
    assert fresh.is_staff is False
    assert fresh.mrn_binding is None


def test_reap_returns_every_dropped_id() -> None:
    book = TrackBook(small_cfg(reap_after_s=5.0))
    book.ingest([obs(1, 0.0), obs(2, 0.0), obs(3, 0.0)])
    assert book.reap(10.0) == [1, 2, 3]
    assert book.tracks() == {}


def test_ingest_reaps_so_a_caller_never_sees_a_stale_track() -> None:
    book = TrackBook(small_cfg(reap_after_s=5.0))
    book.ingest([obs(1, 0.0), obs(2, 0.0), obs(3, 0.0)])
    book.ingest([obs(3, 6.0)])
    assert list(book.tracks()) == [3]
    assert book.reap(6.0) == []  # reaping is idempotent


def test_book_states_inherit_the_book_config() -> None:
    cfg = small_cfg(ring_size=9)
    book = TrackBook(cfg)
    book.ingest([obs(1, 0.0)])
    st = book.tracks()[1]
    assert st.cfg is cfg
    assert st._positions.maxlen == 9


def test_a_long_still_run_stays_bounded_in_memory() -> None:
    """Sanity check on the whole point of the rings: a person who waits for an hour must
    not cost an hour of samples per track."""
    book = TrackBook(small_cfg(ring_size=10, baseline_window_s=1.0, reap_after_s=1000.0))
    for i in range(500):
        book.ingest([obs(1, float(i), cx=100.0, cy=100.0, keypoints=VERTICAL)])
    st = book.tracks()[1]
    assert len(st.positions(10_000.0)) == 10
    assert len(st.posture_angles(10_000.0)) == 10
    assert st.n_observations == 500
    assert st.baseline() is not None
