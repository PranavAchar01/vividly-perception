"""BoT-SORT wrapper behavior pins (2026-08-31 tracker swap,
research/v2-training-stack.md section 7a).

Each test pins one of the adoption's load-bearing behaviors against the real
trackers 2.6.0 library, the same way test_env_smoke pins the ByteTrack wrapper:
the swap was motivated by measured failure modes, so the config knobs that fix
them must be provably wired, not just present in a docstring.
"""

import numpy as np
import pytest

pytest.importorskip("trackers")
pytest.importorskip("supervision")

from vividly.models.base import Detections
from vividly.models.factory import build_tracker
from vividly.models.tracker import BoTSORT


def _one_box(score: float = 0.9) -> Detections:
    return Detections(
        xyxy=np.array([[10.0, 10.0, 50.0, 80.0]], np.float32),
        scores=np.array([score], np.float32),
        class_ids=np.array([0], np.int32),
    )


def test_wrapper_matches_trackers_260_botsort_api() -> None:
    # Constructing and stepping against the real library is the assertion (the
    # ByteTrack wrapper once shipped a parameter the library never had, and nothing
    # noticed until a live run). A steady high-confidence box must hold one
    # confirmed non-sentinel track id within three frames.
    tracker = BoTSORT(frame_rate=15.0)
    out = tracker.update(_one_box(), 0)
    for frame_idx in range(1, 3):
        out = tracker.update(_one_box(), frame_idx)
    assert out.track_ids is not None and len(out) == 1
    assert int(out.track_ids[0]) >= 0
    tracker.reset()
    assert len(tracker.update(Detections.empty(), 0)) == 0


def test_cmc_is_refused_not_defaulted() -> None:
    # CMC needs the raw frame; this protocol is detection-only and the cameras are
    # static. A silent enable would no-op or crash mid-run, so refuse at construction.
    with pytest.raises(ValueError, match="enable_cmc"):
        BoTSORT(enable_cmc=True)


def test_score_floor_cuts_junk_before_the_tracker() -> None:
    # The whole point of the swap: a sub-floor detection must never reach the
    # association passes at all. Feed only junk below the floor; the tracker must
    # see empty frames (no tracks internally), not merely emit nothing.
    tracker = BoTSORT(frame_rate=15.0, score_floor=0.1)
    for frame_idx in range(5):
        out = tracker.update(_one_box(score=0.05), frame_idx)
        assert len(out) == 0
    assert tracker._t.tracks == [], "a below-floor detection leaked into the tracker"


def test_velocity_damping_is_stamped_on_live_tracklets() -> None:
    # Q is rebuilt from _SIGMA_V every tracklet step, so the OC-SORT damping only
    # holds if every live tracklet carries the damped sigma as an instance
    # attribute. variance x0.01 == sigma x0.1.
    from trackers.core.botsort.tracklet import BoTSORTTracklet

    tracker = BoTSORT(frame_rate=15.0, velocity_damping=0.01)
    for frame_idx in range(3):
        tracker.update(_one_box(), frame_idx)
    assert tracker._t.tracks, "steady box produced no tracklet"
    expected = BoTSORTTracklet._SIGMA_V * 0.1
    for tracklet in tracker._t.tracks:
        assert pytest.approx(expected) == tracklet._SIGMA_V
    # Damping 1.0 means off: tracklets keep the library's class constant.
    undamped = BoTSORT(frame_rate=15.0, velocity_damping=1.0)
    for frame_idx in range(3):
        undamped.update(_one_box(), frame_idx)
    for tracklet in undamped._t.tracks:
        assert pytest.approx(BoTSORTTracklet._SIGMA_V) == tracklet._SIGMA_V


def test_buffer_default_is_a_waiting_room_not_a_benchmark() -> None:
    # The library default (30 = 1 s of identity memory) loses every still patient;
    # the wrapper default must be 1800 (60 s) and pass through untranslated.
    tracker = BoTSORT(frame_rate=15.0)
    assert tracker._kwargs["lost_track_buffer"] == 1800
    # The library normalizes to seconds itself, so pin the semantics, not the raw setting:
    # 1800 "30-fps frames" must mean 60 s of identity memory at the 15 fps ingest.
    assert tracker._t.maximum_time_without_update == pytest.approx(60.0)


def test_factory_builds_botsort_from_a_config_spec() -> None:
    # The replay_demo.yaml tracker block, minus comments; the factory must accept
    # exactly these keys (a stale key here means the config and wrapper drifted).
    spec = {
        "backend": "botsort",
        "track_activation_threshold": 0.25,
        "lost_track_buffer": 1800,
        "score_floor": 0.1,
        "velocity_damping": 0.01,
        "frame_rate": 15,
    }
    tracker = build_tracker(spec)
    assert isinstance(tracker, BoTSORT)
