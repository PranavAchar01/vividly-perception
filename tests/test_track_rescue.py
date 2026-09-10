"""Keeping hold of a person the detector stops being sure about (live_stream.py).

Measured on takes/20260909-203345 at 95.3-97.1 s: a person going down is still found on
every frame at 0.40-0.47, below the appliance's 0.5 gate, and on eleven frames nothing is
found near them at all. rescue_tracks continues a known track on the low-score box;
add_followers carries it on its last pose-derived box through the frames with nothing.
Neither may ever start a track.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vividly.models.base import Detections  # noqa: E402


def _appliance():
    spec = importlib.util.spec_from_file_location(
        "live_stream_rescue_under_test", REPO / "scripts" / "live_stream.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ls = _appliance()

STANDING = [425, 453, 533, 707]  # the take's person, seated-to-standing, 94.6 s
PRONE = [314, 528, 521, 678]  # the same person going down, 95.4 s: IoU with STANDING ~0.33
CLUTTER = [900, 100, 960, 200]


def dets(boxes, scores, ids=None) -> Detections:
    return Detections(
        xyxy=np.array(boxes, np.float32).reshape(-1, 4),
        scores=np.array(scores, np.float32),
        class_ids=np.zeros(len(scores), np.int32),
        track_ids=None if ids is None else np.array(ids, np.int32),
    )


def test_a_lost_track_continues_on_an_overlapping_low_score_box() -> None:
    recent: dict = {}
    ls.rescue_tracks(dets([STANDING], [0.9], [7]), dets([STANDING], [0.9]), recent, 0.0, 0.5)
    out = ls.rescue_tracks(dets([], [], []), dets([PRONE, CLUTTER], [0.45, 0.3]), recent, 0.05, 0.5)
    assert list(out.track_ids) == [7]
    assert np.allclose(out.xyxy[0], PRONE)


def test_rescue_never_starts_a_track() -> None:
    out = ls.rescue_tracks(dets([], [], []), dets([CLUTTER], [0.45]), {}, 0.0, 0.5)
    assert len(out) == 0


def test_rescue_does_not_take_a_box_the_tracker_already_used() -> None:
    recent: dict = {}
    ls.rescue_tracks(dets([STANDING], [0.9], [7]), dets([STANDING], [0.9]), recent, 0.0, 0.5)
    # Track 9 now owns that box; track 7 is missing and must not be glued onto it.
    out = ls.rescue_tracks(dets([STANDING], [0.9], [9]), dets([STANDING], [0.9]), recent, 0.05, 0.5)
    assert list(out.track_ids) == [9]


def test_rescue_gives_up_on_a_track_not_seen_recently() -> None:
    recent: dict = {}
    ls.rescue_tracks(dets([STANDING], [0.9], [7]), dets([STANDING], [0.9]), recent, 0.0, 0.5)
    late = ls._RESCUE_RECENT_S + 0.2
    out = ls.rescue_tracks(dets([], [], []), dets([PRONE], [0.45]), recent, late, 0.5)
    assert len(out) == 0


def test_followers_carry_a_lost_track_and_then_let_it_go() -> None:
    follow = {3: np.array(PRONE, np.float32)}
    seen = {3: 0.0}
    out = ls.add_followers(dets([], [], []), follow, seen, 1.0)
    assert list(out.track_ids) == [3] and float(out.scores[0]) == 0.0
    out = ls.add_followers(dets([], [], []), follow, seen, ls._FOLLOW_S + 1.0)
    assert len(out) == 0 and 3 not in follow


def test_followers_do_not_duplicate_a_track_the_detector_has() -> None:
    follow = {3: np.array(PRONE, np.float32)}
    out = ls.add_followers(dets([STANDING], [0.9], [3]), follow, {}, 0.0)
    assert list(out.track_ids) == [3]
    assert np.allclose(out.xyxy[0], STANDING)


def test_skeleton_has_no_face_edges() -> None:
    # Face edges drew a tangle over exactly where a head wound is; the neck stub marks the head.
    assert all(a >= 5 and b >= 5 for a, b in ls.SKELETON)


def test_a_track_refound_under_a_new_id_is_not_kept_alive() -> None:
    # The bleed take: one person lying across two chairs, re-found under a new id. Keeping
    # the old id too drew the person twice and split the red-on-person clock three ways.
    recent: dict = {}
    ls.rescue_tracks(dets([PRONE], [0.9], [1]), dets([PRONE], [0.9]), recent, 0.0, 0.5)
    out = ls.rescue_tracks(
        dets([PRONE], [0.8], [2]), dets([PRONE, PRONE], [0.8, 0.4]), recent, 0.05, 0.5
    )
    assert list(out.track_ids) == [2]
    follow = {1: np.array(PRONE, np.float32)}
    out = ls.add_followers(dets([PRONE], [0.8], [2]), follow, {1: 0.0}, 0.1)
    assert list(out.track_ids) == [2] and 1 not in follow


def test_a_collapse_restarts_the_seizure_clock_unless_a_seizure_is_confirmed() -> None:
    # take1: a candidate from seated shifting at 90.4 s was carried over the 5 s hold by the
    # collapse at 95.5 s and confirmed SEIZURE on a person who fell and never convulsed.
    from vividly.analysis.fall import FallState
    from vividly.analysis.seizure import SeizureState

    idle, falling = FallState(), FallState(state="falling")
    candidate, seizure = SeizureState(state="candidate"), SeizureState(state="seizure")
    assert ls.collapse_began(idle, falling, candidate)  # the frame the collapse latches
    assert ls.collapse_began(None, falling, None)
    assert not ls.collapse_began(falling, falling, candidate)  # only on the transition
    assert not ls.collapse_began(idle, idle, candidate)
    assert not ls.collapse_began(idle, falling, seizure)  # a seizure that causes a fall stays
