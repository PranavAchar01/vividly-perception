"""A CONFIRMED alarm must survive a frame on which the detector found nobody.

The failure this pins down was found in real footage (takes/20260909-203345 at 95.5 s): a
person collapses, the detector loses them for 1.65 s because they are prone and still, and
the fall alert either never reaches the screen or blinks on and off with the detector. An
alarm that flickers while a patient is on the floor is the worst possible time for it to
flicker, so the compositor now keeps drawing the banner from a latched state even when
there are no fresh keypoints behind it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vividly.analysis.fall import FallState  # noqa: E402
from vividly.analysis.seizure import SeizureState  # noqa: E402


def _appliance():
    spec = importlib.util.spec_from_file_location(
        "live_stream_under_test", REPO / "scripts" / "live_stream.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ls = _appliance()

_IDLE_SZ = SeizureState()
_IDLE_FL = FallState()
_FALL = FallState(state="fall", held_s=6.1)
_SEIZ = SeizureState(state="seizure", held_s=5.0)


@pytest.mark.parametrize(
    ("sz", "fl", "want"),
    [
        (_IDLE_SZ, _IDLE_FL, False),
        (SeizureState(state="candidate", held_s=4.9), _IDLE_FL, False),
        (_IDLE_SZ, FallState(state="falling", down_s=3.9), False),
        (_IDLE_SZ, _FALL, True),
        (_SEIZ, _IDLE_FL, True),
        (None, _FALL, True),
        (_SEIZ, None, True),
        (None, None, False),
    ],
)
def test_only_confirmed_states_hold_the_screen(sz, fl, want):
    """A building event is not an alarm. The screen stays quiet until something is certain."""
    assert ls._is_confirmed(sz, fl) is want


def _view(sz=_IDLE_SZ, fl=_IDLE_FL, tid=0):
    box = np.array([100.0, 300.0, 300.0, 430.0], dtype=np.float32)
    return ls.TrackView(tid, box, None, None, sz, fl)


def _banner_pixels(tracks) -> int:
    """How many pixels of the appliance's alert red the banner painted."""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    ls.draw_banner(img, tracks)
    return int(np.count_nonzero(np.all(img == np.array(ls._RED, dtype=np.uint8), axis=2)))


def test_latched_fall_with_no_keypoints_still_paints_the_alarm():
    """The alerts list carries no keypoints at all. The banner must not need them."""
    assert _banner_pixels([_view(fl=_FALL)]) > 0


def test_quiet_scene_paints_nothing():
    assert _banner_pixels([_view()]) == 0
    assert _banner_pixels([]) == 0


def test_a_disabled_analyzer_does_not_crash_the_banner():
    """--no-seizure / --no-fall make an analyzer return None. draw_banner reads .state on
    both slots, so the alert path must substitute a real idle state, never None."""
    for sz, fl in ((None, _FALL), (_SEIZ, None)):
        v = ls.TrackView(0, np.zeros(4, np.float32), None, None, sz or SeizureState(), fl or FallState())
        assert _banner_pixels([v]) > 0


def test_alerts_channel_is_separate_from_posed_tracks():
    """The shared state keeps posed tracks and held alarms apart; merge_alerts joins them."""
    shared = ls.Shared()
    posed, latched = _view(tid=1), _view(fl=_FALL, tid=2)
    shared.put_frame([posed], 20.0, [latched])
    tracks, _ = shared.get_frame()
    alerts = shared.get_alerts()
    assert [t.tid for t in tracks] == [1]
    assert [a.tid for a in alerts] == [2]
    # and the banner, which sees both, reports the latched fall
    assert _banner_pixels(tracks + alerts) > 0


def test_put_frame_without_alerts_clears_the_previous_ones():
    """A stale alert must not outlive the frame that produced it."""
    shared = ls.Shared()
    shared.put_frame([], 20.0, [_view(fl=_FALL)])
    assert shared.get_alerts()
    shared.put_frame([], 20.0)
    assert shared.get_alerts() == []


def _kp():
    kp = np.tile(np.array([[200.0, 350.0]], np.float32), (17, 1))
    kp += np.arange(17, dtype=np.float32)[:, None] * 3.0
    return kp, np.full(17, 0.9, np.float32)


def test_merge_puts_the_alarm_on_the_posed_box_instead_of_a_second_box():
    kp, sc = _kp()
    posed = ls.TrackView(3, np.array([100, 300, 300, 430], np.float32), kp, sc, _IDLE_SZ, _IDLE_FL)
    held = _view(fl=_FALL, tid=3)
    merged = ls.merge_alerts([posed], [held])
    assert len(merged) == 1
    assert merged[0].fall.state == "fall" and merged[0].kp is kp  # red, with its real joints


def test_merge_appends_an_alarm_whose_person_was_not_posed():
    merged = ls.merge_alerts([_view(tid=1)], [_view(fl=_FALL, tid=2)])
    assert [v.tid for v in merged] == [1, 2]


def test_a_held_alarm_draws_its_box_but_never_a_skeleton():
    """No joints this frame means no skeleton; the box is the last place the person was."""
    img = np.zeros((720, 1280, 3), np.uint8)
    ls.draw_tracks(img, [_view(fl=_FALL)], 0.3)
    red = np.all(img == np.array(ls._RED, np.uint8), axis=2)
    assert red.any()  # the red bracket and tag were drawn
    # everything drawn sits on the box outline or its tag, nothing inside the body
    inner = img[320:410, 130:270]
    assert not inner.any()


def test_twin_boxes_on_one_person_draw_as_one():
    body = ls.TrackView(1, np.array([100, 100, 300, 500], np.float32), None, None, _IDLE_SZ, _IDLE_FL)
    head = ls.TrackView(0, np.array([150, 100, 260, 300], np.float32), None, None, _IDLE_SZ, _IDLE_FL)
    out = ls.dedupe_views([head, body])
    assert [v.tid for v in out] == [1]  # the whole-body box survives


def test_a_twin_never_hides_an_alarm():
    body = ls.TrackView(1, np.array([100, 100, 300, 500], np.float32), None, None, _IDLE_SZ, _IDLE_FL)
    head = ls.TrackView(0, np.array([150, 100, 260, 300], np.float32), None, None, _IDLE_SZ, _FALL)
    out = ls.dedupe_views([body, head])
    assert len(out) == 1 and out[0].fall.state == "fall"


def test_two_real_people_stay_two():
    a = ls.TrackView(1, np.array([100, 100, 300, 500], np.float32), None, None, _IDLE_SZ, _IDLE_FL)
    b = ls.TrackView(2, np.array([280, 100, 480, 500], np.float32), None, None, _IDLE_SZ, _IDLE_FL)
    assert len(ls.dedupe_views([a, b])) == 2
