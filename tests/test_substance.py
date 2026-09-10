"""Synthetic-frame tests for the red-substance-on-floor detector.

No video and no sleeping: every frame is a numpy array built here, and time is the explicit
`now` argument, so the whole persistence machine runs deterministically at whatever rate the
test wants.

Frame geometry used throughout: 640x360 display, so proc_width 320 gives a 320x180 processed
frame and floor_y_frac 0.45 puts the floor band at display rows 162..360.
"""

from __future__ import annotations

import numpy as np
import pytest

from vividly.analysis import SubstanceAnalyzer, SubstanceConfig, SubstanceState

W, H = 640, 360
DT = 0.1  # simulated capture interval

FLOOR_BGR = (185, 185, 185)  # grey concrete: near-zero saturation, never red
RED_BGR = (0, 0, 220)  # hue 0, saturation 255: squarely inside hue_band_a

# A pool low on the floor, well below floor_y_frac*H = 162.
POOL = (200, 240, 320, 300)  # x1, y1, x2, y2 in display coords
# The same shape hung on the wall, entirely above the floor line.
WALL = (200, 40, 320, 100)


def floor_frame(fill=FLOOR_BGR) -> np.ndarray:
    return np.full((H, W, 3), fill, np.uint8)


def paint(frame: np.ndarray, box, colour=RED_BGR) -> np.ndarray:
    x1, y1, x2, y2 = box
    frame[y1:y2, x1:x2] = colour
    return frame


def prime(an: SubstanceAnalyzer, t: float, frame_fn, n: int | None = None) -> float:
    """Run the analyzer through priming. Returns the clock after the last primed update."""
    n = an.cfg.prime_frames if n is None else n
    for _ in range(n):
        st = an.update(frame_fn(), t)
        assert st.priming is True
        assert st.state == "idle"
        t += DT
    return t


def run(an: SubstanceAnalyzer, t: float, frame_fn, n: int, boxes=()) -> tuple[float, list]:
    states: list[SubstanceState] = []
    for _ in range(n):
        states.append(an.update(frame_fn(), t, boxes))
        t += DT
    return t, states


@pytest.fixture
def analyzer() -> SubstanceAnalyzer:
    return SubstanceAnalyzer(SubstanceConfig())


def test_plain_floor_is_idle(analyzer):
    t = prime(analyzer, 0.0, floor_frame)
    _, states = run(analyzer, t, floor_frame, 40)
    assert all(s.state == "idle" for s in states)
    assert all(s.priming is False for s in states)
    assert all(s.area_frac == 0.0 and s.n_pools == 0 for s in states)


def test_red_above_the_floor_line_is_idle(analyzer):
    """Place, discriminator 2: red on the wall is never even looked at."""
    t = prime(analyzer, 0.0, floor_frame)
    _, states = run(analyzer, t, lambda: paint(floor_frame(), WALL), 40)
    assert all(s.state == "idle" for s in states)


def test_pool_on_the_floor_goes_candidate_then_confirmed(analyzer):
    t = prime(analyzer, 0.0, floor_frame)
    spill = lambda: paint(floor_frame(), POOL)  # noqa: E731

    first = analyzer.update(spill(), t)
    assert first.state == "candidate"
    assert first.n_pools == 1
    assert first.area_frac > analyzer.cfg.min_area_frac
    assert first.polygons and first.polygons[0].dtype == np.int32

    # Not yet: min_hold_s is 1.5 s and only 1.0 s of it has passed.
    t, mid = run(analyzer, t + DT, spill, 10)
    assert [s.state for s in mid] == ["candidate"] * 10

    t, late = run(analyzer, t, spill, 10)
    assert late[-1].state == "confirmed"
    assert late[-1].held_s >= analyzer.cfg.min_hold_s

    # The polygon comes back in DISPLAY coordinates, so the compositor can draw it directly.
    poly = late[-1].polygons[0].reshape(-1, 2)
    x1, y1, x2, y2 = POOL
    assert x1 - 8 <= poly[:, 0].min() and poly[:, 0].max() <= x2 + 8
    assert y1 - 8 <= poly[:, 1].min() and poly[:, 1].max() <= y2 + 8


def test_pool_survives_a_full_minute_without_being_learned(analyzer):
    """Failure mode 1: an adapting baseline absorbs a stationary pool. At bg_alpha 0.02 an
    unfrozen baseline crosses the novelty threshold in ~35 frames, so this would go quiet
    around second 4 and stay quiet for the rest of the event."""
    t = prime(analyzer, 0.0, floor_frame)
    spill = lambda: paint(floor_frame(), POOL)  # noqa: E731
    _, states = run(analyzer, t, spill, 600)  # 60 s at DT
    assert states[-1].state == "confirmed"
    assert states[-1].area_frac == pytest.approx(states[20].area_frac, rel=0.05)


def test_red_furniture_present_from_the_first_frame_is_never_reported(analyzer):
    """Novelty, discriminator 1: the red step ladder is in frame before the process starts,
    so priming learns it and it can never become news."""
    ladder = lambda: paint(floor_frame(), POOL)  # noqa: E731
    t = prime(analyzer, 0.0, ladder)
    _, states = run(analyzer, t, ladder, 200)
    assert all(s.state == "idle" for s in states)
    assert all(s.n_pools == 0 for s in states)


def test_global_saturation_lift_is_not_reported(analyzer):
    """Failure mode 2: auto-gain lifts saturation across the whole image in one frame, which
    pushes every marginal floor pixel over sat_min at once. That is a camera response, not a
    pool: it appears everywhere simultaneously."""
    dim = (170, 170, 215)  # saturation 53, below sat_min: not red
    lifted = (120, 120, 230)  # saturation 122, above sat_min: the same floor, now "red"

    t = prime(analyzer, 0.0, lambda: floor_frame(dim))
    _, states = run(analyzer, t, lambda: floor_frame(lifted), 40)  # 4 s of the new exposure
    assert all(s.state == "idle" for s in states)
    assert all(s.n_pools == 0 for s in states)


def test_removing_the_pool_clears_the_state_after_release_s(analyzer):
    t = prime(analyzer, 0.0, floor_frame)
    t, _ = run(analyzer, t, lambda: paint(floor_frame(), POOL), 25)
    assert analyzer.state_name() == "confirmed"

    # release_s is 1.0 s, so the state is held across a short dropout and only then cleared.
    t, gone = run(analyzer, t, floor_frame, 5)
    assert gone[-1].state == "confirmed"
    _, cleared = run(analyzer, t, floor_frame, 10)
    assert cleared[-1].state == "idle"
    assert cleared[-1].held_s == 0.0


def test_a_red_object_crossing_the_floor_never_confirms(analyzer):
    """Persistence, discriminator 3: a red shirt is novel and is on the floor, but it does
    not hold one footprint, so its hold clock restarts every time it moves off its seed."""
    t = prime(analyzer, 0.0, floor_frame)
    x = 40

    def crossing():
        nonlocal x
        frame = paint(floor_frame(), (x, 240, x + 60, 300))
        x += 60  # a full body-width per frame: no overlap with the previous footprint
        return frame

    _, states = run(analyzer, t, crossing, 8)
    assert all(s.state != "confirmed" for s in states)


def test_exclude_boxes_mask_a_person_out(analyzer):
    """A tracked person standing over the pool is masked before scoring, and the mask must
    not decay the baseline behind them either."""
    t = prime(analyzer, 0.0, floor_frame)
    spill = lambda: paint(floor_frame(), POOL)  # noqa: E731
    box = np.array([180.0, 220.0, 340.0, 340.0], np.float32)
    _, states = run(analyzer, t, spill, 25, boxes=(box,))
    assert all(s.state == "idle" for s in states)


def test_reset_returns_to_priming(analyzer):
    t = prime(analyzer, 0.0, floor_frame)
    analyzer.update(paint(floor_frame(), POOL), t)
    analyzer.reset()
    assert analyzer.state_name() == "idle"
    assert analyzer.update(floor_frame(), t + DT).priming is True


def test_on_absent_does_not_drop_a_confirmed_pool(analyzer):
    """A spill outlives the person who made it; losing the subject must not clear it."""
    t = prime(analyzer, 0.0, floor_frame)
    t, _ = run(analyzer, t, lambda: paint(floor_frame(), POOL), 25)
    assert analyzer.state_name() == "confirmed"
    analyzer.on_absent(t)
    assert analyzer.state_name() == "confirmed"


def test_a_spreading_pool_that_darkens_at_its_centre_keeps_its_hold(analyzer):
    """Real liquid deep enough to pool goes darker than val_min at the centre, so a
    spreading pool's red mask is an annulus whose hole grows outward until it covers the
    small footprint the hold clock was seeded on. Measured on the raw mask that reads as
    "something moved" and a motionless pool drops back to candidate; measured on the filled
    footprint it does not. Regression for a confirmed pool falling back to candidate 12 s
    into a spread on real footage at proc_width 480."""
    cx, cy = 260, 270  # well below floor_y_frac*H = 162

    def pool_frame(half_w: int, half_h: int, core: int):
        def make():
            f = paint(floor_frame(), (cx - half_w, cy - half_h, cx + half_w, cy + half_h))
            if core > 0:
                # Near-black: under val_min, so the core leaves the red mask entirely.
                f[cy - core // 2 : cy + core // 2, cx - core : cx + core] = (8, 8, 20)
            return f

        return make

    # Learn the bare floor, then seed the clock on a small pool and confirm on it.
    t = prime(analyzer, 0.0, floor_frame)
    t, _ = run(analyzer, t, pool_frame(16, 11, 0), 20)
    assert analyzer.state_name() == "confirmed"

    # Now spread outward while the deep core grows past the original footprint.
    for grow in range(1, 12):
        half_w, half_h = 16 + 5 * grow, 11 + 3 * grow
        core = 2 * grow  # by grow=9 the core is wider than the seed the clock started on
        t, states = run(analyzer, t, pool_frame(half_w, half_h, core), 4)
        assert all(s.state == "confirmed" for s in states), f"dropped at grow={grow}"
