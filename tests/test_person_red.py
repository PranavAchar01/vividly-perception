"""red_substance_on_person: the colour cone, the state machine, part assignment, priming,
identity hand-over, forget(), the room, and the overlay's banner and draw contract.

Every frame here is synthetic: a grey room, one drawn person with pose keypoints that match
the drawing, and dark-red discs painted where a test wants them. No footage is involved.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vividly.analysis.person_red import (
    PersonRedAnalyzer,
    PersonRedConfig,
    PersonRedHit,
    PersonRedState,
)
from vividly.hud.person_red import draw_person_red, person_red_banner

W, H = 1280, 720
DT = 0.1  # 10 updates a second

BG = (70, 70, 70)
SKIN = (120, 145, 185)  # BGR; sRGB (185,145,120): L* 63, a* 11
SHIRT = (150, 110, 60)  # a blue shirt
TROUSERS = (60, 55, 50)
RED = (22, 18, 105)  # BGR; sRGB (105,18,22): L* 22, a* 40, a dark saturated red

# A standing person, facing the camera, head at the top. COCO-17 order.
KP0 = np.array(
    [
        [640, 200], [632, 193], [648, 193], [622, 197], [658, 197],  # nose, eyes, ears
        [600, 250], [680, 250], [588, 320], [692, 320], [584, 390], [696, 390],  # arms
        [612, 380], [668, 380], [612, 480], [668, 480], [612, 580], [668, 580],  # legs
    ],
    np.float32,
)  # fmt: skip
SC0 = np.full(17, 0.9, np.float32)
BOX0 = np.array([560, 140, 720, 610], np.float32)

# Where each part is, as (x, y, radius) of a disc painted onto the drawn person.
SCALP = (640, 170, 10)
FACE = (640, 208, 7)
HAND = (583, 412, 8)
TORSO = (640, 320, 12)
LEG = (612, 530, 9)
ARM = (592, 290, 8)


def draw_person(img: np.ndarray, kp: np.ndarray, shirt=SHIRT) -> None:
    p = [(int(x), int(y)) for x, y in kp]
    cv2.fillConvexPoly(img, np.array([p[5], p[6], p[12], p[11]], np.int32), shirt)
    for a, b in ((11, 13), (13, 15), (12, 14), (14, 16)):
        cv2.line(img, p[a], p[b], TROUSERS, 26)
    for a, b in ((5, 7), (7, 9), (6, 8), (8, 10)):
        cv2.line(img, p[a], p[b], shirt, 20)
    for w in (9, 10):
        cv2.circle(img, p[w], 11, SKIN, -1)
    cv2.circle(img, (p[0][0], p[0][1] - 6), 24, SKIN, -1)


def scene(
    kp=KP0, discs=(), shirt=SHIRT, panel: tuple | None = None, wash: bool = False
) -> np.ndarray:
    img = np.full((H, W, 3), BG, np.uint8)
    if wash:
        img[:] = RED  # the camera's white balance swinging the whole room red
    if panel is not None:
        x1, y1, x2, y2 = panel
        cv2.rectangle(img, (x1, y1), (x2, y2), RED, -1)
    draw_person(img, kp, shirt)
    for x, y, r in discs:
        cv2.circle(img, (int(x), int(y)), int(r), RED, -1)
    return img


def person(tid=1, kp=KP0, box=BOX0, sc=SC0):
    return [(tid, box, kp, sc)]


def run(an, frames, t0=0.0, people_fn=None):
    """Feed (frame, people) pairs at DT and return [(t, state)]."""
    out = []
    for i, (img, ppl) in enumerate(frames):
        t = t0 + i * DT
        out.append((t, an.update(img, t, ppl if people_fn is None else people_fn(i))))
    return out


def hits_of(st: PersonRedState, tid: int = 1):
    return [h for h in st.hits if h.tid == tid]


def primed_analyzer(cfg: PersonRedConfig | None = None, seconds: float = 2.5):
    """An analyzer that has learned the clean person, and the time it reached."""
    an = PersonRedAnalyzer(cfg)
    clean = scene()
    n = round(seconds / DT)
    res = run(an, [(clean, person())] * n)
    assert not res[-1][1].priming
    return an, n * DT


# --- colour ----------------------------------------------------------------------------------


def _lab_patch(bgr) -> np.ndarray:
    return cv2.cvtColor(np.full((4, 4, 3), bgr, np.uint8), cv2.COLOR_BGR2Lab)


@pytest.mark.parametrize(
    "bgr",
    [
        SKIN,
        (100, 110, 190),  # flushed skin under warm light, sRGB (190,110,100): a* 31, L* 55
        (60, 80, 120),  # darker skin
        (140, 172, 224),  # light skin
    ],
)
def test_gate_rejects_skin(bgr) -> None:
    # Lips are deliberately not in this list: bright lips, sRGB (180,80,90), sit at a*/L* 0.89
    # and pass the cone. They are handled by novelty (present from the first frame, and about
    # 1.5% of a head circle, above the learn fraction), which the priming tests cover.
    an = PersonRedAnalyzer()
    assert not an._gate(_lab_patch(bgr)).any()


@pytest.mark.parametrize("bgr", [RED, (20, 20, 160), (10, 10, 100), (30, 25, 130)])
def test_gate_accepts_dark_saturated_red(bgr) -> None:
    an = PersonRedAnalyzer()
    assert an._gate(_lab_patch(bgr)).all()


# --- priming and the state machine ------------------------------------------------------------


def test_priming_reports_nothing_and_says_so() -> None:
    an = PersonRedAnalyzer()
    # Red on the scalp from the very first frame: while the analyzer is priming it must
    # report nothing, and it must say that it is priming.
    frames = [(scene(discs=[SCALP]), person())] * 15
    for t, st in run(an, frames):
        assert st.priming, t
        assert st.hits == ()


def test_red_present_through_priming_is_learned_not_reported() -> None:
    an = PersonRedAnalyzer()
    frames = [(scene(discs=[SCALP]), person())] * 80
    res = run(an, frames)
    assert not res[-1][1].priming
    assert all(st.hits == () for _t, st in res)


def test_confirms_after_min_hold_on_the_head_then_releases() -> None:
    cfg = PersonRedConfig()
    an, t0 = primed_analyzer(cfg)
    red = scene(discs=[SCALP])
    res = run(an, [(red, person())] * 30, t0=t0)

    first = next(t for t, st in res if hits_of(st))
    confirm = next(t for t, st in res if any(h.state == "confirmed" for h in hits_of(st)))
    assert first == pytest.approx(t0)
    assert confirm - first == pytest.approx(cfg.min_hold_s, abs=DT + 1e-6)
    for t, st in res:
        (h,) = hits_of(st)
        assert h.state == ("confirmed" if t >= confirm else "candidate")
        assert h.part == "head"
        assert h.polygons, "a measured hit carries its outline"
        assert 0.0 < h.area_frac < 0.001

    # Gone: the episode is held through release_s (no outline, nothing to draw) and then
    # clears; it is not dropped on the first empty frame.
    t1 = t0 + len(res) * DT
    res2 = run(an, [(scene(), person())] * 20, t0=t1)
    held = [t for t, st in res2 if hits_of(st)]
    assert held and max(held) - t1 < cfg.release_s
    assert all(not h.polygons for t, st in res2 for h in hits_of(st))
    assert res2[-1][1].hits == ()


def test_flicker_below_duty_does_not_confirm() -> None:
    an, t0 = primed_analyzer()
    # Present one update in three: never enough of the hold to be real.
    frames = [(scene(discs=[SCALP] if i % 3 == 0 else ()), person()) for i in range(60)]
    res = run(an, frames, t0=t0)
    assert all(h.state != "confirmed" for _t, st in res for h in st.hits)


def test_a_sparse_start_does_not_lock_out_red_that_becomes_steady() -> None:
    # A thin trickle flickers in and out of shadow before it thickens. Duty is judged over a
    # recent window, so once the red is steady for the hold it confirms; before the fix the
    # early misses stayed in an episode-long denominator and it never could.
    cfg = PersonRedConfig(release_s=2.5)
    an, t0 = primed_analyzer(cfg)
    sparse = [(scene(discs=[SCALP] if i % 3 == 0 else ()), person()) for i in range(30)]
    steady = [(scene(discs=[SCALP]), person())] * 40
    res = run(an, sparse + steady, t0=t0)
    t_steady = t0 + 30 * DT
    confirm = next(t for t, st in res if any(h.state == "confirmed" for h in st.hits))
    assert t_steady < confirm <= t_steady + max(cfg.min_hold_s, 1.0) + DT + 1e-6


def test_red_that_moves_to_another_part_restarts_the_clock() -> None:
    cfg = PersonRedConfig()
    an, t0 = primed_analyzer(cfg)
    frames = [(scene(discs=[SCALP]), person())] * 10 + [(scene(discs=[TORSO]), person())] * 25
    res = run(an, frames, t0=t0)
    t_move = t0 + 10 * DT
    confirm = next(t for t, st in res if any(h.state == "confirmed" for h in st.hits))
    assert confirm >= t_move + cfg.min_hold_s - 1e-6
    assert hits_of(res[-1][1])[0].part == "torso"


def test_scalp_to_face_is_one_place_and_keeps_its_clock() -> None:
    # Red that starts on the scalp and runs down onto the face has not moved anywhere: the
    # nearest keypoint changes from ear to eye, the clock must not restart.
    cfg = PersonRedConfig()
    an, t0 = primed_analyzer(cfg)
    frames = [(scene(discs=[SCALP]), person())] * 8 + [(scene(discs=[FACE]), person())] * 20
    res = run(an, frames, t0=t0)
    confirm = next(t for t, st in res if any(h.state == "confirmed" for h in st.hits))
    assert confirm == pytest.approx(t0 + cfg.min_hold_s, abs=DT + 1e-6)
    assert hits_of(res[-1][1])[0].part == "head", "named for where it started"


def test_confirmed_red_that_moves_stays_confirmed() -> None:
    an, t0 = primed_analyzer()
    frames = [(scene(discs=[SCALP]), person())] * 20 + [(scene(discs=[TORSO]), person())] * 10
    res = run(an, frames, t0=t0)
    after = [hits_of(st)[0] for _t, st in res[20:]]
    assert all(h.state == "confirmed" for h in after)
    assert after[-1].part == "torso"


# --- part assignment ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("disc", "part"),
    [(SCALP, "head"), (FACE, "face"), (HAND, "hand"), (ARM, "arm"), (TORSO, "torso"), (LEG, "leg")],
)
def test_part_is_the_body_part_under_the_region(disc, part) -> None:
    an, t0 = primed_analyzer()
    res = run(an, [(scene(discs=[disc]), person())] * 3, t0=t0)
    (h,) = hits_of(res[-1][1])
    assert h.part == part


def test_polygons_are_in_display_coordinates() -> None:
    an, t0 = primed_analyzer()
    res = run(an, [(scene(discs=[SCALP]), person())] * 2, t0=t0)
    (h,) = hits_of(res[-1][1])
    pts = np.concatenate([np.asarray(p).reshape(-1, 2) for p in h.polygons])
    x, y, r = SCALP
    assert h.polygons[0].dtype == np.int32 and h.polygons[0].shape[1:] == (1, 2)
    assert np.all(np.abs(pts[:, 0] - x) <= r + 3)
    assert np.all(np.abs(pts[:, 1] - y) <= r + 3)


# --- novelty is per part -----------------------------------------------------------------------


def test_red_garment_is_learned_but_does_not_teach_the_head() -> None:
    an = PersonRedAnalyzer()
    shirt = scene(shirt=RED)
    res = run(an, [(shirt, person())] * 60)
    assert all(st.hits == () for _t, st in res), "a red shirt worn from the start is normal"
    t0 = 60 * DT
    res = run(an, [(scene(shirt=RED, discs=[SCALP]), person())] * 25, t0=t0)
    (h,) = hits_of(res[-1][1])
    assert h.state == "confirmed" and h.part == "head"


# --- the room ----------------------------------------------------------------------------------

PANEL = (895, 120, 925, 420)  # a red fixture on the wall, like the step ladder
SHIFT = np.array([300.0, 0.0], np.float32)  # the person later stands in front of it


def _walk_to_panel(an) -> list:
    """Prime with the person away from the panel, then have them stand in front of it."""
    here = [(scene(panel=PANEL), person())] * 40
    there_kp, there_box = KP0 + SHIFT, BOX0 + np.r_[SHIFT, SHIFT]
    there = [(scene(kp=there_kp, panel=PANEL), person(kp=there_kp, box=there_box))] * 40
    return run(an, here + there)


def test_room_red_behind_a_person_is_not_reported() -> None:
    res = _walk_to_panel(PersonRedAnalyzer())
    assert all(st.hits == () for _t, st in res)


def test_room_red_behind_a_person_would_be_reported_without_the_room_map() -> None:
    # The control for the test above: the panel really does reach into the person's body
    # mask, so it is the room map, not luck of geometry, that keeps it out.
    an = PersonRedAnalyzer()
    real_room = an._room

    def no_room(frame, now, people):
        trusted = real_room(frame, now, people)
        an._scene_static[:] = 0
        return trusted

    an._room = no_room
    res = _walk_to_panel(an)
    assert any(h.state == "confirmed" for _t, st in res for h in st.hits)


def swing(img: np.ndarray) -> np.ndarray:
    """A hard warm white-balance swing over the WHOLE frame, person included. Chosen so the
    drawn skin lands inside the colour cone (sRGB (176,51,42): L* 42, a* 50): the worst case
    the room guard exists for."""
    return np.clip(img.astype(np.float32) * np.array([0.35, 0.35, 0.95]), 0, 255).astype(np.uint8)


def test_white_balance_swing_is_relearned_not_reported() -> None:
    an, t0 = primed_analyzer()
    assert an._gate(
        cv2.cvtColor(swing(np.full((2, 2, 3), SKIN, np.uint8)), cv2.COLOR_BGR2Lab)
    ).all()
    res = run(an, [(swing(scene()), person())] * 50, t0=t0)
    assert all(h.state != "confirmed" for _t, st in res for h in st.hits)
    # ...and the detector re-arms under the new colours afterwards. On the torso: on the head
    # the swung skin was re-learned as normal, and in this synthetic swing it is within a
    # histogram bin of the swung red, which is exactly the trade the relearn makes (a colour
    # the camera now renders skin as cannot be told from skin).
    t1 = t0 + 50 * DT
    res = run(an, [(swing(scene(discs=[TORSO])), person())] * 25, t0=t1)
    (h,) = hits_of(res[-1][1])
    assert h.state == "confirmed" and h.part == "torso"


def test_white_balance_swing_would_be_reported_without_the_guard() -> None:
    # The control for the test above: with the lighting guard disabled the swung skin is
    # new, coherent red on the person and it confirms.
    import vividly.analysis.person_red as mod

    old = mod._LIGHTING_JUMP_FRAC, mod._SWING_AB
    mod._LIGHTING_JUMP_FRAC, mod._SWING_AB = 2.0, 1e9  # can never trigger
    try:
        an, t0 = primed_analyzer()
        res = run(an, [(swing(scene()), person())] * 50, t0=t0)
    finally:
        mod._LIGHTING_JUMP_FRAC, mod._SWING_AB = old
    assert any(h.state == "confirmed" for _t, st in res for h in st.hits)


def test_red_arriving_with_a_swing_cannot_confirm_inside_the_window() -> None:
    from vividly.analysis.person_red import _RELEARN_S

    an, t0 = primed_analyzer()
    res = run(an, [(scene(discs=[SCALP], wash=True), person())] * 12, t0=t0)
    assert all(h.state != "confirmed" for t, st in res for h in st.hits if t - t0 < _RELEARN_S)


def test_candidate_seeded_just_before_a_swing_is_dropped() -> None:
    # The swing is seen an update late: the red it painted in the frame before the room
    # noticed must not keep the hold it started.
    an, t0 = primed_analyzer()
    frames = [(scene(discs=[SCALP]), person())] + [(scene(discs=[SCALP], wash=True), person())] * 40
    res = run(an, frames, t0=t0)
    assert hits_of(res[0][1])[0].state == "candidate"
    assert all(h.state != "confirmed" for _t, st in res for h in st.hits)


# --- identity ----------------------------------------------------------------------------------


def test_forget_drops_the_track_and_it_primes_again() -> None:
    cfg = PersonRedConfig()
    an, t0 = primed_analyzer(cfg)
    res = run(an, [(scene(discs=[SCALP]), person())] * 25, t0=t0)
    assert hits_of(res[-1][1])[0].state == "confirmed"
    an.forget(1)
    t1 = t0 + 25 * DT
    res = run(an, [(scene(discs=[SCALP]), person())] * 50, t0=t1)
    assert res[0][1].priming and res[0][1].hits == ()
    # Re-primed with the red already there, so the red is now part of the person.
    assert all(st.hits == () for _t, st in res)


def test_new_id_on_the_same_person_takes_over_model_and_episode() -> None:
    an, t0 = primed_analyzer()
    res = run(an, [(scene(discs=[SCALP]), person())] * 25, t0=t0)
    assert hits_of(res[-1][1])[0].state == "confirmed"
    # The tracker hands the same person to a new id: the alert carries on, no priming.
    t1 = t0 + 25 * DT
    res = run(an, [(scene(discs=[SCALP]), person(tid=7))] * 5, t0=t1)
    for _t, st in res:
        (h,) = [h for h in st.hits if h.tid == 7]
        assert h.state == "confirmed" and h.part == "head"
        assert not st.priming


def test_an_old_id_handed_back_the_person_takes_the_live_episode() -> None:
    # Id 5 has been learned as someone standing elsewhere. Later the tracker hands the person
    # carrying a confirmed episode (id 1, now unposed) to id 5: the alert carries on at once.
    an = PersonRedAnalyzer()
    kp2, box2 = KP0 + SHIFT, BOX0 + np.r_[SHIFT, SHIFT]
    img = scene()
    draw_person(img, kp2)
    run(an, [(img, [(1, BOX0, KP0, SC0), (5, box2, kp2, SC0)])] * 25)
    t0 = 25 * DT
    res = run(an, [(scene(discs=[SCALP]), person())] * 20, t0=t0)
    assert hits_of(res[-1][1])[0].state == "confirmed"
    t1 = t0 + 20 * DT
    res = run(an, [(scene(discs=[SCALP]), person(tid=5))] * 5, t0=t1)
    for _t, st in res:
        (h,) = st.hits
        assert h.tid == 5 and h.state == "confirmed" and h.part == "head"


def test_new_id_elsewhere_primes_from_scratch() -> None:
    an, t0 = primed_analyzer()
    kp2, box2 = KP0 + SHIFT, BOX0 + np.r_[SHIFT, SHIFT]
    both = [(1, BOX0, KP0, SC0), (2, box2, kp2, SC0)]
    img = scene(discs=[SCALP])
    draw_person(img, kp2)
    x, y, r = SCALP
    cv2.circle(img, (int(x + SHIFT[0]), y), r, RED, -1)
    res = run(an, [(img, both)] * 10, t0=t0)
    assert res[0][1].priming, "the new person is learned, not inherited"
    assert all(h.tid == 1 for _t, st in res for h in st.hits)


def test_reset_forgets_everything() -> None:
    an, t0 = primed_analyzer()
    an.reset()
    st = an.update(scene(discs=[SCALP]), t0, person())
    assert st.priming and st.hits == ()


# --- banner and overlay ---------------------------------------------------------------------


def _hit(state="confirmed", part="head", tid=1, held=3.2, growth=0.001, polys=None):
    if polys is None:
        a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        polys = (
            np.stack([640 + 14 * np.cos(a), 170 + 10 * np.sin(a)], 1)
            .astype(np.int32)
            .reshape(-1, 1, 2),
        )
    return PersonRedHit(
        tid=tid, state=state, part=part, area_frac=0.0012, held_s=held,
        growth_frac_per_s=growth, polygons=polys,
    )  # fmt: skip


def test_banner_text() -> None:
    st = PersonRedState(hits=(_hit(),))
    assert person_red_banner(st) == "RED SUBSTANCE ON PERSON   head   person #1   3s   spreading"
    st = PersonRedState(hits=(_hit(growth=0.0, part="hand", tid=4, held=12.4),))
    assert person_red_banner(st) == "RED SUBSTANCE ON PERSON   hand   person #4   12s"


def test_banner_silent_unless_confirmed() -> None:
    assert person_red_banner(None) is None
    assert person_red_banner(PersonRedState()) is None
    assert person_red_banner(PersonRedState(hits=(_hit(state="candidate"),))) is None
    assert person_red_banner(PersonRedState(priming=True)) is None


def test_banner_picks_the_longest_held_person() -> None:
    st = PersonRedState(hits=(_hit(tid=1, held=2.0), _hit(tid=2, held=9.0, part="face")))
    assert person_red_banner(st).startswith("RED SUBSTANCE ON PERSON   face   person #2   9s")


def test_banner_vocabulary_is_observational() -> None:
    text = person_red_banner(PersonRedState(hits=(_hit(),))).lower()
    for word in ("blood", "bleed", "wound", "injur"):
        assert word not in text


def test_draw_is_silent_unless_confirmed() -> None:
    img = scene()
    before = img.copy()
    draw_person_red(img, PersonRedState(hits=(_hit(state="candidate"),)), 1.0)
    draw_person_red(img, PersonRedState(), 1.0)
    draw_person_red(img, None, 1.0)
    assert np.array_equal(img, before)


def test_draw_marks_the_region_and_survives_odd_shapes() -> None:
    img = scene()
    before = img.copy()
    draw_person_red(img, PersonRedState(hits=(_hit(),)), 1.0)
    changed = np.any(img != before, axis=2)
    ys, _xs = np.nonzero(changed)
    assert changed[165:176, 630:651].any(), "the region itself is marked"
    assert ys.min() < 170, "and labelled above it"
    # A confirmed hit with no outline (held through a missed frame), a single point, a line,
    # and a region against the frame edge must neither raise nor draw off-frame.
    for polys in ((), (np.array([[[5, 5]]], np.int32),), (np.array([[[0, 0]], [[40, 0]]], np.int32),),
                  (np.array([[[1270, 2]], [[1279, 2]], [[1279, 12]]], np.int32),)):  # fmt: skip
        draw_person_red(img, PersonRedState(hits=(_hit(polys=polys),)), 2.0)


def test_draw_label_carries_part_and_age() -> None:
    # The chip text is what a viewer reads; check the label string the draw builds. No area
    # percentage: at camera distance it printed "<0.01%", which read as nothing being there.
    import vividly.hud.person_red as hud

    seen = []
    real = hud._chip
    hud._chip = lambda img, anchor, below, text: seen.append(text)
    try:
        img = scene()
        draw_person_red(img, PersonRedState(hits=(_hit(),)), 1.0)
    finally:
        hud._chip = real
    assert len(seen) == 1
    assert seen[0] == "HEAD  3s"
