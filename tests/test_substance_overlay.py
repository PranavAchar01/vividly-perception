"""The substance overlay: silent unless confirmed, local, and unkillable by odd shapes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from vividly.hud.substance import draw_substance, substance_banner

try:  # the analyzer half of this build owns the real type; the overlay only reads attributes
    from vividly.analysis.substance import SubstanceState  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - exercised only before the analyzer lands

    @dataclass(frozen=True)
    class SubstanceState:  # type: ignore[no-redef]
        state: str = "idle"
        area_frac: float = 0.0
        n_pools: int = 0
        held_s: float = 0.0
        growth_frac_per_s: float = 0.0
        polygons: tuple = field(default=())
        priming: bool = False


W, H = 1280, 720


def frame() -> np.ndarray:
    """A non-uniform frame, so a draw cannot hide by happening to match the background."""
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :, 1] = np.arange(W, dtype=np.uint8)[None, :]
    img[:, :, 2] = 40
    return img


def blob(cx: int, cy: int, rx: int = 60, ry: int = 34, n: int = 24) -> np.ndarray:
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], axis=1)
    return pts.astype(np.int32).reshape(-1, 1, 2)


def confirmed(polys, **kw) -> SubstanceState:
    d = {
        "state": "confirmed",
        "area_frac": 0.0084,
        "n_pools": len(polys),
        "held_s": 12.0,
        "polygons": tuple(polys),
    }
    d.update(kw)
    return SubstanceState(**d)


@pytest.mark.parametrize("st", ["idle", "candidate"])
def test_nothing_drawn_unless_confirmed(st: str) -> None:
    img = frame()
    before = img.tobytes()
    draw_substance(
        img,
        SubstanceState(state=st, area_frac=0.004, n_pools=1, polygons=(blob(400, 500),)),
        t_now=3.25,
    )
    assert img.tobytes() == before


def test_priming_draws_nothing() -> None:
    img = frame()
    before = img.tobytes()
    draw_substance(img, SubstanceState(state="idle", priming=True), t_now=0.0)
    assert img.tobytes() == before


def test_confirmed_draws() -> None:
    img = frame()
    before = img.tobytes()
    draw_substance(img, confirmed([blob(400, 500)]), t_now=3.25)
    assert img.tobytes() != before


def test_animation_phase_moves_with_t_now() -> None:
    a, b = frame(), frame()
    draw_substance(a, confirmed([blob(400, 500)]), t_now=0.0)
    draw_substance(b, confirmed([blob(400, 500)]), t_now=0.6)
    assert a.tobytes() != b.tobytes(), "the overlay must animate from the clock it is given"


def test_banner_only_when_confirmed() -> None:
    assert substance_banner(SubstanceState()) is None
    assert substance_banner(SubstanceState(state="candidate", held_s=0.9)) is None
    assert substance_banner(SubstanceState(state="confirmed", priming=True)) is None
    assert substance_banner(None) is None
    line = substance_banner(confirmed([blob(400, 500)], growth_frac_per_s=0.002))
    assert line is not None
    assert "RED SUBSTANCE ON FLOOR" in line
    assert "1 pool" in line and "0.84% of view" in line and "12s" in line
    assert "spreading" in line
    assert "\u2014" not in line and "\u2013" not in line  # no dashes of any width


def test_banner_plural_and_quiet_growth() -> None:
    line = substance_banner(confirmed([blob(300, 500), blob(700, 600)]))
    assert line is not None and "2 pools" in line and "spreading" not in line


@pytest.mark.parametrize(
    "polys",
    [
        [np.array([[[640, 700]]], np.int32)],  # one pixel
        [np.array([[[0, 700]], [[40, 719]], [[-30, 690]]], np.int32)],  # 3 pts, over the edge
        [np.array([[[1270, 715]], [[1400, 800]], [[1290, 719]]], np.int32)],  # off the corner
        [np.array([[[-500, -500]], [[-400, -400]], [[-450, -420]]], np.int32)],  # fully outside
        [blob(200, 500), blob(500, 560), blob(800, 620), blob(1100, 680)],  # four at once
        [np.empty((0, 1, 2), np.int32)],  # empty contour
        [np.array([[[640, 500]], [[660, 500]]], np.int32)],  # two points, no area
    ],
)
def test_degenerate_shapes_do_not_crash(polys) -> None:
    img = frame()
    draw_substance(img, confirmed(polys), t_now=7.5)
    assert img.shape == (H, W, 3) and img.dtype == np.uint8


def test_no_polygons_draws_nothing() -> None:
    img = frame()
    before = img.tobytes()
    draw_substance(img, confirmed([]), t_now=1.0)
    assert img.tobytes() == before


def _changed_bbox(a: np.ndarray, b: np.ndarray):
    ys, xs = np.nonzero(np.any(a != b, axis=2))
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if xs.size else None


def test_draw_stays_inside_the_frame_and_near_its_pool() -> None:
    img = frame()
    before = img.copy()
    poly = blob(1180, 660, rx=70, ry=40)  # pushed into the bottom-right corner
    draw_substance(img, confirmed([poly]), t_now=2.0)
    box = _changed_bbox(before, img)
    assert box is not None
    x1, y1, x2, y2 = box
    assert 0 <= x1 <= x2 < W and 0 <= y1 <= y2 < H
    # Everything drawn belongs to the pool: the contour plus ticks, brackets and the chip,
    # all of which sit within a small neighbourhood of it. A global tint would fail this.
    px1, py1 = 1180 - 70, 660 - 40
    px2, py2 = 1180 + 70, 660 + 40
    assert x1 >= px1 - 240 and y1 >= py1 - 60
    assert x2 <= min(W - 1, px2 + 240) and y2 <= min(H - 1, py2 + 60)


def test_edge_pool_draws_something_inside_the_frame() -> None:
    img = frame()
    before = img.copy()
    poly = np.array([[[-40, 700]], [[120, 700]], [[120, 760]], [[-40, 760]]], np.int32)
    draw_substance(img, confirmed([poly]), t_now=5.0)
    assert _changed_bbox(before, img) is not None
