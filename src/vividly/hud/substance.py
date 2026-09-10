"""Overlay for the `red_substance_on_floor` descriptor: where the pool is and how big it got.

Two jobs, split the way the compositor is split: `draw_substance` marks the pools on the
display frame, `substance_banner` returns the one line the top bar prints. Nothing here
touches the analyzer, and nothing here holds state between frames: every animated value is
derived from the wall clock the caller passes in, so the motion is identical at 12 fps and
at 30 fps and a dropped frame does not rewind it.

What it draws, and why each part is there rather than being decoration:

  * A translucent fill inside the contour. The fill IS the measurement: it covers exactly
    the pixels the analyzer scored as newly-red floor, so a viewer can see the detector
    disagreeing with the room (the red step ladder on the wall never fills) instead of
    taking a box on faith.
  * An outline with regular tick marks along it, spaced in display pixels. Ticks read as a
    caliper rather than a glow, and they make an irregular pool legible on film where a
    plain 2 px outline against a red fill disappears.
  * One slow pass of a bright segment around the perimeter. It says the boundary is being
    traced continuously rather than stamped once, which is the honest description of what
    the analyzer does, and it is the only moving element so it cannot read as urgency.
  * A chip tied to the pool by a leader, carrying size and age. "How much" and "how long"
    are the two numbers that separate a spill from a dropped scarf, and they are the two
    the operator is asked about afterwards.

Palette. The neutral box colour and the confirmed red, nothing between: no amber anywhere,
because a hedging colour on film reads as a system that is not sure. Red is the event; the
white carries the measurement furniture (ticks, sweep) so the instrument and the
alarm are never the same colour. The values are duplicated from scripts/live_stream.py
rather than imported: that script is the integrator's file and importing a script from a
library module would invert the dependency.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# Brand palette (BGR), matching scripts/live_stream.py exactly.
_BOX = (130, 235, 150)  # mint, the neutral instrument colour (people, not substances)
# The measurement furniture on a CONFIRMED pool is white, not mint. Two reasons. Mint is
# what a box gets when nothing is wrong, and lending it to an alert mixes the two things the
# palette exists to keep apart. And on a compressed stream at viewing distance mint reads as
# yellow-green, which is the colour the founder took off this overlay.
_MEASURE = (255, 255, 255)
_RED = (60, 60, 245)  # confirmed
_INK = (24, 22, 18)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE = 0.5  # same as the per-person tag in draw_tracks, so the two chips match
_LABEL_THICK = 1

# The fill breathes between these two alphas. The floor of 0.16 keeps the pool readable on
# a white garage floor after JPEG; the ceiling of 0.30 still lets the texture underneath
# show through, which is what makes it read as a measurement over the floor rather than a
# sticker on top of it.
_FILL_ALPHA_LO = 0.16
_FILL_ALPHA_HI = 0.30
_PULSE_S = 2.4  # slower than a resting pulse: deliberate, not alarming

#: Every pixel constant below is quoted at this frame height and scaled from it at draw
#: time, so the overlay is the same size relative to the picture whether it is drawn on the
#: 720p live view or on a full-resolution render for film. Without this the whole instrument
#: shrinks to half its intended weight on a 1296-line frame, and the label stops being
#: readable at the back of a room.
_REF_H = 720.0


def _k(img) -> float:
    """Scale factor from the reference frame height, clamped so a thumbnail keeps hairlines
    and a very tall frame does not grow slabs."""
    return min(3.0, max(0.6, img.shape[0] / _REF_H))


_OUTLINE_THICK = 2
_TICK_PX = 22.0  # tick spacing in DISPLAY pixels, so ticks stay a constant size on screen
_TICK_LEN = 7  # outward length of a tick
_TICK_GAP = 2  # ticks start clear of the outline, so the boundary itself stays one clean line
_MAX_TICKS = 48  # a huge pool widens its spacing instead of costing hundreds of lines
_SWEEP_PX_S = 170.0  # perimeter travel speed; ~2 s around a metre-wide pool at 1280 wide
_SWEEP_FRAC = 0.16  # fraction of the perimeter lit at once
_SWEEP_STEPS = 40  # perimeter samples used for the sweep, independent of pool size
_SWEEP_THICK = 3

_BRACKET_THICK = 1  # lighter than draw_tracks' 2: the contour is the primary shape here
_MAX_POOLS = 4  # matches SubstanceConfig.max_pools; a bound on draw cost, not a policy

# Below this the pool has no perimeter worth tracing (a one-pixel blob), so it gets a plain
# marker instead of an outline that would collapse into a single dot.
_MIN_PERIMETER_PX = 12.0
_DOT_R = 5

# Growth the analyzer must be reporting before the banner says a pool is still spreading.
# Under this, per-frame contour jitter alone would keep the word on screen permanently.
_SPREAD_MIN_FRAC_PER_S = 0.0002


def _points(poly) -> np.ndarray:
    """A polygon as (N,2) int32, accepting either the (N,1,2) OpenCV contour or plain (N,2)."""
    a = np.asarray(poly)
    if a.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    return a.reshape(-1, 2).astype(np.int32)


def _pct(frac: float) -> str:
    """Area as a percentage of the view, floored so a real pool never prints as 0.00%."""
    p = max(0.0, frac) * 100.0
    return "<0.01%" if 0.0 < p < 0.01 else f"{p:.2f}%"


def _resample(pts: np.ndarray, step: float):
    """Points every `step` px around the closed outline, with an outward normal at each.

    Returns None when the outline is too short to sample, which is the one-pixel-pool case.
    """
    closed = np.vstack([pts, pts[:1]]).astype(np.float32)
    seg = closed[1:] - closed[:-1]
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total < max(step, _MIN_PERIMETER_PX):
        return None
    s = np.arange(0.0, total, step, dtype=np.float32)
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    denom = np.maximum(seg_len[idx], 1e-6)
    frac = (s - cum[idx]) / denom
    p = closed[idx] + seg[idx] * frac[:, None]
    d = seg[idx] / denom[:, None]
    n = np.stack([d[:, 1], -d[:, 0]], axis=1)
    # Contour winding differs between an outer contour and a hole, so the normal's sign is
    # decided per sample by pointing it away from the pool's own centre.
    away = ((p - pts.mean(axis=0)) * n).sum(axis=1)
    n *= np.where(away < 0.0, -1.0, 1.0)[:, None]
    return p, n, total


def _fill(img: np.ndarray, pts: np.ndarray, box: tuple[int, int, int, int], alpha: float) -> None:
    """Translucent red inside the contour, blended over the pool's bounding box only.

    Blending the box rather than the whole frame is what keeps four pools inside the frame
    budget: outside the contour the overlay is a copy of the frame, so those pixels blend
    with themselves and come back unchanged.
    """
    x1, y1, x2, y2 = box
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    overlay = roi.copy()
    cv2.fillPoly(overlay, [pts - (x1, y1)], _RED, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)


def _bracket(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color, t: int) -> None:
    """Corner-bracket box, the same treatment draw_tracks puts on a person."""
    L = max(10, int(0.18 * min(x2 - x1, y2 - y1)))
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, t, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, t, cv2.LINE_AA)


def _chip(img: np.ndarray, anchor: tuple[int, int], below: bool, text: str) -> None:
    """The readout, tied to its pool by a short leader.

    Placement flips below the pool when there is no room above it, and the x is clamped, so
    a pool against the top or right edge still gets a legible number instead of a chip half
    off the frame.
    """
    h, w = img.shape[:2]
    k_s = _k(img)
    label_scale = _LABEL_SCALE * k_s
    label_thick = max(1, round(_LABEL_THICK * k_s))
    (tw, th), _ = cv2.getTextSize(text, _FONT, label_scale, label_thick)
    bw, bh = tw + 8, th + 8
    ax, ay = anchor
    x = min(max(0, ax), max(0, w - bw))
    y = (ay + 10) if below else (ay - 10 - bh)
    y = min(max(0, y), max(0, h - bh))
    leader_y = y + bh if below else y
    cv2.line(img, (x + 6, leader_y), (ax, ay), _RED, max(1, round(k_s)), cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + bw, y + bh), _RED, -1)
    cv2.putText(
        img, text, (x + 4, y + bh - 5), _FONT, label_scale, _INK, label_thick, cv2.LINE_AA
    )


def _draw_pool(img: np.ndarray, pts: np.ndarray, label: str, t_now: float) -> None:
    h, w = img.shape[:2]
    bx, by, bw, bh = cv2.boundingRect(pts)
    x1, y1 = max(0, bx), max(0, by)
    x2, y2 = min(w, bx + bw), min(h, by + bh)
    if x2 <= x1 or y2 <= y1:
        return  # entirely off-frame: nothing to say about a pool nobody can see

    # One sine, evaluated from the caller's clock rather than a frame counter, so the phase
    # is the same for every pool and survives a dropped frame.
    alpha = _FILL_ALPHA_LO + (_FILL_ALPHA_HI - _FILL_ALPHA_LO) * (
        0.5 + 0.5 * math.sin(2.0 * math.pi * t_now / _PULSE_S)
    )
    k_s = _k(img)
    outline_thick = max(1, round(_OUTLINE_THICK * k_s))
    sweep_thick = max(2, round(_SWEEP_THICK * k_s))
    tick_px = _TICK_PX * k_s
    tick_len = max(3, round(_TICK_LEN * k_s))
    tick_gap = max(1, round(_TICK_GAP * k_s))
    if len(pts) >= 3:
        _fill(img, pts, (x1, y1, x2, y2), alpha)
        cv2.polylines(img, [pts], True, _RED, outline_thick, cv2.LINE_AA)

    fine = _resample(pts, max(4.0, cv2.arcLength(pts, True) / _SWEEP_STEPS))
    if fine is not None:
        p, _n, total = fine
        k = len(p)
        lit = max(2, int(_SWEEP_FRAC * k))
        start = int((t_now * _SWEEP_PX_S) % max(total, 1.0) / max(total / k, 1e-6))
        arc = p[(start + np.arange(lit)) % k].astype(np.int32)
        cv2.polylines(img, [arc], False, _MEASURE, sweep_thick, cv2.LINE_AA)

    spacing = max(tick_px, cv2.arcLength(pts, True) / _MAX_TICKS)
    ticks = _resample(pts, spacing)
    if ticks is not None:
        p, n, _total = ticks
        segs = np.stack([p + n * tick_gap, p + n * (tick_gap + tick_len)], axis=1).astype(np.int32)
        cv2.polylines(img, list(segs), False, _MEASURE, max(1, outline_thick - 1), cv2.LINE_AA)
    else:
        # A pool one or two pixels across has no outline to trace; mark the spot instead.
        cv2.circle(img, (x1, y1), max(2, round(_DOT_R * k_s)), _RED, outline_thick, cv2.LINE_AA)

    _bracket(img, x1, y1, max(x1 + 1, x2 - 1), max(y1 + 1, y2 - 1), _RED, max(1, round(_BRACKET_THICK * k_s)))
    _chip(img, (x1, y1), below=y1 < 34, text=label)


def draw_substance(img, state, t_now: float) -> None:
    """Draw the confirmed pools onto a BGR display frame, in place.

    Silent for every other state. A candidate is the analyzer still deciding, and painting
    the screen while it decides is what teaches a viewer to distrust the marks that matter.
    """
    if state is None or getattr(state, "state", "idle") != "confirmed":
        return
    polys = getattr(state, "polygons", ()) or ()
    if not polys:
        return
    h, w = img.shape[:2]
    frame_px = float(max(1, h * w))
    held = float(getattr(state, "held_s", 0.0))
    shown = list(polys)[:_MAX_POOLS]
    for i, poly in enumerate(shown, start=1):
        pts = _points(poly)
        if pts.shape[0] == 0:
            continue
        # Per-pool area is measured off the polygon actually drawn, so the number and the
        # shape can never disagree; state.area_frac is the total and belongs in the banner.
        area = cv2.contourArea(pts) if pts.shape[0] >= 3 else 1.0
        tag = f"POOL {i}" if len(shown) > 1 else "POOL"
        _draw_pool(img, pts, f"{tag}  {_pct(area / frame_px)}  {held:0.0f}s", t_now)


def substance_banner(state) -> str | None:
    """The top line for a confirmed pool, or None when there is nothing to say.

    Observational wording only: the detector sees red on the floor and how much of it, and
    saying anything about what the substance is would be a claim it cannot support.
    """
    if state is None or getattr(state, "state", "idle") != "confirmed":
        return None
    if getattr(state, "priming", False):
        return None
    n = int(getattr(state, "n_pools", 0)) or len(getattr(state, "polygons", ()) or ())
    parts = [
        "RED SUBSTANCE ON FLOOR",
        "1 pool" if n == 1 else f"{n} pools",
        f"{_pct(float(getattr(state, 'area_frac', 0.0)))} of view",
        f"{float(getattr(state, 'held_s', 0.0)):0.0f}s",
    ]
    if float(getattr(state, "growth_frac_per_s", 0.0)) > _SPREAD_MIN_FRAC_PER_S:
        parts.append("spreading")
    return "   ".join(parts)
