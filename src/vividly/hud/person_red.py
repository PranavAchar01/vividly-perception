"""Overlay for the `red_substance_on_person` descriptor: where on the person, how much, how long.

Same split and the same visual language as hud/substance.py, whose primitives it reuses so
the two can never drift apart: `draw_person_red` marks the regions on the display frame and
`person_red_banner` returns the one line the top bar prints. Stateless, and every animated
value comes from the clock the caller passes in.

What it draws on a CONFIRMED region, and why:

  * The outline of the region in the appliance red with a light translucent fill. The fill
    covers exactly the pixels the analyzer scored, so a viewer can see that it is the patch
    on the scalp being measured and not the person's face.
  * White tick marks along the outline and one slow white sweep around it: the measurement
    furniture, in white so that the instrument and the alarm are never the same colour.
  * A red chip tied to the region by a leader, carrying the body part, the size and the age:
    "HEAD  3s". A region on a person is small (a scalp patch is a few dozen display
    pixels across at this distance), so the chip is the part a viewer reads first.

No bracket around the region: the person already carries draw_tracks' box, and a second set
of corners on something twenty pixels wide is clutter, not information.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from vividly.hud.substance import (
    _FILL_ALPHA_HI,
    _FILL_ALPHA_LO,
    _MAX_TICKS,
    _MEASURE,
    _OUTLINE_THICK,
    _PULSE_S,
    _RED,
    _SPREAD_MIN_FRAC_PER_S,
    _SWEEP_FRAC,
    _SWEEP_PX_S,
    _SWEEP_STEPS,
    _SWEEP_THICK,
    _TICK_GAP,
    _TICK_LEN,
    _chip,
    _fill,
    _k,
    _points,
    _resample,
)

# Ticks are spaced tighter than on a floor pool: a region on a person is a fraction of a
# pool's perimeter, and at the pool's 22 px a scalp patch would get two ticks.
_TICK_PX = 12.0
# The fill is lighter than a pool's: the region sits on a face or a scalp, and a viewer must
# still be able to see what is under it.
_FILL_SCALE = 0.35  # light: the blood under it is the thing to see
_DOT_R = 5
_MAX_REGIONS = 3  # matches PersonRedConfig.max_regions; a bound on draw cost


def _draw_region(img: np.ndarray, pts: np.ndarray, label: str | None, t_now: float) -> None:
    h, w = img.shape[:2]
    bx, by, bw, bh = cv2.boundingRect(pts)
    x1, y1 = max(0, bx), max(0, by)
    x2, y2 = min(w, bx + bw), min(h, by + bh)
    if x2 <= x1 or y2 <= y1:
        return
    k_s = _k(img)
    alpha = _FILL_SCALE * (
        _FILL_ALPHA_LO
        + (_FILL_ALPHA_HI - _FILL_ALPHA_LO)
        * (0.5 + 0.5 * math.sin(2.0 * math.pi * t_now / _PULSE_S))
    )
    outline = max(1, round(_OUTLINE_THICK * k_s))
    if len(pts) >= 3:
        _fill(img, pts, (x1, y1, x2, y2), alpha)
        cv2.polylines(img, [pts], True, _RED, outline, cv2.LINE_AA)

    fine = _resample(pts, max(3.0, cv2.arcLength(pts, True) / _SWEEP_STEPS))
    if fine is not None:
        p, _n, total = fine
        n = len(p)
        lit = max(2, int(_SWEEP_FRAC * n))
        start = int((t_now * _SWEEP_PX_S) % max(total, 1.0) / max(total / n, 1e-6))
        arc = p[(start + np.arange(lit)) % n].astype(np.int32)
        cv2.polylines(img, [arc], False, _MEASURE, max(2, round(_SWEEP_THICK * k_s)), cv2.LINE_AA)

    ticks = _resample(pts, max(_TICK_PX * k_s, cv2.arcLength(pts, True) / _MAX_TICKS))
    gap, length = max(1, round(_TICK_GAP * k_s)), max(3, round(_TICK_LEN * k_s))
    if ticks is not None:
        p, nrm, _total = ticks
        segs = np.stack([p + nrm * gap, p + nrm * (gap + length)], axis=1).astype(np.int32)
        cv2.polylines(img, list(segs), False, _MEASURE, max(1, outline - 1), cv2.LINE_AA)
    else:
        # Too small to trace: mark the spot so the chip still has something to point at.
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
        cv2.circle(img, (cx, cy), max(3, round(_DOT_R * k_s)), _MEASURE, outline, cv2.LINE_AA)

    if label is not None:
        # The leader lands on the top edge of the region, where the chip above it points.
        top = pts[np.argmin(pts[:, 1])]
        anchor = (int(top[0]), int(top[1]))
        _chip(img, anchor, below=anchor[1] < round(40 * k_s), text=label)


def draw_person_red(img, state, t_now: float) -> None:
    """Draw every CONFIRMED region onto a BGR display frame, in place. Silent otherwise: a
    candidate is the analyzer still deciding, and the screen stays quiet until it is sure."""
    hits = getattr(state, "hits", ()) if state is not None else ()
    if not hits:
        return
    for hit in hits:
        if getattr(hit, "state", "") != "confirmed":
            continue
        polys = [p for p in (getattr(hit, "polygons", ()) or ()) if np.asarray(p).size]
        if not polys:
            continue
        shown = sorted(
            (_points(p) for p in polys), key=lambda q: -cv2.contourArea(q) if len(q) >= 3 else 0.0
        )
        shown = shown[:_MAX_REGIONS]
        # Part and age only. The area share of the frame was a measurement, not information:
        # at this distance a bleeding scalp is a few hundredths of a percent of the picture,
        # and "<0.01%" on screen read as the system saying there was almost nothing there.
        label = f"{hit.part.upper()}  {float(hit.held_s):0.0f}s"
        # One chip per person, on the largest region; the others are outlined only.
        for i, pts in enumerate(shown):
            _draw_region(img, pts, label if i == 0 else None, t_now)


def person_red_banner(state) -> str | None:
    """The top line for a CONFIRMED hit, or None. Observational wording only.

    With more than one confirmed person, the one that has held longest wins, since it is the
    one the operator has been not-looking-at longest. "spreading" appears only while the
    analyzer reports real growth, under the same threshold the floor banner uses.
    """
    hits = getattr(state, "hits", ()) if state is not None else ()
    conf = [hh for hh in hits if getattr(hh, "state", "") == "confirmed"]
    if not conf:
        return None
    hit = max(conf, key=lambda hh: float(hh.held_s))
    parts = [
        "RED SUBSTANCE ON PERSON",
        hit.part,
        f"person #{hit.tid}",
        f"{float(hit.held_s):0.0f}s",
    ]
    if float(hit.growth_frac_per_s) > _SPREAD_MIN_FRAC_PER_S:
        parts.append("spreading")
    return "   ".join(parts)
