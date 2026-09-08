"""Harness overlay: boxes, COCO-17 skeletons, track ids, rule flashes, provenance HUD.

Pure over plain dicts and a numpy image so tests need no pipeline: jobs.py converts book
state into the ``scene`` mapping and this module only draws. cv2 is imported inside
functions (package import-discipline rule). draw_frame mutates the image in place and
returns it, so the caller owns the array and hands the same buffer to the MP4 writer and
the websocket JPEG.

Language rule: rule ids are printed exactly as recorded, under a ``CANDIDATE EVENT``
label, and no overlay string may ever claim detection of a medical condition. The claim
wording rule in POLICY.md, section 5 governs every string this module draws.
"""

from __future__ import annotations

import colorsys
import math
from typing import Any

# COCO-17 keypoint order: 0 nose, 1/2 eyes L/R, 3/4 ears L/R, 5/6 shoulders, 7/8 elbows,
# 9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles. scripts/live_camera.py carries a
# near-identical edge list of its own, which joins the shoulders to the nose where this one
# joins them to the ears.
COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),  # legs
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),  # hip bar, torso sides, shoulder bar
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),  # arms
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),  # face + ears-to-shoulders
)

KP_CONF_MIN = 0.3  # the repo's kp_conf_min (TrackConfig default)
ALERT_BGR = (94, 77, 255)  # #ff4d5e
CHIP_TEXT_BGR = (245, 245, 245)
HUD_TEXT_BGR = (226, 232, 235)
HUD_SUB_BGR = (166, 149, 139)  # muted #8b95a6 in BGR
FLASH_S = 3.0  # seconds a firing keeps its track in alert dress


def track_color(track_id: int) -> tuple[int, int, int]:
    """BGR chip color; same golden-angle hue walk as the dashboard's track chips."""
    hue = ((int(track_id) * 137) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.62, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = min(1.0, max(0.0, t))
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def _draw_track(cv2: Any, image: Any, tr: dict[str, Any]) -> None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (round(float(v)) for v in tr["xyxy"])
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return

    flash_t = tr.get("flash")
    if flash_t is not None:
        # 2 Hz brightness pulse for the flash window; deterministic in flash_t
        pulse = 0.55 + 0.45 * math.sin(2.0 * math.pi * 2.0 * float(flash_t))
        color = _lerp((0, 0, 120), ALERT_BGR, pulse)
        thickness = 3
    else:
        color = track_color(tr["track_id"])
        thickness = 2
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

    # id chip in a filled tab above the top-left corner
    label = f"#{tr['track_id']}" + (" STAFF" if tr.get("is_staff") else "")
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    ty1 = max(0, y1 - th - 8)
    cv2.rectangle(image, (x1, ty1), (x1 + tw + 8, ty1 + th + 7), color, -1)
    cv2.putText(
        image,
        label,
        (x1 + 4, ty1 + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        CHIP_TEXT_BGR,
        1,
        cv2.LINE_AA,
    )

    kps = tr.get("keypoints")
    if not kps:
        return
    gray = (110, 110, 110)
    for a, b in COCO_SKELETON:
        xa, ya, ca = kps[a]
        xb, yb, cb = kps[b]
        conf = min(float(ca), float(cb))
        if conf < KP_CONF_MIN:
            continue
        edge_color = _lerp(gray, color, conf)  # low confidence fades toward panel gray
        cv2.line(
            image,
            (round(float(xa)), round(float(ya))),
            (round(float(xb)), round(float(yb))),
            edge_color,
            2,
            cv2.LINE_AA,
        )
    for x, y, c in kps:
        if float(c) < KP_CONF_MIN:
            continue
        cv2.circle(image, (round(float(x)), round(float(y))), 3, color, -1, lineType=cv2.LINE_AA)


ZONE_BLACK = (0, 0, 0)
ZONE_WHITE = (255, 255, 255)
ZONE_DASH_PX = 9
ZONE_GAP_PX = 6


def _dashed_line(
    cv2: Any,
    image: Any,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Dashes along one segment. LINE_8 on purpose: the zone overlay is a strict
    black-and-white surface, and antialiasing would blend gray into it."""
    (x0, y0), (x1, y1) = p0, p1
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist <= 0:
        return
    step = ZONE_DASH_PX + ZONE_GAP_PX
    n = max(1, math.ceil(dist / step))
    for i in range(n):
        t0 = (i * step) / dist
        t1 = min(1.0, (i * step + ZONE_DASH_PX) / dist)
        if t0 >= 1.0:
            break
        a = (round(x0 + (x1 - x0) * t0), round(y0 + (y1 - y0) * t0))
        b = (round(x0 + (x1 - x0) * t1), round(y0 + (y1 - y0) * t1))
        cv2.line(image, a, b, color, thickness)


def _draw_zones(cv2: Any, image: Any, zones: list[dict[str, Any]]) -> None:
    """Exit-zone outlines and labels: strictly white and black, under the monochrome
    overlay rule, with a solid black halo under a white dashed outline so the region
    reads on any footage. Drawn before the tracks so the door region sits under the
    people, never over them."""
    for zone in zones:
        pts = [(round(float(x)), round(float(y))) for x, y in zone.get("polygon", [])]
        if len(pts) < 3:
            continue
        for i, p0 in enumerate(pts):
            p1 = pts[(i + 1) % len(pts)]
            cv2.line(image, p0, p1, ZONE_BLACK, 3)
            _dashed_line(cv2, image, p0, p1, ZONE_WHITE, 1)
        label = f"EXIT ZONE  {zone.get('name', '')}".rstrip()
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        lx = max(0, min(pts, key=lambda p: (p[1], p[0]))[0])
        ly = max(0, min(p[1] for p in pts) - th - 10)
        cv2.rectangle(image, (lx, ly), (lx + tw + 10, ly + th + 8), ZONE_BLACK, -1)
        cv2.putText(
            image,
            label,
            (lx + 5, ly + th + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            ZONE_WHITE,
            1,
        )


def _draw_banners(cv2: Any, image: Any, banners: list[str]) -> None:
    h, w = image.shape[:2]
    y = 0
    strip_h = 26
    for text in banners[:3]:  # more than 3 simultaneous banners is noise, not information
        if y + strip_h > h:
            break
        roi = image[y : y + strip_h, 0:w]
        roi[:] = roi // 3  # darken without an alpha channel
        cv2.rectangle(image, (0, y), (5, y + strip_h), ALERT_BGR, -1)
        cv2.putText(
            image,
            text,
            (12, y + strip_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            ALERT_BGR,
            1,
            cv2.LINE_AA,
        )
        y += strip_h


def _draw_hud(cv2: Any, image: Any, hud: dict[str, str]) -> None:
    h, w = image.shape[:2]
    bar_h = min(40, max(28, h // 12))
    roi = image[h - bar_h : h, 0:w]
    roi[:] = roi // 4  # translucent dark bar
    left = hud.get("left", "")
    right = hud.get("right", "")
    sub = hud.get("sub", "")
    right_x = w - 8
    if right:
        (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        right_x = max(8, w - tw - 8)
        cv2.putText(
            image,
            right,
            (right_x, h - bar_h + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            HUD_TEXT_BGR,
            1,
            cv2.LINE_AA,
        )
    # truncate the left text so it never collides with the right-aligned block
    budget = right_x - 24
    while left and cv2.getTextSize(left, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0] > budget:
        left = left[: max(0, len(left) - 4)].rstrip() + "..." if len(left) > 8 else ""
    if left:
        cv2.putText(
            image,
            left,
            (8, h - bar_h + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            HUD_TEXT_BGR,
            1,
            cv2.LINE_AA,
        )
    if sub:
        cv2.putText(
            image,
            sub,
            (8, h - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            HUD_SUB_BGR,
            1,
            cv2.LINE_AA,
        )


def draw_frame(image: Any, scene: dict[str, Any], hud: dict[str, str]) -> Any:
    """Draw the harness onto ``image`` (HxWx3 uint8 BGR), in place, and return it.

    scene = {"tracks": [{"track_id", "xyxy", "keypoints" (17x(x,y,conf) or None),
                         "is_staff", "flash" (seconds since firing, None when quiet)}],
             "banners": ["CANDIDATE EVENT  <rule_id> - track N - onset Xs", ...],
             "zones": [{"name", "polygon": [[x, y], ...]}, ...]  (optional; egress)}
    hud   = {"left": models line, "right": frame/latency line, "sub": disclaimer}
    """
    import cv2  # heavy: function-local by package rule

    zones = scene.get("zones", [])
    if zones:
        _draw_zones(cv2, image, zones)
    for tr in scene.get("tracks", []):
        _draw_track(cv2, image, tr)
    banners = scene.get("banners", [])
    if banners:
        _draw_banners(cv2, image, banners)
    if hud:
        _draw_hud(cv2, image, hud)
    return image
