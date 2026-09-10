"""Composite a spreading pool of red liquid into a clip, so the substance detector has a
true positive to be scored against before real spilled footage exists.

    uv run python scripts/substance_synth.py \
        --in takes/cuts/03-bleed.mp4 --out /tmp/03-bleed-spill.mp4 \
        --start 12 --grow 6 --seed 20260909

The pool starts at --start seconds, spreads over --grow seconds and then holds, which is
exactly the shape the detector's novelty + persistence gates are built to catch. Rendering
is deterministic in --seed, and the final outline plus the appearance time are written to a
sidecar JSON beside the output so an evaluator can score delay and area against them.

Why an implicit field and not cv2.ellipse: a flat filled ellipse has a hard edge, a flat
interior and one colour, and every one of those reads as a sticker rather than a liquid.
Here a scalar field carries the thickness of the liquid, and the shading, the opacity and
the soft edge are all read off that field, so the pool darkens where it is deep, thins to
translucency at the rim, and catches a highlight the way a wet surface does.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

SIDECAR_SCHEMA = "vividly.substance_synth/1"

# Blood on a bright garage floor under fluorescent light: nearly black where it pools deep,
# a saturated red where it thins out. Blue sits just under green in both so the hue lands a
# few degrees off pure red (OpenCV H around 3) instead of straddling the 0/179 wrap, which
# would split the detector's two hue bands for no reason.
# The garage floor sits between V 65 and V 90 in this footage, and a pool of blood is
# markedly darker than the floor it lies on: without that luminance drop the composite reads
# as red paint. Only the deepest core, about 6 percent of the area, falls under the
# detector's val_min of 60, so this stays a fair positive rather than one brightened to pass.
CENTER_BGR = np.array([11.0, 15.0, 52.0], np.float32)
EDGE_BGR = np.array([32.0, 42.0, 100.0], np.float32)
MAX_ALPHA = 0.97  # deep liquid still lets a trace of floor through
# Surface tension holds a real edge to a couple of millimetres, so the translucent rim is a
# few percent of the radius and no more. Anything wider renders as airbrushed fog.
EDGE_SOFT = 0.035


def _ease_out(u: float) -> float:
    """Spread fast, then settle. A poured volume covers ground quickest at the start and
    slows as it thins toward the edge, so radius must not grow linearly."""
    return 1.0 - (1.0 - u) ** 2.0


class Pool:
    """One liquid pool, rendered from an implicit field over its own bounding box only."""

    def __init__(
        self,
        w: int,
        h: int,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        rng: np.random.Generator,
        persp: float = 0.30,
        n_harm: int = 5,
    ) -> None:
        self.cx, self.cy, self.rx, self.ry = cx, cy, rx, ry
        # Padding leaves room for the soft rim and for lobes that push past the nominal
        # radius; nothing outside this box is ever touched, which is what keeps the whole
        # composite cheap on a 2304x1296 frame.
        pad = 1.5
        self.x0 = max(0, int(cx - rx * pad))
        self.x1 = min(w, int(cx + rx * pad) + 1)
        self.y0 = max(0, int(cy - ry * pad))
        self.y1 = min(h, int(cy + ry * pad) + 1)
        ys, xs = np.mgrid[self.y0 : self.y1, self.x0 : self.x1]
        dx = (xs - cx).astype(np.float32)
        dy = (ys - cy).astype(np.float32)

        # The floor recedes upward in this camera, so a fixed physical width covers fewer
        # pixels the further up the pool reaches. Widening the near (lower) half by persp is
        # what makes the shape sit ON the ground instead of looking pasted on a wall.
        self.u = dx / (rx * (1.0 + persp * dy / ry))
        self.v = dy / ry
        theta = np.arctan2(self.v, self.u)

        # Low-order harmonics only. Orders 2..6 give an outline that is irregular but still
        # closed and smooth, which is what surface tension actually produces; higher orders
        # would look like noise on the rim.
        dev = np.zeros_like(theta, np.float32)
        for k in range(2, 2 + n_harm):
            amp = rng.uniform(0.05, 0.13) * (2.0 / k) ** 1.3
            dev += amp * np.cos(k * theta + rng.uniform(0.0, 2.0 * math.pi))
        self.dev = np.clip(dev, -0.35, 0.45).astype(np.float32)

        # Spatter. A spill throws satellite droplets ahead of the front, and they are the
        # detail that most separates a spill from a shape someone drew: they sit at fixed
        # ground positions, so the pool overruns the nearest of them as it spreads.
        self.sats = [
            (
                float(r * math.cos(a)),
                float(r * math.sin(a)),
                float(rng.uniform(0.05, 0.11)),
            )
            for r, a in (
                (rng.uniform(1.02, 1.45), rng.uniform(0.0, 2.0 * math.pi))
                for _ in range(int(rng.integers(3, 6)))
            )
        ]

        # A faint low-frequency thickness texture, so the interior is not a smooth dome.
        small = rng.random((6, 12), np.float32) - 0.5
        self.texture = cv2.resize(
            small, (self.u.shape[1], self.u.shape[0]), interpolation=cv2.INTER_CUBIC
        ).astype(np.float32)

        # Downhill creep: a settling pool keeps inching toward the camera long after it has
        # stopped spreading sideways, so the centroid drifts down by a fraction of ry.
        self.drift_max = 0.10

    def _fields(self, s: float, w: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(normalized distance, thickness, alpha) for a pool grown to radius scale s.

        w scales the outline's irregularity: a small pool is held round by surface tension
        and only develops lobes as it spreads, so the harmonics fade in with the growth.
        """
        v = self.v - self.drift_max * w
        d = np.hypot(self.u, v) / np.maximum(s * (1.0 + w * self.dev), 1e-3)
        # Droplets are ground truth too: fold them into the same field with a min, so one
        # contour comes back out and the shading treats a merged droplet as pool.
        for su, sv, sr in self.sats:
            np.minimum(d, np.hypot(self.u - su, v - sv) / (sr * max(s, 0.35)), out=d)

        thick = np.clip(1.0 - d, 0.0, 1.0) ** 0.65
        thick = np.clip(thick * (1.0 + 0.16 * self.texture), 0.0, 1.0)
        alpha = np.clip((1.0 - d) / EDGE_SOFT, 0.0, 1.0)

        # Blur in pixels, not in normalized units, so the rim stays a believable couple of
        # pixels wide at every size instead of scaling with the pool.
        sigma = max(0.7, 0.006 * self.rx * s)
        k = int(sigma * 4) | 1
        thick = cv2.GaussianBlur(thick, (k, k), sigma)
        alpha = cv2.GaussianBlur(alpha, (k, k), sigma)
        # Thin liquid is translucent and deep liquid is not, so opacity tracks thickness.
        alpha = alpha * (0.80 + 0.20 * thick) * MAX_ALPHA
        return d, thick, alpha

    def paint(self, frame: np.ndarray, s: float, w: float, t: float) -> None:
        """Composite the pool at growth (s, w) into frame in place."""
        d, thick, alpha = self._fields(s, w)
        base = frame[self.y0 : self.y1, self.x0 : self.x1].astype(np.float32)

        col = EDGE_BGR + (CENTER_BGR - EDGE_BGR) * thick[..., None]
        # Depth varies over the pool, so brightness must vary with it; a perfectly even
        # interior is the other half of what makes a composite look like a decal.
        col = col * (1.0 + 0.13 * self.texture)[..., None]
        # The meniscus stands proud of the pool and so is optically deeper than the film
        # just inside it: a real pool is outlined in a darker line, not fringed in a lighter
        # one, and that line is most of what stops the shape reading as a decal.
        band = np.exp(-(((d - 0.95) / 0.045) ** 2))
        col = col * (1.0 - 0.40 * band)[..., None]
        a3 = alpha[..., None]
        out = base * (1.0 - a3) + col * a3

        # Specular sheen. The ceiling strip is up and to the left of this floor patch, so a
        # tight highlight sits off-centre that way, with a second stretched reflection of the
        # strip nearer the camera; a 0.7 Hz sub-pixel wander is what separates a wet surface
        # from painted-on colour. The rim term is the light the meniscus catches, and it is
        # the single strongest wetness cue available.
        wob = 0.012 * math.sin(2.0 * math.pi * 0.7 * t)
        vv = self.v - self.drift_max * w
        gu = (self.u - (-0.26 + wob) * s) / (0.17 * s)
        gv = (vv - (-0.34) * s) / (0.045 * s)
        spec = np.exp(-0.5 * (gu * gu + gv * gv)) * 42.0
        hu = (self.u - (0.26 - wob) * s) / (0.20 * s)
        hv = (vv - 0.22 * s) / (0.035 * s)
        spec += np.exp(-0.5 * (hu * hu + hv * hv)) * 12.0
        rim = np.exp(-(((d - 0.92) / 0.030) ** 2)) * 38.0
        out += (spec + rim)[..., None] * a3

        frame[self.y0 : self.y1, self.x0 : self.x1] = np.clip(out, 0, 255).astype(np.uint8)

    def polygon(self, s: float = 1.0, w: float = 1.0) -> np.ndarray:
        """Outline of the pool at growth (s, w), in display pixel coords, as (N,2) int32."""
        _, _, alpha = self._fields(s, w)
        mask = (alpha > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((0, 2), np.int32)
        c = max(contours, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 0.003 * cv2.arcLength(c, True), True)
        pts = c.reshape(-1, 2).astype(np.int32)
        pts[:, 0] += self.x0
        pts[:, 1] += self.y0
        return pts

    def area_px(self, s: float = 1.0, w: float = 1.0) -> float:
        _, _, alpha = self._fields(s, w)
        return float((alpha > 0.5).sum())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out", dest="dst", required=True, type=Path)
    ap.add_argument("--start", type=float, default=12.0, help="seconds; pool first appears")
    ap.add_argument("--grow", type=float, default=6.0, help="seconds; spread to stable size")
    ap.add_argument("--seed", type=int, default=20260909)
    # Defaults put the pool on the open floor just right of the chair in the garage takes:
    # near the subject, clear of the body, and well below the detector's floor_y_frac line.
    ap.add_argument("--center-frac", default="0.547,0.910", help="pool centre, x,y of frame")
    ap.add_argument("--radius-frac", type=float, default=0.063, help="pool half-width / width")
    ap.add_argument(
        "--squash",
        type=float,
        default=0.42,
        help="half-height / half-width; foreshortening of a round pool on this floor",
    )
    args = ap.parse_args(argv)

    src = args.src if args.src.is_absolute() else REPO_ROOT / args.src
    dst = args.dst if args.dst.is_absolute() else REPO_ROOT / args.dst
    dst.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    cfx, cfy = (float(v) for v in args.center_frac.split(","))
    rx = args.radius_frac * w
    rng = np.random.default_rng(args.seed)
    pool = Pool(w, h, cfx * w, cfy * h, rx, rx * args.squash, rng)

    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"cannot write {dst}")

    growth: list[dict] = []
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = n / fps
        u = (t - args.start) / args.grow if args.grow > 0 else 1.0
        if t >= args.start:
            u = min(max(u, 0.0), 1.0)
            e = _ease_out(u)
            # A visible starting size, so the first frame of the spill is a splash rather
            # than a single pixel that no detector could or should see.
            s = 0.10 + 0.90 * e
            shape_w = 0.30 + 0.70 * e
            pool.paint(frame, s, shape_w, t)
            if n % max(1, round(fps / 2)) == 0:
                a = pool.area_px(s, shape_w)
                growth.append(
                    {"t": round(t, 3), "scale": round(s, 4), "area_px": a, "area_frac": a / (w * h)}
                )
        writer.write(frame)
        n += 1
    cap.release()
    writer.release()

    poly = pool.polygon()
    area = pool.area_px()
    sidecar = dst.with_suffix(".substance.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema": SIDECAR_SCHEMA,
                "source": str(src),
                "output": str(dst),
                "seed": args.seed,
                "video": {"width": w, "height": h, "fps": fps, "n_frames": n},
                "appearance_s": args.start,
                "grow_s": args.grow,
                "stable_s": args.start + args.grow,
                "pool": {
                    "center_xy": [cfx * w, cfy * h],
                    "radius_xy": [rx, rx * args.squash],
                    "polygon": poly.tolist(),
                    "area_px": area,
                    "area_frac": area / (w * h),
                },
                "growth": growth,
            },
            indent=1,
        )
    )
    print(f"wrote {dst} ({n} frames, {w}x{h} @ {fps:.2f} fps)")
    print(f"wrote {sidecar}")
    print(
        f"pool appears at {args.start:.2f}s, stable at {args.start + args.grow:.2f}s, "
        f"final area_frac {area / (w * h):.5f} ({int(area)} px, {len(poly)}-point polygon)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
