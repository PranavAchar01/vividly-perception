"""Composite a scalp laceration onto a real take, the way a film compositor would, locally.

    uv run python scripts/bleed_synth.py --in takes/20260909-205301-bleed.mp4 \\
        --out bleed-head.mp4 --json bleed-head.json --onset 34.0 --end 50.0 --seed 7

Everything runs on this machine: the repo's own detector and RTMPose for the skeleton,
OpenCV's DIS flow and guided filter, numpy and numba. No frame leaves the process.

What it does, in the order a compositor would:

1. TRACK. Detector + RTMPose over the window, zero-phase One Euro smoothing of every joint.
   The right hand is rotoscoped as keyframes on the tracked wrist (HAND_TIP_KEYS; COCO-17
   has no hand joints). Then every surface that carries blood (head, right hand, right
   forearm, left hand) gets its own planar track: a similarity per frame, fitted with RANSAC
   to the forward/backward-checked DIS flow (MEDIUM, full resolution) of the pixels of that
   part's own visible matte, chained from a reference frame. The blood on a part lives in
   that part's canonical coordinates, so it rides the part rigidly (the hand can move across
   the face without dragging the face's blood with it), and it is resampled exactly once per
   frame, when it is drawn.
2. MATTES. A clean background from an early and a late stretch of the take (re-registered
   to the chair, which is nudged ~2 px when the person lands on it); a shadow-rejecting
   difference matte gives the person; the parts come from capsules and colour classes
   refined edge-aware with a guided filter against the plate. Each part has a SUPPORT (where
   the surface is, including under something in front of it) and a VISIBILITY. Supports are
   smoothed over time in the part's own coordinates, so a matte edge cannot shimmer.
3. BLOOD. The wound is where the hand lands: the wrist that sits on the head over 35-36 s,
   90% of the way to its fingertips, in head coordinates. The palm is pressed on it, so the
   blood runs between palm and scalp (a hidden smear, revealed if the hand moves), oozes out
   all along the palm's rim below the wound (a wet margin on the skin round the hand), and
   leaves the rim in a fan of channels, mostly straight down, onto whatever surface is
   there; while the fingers point up some seeps between them and runs down the back of the
   hand to the wrist. Rivulets are Lagrangian heads on a part's canvas: they run in image +y
   (gravity, carried into each part's coordinates) at 1-3 cm/s scaled by their volume
   against contact-line pinning (px/mm from the measured head width), meander along a smooth
   noise field fixed to the surface, steer into wet channels, and lay down a height field: a
   residual film (max-combined) and the depth of the live stream behind them, which drains
   when the feed stops; a leading bead rides at each tip, and one too small to overcome
   pinning stops as an elongated tip on its trail. Blood leaves a surface only at a lower
   edge: onto a part it touches, round a silhouette out of sight, or as a pendant that lets
   go of one drop at a time (irregular cadence). Lying down, the face rests on the cupped
   left palm: there the blood spreads as one sheet (a thin-film flow on the palm's hollow,
   tipped toward its lowest edge) and leaves over that edge, at one spout, onto the seat.
   The seat is a plane (a homography fitted to its corners, so area is real area): the
   upholstery absorbs up to a capacity per area and wicks by a porous-medium law with a
   heterogeneous, weave-anisotropic diffusivity (a sharp, fibrous front advancing like
   sqrt(t), with a paler diluted fringe); the rivulet arriving off the palm keeps going as
   one channel down the seat's forward tilt, soaking the fabric along it, over the front
   edge and down the painted rim as at most two rivulets, which drip. Drops fall under
   gravity (drawn over the shutter) and land as one spot with a crown, then grow one small
   drip puddle whose area follows its volume.
4. SHADE, in linear light, at several instants across the measured shutter and averaged,
   so blood on a moving part carries the same motion blur as the plate under it. The plate
   is multiplied by exp(-sigma_rgb * h) (green and blue absorbed ~8-10x more than red),
   plus the blood's body reflectance, a Fresnel environment term and a narrow specular lobe
   from the key light estimated off the head's own shading, all scaled by the local plate
   illumination (level wet surfaces also mirror the ceiling fixtures); fabric-held blood
   darkens with saturation, is matte and lets the weave show through; wet black hair shows
   its own strand highlights clumped and sharpened.
5. MATCH THE PLATE. Blood never goes darker than the plate's black level; its chroma is held
   inside the red envelope this camera actually produces (measured per luma band, and
   checked again on the final 4:2:0 samples against the darkest luma each one covers); the
   plate's fine detail is partly restored under the film and the luma noise topped back up
   to the measured noise-vs-level curve, chroma noise likewise; the chroma delta goes
   through the same 4:2:0 subsampling and softness as the plate. Only a delta on the decoded
   YUV planes is applied, so every other pixel is the decoded source.
6. ENCODE with libx264, yuv420p, GOP 20 and no B-frames like the camera, at the source's
   bitrate, every frame carrying the source's own PTS. No audio. A JSON sidecar gives the
   onset, the wound, per-frame visible-blood boxes and areas by part, events, every param.

Blood on the person is drawn through --end. The stains on the chair and the floor are then
kept, unchanged, through the end of the clip (--no-persist to stop them at --end too): a
stain does not vanish when the effect window closes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
import zlib
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]

# COCO-17 indices
NOSE, LEYE, REYE, LEAR, REAR, LSHO, RSHO, LELB, RELB, LWRI, RWRI = range(11)
HEAD_KP = (NOSE, LEYE, REYE, LEAR, REAR)


# --------------------------------------------------------------------------------------------
# source probing
# --------------------------------------------------------------------------------------------


@dataclass
class Source:
    path: Path
    width: int
    height: int
    n_frames: int
    fps: float  # the presentation clock the rest of the repo uses: t = index / fps
    pts: np.ndarray  # container PTS in time_base units, one per frame, decode order == display
    time_base: Fraction
    bit_rate: int


def probe(path: Path) -> Source:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        pts = [p.pts for p in c.demux(s) if p.pts is not None]
        fps = float(s.average_rate)
        pts_arr = np.sort(np.array(pts, np.int64))
        return Source(
            path=path,
            width=s.codec_context.width,
            height=s.codec_context.height,
            n_frames=len(pts_arr),
            fps=fps,
            pts=pts_arr,
            time_base=s.time_base,
            bit_rate=int(s.bit_rate or c.bit_rate),
        )


def iter_yuv(path: Path):
    """Yield (index, Y, U, V) uint8 planes in decode order (no B-frames in these takes)."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for i, f in enumerate(c.decode(s)):
            a = f.to_ndarray(format="yuv420p")
            h = f.height
            w = f.width
            y = a[:h]
            u = a[h : h + h // 4].reshape(h // 2, w // 2)
            v = a[h + h // 4 :].reshape(h // 2, w // 2)
            yield i, y, u, v


# --------------------------------------------------------------------------------------------
# colour: BT.709 limited range <-> RGB, sRGB transfer
# --------------------------------------------------------------------------------------------

KR, KB = 0.2126, 0.0722
KG = 1.0 - KR - KB


def yuv_to_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Planes at the same size, float32 code values -> RGB in [0,1] nominal (not clipped)."""
    yy = (y.astype(np.float32) - 16.0) / 219.0
    cb = (u.astype(np.float32) - 128.0) / 224.0
    cr = (v.astype(np.float32) - 128.0) / 224.0
    r = yy + 2.0 * (1.0 - KR) * cr
    b = yy + 2.0 * (1.0 - KB) * cb
    g = (yy - KR * r - KB * b) / KG
    return np.stack([r, g, b], -1)


def rgb_to_yuv(rgb: np.ndarray):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    yy = KR * r + KG * g + KB * b
    cb = (b - yy) / (2.0 * (1.0 - KB))
    cr = (r - yy) / (2.0 * (1.0 - KR))
    return 16.0 + 219.0 * yy, 128.0 + 224.0 * cb, 128.0 + 224.0 * cr


def srgb_to_lin(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(
        np.float32
    )


# --------------------------------------------------------------------------------------------
# pose: detector + RTMPose over the window, cached; then zero-phase One Euro smoothing
# --------------------------------------------------------------------------------------------


def run_pose(src: Source, i0: int, i1: int, cache: Path) -> dict:
    """Keypoints for frames i0..i1 inclusive. The cache is reused when it covers the range."""
    if cache.exists():
        d = np.load(cache)
        idx = d["idx"]
        if idx.min() <= i0 and idx.max() >= i1:
            sel = (idx >= i0) & (idx <= i1)
            return {"idx": idx[sel], "k": d["k"][sel], "s": d["s"][sel], "b": d["b"][sel]}
    import yaml

    sys.path.insert(0, str(REPO / "src"))
    from vividly.models.factory import build_detector, build_pose

    cfg = yaml.safe_load((REPO / "configs/eval/live_final.yaml").read_text())
    det = build_detector(dict(cfg["detector"]))
    pose = build_pose(dict(cfg["pose"]))
    K, S, B, idx = [], [], [], []
    prev_c = None
    with av.open(str(src.path)) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i < i0:
                continue
            if i > i1:
                break
            im = f.to_ndarray(format="bgr24")
            d = det.detect_one(im).filter(0.5)
            idx.append(i)
            if len(d) == 0:
                K.append(np.full((17, 2), np.nan, np.float32))
                S.append(np.zeros(17, np.float32))
                B.append(np.full(4, np.nan, np.float32))
                continue
            xy = d.xyxy.astype(np.float32)
            area = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
            if prev_c is None:
                j = int(np.argmax(area))
            else:  # stay on the same person: nearest box centre to the last one
                ctr = np.stack([(xy[:, 0] + xy[:, 2]) / 2, (xy[:, 1] + xy[:, 3]) / 2], 1)
                j = int(np.argmin(np.linalg.norm(ctr - prev_c, axis=1) - 1e-4 * area))
            prev_c = np.array([(xy[j, 0] + xy[j, 2]) / 2, (xy[j, 1] + xy[j, 3]) / 2])
            p = pose.estimate(im, xy[j : j + 1])
            K.append(p.keypoints[0].astype(np.float32))
            S.append(p.scores[0].astype(np.float32))
            B.append(xy[j])
    out = {"idx": np.array(idx), "k": np.array(K), "s": np.array(S), "b": np.array(B)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def _one_euro(x: np.ndarray, valid: np.ndarray, fps: float, fmin: float, beta: float):
    """Causal One Euro filter over a (T, 2) track; invalid samples hold the prediction."""
    out = np.zeros_like(x)
    te = 1.0 / fps
    xh = None
    dxh = np.zeros(2)
    for t in range(len(x)):
        if xh is None:
            if valid[t]:
                xh = x[t].copy()
            out[t] = xh if xh is not None else np.nan
            continue
        if not valid[t]:
            out[t] = xh
            continue
        dx = (x[t] - xh) / te
        a_d = 1.0 / (1.0 + (1.0 / (2 * math.pi * 1.0)) / te)
        dxh = a_d * dx + (1 - a_d) * dxh
        fc = fmin + beta * float(np.linalg.norm(dxh))
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * fc)) / te)
        xh = a * x[t] + (1 - a) * xh
        out[t] = xh
    return out


def smooth_keypoints(k: np.ndarray, s: np.ndarray, fps: float, min_score: float = 0.3):
    """Zero-phase One Euro: run forward and backward and average, so a still pose is held
    tight (0.7 Hz cutoff) and a slump is followed without lag (cutoff rises with speed)."""
    T = len(k)
    out = np.full_like(k, np.nan)
    for j in range(17):
        x = k[:, j].astype(np.float64)
        v = (s[:, j] >= min_score) & np.isfinite(x).all(1)
        if v.sum() < 2:
            continue
        # fill invalid samples by linear interpolation first so both passes see a track
        ti = np.arange(T)
        xf = np.stack([np.interp(ti, ti[v], x[v, c]) for c in range(2)], 1)
        f = _one_euro(xf, v | True, fps, 0.7, 0.012)
        b = _one_euro(xf[::-1], v[::-1] | True, fps, 0.7, 0.012)[::-1]
        out[:, j] = 0.5 * (f + b)
    return out.astype(np.float32)


# --------------------------------------------------------------------------------------------
# small image helpers
# --------------------------------------------------------------------------------------------


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def capsule(shape, a, b, ra: float, rb: float) -> np.ndarray:
    """Signed-distance capsule with a linearly tapered radius, as a 0..1 matte (1 px AA)."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    ab = b - a
    L2 = float(ab @ ab) + 1e-6
    t = np.clip(((xs - a[0]) * ab[0] + (ys - a[1]) * ab[1]) / L2, 0.0, 1.0)
    px = a[0] + t * ab[0]
    py = a[1] + t * ab[1]
    d = np.sqrt((xs - px) ** 2 + (ys - py) ** 2) - (ra + (rb - ra) * t)
    return np.clip(0.5 - d, 0.0, 1.0)


def disk(shape, c, r: float) -> np.ndarray:
    return capsule(shape, c, c, r, r)


def guided(guide: np.ndarray, src: np.ndarray, r: int = 3, eps: float = 2e-3) -> np.ndarray:
    return np.clip(
        cv2.ximgproc.guidedFilter(guide.astype(np.float32), src.astype(np.float32), r, eps),
        0.0,
        1.0,
    )


def sample(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear sample of a 2-D (or H,W,C) array at float (x,y) points; clamps at borders."""
    if len(pts) == 0:
        return np.zeros((0, *img.shape[2:]), np.float32)
    m = pts.astype(np.float32).reshape(-1, 1, 2)
    out = cv2.remap(
        img.astype(np.float32),
        m[..., 0],
        m[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return out.reshape((len(pts), *img.shape[2:]))


def norm_conv(field: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Normalized convolution: fill a (H,W,2) field from the pixels a weight map trusts."""
    k = int(sigma * 3) | 1
    w = cv2.GaussianBlur(weight, (k, k), sigma) + 1e-4
    out = np.empty_like(field)
    for c in range(field.shape[2]):
        out[..., c] = cv2.GaussianBlur(field[..., c] * weight, (k, k), sigma) / w
    return out


# --------------------------------------------------------------------------------------------
# pass A: the window's plates in memory, clean background, plate statistics
# --------------------------------------------------------------------------------------------


@dataclass
class Roi:
    x0: int
    y0: int
    w: int
    h: int

    @property
    def sl(self):
        return slice(self.y0, self.y0 + self.h), slice(self.x0, self.x0 + self.w)

    @property
    def sl2(self):
        return (
            slice(self.y0 // 2, (self.y0 + self.h) // 2),
            slice(self.x0 // 2, (self.x0 + self.w) // 2),
        )


class Plates:
    """ROI planes of every frame in [i0, i1], kept in memory (about 0.5 MB a frame)."""

    def __init__(self, src: Source, roi: Roi, i0: int, i1: int):
        self.roi, self.i0, self.i1 = roi, i0, i1
        self.Y, self.U, self.V = {}, {}, {}
        ys, xs = roi.sl
        ys2, xs2 = roi.sl2
        for i, y, u, v in iter_yuv(src.path):
            if i < i0:
                continue
            if i > i1:
                break
            self.Y[i] = y[ys, xs].copy()
            self.U[i] = u[ys2, xs2].copy()
            self.V[i] = v[ys2, xs2].copy()

    def rgb(self, i: int) -> np.ndarray:
        """Gamma-encoded RGB float in [0,1] nominal at ROI full resolution."""
        h, w = self.Y[i].shape
        u = cv2.resize(self.U[i], (w, h), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(self.V[i], (w, h), interpolation=cv2.INTER_LINEAR)
        return yuv_to_rgb(self.Y[i], u, v)

    def gray(self, i: int) -> np.ndarray:
        return np.ascontiguousarray(self.Y[i])


def measure_noise(plates: Plates, frames: list[int], still_mask: np.ndarray) -> dict:
    """Temporal std of Y in static, flat regions, binned by mean Y (code values)."""
    stack = np.stack([plates.Y[i].astype(np.float32) for i in frames])
    m = stack.mean(0)
    s = stack.std(0)
    gx = cv2.Sobel(m, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(m, cv2.CV_32F, 0, 1)
    flat = (np.hypot(gx, gy) < 12.0) & still_mask
    edges = np.arange(0, 256 + 16, 16)
    centers, sig = [], []
    for lo, hi in itertools.pairwise(edges):
        sel = flat & (m >= lo) & (m < hi)
        if sel.sum() > 300:
            centers.append(0.5 * (lo + hi))
            sig.append(float(np.median(s[sel])))
    # chroma temporal std too
    su = np.stack([plates.U[i].astype(np.float32) for i in frames]).std(0)
    return {
        "y_centers": centers,
        "y_sigma": sig,
        "c_sigma": float(np.median(su)),
    }


def measure_blur(src: Source, frames: list[int], box=(950, 540, 1150, 585)) -> float:
    """Gaussian sigma of the plate's edge spread, measured on a hard edge: the bottom of the
    dark door windows against the white frame. For every column the step is located, its low
    and high levels are read just outside it, and the 10-90 percent rise distance is found
    by interpolation; sigma = rise / 2.563, median over columns. The camera's sharpening
    overshoot sits outside the 10-90 band and does not bias it."""
    acc = None
    n = 0
    want = set(frames)
    for i, y, _u, _v in iter_yuv(src.path):
        if i in want:
            acc = y.astype(np.float64) if acc is None else acc + y
            n += 1
        if i >= max(frames):
            break
    m = acc / max(n, 1)
    x0, y0, x1, y1 = box
    band = m[y0:y1, x0:x1]
    rises = []
    for c in range(band.shape[1]):
        col = band[:, c]
        g = np.diff(col)
        r = int(np.argmax(g))
        if g[r] < 60 or r < 9 or r > len(col) - 5:
            continue
        lo = float(np.median(col[r - 9 : r - 3]))
        hi = float(np.max(col[r + 1 : r + 4]))
        hi = float(np.median(col[r + 1 : r + 4])) if hi - lo > 0 else hi
        seg = col[r - 3 : r + 4]
        xs = np.arange(r - 3, r + 4, dtype=np.float64)
        nrm = (seg - lo) / (hi - lo + 1e-9)
        try:
            j10 = next(k for k in range(1, len(nrm)) if nrm[k - 1] < 0.1 <= nrm[k])
            j90 = next(k for k in range(1, len(nrm)) if nrm[k - 1] < 0.9 <= nrm[k])
        except StopIteration:
            continue
        x10 = xs[j10 - 1] + (0.1 - nrm[j10 - 1]) / (nrm[j10] - nrm[j10 - 1])
        x90 = xs[j90 - 1] + (0.9 - nrm[j90 - 1]) / (nrm[j90] - nrm[j90 - 1])
        rises.append(x90 - x10)
    if len(rises) < 10:
        return 0.55
    return float(np.clip(np.median(rises) / 2.563, 0.3, 2.0))


def measure_gamut(src: Source, frame: int) -> dict:
    """How red this camera ever renders anything, per luma band: the 99.9th percentile of Cr
    (and of |Cb|) over a whole frame. In this garage the reddest object is a dark red ladder
    (Y ~37, Cr ~+37); nothing brighter gets past Cr ~+22. Blood is then held inside that
    envelope, because a composite redder than anything the sensor produced reads as paint."""
    for i, y_, u_, v_ in iter_yuv(src.path):
        if i == frame:
            y, u, v = y_, u_, v_
            break
    uu = cv2.resize(u, (y.shape[1], y.shape[0])).astype(np.float32) - 128
    vv = cv2.resize(v, (y.shape[1], y.shape[0])).astype(np.float32) - 128
    cen, lv, lu = [], [], []
    for lo in range(0, 256, 16):
        sel = (y >= lo) & (y < lo + 16)
        if sel.sum() < 500:
            continue
        cen.append(lo + 8.0)
        lv.append(float(max(np.percentile(vv[sel], 99.9), 8.0)))
        lu.append(float(max(np.percentile(np.abs(uu[sel]), 99.9), 8.0)))
    return {"y": cen, "cr_max": lv, "cb_abs_max": lu}


# --------------------------------------------------------------------------------------------
# set pieces of this take, measured on the clean plate (full-resolution pixel coordinates)
# --------------------------------------------------------------------------------------------

# The right-hand folding chair: the fabric top of the seat, and the painted rim below its
# front edge that blood runs over before it drips. Measured at 6x zoom on the 31 s plate,
# where the chair is empty; the chair is then tracked per frame (it is nudged ~2 px when
# the person lands on it) and the polygons are snapped to the plate's own edges by a guided
# filter, so they only have to be right to a pixel or two.
SEAT_FABRIC = [
    (945, 1080), (958, 1063), (979, 1050), (1077, 1057), (1084, 1067),
    (1078, 1086), (1073, 1090), (948, 1084),
]  # fmt: skip
SEAT_RIM = [
    (941, 1086), (948, 1084), (1073, 1090), (1078, 1086), (1084, 1068), (1087, 1076),
    (1082, 1098), (1068, 1105), (1018, 1102), (947, 1096),
]  # fmt: skip
# Legs region used to measure the chair's drift (no person ever covers the lower legs).
CHAIR_LEGS_BOX = (950, 1112, 1110, 1225)  # x0, y0, x1, y1
# The whole right chair, where the clean plate is re-registered to the chair's drift.
CHAIR_BOX = (932, 960, 1118, 1232)
# Floor directly below the seat's front edge: the front feet stand on the floor 112 px
# below the rim, i.e. the rim is ~45 cm up at ~2.5 px/cm of vertical foreshortening.
SEAT_EDGE_TO_FLOOR_PX = 112.0
FLOOR_TOP_Y = 1112  # floor visible below the seat from here down
# The seat top's four corners (back-left, back-right, front-right, front-left), for a
# homography from a 38 x 38 cm seat plane: ~3.4 px/cm across but under 1 px/cm in depth.
SEAT_QUAD = [(972, 1051), (1081, 1058), (1077, 1089), (946, 1084)]

# Roto keyframes for the right hand. RTMPose-m COCO-17 has no hand keypoints, and the hand
# bends at the wrist (it points straight up the cheek while the forearm points up and to
# the right), so the hand axis is keyframed the way a compositor rotos a limb: at each key
# time the visible fingertip end of the hand was read off a 4x zoom, relative to the
# smoothed wrist, in full-resolution pixels. Keys are converted to forearm-local coordinates
# and interpolated, so between keys the hand rides on the tracked wrist and elbow.
HAND_TIP_KEYS = [  # (t_s, dx, dy) or (t_s, None, None) for "straight along the forearm"
    (33.45, None, None), (33.8, 7, -48), (34.0, 4, -43), (34.5, 2.5, -41), (35.25, 11, -41),
    (36.0, 8, -36), (36.5, 17, -30), (36.8, 31, -30), (37.2, 30, -33), (37.6, 25, -33),
    (38.0, 32, -35), (38.4, 37, -33), (38.8, 45, 7), (39.2, 42, 17), (40.0, 37, 10),
    (44.0, 45, 10), (48.3, 45, 12), (48.6, None, None), (49.0, -38, 30), (49.6, None, None),
]  # fmt: skip
HAND_STRAIGHT = 0.62  # hand length / forearm length when no key applies


# --------------------------------------------------------------------------------------------
# tracks: keypoints, head, hand, chair
# --------------------------------------------------------------------------------------------


class Tracks:
    def __init__(self, src: Source, pose: dict, roi: Roi):
        self.fps = src.fps
        self.roi = roi
        self.idx = {int(i): n for n, i in enumerate(pose["idx"])}
        self.raw = pose["k"]
        self.score = pose["s"]
        self.k = smooth_keypoints(pose["k"], pose["s"], src.fps)
        # head scale from upright frames with both ears confident
        ee = np.linalg.norm(self.raw[:, LEAR] - self.raw[:, REAR], axis=1)
        good = (self.score[:, LEAR] > 0.8) & (self.score[:, REAR] > 0.8)
        self.ear_px = float(np.median(ee[good])) if good.sum() > 5 else 48.0
        self.head_w_px = 1.3 * self.ear_px  # full head breadth vs the ear-to-ear span
        self.px_per_mm = self.head_w_px / 180.0
        self.head_r = 0.5 * self.head_w_px * 1.08
        self._hand_keys()

    def kp(self, i: int) -> np.ndarray:
        """Smoothed keypoints of frame i in ROI coordinates."""
        return self.k[self.idx[i]] - np.array([self.roi.x0, self.roi.y0], np.float32)

    def kp_score(self, i: int) -> np.ndarray:
        return self.score[self.idx[i]]

    def _hand_keys(self):
        ts, loc = [], []
        for t, dx, dy in HAND_TIP_KEYS:
            i = round(t * self.fps)
            if i not in self.idx:
                continue
            k = self.k[self.idx[i]]
            e, w = k[RELB], k[RWRI]
            L = float(np.linalg.norm(w - e)) + 1e-6
            u = (w - e) / L
            n = np.array([-u[1], u[0]])
            if dx is None:
                loc.append((HAND_STRAIGHT, 0.0))
            else:
                d = np.array([dx, dy], np.float32)
                loc.append((float(d @ u) / L, float(d @ n) / L))
            ts.append(t)
        self.hand_t = np.array(ts)
        self.hand_loc = np.array(loc) if loc else np.zeros((0, 2))

    def hand(self, i: int):
        """(elbow, wrist, tip) in ROI coordinates for the right arm."""
        k = self.kp(i)
        e, w = k[RELB], k[RWRI]
        L = float(np.linalg.norm(w - e)) + 1e-6
        u = (w - e) / L
        n = np.array([-u[1], u[0]], np.float32)
        t = i / self.fps
        if len(self.hand_t) and self.hand_t[0] <= t <= self.hand_t[-1]:
            a = float(np.interp(t, self.hand_t, self.hand_loc[:, 0]))
            b = float(np.interp(t, self.hand_t, self.hand_loc[:, 1]))
        else:
            a, b = HAND_STRAIGHT, 0.0
        tip = w + (a * u + b * n) * L
        return e.astype(np.float32), w.astype(np.float32), tip.astype(np.float32)

    def head_kp_center(self, i: int) -> np.ndarray:
        k = self.kp(i)
        s = self.kp_score(i)
        sel = [j for j in HEAD_KP if s[j] > 0.3]
        if not sel:
            sel = list(HEAD_KP)
        return k[sel].mean(0)


def skeleton_mask(tr: Tracks, i: int, shape, head_c=None, pad: float = 0.0) -> np.ndarray:
    """Coarse person region from the smoothed skeleton: generous capsules, used only to
    confine the difference matte (so shadows and plate noise elsewhere never count)."""
    k = tr.kp(i)
    m = np.zeros(shape, np.float32)
    limbs = [(LSHO, LELB), (LELB, LWRI), (RSHO, RELB), (RELB, RWRI), (LSHO, RSHO),
             (11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (LSHO, 11), (RSHO, 12)]  # fmt: skip
    for a, b in limbs:
        m = np.maximum(m, capsule(shape, k[a], k[b], 30 + pad, 30 + pad))
    torso = np.array([k[LSHO], k[RSHO], k[12], k[11]], np.int32)
    cv2.fillConvexPoly(m, torso, 1.0)
    _e, w, tip = tr.hand(i)
    m = np.maximum(m, capsule(shape, w, tip, 22 + pad, 22 + pad))
    hc = tr.head_kp_center(i) if head_c is None else head_c
    m = np.maximum(m, disk(shape, hc, 2.0 * tr.head_r + pad))
    m = np.maximum(m, disk(shape, tr.head_kp_center(i), 1.6 * tr.head_r + pad))
    return m


def poly_mask(shape, pts, off, roi: Roi, ss: int = 4) -> np.ndarray:
    """Anti-aliased polygon (supersampled) in ROI coordinates, shifted by off=(dx,dy)."""
    h, w = shape
    p = (np.array(pts, np.float64) - [roi.x0, roi.y0] + np.asarray(off)) * ss
    big = np.zeros((h * ss, w * ss), np.uint8)
    cv2.fillPoly(big, [np.round(p).astype(np.int32)], 255)
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def luminance(lin: np.ndarray) -> np.ndarray:
    return (KR * lin[..., 0] + KG * lin[..., 1] + KB * lin[..., 2]).astype(np.float32)


PARTS = ("rhand", "head", "cloth", "seat", "floor", "lhand", "rfore", "rim")
RHAND, HEAD, CLOTH, SEAT, FLOOR, LHAND, RFORE, RIM = range(8)
NPART = len(PARTS)
BODY = (
    HEAD,
    RHAND,
    RFORE,
    LHAND,
)  # surfaces that move, each with its own tracked transform
# Which surface blood can run onto (at the lower edge of the one it is on) because the two are
# actually in contact. The right forearm is held in the air, so blood on it runs to its low
# point and drips; the painted rim under the seat's front edge only drips.
CONTACT = {
    RHAND: (HEAD, RFORE, SEAT),
    HEAD: (LHAND, SEAT, CLOTH),
    LHAND: (SEAT, CLOTH),
    RFORE: (),
    RIM: (),
}


class Scene:
    """Clean background, chair drift, the person matte and the per-part mattes."""

    def __init__(self, plates: Plates, tr: Tracks, early: tuple, late: tuple, log):
        self.p, self.tr, self.roi = plates, tr, plates.roi
        fps = tr.fps
        h, w = plates.Y[plates.i0].shape
        self.shape = (h, w)
        f1 = [i for i in range(round(early[0] * fps), round(early[1] * fps) + 1, 2)]
        f2 = [i for i in range(round(late[0] * fps), round(late[1] * fps) + 1, 2)]
        c1 = np.median(np.stack([plates.rgb(i) for i in f1]), 0).astype(np.float32)
        c2 = np.median(np.stack([plates.rgb(i) for i in f2]), 0).astype(np.float32)
        m1 = np.max(np.stack([skeleton_mask(tr, i, self.shape, pad=10) for i in f1]), 0) > 0.5
        m2 = np.max(np.stack([skeleton_mask(tr, i, self.shape, pad=10) for i in f2]), 0) > 0.5
        self.c1 = c1
        self.bg = np.where((m1 & ~m2)[..., None], c2, c1).astype(np.float32)
        self.bg_unknown = m1 & m2
        self.bg_lin = srgb_to_lin(self.bg)
        log(
            f"clean plate: early {len(f1)} frames, late {len(f2)} frames, "
            f"{self.bg_unknown.mean() * 100:.1f}% of the ROI never uncovered"
        )
        # chair drift against the early plate, measured on the legs
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.g1 = np.ascontiguousarray((luminance(c1) * 255).astype(np.uint8))
        frames = sorted(plates.Y)
        offs = []
        for i in frames:
            sk = skeleton_mask(tr, i, self.shape, pad=6) < 0.5
            offs.append(self.chair_drift(plates.gray(i), sk))
        offs = np.array(offs)
        k = 9
        pad = np.pad(offs, ((k // 2, k // 2), (0, 0)), mode="edge")
        sm = np.array([np.median(pad[j : j + k], 0) for j in range(len(offs))])
        self.chair_off = {i: sm[n].astype(np.float32) for n, i in enumerate(frames)}
        log(f"chair drift: max |off| {np.abs(sm).max(0).round(2).tolist()} px")
        # floor: thin dark structures (chair legs, crossbars) are not floor
        lum = luminance(srgb_to_lin(c1))
        close = cv2.morphologyEx(lum, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        thin = smoothstep(0.03, 0.08, close - lum)
        fl_region = np.zeros(self.shape, np.float32)
        fl_region[max(FLOOR_TOP_Y - self.roi.y0, 0) :, :] = 1.0
        self.floor_static = guided(c1, fl_region * (1.0 - thin), 2, 1e-3)
        # fabric texture of the seat (high-pass of the clean plate): the weave shows through
        lum_g = luminance(c1)
        self.fabric_tex = (lum_g - cv2.GaussianBlur(lum_g, (0, 0), 2.0)).astype(np.float32)
        self.head_c: dict[int, np.ndarray] = {}

    def chair_drift(self, gray: np.ndarray, free: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = (np.array(CHAIR_LEGS_BOX) - [self.roi.x0, self.roi.y0] * 2).tolist()
        fl = self.dis.calc(self.g1, gray, None)
        f = fl[y0:y1, x0:x1][free[y0:y1, x0:x1]]
        return np.median(f, 0) if len(f) > 200 else np.zeros(2)

    def background(self, i: int, off=None) -> np.ndarray:
        off = self.chair_off.get(i, np.zeros(2, np.float32)) if off is None else off
        if float(np.abs(off).max()) < 0.05:
            return self.bg
        x0, y0, x1, y1 = (np.array(CHAIR_BOX) - [self.roi.x0, self.roi.y0] * 2).tolist()
        M = np.float32([[1, 0, off[0]], [0, 1, off[1]]])
        sh = cv2.warpAffine(
            self.c1,
            M,
            (self.shape[1], self.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        out = self.bg.copy()
        box = np.zeros(self.shape, bool)
        box[y0:y1, x0:x1] = True
        box &= ~self.bg_unknown
        out[box] = sh[box]
        return out

    def diff_matte(self, rgb, lin, bg) -> np.ndarray:
        """Shadow-rejecting difference against the clean plate: anything that is not the
        background, where a cast shadow (every channel scaled by one factor) does not count."""
        d = np.abs(rgb - bg).max(-1)
        d = cv2.GaussianBlur(d, (0, 0), 1.0)
        bl = srgb_to_lin(bg)
        ratio = (lin + 0.004) / (bl + 0.004)
        rm = ratio.mean(-1)
        spread = (ratio.max(-1) - ratio.min(-1)) / (rm + 1e-3)
        shadow = (
            smoothstep(0.25, 0.12, spread) * smoothstep(0.18, 0.3, rm) * smoothstep(0.97, 0.9, rm)
        )
        return smoothstep(0.06, 0.15, d) * (1.0 - shadow)

    def person(self, i: int, rgb: np.ndarray, lin: np.ndarray) -> np.ndarray:
        p = self.diff_matte(rgb, lin, self.background(i))
        p *= skeleton_mask(self.tr, i, self.shape, self.head_c.get(i), pad=14)
        p = cv2.morphologyEx(p, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return guided(rgb, p, 3, 1e-3)

    def occluder(self, rgb: np.ndarray, lin: np.ndarray, off) -> np.ndarray:
        """After the effect window (no skeleton any more): whatever is in front of the chair
        and the floor under it, from the difference matte alone, inside the chair box."""
        p = self.diff_matte(rgb, lin, self.background(-1, off))
        x0, y0, x1, y1 = (np.array(CHAIR_BOX) - [self.roi.x0, self.roi.y0] * 2).tolist()
        box = np.zeros(self.shape, np.float32)
        box[y0 - 20 : y1 + 60, x0 - 20 : x1 + 20] = 1.0
        p = cv2.morphologyEx(p * box, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        p = cv2.dilate(p, np.ones((5, 5), np.uint8))
        return guided(rgb, p, 3, 1e-3)

    @staticmethod
    def navy(lin: np.ndarray) -> np.ndarray:
        """The navy shirt. Black hair and the navy cotton are both near black here and both
        read slightly blue, so the test is on gamma-encoded chroma, where the shirt sits
        ~28 codes bluer than red and the hair ~9."""
        g = lin_to_srgb(lin)
        return smoothstep(0.05, 0.095, g[..., 2] - g[..., 0]) * smoothstep(0.4, 0.25, luminance(g))

    @staticmethod
    def skinlike(rgb: np.ndarray) -> np.ndarray:
        """Skin in this light: red over green over blue by a margin, and not near black."""
        g = np.clip(rgb, 0, 1)
        r, gg, b = g[..., 0], g[..., 1], g[..., 2]
        warm = (
            smoothstep(0.05, 0.1, r - b)
            * smoothstep(-0.02, 0.03, r - gg)
            * smoothstep(-0.03, 0.02, gg - b)
        )
        return warm * smoothstep(0.1, 0.17, luminance(g))

    def hairlike(self, lin: np.ndarray) -> np.ndarray:
        return smoothstep(0.03, 0.010, luminance(lin)) * (1.0 - self.navy(lin))

    def track_head(self, log):
        """Head centre per frame (used for the head disk the mattes are built in): the median
        dense flow inside the head disk (the hand excluded) carries the centre from frame to
        frame, and a complementary filter pulls it toward the keypoint head only while those
        keypoints are confident (they slide onto the forearm once the person lies face down
        on it). Then zero-phase smoothed."""
        frames = sorted(self.p.Y)
        R = self.tr.head_r
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        c = self.tr.head_kp_center(frames[0]).astype(np.float64)
        raw = [c.copy()]
        prev = self.p.gray(frames[0])
        for i in frames[1:]:
            g = self.p.gray(i)
            fl = dis.calc(prev, g, None)
            e, w, tip = self.tr.hand(i - 1)
            hand = np.maximum(capsule(self.shape, e, w, 11, 8), capsule(self.shape, w, tip, 11, 9))
            m = disk(self.shape, c, R) * (1.0 - hand)
            f = fl[m > 0.5]
            if len(f) > 50:
                c = c + np.median(f, 0)
            s = self.tr.kp_score(i)
            conf = float(smoothstep(0.72, 0.9, np.array(np.mean(s[list(HEAD_KP)]))))
            c = c + 0.2 * conf * (self.tr.head_kp_center(i) - c)
            raw.append(c.copy())
            prev = g
        raw = np.array(raw, np.float64)
        v = np.ones(len(raw), bool)
        f = _one_euro(raw, v, self.tr.fps, 1.0, 0.01)
        b = _one_euro(raw[::-1], v[::-1], self.tr.fps, 1.0, 0.01)[::-1]
        sm = 0.5 * (f + b)
        self.head_c = {i: sm[n].astype(np.float32) for n, i in enumerate(frames)}
        log("head track done")

    def left_hand(self, i: int):
        """(elbow, wrist, tip) of the left arm. The hand is not a COCO joint; it continues the
        forearm by ~0.5 of its length (it lies curled on the seat, palm up)."""
        k = self.tr.kp(i)
        le, lw = k[LELB].astype(np.float32), k[LWRI].astype(np.float32)
        return le, lw, (lw + 0.5 * (lw - le)).astype(np.float32)

    def mattes(self, i: int, rgb: np.ndarray, lin: np.ndarray) -> dict:
        tr = self.tr
        sh = self.shape
        P = self.person(i, rgb, lin)
        e, w, tip = tr.hand(i)
        mid = w + 0.55 * (tip - w)
        hand_c = np.maximum(capsule(sh, w, mid, 8.5, 11.5), capsule(sh, mid, tip, 11.5, 7.0))
        fore_c = capsule(sh, e, w, 10.5, 8.0)
        rh = guided(rgb, hand_c * P, 3, 1e-3)
        # an occluder is opaque once it is more than half there: a guided matte is soft
        # inside textured skin, and blood under the hand must not show through it
        rh_o = smoothstep(0.2, 0.55, rh)
        rf = guided(rgb, fore_c * P, 3, 1e-3) * (1.0 - rh_o)
        hc = self.head_c.get(i, tr.head_kp_center(i))
        nav = self.navy(lin)
        skin = self.skinlike(rgb)
        lumg = luminance(np.clip(rgb, 0, 1))
        # The left hand and the end of the left forearm: lit skin near the left wrist. (Lying
        # on the seat the left elbow keypoint wanders, so no capsule on the arm: the hand is
        # the palest skin in the picture and the face resting on it is in shadow.)
        le, lw, ltip = self.left_hand(i)
        lh = disk(sh, lw, 42.0) * P * skin * smoothstep(0.2, 0.3, lumg) * (1.0 - nav)
        lh = guided(rgb, lh, 3, 1e-3) * (1.0 - rh_o) * (1.0 - smoothstep(0.2, 0.55, rf))
        # Skin anywhere near the head (face, neck) is a surface blood runs on; near-black
        # pixels count only inside a tight disk (that is the hair), because the navy shirt in
        # shadow is just as dark and blood soaks into cotton instead of running over it.
        dark = smoothstep(0.16, 0.09, lumg)
        tight = disk(sh, hc, 1.3 * tr.head_r)
        reach = disk(sh, hc, 2.3 * tr.head_r)
        # black hair is crushed below the video black level (Y < 13) in this camera, the
        # shadowed shirt next to it sits just above it (Y 13-25): outside the tight disk only
        # the crushed blacks count as hair
        crushed = smoothstep(-0.008, -0.02, luminance(rgb))
        surf = np.maximum(skin * (1.0 - nav), (1.0 - nav) * np.maximum(dark * tight, crushed))
        hd_sup = reach * np.maximum(P * surf, hand_c * P * tight)
        hd_sup = cv2.morphologyEx(hd_sup, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        hd_sup = guided(rgb, hd_sup, 3, 1e-3) * (1.0 - rf) * (1.0 - lh)
        hd = hd_sup * (1.0 - rh_o)
        cl_sup = np.clip(np.maximum(P, hand_c) - hd_sup - lh - rf - rh, 0.0, 1.0)
        cl = cl_sup * (1.0 - rh)
        off = self.chair_off.get(i, np.zeros(2, np.float32))
        fab = poly_mask(sh, SEAT_FABRIC, off, self.roi)
        rim = poly_mask(sh, SEAT_RIM, off, self.roi)
        seat_full = guided(rgb, np.clip(fab + rim, 0, 1), 2, 1e-3)
        s_fab = seat_full * fab / np.maximum(fab + rim, 1e-3)
        s_rim = seat_full - s_fab
        floor_sup = self.floor_static * (1.0 - seat_full)
        sup = np.stack([rh, hd_sup, cl_sup, s_fab, floor_sup, lh, rf, s_rim]).astype(np.float32)
        vis = np.stack(
            [rh, hd, cl, s_fab * (1 - P), floor_sup * (1 - P), lh, rf, s_rim * (1 - P)]
        ).astype(np.float32)
        return {
            "P": P,
            "m": vis,
            "sup": sup,
            "seat_full": seat_full,
            "hair": self.hairlike(lin) * hd_sup,
            "lum": lumg,
            "navy": nav,
            "head_c": hc,
            "hand": (e, w, tip),
            "lhand": (le, lw, ltip),
            "off": off,
        }


# --------------------------------------------------------------------------------------------
# numba kernels: capsule stamps into height fields, and exposure-smeared drop splats
# --------------------------------------------------------------------------------------------

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is installed in the harness

    def njit(*a, **k):
        def wrap(f):
            return f

        return wrap if not (a and callable(a[0])) else a[0]


@njit(cache=True)
def _stamp(grid, ax, ay, bx, by, ra, rb, h, expo, add):
    """Stamp a tapered capsule (radius ra at a, rb at b) into a 2-D height field, with the
    cross-section of a rivulet: h * (1 - (d/r)^2)^expo, a cap that meets the surface at a
    contact angle instead of fading out. add=0 takes the max (a residual film does not pile
    up where a path crosses itself), add=1 accumulates."""
    H, W = grid.shape
    rm = max(ra, rb)
    x0 = math.floor(min(ax, bx) - rm - 1.0)
    x1 = math.ceil(max(ax, bx) + rm + 1.0)
    y0 = math.floor(min(ay, by) - rm - 1.0)
    y1 = math.ceil(max(ay, by) + rm + 1.0)
    dx = bx - ax
    dy = by - ay
    L2 = dx * dx + dy * dy
    for y in range(max(y0, 0), min(y1 + 1, H)):
        for x in range(max(x0, 0), min(x1 + 1, W)):
            t = 0.0
            if L2 > 1e-9:
                t = ((x - ax) * dx + (y - ay) * dy) / L2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
            px = ax + t * dx - x
            py = ay + t * dy - y
            r = ra + (rb - ra) * t
            if r < 0.35:
                r = 0.35
            d2 = (px * px + py * py) / (r * r)
            if d2 < 1.0:
                v = h * (1.0 - d2) ** expo
                if add == 1:
                    grid[y, x] += v
                elif v > grid[y, x]:
                    grid[y, x] = v


@njit(cache=True)
def _splat(grid, xs, ys, dxs, dys, sig, amp, ox, oy, ss):
    """Add Gaussian blobs into grid (ss-supersampled). (xs, ys) is the blob centre at
    mid-exposure in ROI pixel coordinates; (dxs, dys) is its travel during the exposure,
    along which the blob is smeared exactly as the shutter smears it."""
    H, W = grid.shape
    for k in range(xs.shape[0]):
        s = sig[k] * ss
        if s < 0.35:
            s = 0.35
        L = math.hypot(dxs[k], dys[k]) * ss
        n = int(L / max(0.6 * s, 0.5)) + 2
        if n > 64:
            n = 64
        a = amp[k] / n
        rad = int(3.0 * s) + 2
        inv = 1.0 / (2.0 * s * s)
        for j in range(n):
            f = (j + 0.5) / n - 0.5 if n > 1 else 0.0
            cx = (xs[k] + dxs[k] * f - ox + 0.5) * ss - 0.5
            cy = (ys[k] + dys[k] * f - oy + 0.5) * ss - 0.5
            ix = int(cx)
            iy = int(cy)
            for y in range(iy - rad, iy + rad + 1):
                if y < 0 or y >= H:
                    continue
                ddy = (y - cy) * (y - cy)
                for x in range(ix - rad, ix + rad + 1):
                    if x < 0 or x >= W:
                        continue
                    ddx = (x - cx) * (x - cx)
                    grid[y, x] += a * math.exp(-(ddx + ddy) * inv)


def value_noise(shape, scale: float, rng: np.random.Generator, aniso=(1.0, 1.0)) -> np.ndarray:
    """Smooth band-limited noise in [-1, 1] at a feature size of `scale` px."""
    h, w = shape
    gh = max(int(h / (scale * aniso[1])) + 3, 3)
    gw = max(int(w / (scale * aniso[0])) + 3, 3)
    g = rng.standard_normal((gh, gw)).astype(np.float32)
    n = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(n / (np.abs(n).max() + 1e-6), -1, 1)


# --------------------------------------------------------------------------------------------
# tracking: one similarity transform per body part per frame, from the plate's dense flow
# --------------------------------------------------------------------------------------------

I23 = np.float32([[1, 0, 0], [0, 1, 0]])


def compose(D: np.ndarray, A: np.ndarray) -> np.ndarray:
    """D after A, both 2x3."""
    return (np.vstack([D, [0, 0, 1]]) @ np.vstack([A, [0, 0, 1]]))[:2].astype(np.float32)


def apply(A: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    return (pts @ A[:, :2].T + A[:, 2]).astype(np.float32)


def sim_from_segments(a0, b0, a1, b1) -> np.ndarray:
    """The similarity that carries segment a0-b0 onto a1-b1."""
    v0 = np.asarray(b0, np.float64) - a0
    v1 = np.asarray(b1, np.float64) - a1
    z = complex(v1[0], v1[1]) / (complex(v0[0], v0[1]) + 1e-9)
    a, b = z.real, z.imag
    t = np.asarray(a1, np.float64) - np.array([a * a0[0] - b * a0[1], b * a0[0] + a * a0[1]])
    return np.float32([[a, -b, t[0]], [b, a, t[1]]])


class PartTrack:
    """canonical -> ROI image similarity per frame. The canonical frame of a part is the ROI
    image at its reference frame. Blood lives in canonical coordinates, so it is carried
    rigidly by the part it is on and is resampled exactly once, when it is drawn."""

    def __init__(self, T: dict, ref: int):
        self.T = T
        self.ref = ref

    def M(self, i: int, tau: float = 0.0) -> np.ndarray:
        if i not in self.T:
            i = min(self.T, key=lambda j: abs(j - i))
        if abs(tau) < 1e-6:
            return self.T[i]
        j = i + (1 if tau > 0 else -1)
        if j not in self.T:
            return self.T[i]
        f = abs(tau)
        # a similarity's (a, b, tx, ty) interpolate linearly for inter-frame motion
        return ((1 - f) * self.T[i] + f * self.T[j]).astype(np.float32)

    def scale(self, i: int) -> float:
        A = self.M(i)
        return float(math.hypot(A[0, 0], A[1, 0]))

    def g_can(self, i: int) -> np.ndarray:
        """Image gravity (0, +1) expressed as a unit direction in canonical coordinates."""
        A = self.M(i)
        a, b = float(A[0, 0]), float(A[1, 0])
        n = math.hypot(a, b) + 1e-9
        return np.array([b / n, a / n], np.float32)

    def inv(self, i: int, pts) -> np.ndarray:
        return apply(cv2.invertAffineTransform(self.M(i)), pts)


def fit_increment(flow, fb, mask, fallback, rng):
    """Similarity from frame i to i+1, fitted robustly to the flow of the pixels of one part
    that pass the forward/backward check. Falls back to the skeleton's own motion."""
    ys, xs = np.nonzero(mask & (fb < 0.8))
    if len(xs) < 40:
        return fallback, 0
    if len(xs) > 2500:
        sel = rng.choice(len(xs), 2500, replace=False)
        ys, xs = ys[sel], xs[sel]
    P = np.stack([xs, ys], 1).astype(np.float32)
    Q = P + flow[ys, xs]
    A, inl = cv2.estimateAffinePartial2D(
        P,
        Q,
        method=cv2.RANSAC,
        ransacReprojThreshold=0.8,
        maxIters=2000,
        confidence=0.995,
    )
    if A is None or int(inl.sum()) < 25:
        return fallback, 0
    return A.astype(np.float32), int(inl.sum())


# --------------------------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------------------------


@dataclass
class Params:
    seed: int = 7
    onset: float = 34.0
    end: float = 50.0
    flow_ml_s: float = 1.05  # a brisk scalp laceration under a pressing hand (~17 ml)
    initial_boost: float = 1.0  # a fresh laceration bleeds hardest in its first seconds
    boost_tau_s: float = 2.5
    ramp_s: float = 0.8
    pulse_hz: float = 1.25
    pulse_amp: float = 0.2
    under_frac: float = 0.10  # trapped between palm and scalp (a hidden smear)
    dorsal_frac: float = 0.22  # out between the fingers onto the back of the hand
    bead_ml: float = 0.07  # volume a rivulet head is released with
    v_max_mm_s: float = 30.0  # thick stream on skin (spec: 1-3 cm/s)
    v_pin_dry: float = 5.0  # mm^3: contact-line pinning on dry skin
    v_pin_wet: float = 1.6  # mm^3: on an already wetted channel
    v_full: float = 70.0  # mm^3: at or above this a bead runs at v_max
    film_dry_mm: float = 0.06  # residual film a drained rivulet leaves on dry skin
    film_wet_mm: float = 0.012  # extra where the channel is already wet
    hair_soak: float = 2.6  # hair holds this much more than skin per path length
    live_tau_s: float = 3.0  # a fed stream drains to its residual film this fast
    drip_mm3: float = 40.0  # a pendant lets go at this volume
    seat_c_sat_mm: float = 0.22  # blood the upholstery's top layer holds (0.022 ml/cm^2)
    seat_run_mm_s: float = 90.0  # a rivulet of excess down the loaded seat's forward tilt
    seat_pin_mm: float = 0.08  # standing liquid deeper than this leaves as a rivulet
    seat_D_mm2_s: float = 45.0  # wicking diffusivity at saturation (a drop's stain in ~2 s)
    seat_abs_tau_s: float = 3.0  # standing liquid soaks into the fabric under it this fast
    seat_spread_mm2_s: float = 12.0  # a standing puddle on the seat spreads sideways
    px_per_mm_seat: float = 0.34  # across the seat, at the chair's depth (seat ~38 cm wide)
    floor_fore: float = 0.3  # floor/seat foreshortening in depth
    g_px_s2: float = 2443.0  # 9.81 m/s^2 at 0.249 px/mm vertical (chair legs)
    shutter: float = 0.55  # exposure / frame interval, from the plate's own motion smear
    # Whole blood, double pass (in to the surface and back out), camera RGB bands. Oxy-
    # haemoglobin absorbs about as much in the blue band as in green (its Soret and Q bands)
    # and ~1/10 of that at the red band's centre, so a thin film is scarlet, never magenta.
    sigma_rgb: tuple = (2.4, 19.0, 23.0)
    rho_rgb: tuple = (
        0.035,
        0.0025,
        0.0022,
    )  # diffuse body reflectance of a thick layer
    env_reflect: float = 0.6  # Fresnel reflection of the (grey, bright) garage surroundings
    spec_k: float = 0.3
    spec_alpha: float = 60.0
    ss: int = 3
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# canvases: height fields that live in a body part's canonical coordinates
# --------------------------------------------------------------------------------------------


class Canvas:
    """Blood on one body part, as height fields (mm) on a supersampled grid in the part's
    canonical coordinates: `res` the residual film a rivulet leaves (max-combined, it never
    moves again), `live` the depth of a stream while it is fed (drains exponentially), and
    `bead` the leading beads of this frame. Fields only ever gain or drain mass; they are
    never re-thresholded or re-advected, so nothing boils from frame to frame."""

    def __init__(self, part: int, box, ss: int, rng: np.random.Generator):
        self.part = part
        self.x0, self.y0, self.x1, self.y1 = (int(v) for v in box)
        self.ss = ss
        self.W = (self.x1 - self.x0) * ss
        self.H = (self.y1 - self.y0) * ss
        sh = (self.H, self.W)
        self.res = np.zeros(sh, np.float32)
        self.live = np.zeros(sh, np.float32)
        self.bead = np.zeros(sh, np.float32)
        self.sup = None  # support, temporally smoothed in canonical space
        self.hair = np.zeros(sh, np.float32)
        self.wander = [value_noise(sh, 9.0 * ss, rng), value_noise(sh, 3.0 * ss, rng)]
        self.S = np.float32(
            [
                [1.0 / ss, 0, self.x0 + 0.5 / ss - 0.5],
                [0, 1.0 / ss, self.y0 + 0.5 / ss - 0.5],
            ]
        )

    def c2i(self, T: np.ndarray) -> np.ndarray:
        return compose(T, self.S)

    def to_c(self, pts) -> np.ndarray:
        p = np.asarray(pts, np.float32).reshape(-1, 2)
        return ((p - [self.x0, self.y0] + 0.5) * self.ss - 0.5).astype(np.float32)

    def pull(self, img: np.ndarray, T: np.ndarray) -> np.ndarray:
        """An image-space map resampled into this canvas at transform T."""
        return cv2.warpAffine(
            img.astype(np.float32),
            self.c2i(T),
            (self.W, self.H),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def update_support(self, sup_img, hair_img, T, a=0.5):
        s = self.pull(sup_img, T)
        self.sup = s if self.sup is None else (1 - a) * self.sup + a * s
        if hair_img is not None:
            self.hair = (1 - a) * self.hair + a * self.pull(hair_img, T)

    def at(self, arr: np.ndarray, pts) -> np.ndarray:
        return sample(arr, self.to_c(pts))

    def height(self) -> np.ndarray:
        return self.res + self.live + self.bead

    def push(self, T: np.ndarray, shape, win, arr=None):
        """Draw this canvas into the ROI window (x0, y0, x1, y1) at transform T, prefiltered
        so the 3x supersampled field is area-averaged, not point-sampled."""
        a = self.height() if arr is None else arr
        a = cv2.GaussianBlur(a, (0, 0), 0.45 * self.ss)
        x0, y0, x1, y1 = win
        M = self.c2i(T).copy()
        M[0, 2] -= x0
        M[1, 2] -= y0
        return cv2.warpAffine(
            a,
            M,
            (x1 - x0, y1 - y0),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def bbox_img(self, T: np.ndarray, thr=1e-4):
        """ROI-image bbox of whatever this canvas holds, or None."""
        h = self.height()
        ys, xs = np.nonzero(h > thr)
        if len(xs) == 0:
            return None
        pts = np.float32(
            [
                [xs.min(), ys.min()],
                [xs.max(), ys.min()],
                [xs.min(), ys.max()],
                [xs.max(), ys.max()],
            ]
        )
        q = apply(self.c2i(T), pts)
        return q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()


# --------------------------------------------------------------------------------------------
# the seat: blood wicking into the upholstery, on the seat plane
# --------------------------------------------------------------------------------------------


class SeatField:
    """The seat as a plane (38 x 38 cm at 2 mm cells), mapped to the image by a homography
    fitted to its four corners, so a stain's footprint in the picture follows from its real
    area (the seat is ~3.4 px/cm across but under 1 px/cm in depth).

    Blood arrives as surface liquid. The fabric absorbs it up to a capacity per area, and
    what is absorbed wicks outward by a porous-medium law, dc/dt = div(D(x) (c/c_sat)^2
    grad c): a sharp front that advances like sqrt(t), heterogeneous D (smooth noise times the
    plate's own weave texture, faster along the ribs) giving an irregular fibrous edge. Liquid
    the fabric cannot take yet runs down the seat's slight forward tilt as a narrow tongue
    (with a little lateral wander) and, at the front edge, runs over onto the rim."""

    CELL = 2.0
    N = 190

    def __init__(self, sc: Scene, prm: Params, rng: np.random.Generator):
        self.prm = prm
        self.roi = sc.roi
        n, cell = self.N, self.CELL
        L = n * cell
        self.Hm = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [L, 0], [L, L], [0, L]]), np.float32(SEAT_QUAD)
        )
        self.Hinv = np.linalg.inv(self.Hm)
        self.c = np.zeros((n, n), np.float32)  # absorbed (mm of blood per unit area)
        self.liq = np.zeros((n, n), np.float32)  # on the surface
        # the plate's weave, projected onto the seat plane
        G = self._grid_to_roi(np.zeros(2))
        tex = cv2.warpPerspective(
            sc.fabric_tex, G, (n, n), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        )
        tex = np.clip(tex / (np.std(tex) * 2.0 + 1e-6), -1.5, 1.5)
        # lobes ~1.5 x 3 cm and fibre bundles ~5 x 15 mm (the ribs run front to back)
        nz = 0.9 * value_noise((n, n), 8.0, rng, (0.75, 1.5)) + 0.6 * value_noise(
            (n, n), 4.0, rng, (0.6, 1.9)
        )
        # heterogeneity of the padding and the weave: what makes a stain's edge irregular
        self.het = np.exp(nz + 0.45 * tex).astype(np.float32)
        self.D = (prm.seat_D_mm2_s * self.het).astype(np.float32)
        self.cap = (prm.seat_c_sat_mm * (1.0 + 0.12 * value_noise((n, n), 5.0, rng))).astype(
            np.float32
        )
        self.drift = value_noise((n, n), 25.0, rng)  # lateral wander of the running liquid
        self.volume = 0.0
        self.over = {}  # front-edge overflow collected per rim channel
        self.channels: list[float] = []  # u (mm) of the (at most two) rim rivulets
        self.runners: list[dict] = []
        self.entry = None  # (u, v) where the main rivulet comes onto the seat
        self.chan = None  # the polyline the runners follow, (u, v) mm
        self.s_wet = 0.0
        self.chan_len = 0.0

    def _grid_to_roi(self, off) -> np.ndarray:
        cell = self.CELL
        Gs = np.float64([[cell, 0, 0.5 * cell], [0, cell, 0.5 * cell], [0, 0, 1]])
        Tr = np.float64([[1, 0, off[0] - self.roi.x0], [0, 1, off[1] - self.roi.y0], [0, 0, 1]])
        return Tr @ self.Hm @ Gs

    def uv(self, xy_roi, off) -> np.ndarray:
        p = np.array([xy_roi[0] + self.roi.x0 - off[0], xy_roi[1] + self.roi.y0 - off[1], 1.0])
        q = self.Hinv @ p
        return q[:2] / q[2]

    def img(self, u, v, off) -> np.ndarray:
        q = self.Hm @ np.array([u, v, 1.0])
        return np.array(
            [q[0] / q[2] - self.roi.x0 + off[0], q[1] / q[2] - self.roi.y0 + off[1]],
            np.float32,
        )

    def add(self, xy_roi, V: float, off, runs: bool = True) -> bool:
        """Blood arriving at a point of the seat. runs=True: a rivulet (off the palm) that
        keeps going across the fabric; runs=False: it arrives along a contact line (hair
        lying on the seat) and soaks in where it lands, joining the main stain when it is
        close to it."""
        u, v = self.uv(xy_roi, off)
        if (
            not runs
            and self.entry is not None
            and math.hypot(u - self.entry[0], v - self.entry[1]) < 70.0
        ):
            # the contact line of the hair on the seat leads down to the palm's edge, where
            # the fabric is already wet: half runs along it to the main stain
            self.add_uv(self.entry[0], self.entry[1], 0.5 * V, runs=False)
            V = 0.5 * V
        return self.add_uv(u, v, V, runs)

    def add_uv(self, u, v, V, runs=True) -> bool:
        n, cell = self.N, self.CELL
        if not (-6 <= u <= n * cell + 6 and -6 <= v <= n * cell + 6):
            return False
        j = float(np.clip(u / cell - 0.5, 1, n - 2))
        k = float(np.clip(v / cell - 0.5, 1, n - 2))
        s = 2.2  # cells: blood arriving from a finger edge covers ~1 cm
        j0, j1 = int(max(j - 3 * s, 0)), int(min(j + 3 * s + 1, n))
        k0, k1 = int(max(k - 3 * s, 0)), int(min(k + 3 * s + 1, n))
        yy, xx = np.mgrid[k0:k1, j0:j1].astype(np.float32)
        g = np.exp(-((xx - j) ** 2 + (yy - k) ** 2) / (2 * s * s))
        g /= g.sum()
        # A rivulet arriving off the palm keeps going across the fabric (most of it); the
        # rest spreads where it lands and soaks in there.
        self.volume += V
        if runs and self.entry is None:
            self.entry = (u, v)
        if runs and V >= 8.0:
            s0 = 0.0
            if self.chan is not None:
                dch = np.hypot(self.chan[:, 0] - u, self.chan[:, 1] - v)
                kn = int(np.argmin(dch))
                if dch[kn] < 30.0:  # joins the channel already running across the seat
                    s0 = float(self.chan_s[kn])
                elif v > self.chan[0, 1] + 20.0:  # arrives further forward: a new channel
                    self.chan = None
            if self.chan is None:
                self._make_channel(u, v + 3.0)
                self.s_wet = 0.0
                self.runners = []
            self.runners.append({"s": s0, "V": 0.4 * V})
            V = 0.6 * V
        self.liq[k0:k1, j0:j1] += (V / (cell * cell)) * g
        return True

    def _diffuse(self, f, D, dt, ax=1.0, ay=1.0):
        """Explicit conservative diffusion of f with a per-cell diffusivity D (mm^2/s)."""
        cell = self.CELL
        n_sub = min(math.ceil(dt * float(D.max()) * 2.0 * (ax + ay) / (cell * cell) / 0.9) + 1, 60)
        h = dt / n_sub
        for _ in range(n_sub):
            fx = -0.5 * (D[:, 1:] + D[:, :-1]) * ax * (f[:, 1:] - f[:, :-1]) / (cell * cell)
            fy = -0.5 * (D[1:, :] + D[:-1, :]) * ay * (f[1:, :] - f[:-1, :]) / (cell * cell)
            div = np.zeros_like(f)
            div[:, :-1] -= fx
            div[:, 1:] += fx
            div[:-1, :] -= fy
            div[1:, :] += fy
            f = np.maximum(f + h * div, 0.0)
        return f

    def step(self, dt: float):
        """Advance by dt; returns [(u_mm, V_mm3)] that ran over the front edge this step."""
        prm = self.prm
        cell = self.CELL
        c, liq = self.c, self.liq
        if float(liq.max()) < 1e-6 and float(c.max()) < 1e-6 and not self.runners:
            return []
        # absorption: surface liquid soaks into whatever capacity is left under it
        room = np.maximum(self.cap - c, 0.0)
        ab = np.minimum(liq * (1.0 - math.exp(-dt / prm.seat_abs_tau_s)), room)
        c = c + ab
        liq = liq - ab
        # wicking (porous-medium, (c/c_sat)^2), anisotropic: the ribs run front to back
        De = self.D * (np.minimum(c / self.cap, 1.2) ** 2 + 0.002)
        c = self._diffuse(c, De, dt, 0.8, 1.35)
        # standing liquid spreads under its own weight (a gravity current)
        Dl = self.het * prm.seat_spread_mm2_s * np.clip(liq / (2 * prm.seat_pin_mm), 0.0, 1.5) ** 2
        if float(Dl.max()) > 1e-6:
            liq = self._diffuse(liq, Dl, dt)
        # What the fabric cannot take yet leaves the puddle as a rivulet down the seat's
        # forward tilt, from its front-most standing liquid, in one channel: each runner
        # crosses the part of the channel already soaked without losing anything, then soaks
        # the dry fabric ahead of it until it is spent, so the channel grows toward the edge.
        stand = np.maximum(liq - prm.seat_pin_mm, 0.0)
        tot = float(stand.sum()) * cell * cell
        out = []
        if tot > 20.0:
            if self.chan is None:
                k, j = np.unravel_index(
                    int(np.argmax(stand * (np.arange(self.N)[:, None] + 1))), stand.shape
                )
                self._make_channel((j + 0.5) * cell, (k + 1.0) * cell)
            V = min(tot, 30.0)
            liq = liq - stand * (V / tot)
            self.runners.append({"s": 0.0, "V": V})
        keep = []
        w = 2.5  # half-width of the channel, mm
        for r in self.runners:
            travel = prm.seat_run_mm_s * dt
            while travel > 0 and r["V"] > 0.5:
                if r["s"] < self.s_wet:
                    d = min(travel, self.s_wet - r["s"])
                    r["s"] += d
                    travel -= d
                    continue
                d = min(travel, cell, r["V"] / (2 * w * prm.seat_c_sat_mm))
                r["s"] += d
                travel -= d
                r["V"] -= d * 2 * w * prm.seat_c_sat_mm
                self.s_wet = r["s"]
                u, v = self._chan_at(r["s"])
                _stamp(
                    c,
                    u / cell - 0.5,
                    v / cell - 0.5,
                    u / cell - 0.5,
                    v / cell - 0.5,
                    w / cell + 0.5,
                    w / cell + 0.5,
                    1.0,
                    0.3,
                    0,
                )
                np.minimum(c, self.cap, out=c)
                if r["s"] >= self.chan_len:
                    break
            if r["s"] >= self.chan_len - 1e-3:
                if r["V"] > 0.5:
                    out.append((self._chan_at(self.chan_len)[0], r["V"]))
            elif r["V"] > 0.5:
                keep.append(r)
        self.runners = keep
        self.c, self.liq = c.astype(np.float32), liq.astype(np.float32)
        # what ran over the front edge, gathered into at most two rim channels
        res = []
        for u, V in out:
            if not self.channels or (
                len(self.channels) < 2 and min(abs(u - c_) for c_ in self.channels) > 30.0
            ):
                self.channels.append(u)
            ch = min(range(len(self.channels)), key=lambda q: abs(self.channels[q] - u))
            self.over[ch] = self.over.get(ch, 0.0) + V
        for ch, v in list(self.over.items()):
            if v > 10.0:
                res.append((self.channels[ch], v))
                self.over[ch] = 0.0
        return res

    def _make_channel(self, u0, v0):
        """Down the tilt (+v) from (u0, v0) to the front edge, meandering a little sideways
        along the smooth drift field (the weave and the padding's dips)."""
        L = self.N * self.CELL
        pts = [(u0, v0)]
        u, v = u0, v0
        while v < L - 0.5 * self.CELL:
            j = int(np.clip(u / self.CELL, 0, self.N - 1))
            k = int(np.clip(v / self.CELL, 0, self.N - 1))
            u += 0.6 * float(self.drift[k, j]) * self.CELL
            v += self.CELL
            pts.append((u, v))
        self.chan = np.array(pts, np.float64)
        seg = np.linalg.norm(np.diff(self.chan, axis=0), axis=1)
        self.chan_s = np.concatenate([[0.0], np.cumsum(seg)])
        self.chan_len = float(self.chan_s[-1])

    def _chan_at(self, s_):
        s_ = float(np.clip(s_, 0.0, self.chan_len))
        return (
            float(np.interp(s_, self.chan_s, self.chan[:, 0])),
            float(np.interp(s_, self.chan_s, self.chan[:, 1])),
        )

    def runner_field(self) -> np.ndarray:
        g = np.zeros_like(self.liq)
        cell = self.CELL
        for r in self.runners:
            u_, v_ = self._chan_at(r["s"])
            j, k = u_ / cell - 0.5, v_ / cell - 0.5
            rad = 2.2 + 0.8 * min(r["V"] / 30.0, 1.5)
            _stamp(
                g, j, k, j, k - 1.5, rad, rad * 0.8, 0.6 * min(1.0, (r["V"] / 30.0) ** 0.4), 0.5, 0
            )
        return g

    def field(self, win, off, ss=3):
        """Equivalent blood thickness over the ROI window, rendered at ss x and area-reduced."""
        x0, y0, x1, y1 = win
        G = self._grid_to_roi(off)
        S = np.float64(
            [
                [ss, 0, (0.5 * ss - 0.5) - x0 * ss],
                [0, ss, (0.5 * ss - 0.5) - y0 * ss],
                [0, 0, 1],
            ]
        )
        M = S @ G
        W, H = (x1 - x0) * ss, (y1 - y0) * ss
        # The wet front of a stain in fabric runs ahead of the blood as a thin, diluted
        # fringe (plasma wicks further than the cells): a lighter band a few mm wide.
        # Optically, fibres soaked with blood are dark as soon as they are wet through (the
        # light crosses many fibre walls): darkness follows saturation, not volume.
        m = smoothstep(0.04, 0.4, self.c / self.cap)
        fr = np.clip(cv2.GaussianBlur(m, (0, 0), 2.2) * 1.8 - m, 0.0, 1.0)
        f = (0.32 * m**0.7 + 0.03 * fr * (0.7 + 0.3 * self.het / self.het.max())).astype(np.float32)
        big = cv2.warpPerspective(
            f, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        liq = cv2.warpPerspective(
            (self.liq + self.runner_field()).astype(np.float32),
            M,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return (
            cv2.resize(big, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA),
            cv2.resize(liq, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA),
        )

    def bbox_img(self, off, thr=0.01):
        ys, xs = np.nonzero((self.c + self.liq + self.runner_field()) > thr)
        if len(xs) == 0:
            return None
        cell = self.CELL
        pts = [
            self.img((x + 0.5 + dx) * cell, (y + 0.5 + dy) * cell, off)
            for x, y in (
                (xs.min(), ys.min()),
                (xs.max(), ys.max()),
                (xs.min(), ys.max()),
                (xs.max(), ys.min()),
            )
            for dx, dy in ((-1, -1), (1, 1))
        ]
        p = np.array(pts)
        return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()


class Sheet:
    """Blood on the nearly level, cupped left palm the face rests on. There it does not run
    in rivulets: it spreads as one sheet under its own weight (a gravity current, spreading
    faster the deeper it stands), drifts slowly down the palm's tilt, stays inside the hand,
    and leaves over the hand's lowest edge onto whatever is below. A height field on the
    hand's canvas at image resolution (the hand does not move while it lies there)."""

    def __init__(self, c: Canvas, ppm: float):
        self.c = c
        self.k = c.ss
        self.H, self.W = c.H // self.k, c.W // self.k
        self.q = np.zeros((self.H, self.W), np.float32)  # mm
        self.wet = np.zeros((self.H, self.W), np.float32)
        self.ppm = ppm
        self.out_acc = 0.0
        self.spout = None

    def add(self, xy_can, V):
        c = self.c
        u, v = np.asarray(xy_can, np.float32) - [c.x0, c.y0]
        j0, j1 = int(max(u - 5, 0)), int(min(u + 6, self.W))
        k0, k1 = int(max(v - 5, 0)), int(min(v + 6, self.H))
        if j1 - j0 < 3 or k1 - k0 < 3:
            return
        yy, xx = np.mgrid[k0:k1, j0:j1].astype(np.float32)
        g = np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * 1.3**2))
        area = (1.0 / self.ppm) ** 2  # mm^2 per cell
        self.q[k0:k1, j0:j1] += V * g / (g.sum() * area)

    def _bowl(self, sup):
        """The palm's surface height (mm): a hollow around the palm's middle, rising toward
        its edges, and tipped slightly toward the edge that rests lowest (image down)."""
        ys, xs = np.nonzero(sup)
        cy, cx = self.H / 2.0, self.W / 2.0
        near = (ys - cy) ** 2 + (xs - cx) ** 2 < 20.0**2
        if near.sum() > 20:
            cy, cx = ys[near].mean(), xs[near].mean() + 4.0  # the hollow is past the heel
        yy, xx = np.mgrid[0 : self.H, 0 : self.W].astype(np.float32)
        r2 = ((xx - cx) / 1.4) ** 2 + (yy - cy) ** 2
        self.z = (np.minimum(r2 / 11.0**2, 1.5) * 4.0 - 0.12 * (yy - cy)).astype(np.float32)

    def step(self, dt, g_can):
        """Returns [(xy_can, V)] leaving over the edge of the hand."""
        sup = cv2.resize(self.c.sup, (self.W, self.H), interpolation=cv2.INTER_AREA) > 0.5
        if not hasattr(self, "z"):
            self._bowl(sup)
        q = self.q * sup
        if float(q.max()) < 1e-4:
            self.q = q
            return []
        mm = 1.0 / self.ppm  # mm per cell
        # thin-film flow down the gradient of the free surface (palm height + depth), with a
        # mobility that grows with depth above the contact line's pinning depth; upwind
        # (the donor's mobility), conservative; past the edge of the hand the surface drops
        z = self.z
        sp = np.pad(sup, 1)
        zp = np.pad(z, 1, mode="edge") - 1.0
        left = np.zeros_like(q)
        t_left, n = dt, 0
        while t_left > 1e-6 and n < 80:
            Dm = 30.0 * np.clip((q - 0.12) / 0.4, 0.0, 2.5) ** 2 / (mm * mm)  # cells^2 / s
            h = min(t_left, 0.2 / max(float(Dm.max()) * 4.0, 1e-6))
            e = q + z
            ep = np.where(sp, np.pad(e, 1), zp)
            Dp = np.pad(Dm, 1)
            flux = np.zeros_like(q)
            outflow = np.zeros_like(q)
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                sl_ = (slice(1 + dy, 1 + dy + self.H), slice(1 + dx, 1 + dx + self.W))
                en, Dn, inn = ep[sl_], Dp[sl_], sp[sl_]
                up = np.where(e > en, Dm, Dn)
                flux += np.where(inn, up * (en - e), 0.0)
                outflow += np.where(~inn & (e > en), Dm * (e - en), 0.0)
            outflow *= sup
            q = np.maximum(q + h * (flux - outflow), 0.0)
            left += h * outflow
            t_left -= h
            n += 1
        lv_tot = left
        out_tot = float(lv_tot.sum()) * mm * mm
        out = []
        self.out_acc += out_tot
        if self.out_acc > 25.0:
            ys, xs = np.nonzero(lv_tot > 0)
            if len(xs):
                if self.spout is None:  # the first place it goes over stays the channel
                    j = int(np.argmax(lv_tot[ys, xs]))
                    self.spout = np.array(
                        [xs[j] + self.c.x0 + 0.5, ys[j] + self.c.y0 + 1.5], np.float32
                    )
                out.append((self.spout.copy(), self.out_acc))
            self.out_acc = 0.0
        self.q = (q * sup).astype(np.float32)
        np.maximum(self.wet, np.minimum(self.q, 0.06), out=self.wet)
        return out

    def field(self) -> np.ndarray:
        """The sheet and the film it has wetted, at the canvas' resolution."""
        f = np.maximum(self.q, self.wet)
        return cv2.resize(f, (self.c.W, self.c.H), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------------------------
# the blood: rivulet heads on the body parts' canvases, the seat, the rim,
# drops and the floor
# --------------------------------------------------------------------------------------------


class Blood:
    def __init__(self, prm: Params, tr: Tracks, sc: Scene, tracks: dict, boxes: dict, fps: float):
        self.prm, self.tr, self.sc, self.fps = prm, tr, sc, fps
        self.rng = np.random.default_rng(prm.seed)
        self.ppm = tr.px_per_mm
        self.tracks = tracks
        self.cv = {p: Canvas(p, boxes[p], prm.ss, self.rng) for p in (*BODY, RIM, FLOOR)}
        self.seat = SeatField(sc, prm, self.rng)
        self.sheet = Sheet(self.cv[LHAND], self.ppm)
        # rivulet heads ("beads"): canonical position on their part, volume (mm^3), ...
        self.b_pos = np.zeros((0, 2), np.float32)
        self.b_part = np.zeros(0, np.int8)
        self.b_V = np.zeros(0, np.float32)
        self.b_ph = np.zeros(0, np.float32)
        self.b_thk = np.zeros(0, np.float32)
        self.drops: list[dict] = []
        self.shown_drops: list[dict] = []
        self.spots: list[dict] = []
        self.wound = None
        self.wound_dir = None
        self.emitted = 0.0
        self.acc_open = 0.0
        self.acc_dors = 0.0
        self.events: list[str] = []
        self.last_drip: dict = {}
        self.off = np.zeros(2, np.float32)
        self.i = None
        self.t = 0.0
        self.to_seat = 0.0
        self.seat_log: list = []
        self.to_cloth = 0.0
        # where blood leaves the pressed palm: a few channels a fixed sideways offset apart
        # (the gaps under the fingers), weighted unevenly so one or two carry most of it
        self.ch_ang = np.deg2rad(np.array([-75.0, -40.0, -10.0, 20.0, 50.0, 80.0], np.float32))
        # most leaves straight down (the lowest point of the rim), less the further to the side
        self.ch_w = self.rng.dirichlet(np.ones(len(self.ch_ang)) * 2.0) * (
            np.cos(self.ch_ang) ** 2 + 0.12
        )
        self.ch_acc = np.zeros(len(self.ch_ang), np.float32)
        self.ch_first = True

    # ------------------------------------------------------------------ transforms
    def T(self, p: int, i: int, tau: float = 0.0) -> np.ndarray:
        if p in BODY:
            return self.tracks[p].M(i, tau)
        if p == RIM:
            o = self.sc.chair_off.get(i, self.off)
            return np.float32([[1, 0, o[0]], [0, 1, o[1]]])
        return I23

    def scale(self, p: int, i: int) -> float:
        return self.tracks[p].scale(i) if p in BODY else 1.0

    def g_can(self, p: int, i: int) -> np.ndarray:
        return self.tracks[p].g_can(i) if p in BODY else np.array([0.0, 1.0], np.float32)

    def to_img(self, p: int, pts) -> np.ndarray:
        return apply(self.T(p, self.i), pts)

    def to_can(self, p: int, pts) -> np.ndarray:
        return apply(cv2.invertAffineTransform(self.T(p, self.i)), pts)

    def ppm_of(self, p: int) -> float:
        # px per mm along the direction blood runs: skin at the head's scale; the rim is a
        # vertical face, at the chair legs' vertical scale
        return self.prm.g_px_s2 / 9810.0 if p == RIM else self.ppm

    def init_wound(self, xy_can, tangent):
        self.wound = np.asarray(xy_can, np.float32)
        self.wound_dir = np.asarray(tangent, np.float32)
        _e, w, tip = self.tr.hand(self.tracks[RHAND].ref)
        ax = tip - w
        n = np.array([-ax[1], ax[0]], np.float32)
        # two finger gaps on the back of the hand, near the knuckles (hand canonical)
        self.gaps = [w + 0.64 * ax + 0.13 * n, w + 0.5 * ax - 0.16 * n]
        self.gap_w = np.array([0.8, 0.2])

    # ------------------------------------------------------------------ beads
    def _add_beads(self, pos, V, part):
        pos = np.asarray(pos, np.float32).reshape(-1, 2)
        n = len(pos)
        self.b_pos = np.vstack([self.b_pos, pos])
        self.b_part = np.concatenate([self.b_part, np.full(n, part, np.int8)])
        self.b_V = np.concatenate([self.b_V, np.asarray(V, np.float32).reshape(-1)])
        self.b_ph = np.concatenate([self.b_ph, self.rng.uniform(0, 6.28, n).astype(np.float32)])
        self.b_thk = np.concatenate(
            [self.b_thk, self.rng.uniform(0.75, 1.25, n).astype(np.float32)]
        )

    def _keep(self, keep):
        self.b_pos, self.b_part, self.b_V = (
            self.b_pos[keep],
            self.b_part[keep],
            self.b_V[keep],
        )
        self.b_ph, self.b_thk = self.b_ph[keep], self.b_thk[keep]

    def _width_mm(self, V):
        return np.clip(1.3 * np.cbrt(np.maximum(V, 0.1)), 2.0, 7.0)

    # ------------------------------------------------------------------ frame step
    def step(self, i: int, t: float, M: dict):
        prm = self.prm
        dt = 1.0 / self.fps
        self.i, self.t, self.M = i, t, M
        self.off = M["off"]
        for p in BODY:
            c = self.cv[p]
            c.update_support(M["sup"][p], M["hair"] if p == HEAD else None, self.T(p, i))
        self.cv[RIM].update_support(M["sup"][RIM], None, self.T(RIM, i), a=1.0)
        self.cv[FLOOR].update_support(M["sup"][FLOOR], None, I23, a=1.0)
        k = math.exp(-dt / prm.live_tau_s)
        for c in self.cv.values():
            c.live *= k
        if self.wound is not None and t >= prm.onset:
            self._emit(t, dt, M)
        for _ in range(2):
            self._beads(t, dt / 2, M)
        self._merge()
        for pt, V in self.sheet.step(dt, self.g_can(LHAND, i)):
            # over the edge of the hand: onto the seat it rests on, or a drop if in the air
            q_img = self.to_img(LHAND, pt[None])[0] + np.float32([0.0, 1.5])
            if float(sample(M["sup"][SEAT], q_img[None])[0]) > 0.3 and self.seat.add(
                q_img, V, self.off
            ):
                self._seat_event(q_img, t)
                self.to_seat += V
            else:
                self._release_drop(q_img, V, LHAND, t)
        self._drops(t, dt, M)
        for u, V in self.seat.step(dt):
            q = self.seat.img(u, self.seat.N * self.seat.CELL, self.off) + np.float32([0, 2.0])
            q_can = q - self.off
            self._add_beads([q_can], [V], RIM)
            if not getattr(self, "_rim_started", False):
                self._rim_started = True
                self.events.append(f"{t:.2f}s the seat stain runs over the front edge")
        self._stamp_beads()

    def _emit(self, t, dt, M):
        prm = self.prm
        tt = t - prm.onset
        ramp = 0.35 + 0.65 * min(1.0, tt / prm.ramp_s)
        boost = 1.0 + prm.initial_boost * math.exp(-tt / prm.boost_tau_s)
        pulse = 1.0 + prm.pulse_amp * math.sin(2 * math.pi * prm.pulse_hz * tt)
        vol = prm.flow_ml_s * 1000.0 * ramp * boost * pulse * dt
        self.emitted += vol
        Vb = prm.bead_ml * 1000.0
        w_img = self.to_img(HEAD, self.wound)[0]
        sup_rh = M["sup"][RHAND]
        covered = float(sample(sup_rh, w_img[None])[0]) > 0.5
        if not covered:
            self.acc_open += vol
            while self.acc_open >= Vb * 0.6:
                V = float(self.rng.uniform(0.7, 1.3) * Vb)
                self.acc_open -= V
                p = self.wound + self.rng.uniform(-0.5, 0.5) * self.wound_dir
                self._add_beads([p], [V], HEAD)
            return
        # Between the fingers and onto the back of the hand only while the fingers point up
        # (blood runs down the backs of the fingers to the wrist). Lying on the side, the
        # fingers curl down over the head and what seeps between them runs onto the scalp.
        _e, w_, tip_ = M["hand"]
        ax = tip_ - w_
        up = float(-ax[1] / (np.linalg.norm(ax) + 1e-6))
        # (it takes a moment to fill the gaps between the fingers before it shows)
        dors = prm.dorsal_frac * float(smoothstep(0.3, 0.7, np.array(up))) * min(1.0, tt / 0.8)
        v_dors = vol * dors
        v_under = vol * prm.under_frac
        v_rim = vol - v_dors - v_under
        # The blood under the pressed palm runs between palm and skin (hidden) until it
        # reaches the rim of the hand and leaves there, in a fan of channels around the down
        # direction (it squeezes out sideways too, wherever the seal is weakest).
        exits = []
        for ang in self.ch_ang:
            dv = np.float32([math.sin(ang), math.cos(ang)]) * 0.5
            p = w_img.copy()
            path = [p.copy()]
            inside = True
            n = 0
            while inside and n < 130:
                p = p + dv
                path.append(p.copy())
                inside = float(sample(sup_rh, p[None])[0]) > 0.15
                n += 1
            if inside:
                exits.append(None)
                continue
            ex = p + dv * 1.5
            recv = None
            for r in (HEAD, LHAND, RFORE):
                if float(sample(M["m"][r], ex[None])[0]) > 0.5:
                    recv = r
                    break
            if recv is None:  # the rim of the hand is its outer edge: down the side of the hand
                recv, ex = RHAND, path[max(len(path) - 5, 0)]
            exits.append((recv, ex, path))
        # Where the rim meets skin the blood shows (face, the other hand); off the wrist it
        # runs on under the forearm; off the hand's free edge (upright only: lying, the edge
        # of the hand is on the scalp) it runs down the side of the hand.
        pref = {
            HEAD: 1.0,
            LHAND: 1.0,
            RFORE: 0.5,
            RHAND: 0.3 * float(smoothstep(0.3, 0.7, np.array(up))),
        }
        wts = np.array(
            [0.0 if e is None else self.ch_w[k] * pref[e[0]] for k, e in enumerate(exits)]
        )
        # The seal gives way in one or two places at a time, not all round: the two
        # heaviest open channels carry it (the others stay pressed shut for now).
        if (wts > 0).sum() > 2:
            wts[np.argsort(wts)[:-2]] = 0.0
        if wts.sum() > 0:
            wts /= wts.sum()
            if self.ch_first:
                # the first trickle shows at the onset, in the main channel onto the face
                face = [k for k, e in enumerate(exits) if e is not None and e[0] == HEAD]

                # where it first shows: a channel onto the face, on lit skin (in the hand's
                # shadow a first trickle is there but unseen), short of the wound
                def _score(k):
                    ex_ = exits[k][1]
                    return float(sample(M["lum"], ex_[None])[0]) - 0.004 * len(exits[k][2])

                k0 = max(face, key=_score) if face else int(np.argmax(wts))
                self.ch_acc[k0] += Vb * 2.5
            for k, e in enumerate(exits):
                if e is None or wts[k] == 0:
                    continue
                recv, ex, path = e
                if len(path) > 2 and self.i % 2 == 0:
                    pc = self.to_can(HEAD, np.array(path[:: max(len(path) // 8, 1)]))
                    self._film_path(HEAD, pc, 0.025, 3.5)
                self.ch_acc[k] += v_rim * wts[k]
                thr = Vb * 0.5 if self.ch_first else Vb
                if self.ch_acc[k] >= thr:
                    V = float(self.ch_acc[k])
                    self.ch_acc[k] = 0.0
                    self._add_beads(self.to_can(recv, ex[None]), [V], recv)
                    if self.ch_first:
                        self.ch_first = False
                        self.events.append(
                            f"{t:.2f}s first blood leaves the palm's rim onto {PARTS[recv]} at "
                            f"({ex[0] + self.sc.roi.x0:.0f},{ex[1] + self.sc.roi.y0:.0f})"
                        )
        # Blood trapped under the pressed palm also oozes out all along the palm's rim below
        # the wound: a wet margin a few mm wide on the skin around the hand, deepening over
        # the first seconds. It stays on the skin where it formed if the hand shifts.
        if self.i % 2 == 0:
            self._rim_band(w_img, tt, M, v_under)
        # some seeps out between the fingers onto the back of the hand
        self.acc_dors += v_dors
        if self.acc_dors >= Vb * 0.8:
            j = int(self.rng.choice(len(self.gaps), p=self.gap_w))
            V = float(self.acc_dors)
            self.acc_dors = 0.0
            g = self.gaps[j] + self.rng.normal(0, 0.4, 2).astype(np.float32)
            if float(self.cv[RHAND].at(self.cv[RHAND].sup, g[None])[0]) > 0.5:
                self._add_beads([g], [V], RHAND)
        self.under_vol = getattr(self, "under_vol", 0.0) + v_under

    def _rim_band(self, w_img, tt, M, vol):
        rh = M["sup"][RHAND]
        x0 = int(max(w_img[0] - 50, 0))
        x1 = int(min(w_img[0] + 50, rh.shape[1]))
        y0 = int(max(w_img[1] - 30, 0))
        y1 = int(min(w_img[1] + 60, rh.shape[0]))
        if x1 <= x0 or y1 <= y0:
            return
        r = rh[y0:y1, x0:x1]
        ring = (cv2.dilate((r > 0.35).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & (r < 0.15)
        ring &= M["m"][HEAD][y0:y1, x0:x1] > 0.5
        ys, xs = np.nonzero(ring)
        if len(xs) == 0:
            return
        pts = np.stack([xs + x0, ys + y0], 1).astype(np.float32)
        dy = pts[:, 1] - w_img[1]
        dist = np.linalg.norm(pts - w_img, axis=1)
        wgt = smoothstep(-12.0, 6.0, dy) * smoothstep(48.0, 20.0, dist)
        sel = wgt > 0.05
        if not sel.any():
            return
        pts, wgt = pts[sel], wgt[sel]
        h = 0.18 * min(1.0, tt / 2.5)
        c = self.cv[HEAD]
        pc = c.to_c(self.to_can(HEAD, pts))
        rad = 1.1 * c.ss / self.scale(HEAD, self.i)
        for (qx, qy), wv in zip(pc, wgt, strict=True):
            _stamp(c.res, qx, qy, qx, qy, rad, rad, float(h * wv), 0.5, 0)

    def _film_path(self, p, pts_can, h_mm, w_mm):
        c = self.cv[p]
        pc = c.to_c(pts_can)
        r = 0.5 * w_mm * self.ppm_of(p) / self.scale(p, self.i) * c.ss
        for a, b in itertools.pairwise(pc):
            _stamp(c.res, a[0], a[1], b[0], b[1], r, r, h_mm, 0.5, 0)

    def _beads(self, t, dt, M):
        prm = self.prm
        n = len(self.b_V)
        if n == 0:
            return
        grads = {}
        kill = np.zeros(n, bool)
        i = self.i
        for k in range(n):
            p = int(self.b_part[k])
            c = self.cv[p]
            V = float(self.b_V[k])
            x = self.b_pos[k].copy()
            cx = c.to_c(x[None])[0]
            if not (2 <= cx[0] < c.W - 3 and 2 <= cx[1] < c.H - 3):
                kill[k] = True
                continue
            sc_ = self.scale(p, i)
            ppm = self.ppm_of(p)
            wet = float(sample(c.res, cx[None])[0]) > 0.02
            pin = prm.v_pin_wet if wet else prm.v_pin_dry
            vfull = prm.v_full
            surf = 1.0
            hair = 0.0
            if p == HEAD:
                hair = float(sample(c.hair, cx[None])[0])
                surf *= 1.0 - 0.45 * hair
                if float(sample(M["sup"][RHAND], self.to_img(p, x[None]))[0]) > 0.5:
                    surf *= 0.8  # squeezed along the gap between palm and skin
            elif p == LHAND:
                surf *= 0.5  # the palm is nearly level
            elif p == RIM:
                pin, vfull, surf = 1.0, 25.0, 1.6  # smooth paint, already running wet
            s = min(1.0, max(0.0, (V - pin) / (vfull - pin))) ** 0.5
            speed = prm.v_max_mm_s * s * surf * ppm / sc_  # canonical px / s
            if speed * dt < 1e-3:
                # too little left to overcome the contact line: it stops as a small bead of
                # residue and is no longer a rivulet
                # (an elongated tip on the end of its own trail, not a round dot)
                g0 = self.g_can(p, i)
                ca, cb = c.to_c(np.stack([x - g0 * 1.2 / sc_, x]))
                r = 0.45 * float(self._width_mm(V)) * ppm / sc_ * c.ss
                h = min(0.15, V / (math.pi * (0.5 * float(self._width_mm(V))) ** 2))
                _stamp(c.res, ca[0], ca[1], cb[0], cb[1], 0.7 * r, r, h, 0.5, 0)
                kill[k] = True
                continue
            g = self.g_can(p, i)
            wx = float(sample(c.wander[0], cx[None])[0])
            wy = float(sample(c.wander[1], cx[None])[0])
            th = 0.35 * wx + 0.12 * wy
            if p != RIM:  # (each rivulet's own slow meander; on the rim, one channel)
                th += 0.15 * math.sin(float(self.b_ph[k]) + 0.35 * t)
            cs, sn = math.cos(th), math.sin(th)
            d = np.array([cs * g[0] - sn * g[1], sn * g[0] + cs * g[1]], np.float32)
            if p not in grads:
                hb = cv2.GaussianBlur(c.res, (0, 0), 1.2 * c.ss)
                grads[p] = (
                    cv2.Sobel(hb, cv2.CV_32F, 1, 0),
                    cv2.Sobel(hb, cv2.CV_32F, 0, 1),
                )
            gr = np.array(
                [
                    float(sample(grads[p][0], cx[None])[0]),
                    float(sample(grads[p][1], cx[None])[0]),
                ]
            )
            lat = np.array([-g[1], g[0]], np.float32)
            kst = 8.0  # (a stream follows the one already wet)
            d = d + lat * float(np.clip((gr @ lat) * kst, -0.6, 0.6))  # into a wet channel
            d /= np.linalg.norm(d) + 1e-6
            step = speed * dt
            q = x + d * step
            moved = None
            if float(c.at(c.sup, q[None])[0]) >= 0.5:
                moved = q
            else:
                q_img = self.to_img(p, q[None])[0]
                probe = q_img + np.float32([0.0, 2.0])
                recv = None
                for r in CONTACT.get(p, ()):
                    if float(sample(M["sup"][r], probe[None])[0]) > 0.5:
                        recv = r
                        break
                if recv == LHAND:  # onto the level palm: it spreads there as a sheet
                    self.sheet.add(self.to_can(LHAND, probe[None])[0], V)
                    self._palm_event(probe, t)
                    kill[k] = True
                    continue
                if recv in BODY:
                    self.b_part[k] = recv
                    self.b_pos[k] = self.to_can(recv, probe[None])[0]
                    continue
                if recv == SEAT:
                    if self.seat.add(probe, V, self.off, runs=p != HEAD):
                        self.seat_log.append(
                            (round(t, 2), PARTS[p], probe.round(1).tolist(), round(V))
                        )
                        self._seat_event(probe, t)
                        self.to_seat += V
                    kill[k] = True
                    continue
                if recv == CLOTH:
                    self.to_cloth += V
                    kill[k] = True
                    continue
                # no surface in contact below: slide along the edge, drip, or run round out
                # of sight at a silhouette
                sw = c.sup[int(cx[1]) - 2 : int(cx[1]) + 3, int(cx[0]) - 2 : int(cx[0]) + 3]
                gx_m = float(cv2.Sobel(sw, cv2.CV_32F, 1, 0, ksize=3)[2, 2])
                gy_m = float(cv2.Sobel(sw, cv2.CV_32F, 0, 1, ksize=3)[2, 2])
                nin = np.array([gx_m, gy_m], np.float32)
                nn = float(np.linalg.norm(nin))
                nin = nin / nn if nn > 1e-6 else -d
                gd = float(d @ nin)
                tv = d - gd * nin
                tn = float(np.linalg.norm(tv))
                if tn > 0.3:
                    q2 = x + tv * step
                    if float(c.at(c.sup, q2[None])[0]) >= 0.45:
                        moved = q2
                if moved is None:
                    key = (p, round(float(q_img[0])) // 3)
                    ready = self.last_drip.get(key, -9.0)
                    drip_at = prm.drip_mm3 * (0.7 if p == RIM else 1.0)  # a thin painted edge
                    if gd < -0.3 and drip_at < V and t > ready:
                        # the pendant necks and refills before the next one: irregular cadence
                        self.last_drip[key] = t + float(self.rng.uniform(0.3, 0.75))
                        dv = min(drip_at * 0.85 * float(self.rng.uniform(0.85, 1.15)), V - 4.0)
                        self.b_V[k] -= dv
                        self._release_drop(q_img, dv, p, t)
                    elif (
                        -0.5 < gd < 0.25 and p in (HEAD, RHAND, RFORE) and self.rng.random() < 0.25
                    ):
                        kill[k] = True  # runs on round the side of the head/arm, out of view
                        continue
            if moved is not None:
                self._deposit(k, p, x, moved, V, wet, hair, sc_, ppm)
                self.b_pos[k] = moved
            if self.b_V[k] < 1.5:
                kill[k] = True
        if kill.any():
            self._keep(~kill)

    def _deposit(self, k, p, a, b, V, wet, hair, sc_, ppm):
        """Lay the rivulet down along the step: its residual film (max-combined, so a channel
        does not pile up where it is re-used) and the depth of the live stream behind the
        bead. The residual comes out of the bead's volume; the live stream is liquid still
        moving with it."""
        prm = self.prm
        c = self.cv[p]
        w_mm = float(self._width_mm(V))
        if p == RHAND:
            w_mm *= 0.85
        if p == LHAND:  # the palm is nearly level and cupped: the blood spreads and stands
            w_mm *= 1.4
        r = 0.5 * w_mm * ppm / sc_ * c.ss
        h_f = prm.film_wet_mm if wet else prm.film_dry_mm * (0.4 if p in (RHAND, RFORE) else 1.0)
        if p == HEAD:
            h_f *= 1.0 + (prm.hair_soak - 1.0) * hair
        elif p == LHAND:
            h_f *= 2.5
        ca, cb = c.to_c(np.stack([a, b]))
        _stamp(c.res, ca[0], ca[1], cb[0], cb[1], 0.8 * r, 0.8 * r, max(h_f, 0.05), 0.5, 0)
        # a rivulet a few mm wide stands ~0.2-0.5 mm high (contact angle ~15 degrees): deep
        # enough to go dark crimson, thin enough to stay red rather than black
        h_live = min(0.55, max(0.15, 0.32 * (V / 40.0) ** 0.5)) * float(self.b_thk[k])
        _stamp(c.live, ca[0], ca[1], cb[0], cb[1], r, r, h_live, 0.55, 0)
        L_mm = float(np.linalg.norm(b - a)) * sc_ / ppm
        self.b_V[k] -= h_f * w_mm * L_mm

    def _palm_event(self, xy, t):
        if not getattr(self, "_palm_started", False):
            self._palm_started = True
            self.events.append(
                f"{t:.2f}s blood runs onto the left palm at "
                f"({xy[0] + self.sc.roi.x0:.0f},{xy[1] + self.sc.roi.y0:.0f})"
            )

    def _stamp_beads(self):
        for c in self.cv.values():
            c.bead[:] = 0.0
        self.cv[LHAND].bead[:] = self.sheet.field()
        for k in range(len(self.b_V)):
            p = int(self.b_part[k])
            c = self.cv[p]
            V = float(self.b_V[k])
            r = 0.62 * float(self._width_mm(V)) * self.ppm_of(p) / self.scale(p, self.i) * c.ss
            h = min(0.7, max(0.25, 0.4 * (V / 40.0) ** 0.4))
            q = c.to_c(self.b_pos[k][None])[0]
            _stamp(c.bead, q[0], q[1], q[0], q[1] - 0.6 * r, r, 0.8 * r, h, 0.45, 0)

    def _merge(self):
        n = len(self.b_V)
        if n < 2:
            return
        dead = np.zeros(n, bool)
        for p in np.unique(self.b_part):
            idx = np.nonzero(self.b_part == p)[0]
            if len(idx) < 2:
                continue
            sc_ = self.scale(int(p), self.i)
            r = 0.5 * self._width_mm(self.b_V[idx]) * self.ppm_of(int(p)) / sc_
            if p == RIM:
                r = np.maximum(r, 2.5)  # the rim collects into one pendant
            P = self.b_pos[idx]
            d = np.linalg.norm(P[:, None] - P[None], axis=-1)
            close = d < (r[:, None] + r[None]) * 0.9
            np.fill_diagonal(close, False)
            g = self.g_can(int(p), self.i)
            depth = P @ g
            for a in range(len(idx)):
                if dead[idx[a]]:
                    continue
                for b in np.nonzero(close[a])[0]:
                    if dead[idx[b]]:
                        continue
                    keep, gone = (a, b) if depth[a] >= depth[b] else (b, a)
                    self.b_V[idx[keep]] += self.b_V[idx[gone]]
                    dead[idx[gone]] = True
        if dead.any():
            self._keep(~dead)

    def _seat_event(self, xy, t):
        if not getattr(self, "_seat_started", False):
            self._seat_started = True
            self.events.append(
                f"{t:.2f}s blood reaches the seat fabric at "
                f"({xy[0] + self.sc.roi.x0:.0f},{xy[1] + self.sc.roi.y0:.0f})"
            )

    # ------------------------------------------------------------------ drops and the floor
    def _release_drop(self, xy_img, V, src_part, t):
        y_land = float(xy_img[1]) + (SEAT_EDGE_TO_FLOOR_PX - 4.0 if src_part == RIM else 600.0)
        self.drops.append(
            {"p": np.array(xy_img, np.float64), "v": np.zeros(2), "V": V, "src": src_part,
             "fallen": 0.0, "y_land": y_land, "t0": t, "hist": [np.array(xy_img, np.float64)]}
        )  # fmt: skip
        self.events.append(f"{t:.2f}s drop ({V:.0f} mm3) leaves {PARTS[src_part]}")

    def _drops(self, t, dt, M):
        prm = self.prm
        keep = []
        self.shown_drops = []
        for d in self.drops:
            landed = False
            nsub = 10
            for _ in range(nsub):
                h = dt / nsub
                d["v"][1] += prm.g_px_s2 * h
                d["p"] = d["p"] + d["v"] * h
                d["fallen"] += float(d["v"][1] * h)
                xy = d["p"].astype(np.float32)
                if d["fallen"] > 4.0 and d["src"] != RIM:
                    for r in (CLOTH, LHAND, SEAT):
                        if float(sample(M["sup"][r], xy[None])[0]) > 0.5:
                            if r == SEAT and self.seat.add(xy, d["V"], self.off, runs=False):
                                self.seat_log.append(
                                    (round(t, 2), "drop", xy.round(1).tolist(), round(d["V"]))
                                )
                                self._seat_event(xy, t)
                                self.to_seat += d["V"]
                            elif r == LHAND:
                                self.sheet.add(self.to_can(LHAND, xy[None])[0], d["V"])
                            landed = True
                            break
                if not landed and d["p"][1] >= d["y_land"]:
                    self._land(np.array([d["p"][0], d["y_land"]], np.float32), d["V"], t)
                    landed = True
                if landed or d["p"][1] > self.sc.shape[0] + 5:
                    break
            d["hist"].append(d["p"].copy())
            self.shown_drops.append(d)
            if not landed and d["p"][1] <= self.sc.shape[0] + 5:
                keep.append(d)
        self.drops = keep

    def _land(self, xy, V, t):
        prm = self.prm
        rng = self.rng
        c = self.cv[FLOOR]
        ppm = prm.px_per_mm_seat
        fore = prm.floor_fore
        if self.spots:
            s = self.spots[0]
            s["V"] += V
            s["n"] += 1
            # later drops land in the same place and grow one small puddle (area ~ volume)
            r_mm = math.sqrt(s["V"] / (math.pi * 0.9))
            self._floor_disc(s["c"], r_mm * ppm, r_mm * ppm * fore, 0.9, s["seed"])
            self.events.append(f"{t:.2f}s drop lands in the drip puddle ({s['n']} drops)")
            return
        # a ~35 uL drop after a ~45 cm fall: a ~13 mm spot with a fine crown of satellites
        d_mm = 13.0 * (V / 35.0) ** (1 / 3)
        s = {"c": xy.copy(), "V": V, "n": 1, "seed": float(rng.uniform(0, 100))}
        self.spots.append(s)
        self._floor_disc(xy, 0.5 * d_mm * ppm, 0.5 * d_mm * ppm * fore, 0.6, s["seed"])
        cc = c.to_c(xy[None])[0]
        for _ in range(int(rng.integers(5, 9))):
            a = rng.uniform(0, 2 * np.pi)
            rr = 0.5 * d_mm * ppm * rng.uniform(1.3, 2.4)
            q = cc + np.float32([rr * math.cos(a), rr * math.sin(a) * fore]) * c.ss
            r_s = rng.uniform(0.25, 0.5) * c.ss
            _stamp(
                c.res,
                q[0],
                q[1],
                q[0],
                q[1],
                r_s,
                r_s,
                float(rng.uniform(0.15, 0.35)),
                0.5,
                0,
            )
        self.events.append(
            f"{t:.2f}s first drop lands on the floor at "
            f"({xy[0] + self.sc.roi.x0:.0f},{xy[1] + self.sc.roi.y0:.0f})"
        )

    def _floor_disc(self, xy, rx, ry, h, seed):
        c = self.cv[FLOOR]
        cc = c.to_c(np.asarray(xy)[None])[0]
        ss = c.ss
        x0, x1 = (
            int(max(cc[0] - 2 * rx * ss - 3, 0)),
            int(min(cc[0] + 2 * rx * ss + 4, c.W)),
        )
        y0, y1 = (
            int(max(cc[1] - 2 * ry * ss - 3, 0)),
            int(min(cc[1] + 2 * ry * ss + 4, c.H)),
        )
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        ang = np.arctan2((yy - cc[1]) / max(ry, 1e-3), (xx - cc[0]) / max(rx, 1e-3))
        wob = 1.0 + 0.08 * np.sin(3 * ang + seed) + 0.05 * np.sin(5 * ang + 2 * seed)
        d = np.sqrt(((xx - cc[0]) / (rx * ss)) ** 2 + ((yy - cc[1]) / (ry * ss)) ** 2) / wob
        val = h * np.clip(1.0 - d * d, 0.0, 1.0) ** 0.3
        sl = c.res[y0:y1, x0:x1]
        np.maximum(sl, val.astype(np.float32), out=sl)


# --------------------------------------------------------------------------------------------
# render + composite (linear light, motion blur over the shutter) + plate matching
# --------------------------------------------------------------------------------------------


def estimate_light(rgb: np.ndarray, head_sup: np.ndarray, hair: np.ndarray, c, R) -> np.ndarray:
    """Key light direction from the plate's own shading on the head, modelled as a sphere:
    skin luminance is regressed on the sphere normals (x right, y down, z to camera)."""
    lin = srgb_to_lin(rgb)
    lum = luminance(lin)
    ys, xs = np.nonzero((head_sup > 0.8) & (hair < 0.3) & (lum > 0.004))
    if len(xs) < 80:
        return np.array([0.0, -0.55, 0.83], np.float32)
    nx = (xs - c[0]) / (1.15 * R)
    ny = (ys - c[1]) / (1.15 * R)
    r2 = nx * nx + ny * ny
    ok = r2 < 0.95
    nx, ny = nx[ok], ny[ok]
    nz = np.sqrt(1 - nx * nx - ny * ny)
    A = np.stack([np.ones_like(nx), nx, ny, nz], 1)
    coef, *_ = np.linalg.lstsq(A, lum[ys[ok], xs[ok]], rcond=None)
    L = coef[1:]
    n = np.linalg.norm(L)
    if n < 1e-6:
        return np.array([0.0, -0.55, 0.83], np.float32)
    L = L / n
    if L[2] < 0.15:  # the camera sees lit skin, so the key cannot be behind the surface
        L[2] = 0.15
        L /= np.linalg.norm(L)
    return L.astype(np.float32)


class Renderer:
    def __init__(self, prm, sc, blur_sigma, noise, light, wb, yblack, gamut, fps):
        self.prm, self.sc = prm, sc
        self.gamut = gamut
        self.blur = blur_sigma
        self.noise = noise
        self.fps = fps
        self.L = np.asarray(light, np.float32)
        H = self.L + np.array([0.0, 0.0, 1.0], np.float32)
        self.H = H / np.linalg.norm(H)
        # A level wet surface seen from this high camera mirrors the ceiling above the far
        # side of the room, where the garage's fixtures are: a second half-vector for those.
        H2 = np.array([0.0, -0.95, 0.3], np.float32) + np.array([0.0, 0.0, 1.0], np.float32)
        self.H2 = H2 / np.linalg.norm(H2)
        self.wb = np.asarray(wb, np.float32)
        self.yblack = float(yblack)
        tex = sc.fabric_tex
        self.tex_n = np.clip(tex / (np.std(tex) * 2.5 + 1e-6), -1, 1)

    # ---------------------------------------------------------------- layers
    def taus(self, bl: Blood, i: int):
        """Sub-frame times across the exposure. The blood on a moving part is drawn at each
        and averaged, so it carries exactly the motion blur the plate carries under it."""
        s = self.prm.shutter
        mx = 0.0
        for p in BODY:
            c = bl.cv[p]
            if not c.height().any():
                continue
            corners = np.float32([[0, 0], [c.W, 0], [0, c.H], [c.W, c.H]])
            a = apply(c.c2i(bl.T(p, i, -0.5 * s)), corners)
            b = apply(c.c2i(bl.T(p, i, 0.5 * s)), corners)
            mx = max(mx, float(np.abs(a - b).max()))
        if mx < 0.4:
            return [0.0], mx
        k = int(np.clip(math.ceil(mx / 0.6) + 1, 3, 15))
        return list(np.linspace(-0.5 * s, 0.5 * s, k)), mx

    def window(self, bl: Blood, i: int, taus, body=True, drops=True):
        boxes = []
        if body:
            for p in BODY:
                for tau in (taus[0], taus[-1]):
                    b = bl.cv[p].bbox_img(bl.T(p, i, tau))
                    if b is not None:
                        boxes.append(b)
        for p in (RIM, FLOOR):
            b = bl.cv[p].bbox_img(bl.T(p, i))
            if b is not None:
                boxes.append(b)
        b = bl.seat.bbox_img(bl.off)
        if b is not None:
            boxes.append(b)
        if drops:
            for d in bl.shown_drops:
                for q in d["hist"][-2:]:
                    boxes.append((q[0], q[1], q[0], q[1]))
        if not boxes:
            return None
        B = np.array(boxes)
        h, w = self.sc.shape
        x0 = int(max(np.floor(B[:, 0].min()) - 8, 0))
        y0 = int(max(np.floor(B[:, 1].min()) - 8, 0))
        x1 = int(min(np.ceil(B[:, 2].max()) + 9, w))
        y1 = int(min(np.ceil(B[:, 3].max()) + 9, h))
        x0 -= x0 % 2
        y0 -= y0 % 2
        x1 += x1 % 2
        y1 += y1 % 2
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        return x0, y0, x1, y1

    def _blur(self, a):
        return cv2.GaussianBlur(a, (0, 0), self.blur) if self.blur > 0.3 and a.any() else a

    def static_layers(self, bl: Blood, i: int, win, drops=True):
        x0, y0, x1, y1 = win
        fab, liq = bl.seat.field(win, bl.off)
        rim = bl.cv[RIM].push(bl.T(RIM, i), None, win)
        flo = bl.cv[FLOOR].push(I23, None, win)
        air = np.zeros((y1 - y0, x1 - x0), np.float32)
        if drops and bl.shown_drops:
            ss = 3
            grid = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), np.float32)
            xs, ys, dxs, dys, sig, amp = [], [], [], [], [], []
            s = self.prm.shutter
            for d in bl.shown_drops:
                p1 = d["hist"][-1]
                p0 = d["hist"][-2] if len(d["hist"]) > 1 else p1
                # the exposure is centred on the frame time and lasts shutter x frame interval
                mid = p1 - 0.5 * (1 - s) * (p1 - p0) * 0.0
                trav = (p1 - p0) * s
                r_px = (3 * d["V"] / (4 * math.pi)) ** (1 / 3) * self.prm.px_per_mm_seat
                xs.append(mid[0] - 0.5 * trav[0])
                ys.append(mid[1] - 0.5 * trav[1])
                dxs.append(trav[0])
                dys.append(trav[1])
                sig.append(max(r_px / 1.1, 0.45))
                amp.append(1.0)
            _splat(
                grid,
                np.array(xs, np.float64),
                np.array(ys, np.float64),
                np.array(dxs, np.float64),
                np.array(dys, np.float64),
                np.array(sig, np.float64),
                np.array(amp, np.float64),
                float(x0),
                float(y0),
                ss,
            )
            air = cv2.resize(grid, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        return fab, liq, rim, flo, np.clip(air, 0.0, 1.0)

    # ---------------------------------------------------------------- shading
    def shade(self, plate, lum, E, g, hh, fb, a, hair, n0, px_per_mm):
        """Linear-light composite of one sub-frame: the plate under an absorbing layer."""
        prm = self.prm
        sig = np.array(prm.sigma_rgb, np.float32)
        rho = np.array(prm.rho_rgb, np.float32)
        fbv = fb * smoothstep(0.004, 0.02, fb)
        T = np.exp(-(g + 0.7 * hh + 2.6 * fbv)[..., None] * sig)
        wet = 1.0 - 0.3 * smoothstep(0.0, 0.2, fbv) - 0.25 * smoothstep(0.0, 0.25, hh)
        body = E * rho * (1.0 - np.exp(-2.2 * (g + 0.35 * hh + 1.5 * fbv)))[..., None]
        fib = np.clip(0.05 * (1.0 + 1.5 * self._tex), 0.01, 0.12) * smoothstep(0.0, 0.05, fbv)
        Tf = T * (1.0 - fib[..., None]) + fib[..., None]
        # Under a film thicker than a smear the skin's fine structure is lost (the film has its
        # own smooth surface): the plate's fine detail is seen through thin blood only. (The
        # luma grain comes back afterwards, from the plate-matched regrain.)
        pb = cv2.GaussianBlur(plate, (0, 0), 1.2)
        keep = np.exp(-5.0 * (g + hh))[..., None]
        out = (pb + (plate - pb) * keep) * Tf * wet[..., None] + body
        # wet specular: film normals from the thickness gradient, on the base surface normal
        hs = cv2.GaussianBlur(g + hh, (0, 0), 0.7)
        k = px_per_mm / 8.0
        gx = cv2.Sobel(hs, cv2.CV_32F, 1, 0, ksize=3) * k
        gy = cv2.Sobel(hs, cv2.CV_32F, 0, 1, ksize=3) * k
        n0x, n0y, n0z, horiz = n0
        nxx = n0x - gx * 2.5
        nyy = n0y - gy * 2.5
        nn = np.sqrt(nxx**2 + nyy**2 + n0z**2)
        nxx, nyy, nzz = nxx / nn, nyy / nn, n0z / nn
        ndh = np.clip(nxx * self.H[0] + nyy * self.H[1] + nzz * self.H[2], 0, 1)
        ndh2 = np.clip(nxx * self.H2[0] + nyy * self.H2[1] + nzz * self.H2[2], 0, 1)
        F = np.minimum(0.025 + 0.975 * (1 - np.clip(nzz, 0, 1)) ** 5, 0.08)
        # (the fixtures are small: only where the meniscus tilts a level surface just so)
        D = np.where(horiz, ndh2**400.0, ndh**prm.spec_alpha)
        gloss = smoothstep(0.1, 0.35, g + hh) * (1 - 0.5 * hair)
        spec = prm.spec_k * 12.0 * E * (F * D * gloss)[..., None]
        out = out + spec
        env = prm.env_reflect * F * gloss * np.clip(E.mean(-1) / 0.25, 0.0, 1.5) * 0.24
        out = out + env[..., None] * self.wb
        # wet black hair: the strands' own highlights clump and sharpen (anisotropic streaks
        # along the hair, from the plate itself, so they follow the real strands)
        wh = smoothstep(0.03, 0.2, hh) * hair
        if wh.any():
            hp = np.maximum(self._hair_hp, 0.0)[..., None]
            out = out + (7.0 * hp * wh[..., None]) * self.wb
        drop_col = E * rho * 0.9 + spec * 0.3
        out = out * (1 - a[..., None]) + a[..., None] * drop_col
        return out

    def base_normals(self, M, win, body=True):
        x0, y0, x1, y1 = win
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        n0x = np.zeros_like(xx)
        n0y = np.zeros_like(xx)
        vis = M["m"]
        sl = (slice(y0, y1), slice(x0, x1))
        if body:
            hc = M["head_c"]
            R = self.sc.tr.head_r
            nx = (xx - hc[0]) / (1.2 * R)
            ny = (yy - hc[1]) / (1.2 * R)
            on_head = (vis[HEAD][sl] > 0.3) & (nx * nx + ny * ny < 0.97)
            n0x = np.where(on_head, nx, 0.0)
            n0y = np.where(on_head, ny, 0.0)
            e_, w_, tip_ = M["hand"]
            best = np.full(xx.shape, 1e9, np.float32)
            vx = np.zeros_like(xx)
            vy = np.zeros_like(xx)
            for a_, b_, rad in ((e_, w_, 9.0), (w_, tip_, 10.0)):
                ab = b_ - a_
                L2 = float(ab @ ab) + 1e-6
                tt = np.clip(((xx - a_[0]) * ab[0] + (yy - a_[1]) * ab[1]) / L2, 0, 1)
                ox = xx - (a_[0] + tt * ab[0])
                oy = yy - (a_[1] + tt * ab[1])
                dd = np.hypot(ox, oy)
                closer = dd < best
                best = np.where(closer, dd, best)
                vx = np.where(closer, ox / rad, vx)
                vy = np.where(closer, oy / rad, vy)
            vr = np.hypot(vx, vy)
            scv = np.where(vr > 0.95, 0.95 / np.maximum(vr, 1e-6), 1.0)
            on_hand = (vis[RHAND][sl] + vis[RFORE][sl]) > 0.3
            n0x = np.where(on_hand, vx * scv, n0x)
            n0y = np.where(on_hand, vy * scv, n0y)
        horiz = (M["seat_full"][sl] + self.sc.floor_static[sl] + vis[LHAND][sl]) > 0.5
        n0y = np.where(horiz, -0.72, n0y)
        n0x = np.where(horiz, 0.0, n0x)
        n0z = np.sqrt(np.clip(1 - n0x**2 - n0y**2, 0.05, 1))
        return n0x, n0y, n0z, horiz

    # ---------------------------------------------------------------- composite
    def composite(self, i, Y, U, V, rgb, M, bl: Blood, body=True):
        """Return new (Y, U, V) ROI planes, the visible-blood mask and per-part pixel counts."""
        prm = self.prm
        taus, motion = self.taus(bl, i) if body else ([0.0], 0.0)
        win = self.window(bl, i, taus, body)
        if win is None:
            return Y, U, V, np.zeros(Y.shape, bool), {}, 0
        x0, y0, x1, y1 = win
        sl = (slice(y0, y1), slice(x0, x1))
        vis = M["m"]
        fab, liq, rim, flo, air = self.static_layers(bl, i, win)
        fab = self._blur(fab * vis[SEAT][sl])
        liq = liq * vis[SEAT][sl]
        rim = rim * vis[RIM][sl]
        flo = flo * vis[FLOOR][sl]
        occl = np.clip(sum(vis[p][sl] for p in (HEAD, RHAND, RFORE, LHAND)), 0, 1) if body else 0.0
        air = self._blur(air) * (1.0 - occl)
        stat_gloss = self._blur(rim + flo + liq)
        # the hair class is noisy pixel to pixel (dark skin in shadow, stray strands): blood
        # on the head reads as wet hair or as blood on skin by region, not by pixel
        hair = smoothstep(0.25, 0.6, cv2.GaussianBlur(M["hair"][sl], (0, 0), 1.3))
        plate = srgb_to_lin(rgb[sl])
        lum = luminance(plate)
        # local illumination: plate luminance over a class albedo guess
        alb = np.full(lum.shape, 0.3, np.float32)
        alb = alb * (1 - hair) + 0.05 * hair
        alb = alb * (1 - M["navy"][sl]) + 0.06 * M["navy"][sl]
        seatf = M["seat_full"][sl]
        alb = alb * (1 - seatf) + 0.45 * seatf
        flr = self.sc.floor_static[sl] * (1 - seatf)
        alb = alb * (1 - flr) + 0.5 * flr
        E = cv2.GaussianBlur(lum, (0, 0), 4.0) / cv2.GaussianBlur(alb, (0, 0), 4.0)
        E = np.clip(E, 0.01, 0.8)[..., None] * self.wb
        self._tex = self.tex_n[sl]
        lg = np.log(np.maximum(lum, 1e-4))
        self._hair_hp = np.clip(np.exp(lg) - np.exp(cv2.GaussianBlur(lg, (0, 0), 1.5)), 0, 0.03)
        n0 = self.base_normals(M, win, body)
        acc = np.zeros_like(plate)
        g_mid = h_mid = None
        parts_h = {}
        for tau in taus:
            g = stat_gloss.copy()
            hh = np.zeros_like(g)
            if body:
                for p in BODY:
                    c = bl.cv[p]
                    if not c.height().any():
                        continue
                    lay = c.push(bl.T(p, i, tau), None, win) * vis[p][sl]
                    lay = self._blur(lay)
                    if p == HEAD:
                        g += lay * (1 - hair)
                        hh += lay * hair
                    else:
                        g += lay
                    parts_h[p] = parts_h.get(p, 0) + lay / len(taus)
            acc += self.shade(plate, lum, E, g, hh, fab, air, hair, n0, bl.ppm)
            if g_mid is None or abs(tau) < 1e-6:
                g_mid, h_mid = g, hh
        out = acc / len(taus)
        tot = g_mid + h_mid + fab + air
        # black level: blood never drives a pixel below the plate's own darkest value
        rgb_out = lin_to_srgb(out)
        rgb_in = np.clip(rgb[sl], 0, 1)
        yo, uo, vo = rgb_to_yuv(rgb_out)
        yi, ui, vi = rgb_to_yuv(rgb_in)
        yo = np.maximum(yo, np.minimum(yi, self.yblack))
        if self.gamut is not None:
            # Hold the blood inside the red envelope this camera actually produces. A colour
            # outside it is moved toward black along its own hue (a thicker, darker layer of the
            # same blood) rather than greyed, which would read as red light instead of blood.
            g_y = np.array(self.gamut["y"])
            g_cr = np.array(self.gamut["cr_max"])
            dv = vo - 128.0
            du = uo - 128.0
            ye = yo - 16.0
            sc_ = np.ones_like(yo)
            for sv in np.linspace(1.0, 0.4, 13):
                ok = dv * sv <= np.interp(16.0 + ye * sv, g_y, g_cr) * 1.05
                sc_ = np.where((sc_ == 1.0) & ~ok, sv - 0.05, sc_)
            sc_ = np.clip(sc_, 0.4, 1.0)
            yo = 16.0 + ye * sc_
            dv = dv * sc_
            du = du * sc_
            lim = np.interp(yo, g_y, g_cr) * 1.05
            dv = np.where(dv > lim, lim + 0.2 * (dv - lim), dv)
            vo = 128.0 + dv
            uo = 128.0 + du
        dY = yo - yi
        dU = uo - ui
        dV = vo - vi
        touched = smoothstep(0.3, 1.5, np.abs(dY)).astype(np.float32)
        # The layer multiplies the plate's texture down with its light. What the camera sees
        # of a film on skin still carries the skin's relief (the film follows it and its
        # sheen breaks up on it), and its own sensor noise is added after, not attenuated:
        # restore part of the plate's fine detail, then top the noise back up to the
        # measured noise-vs-level curve.
        gain = np.clip((yo - 16) / np.maximum(yi - 16, 1.0), 0, 1)
        hp = yi - cv2.GaussianBlur(yi, (0, 0), 0.9)
        dY = dY + hp * (1.0 - gain) * 0.55 * touched
        cen = np.array(self.noise["y_centers"])
        sg = np.array(self.noise["y_sigma"])
        s_out = np.interp(yo, cen, sg)
        s_in = np.interp(yi, cen, sg)
        add = np.sqrt(np.maximum(s_out**2 - (gain * s_in) ** 2, 0.0))
        rng = np.random.default_rng(prm.seed * 100003 + i)
        nz = cv2.GaussianBlur(rng.standard_normal(yo.shape).astype(np.float32), (0, 0), 0.6)
        nz /= nz.std() + 1e-6
        dY = dY + nz * add * touched
        Yn = Y.astype(np.float32).copy()
        Yn[sl] = Yn[sl] + dY
        # chroma: the delta goes through the same 4:2:0 subsampling and softness as the plate,
        # and gets the plate's own chroma noise back where the layer took it away
        h2, w2 = (y1 - y0) // 2, (x1 - x0) // 2
        dU2 = cv2.GaussianBlur(
            cv2.resize(dU.astype(np.float32), (w2, h2), interpolation=cv2.INTER_AREA),
            (0, 0),
            0.55,
        )
        dV2 = cv2.GaussianBlur(
            cv2.resize(dV.astype(np.float32), (w2, h2), interpolation=cv2.INTER_AREA),
            (0, 0),
            0.55,
        )
        t2 = cv2.resize(touched * (1.0 - gain), (w2, h2), interpolation=cv2.INTER_AREA)
        cs = float(self.noise.get("c_sigma", 0.45))
        nu = cv2.GaussianBlur(rng.standard_normal((h2, w2)).astype(np.float32), (0, 0), 0.6)
        nv = cv2.GaussianBlur(rng.standard_normal((h2, w2)).astype(np.float32), (0, 0), 0.6)
        dU2 += nu / (nu.std() + 1e-6) * cs * t2
        dV2 += nv / (nv.std() + 1e-6) * cs * t2
        Un = U.astype(np.float32).copy()
        Vn = V.astype(np.float32).copy()
        Un[y0 // 2 : y1 // 2, x0 // 2 : x1 // 2] += dU2
        Vn[y0 // 2 : y1 // 2, x0 // 2 : x1 // 2] += dV2
        if self.gamut is not None:
            # Final guard, on what is actually encoded: each chroma sample shared by a 2x2
            # block of luma must sit inside the camera's envelope for the darkest of them
            # (after the regrain moved the luma), where the blood touched the block.
            yb = Yn[y0:y1, x0:x1].reshape(h2, 2, w2, 2).min(axis=(1, 3))
            lim = np.interp(yb, np.array(self.gamut["y"]), np.array(self.gamut["cr_max"]))
            vv = Vn[y0 // 2 : y1 // 2, x0 // 2 : x1 // 2]
            v_pl = V[y0 // 2 : y1 // 2, x0 // 2 : x1 // 2].astype(np.float32)
            tb = cv2.resize(touched, (w2, h2), interpolation=cv2.INTER_AREA) > 0.05
            cap_v = 128.0 + np.maximum(0.82 * lim, v_pl - 128.0)  # (headroom for the encoder)
            vv[:] = np.where(tb, np.minimum(vv, cap_v), vv)
        Yq = np.clip(np.round(Yn), 0, 255).astype(np.uint8)
        Uq = np.clip(np.round(Un), 0, 255).astype(np.uint8)
        Vq = np.clip(np.round(Vn), 0, 255).astype(np.uint8)
        # ground truth: where the blood visibly changed the picture
        diff = np.abs(rgb_out - rgb_in).max(-1) * 255
        visible = np.zeros(Y.shape, bool)
        visible[sl] = (diff > 6.0) & (tot > 1e-3)
        parts = {}
        hand_h = sum(parts_h.get(p, 0) for p in (RHAND, RFORE, LHAND))
        for name, fld in (
            ("head", parts_h.get(HEAD, 0)),
            ("hand", hand_h),
            ("seat", fab + liq + rim),
            ("floor", flo),
            ("drops", air),
        ):
            if isinstance(fld, int):
                parts[name] = 0
                continue
            parts[name] = int(((fld > 0.01) & visible[sl]).sum())
        return Yq, Uq, Vq, visible, parts, motion


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------


def track_parts(src, sc: Scene, plates: Plates, fa: int, fb: int, refs: dict, log) -> dict:
    """Pass 1: a similarity per body part per frame, chained from the plate's DIS flow
    (forward/backward checked) inside each part's own visible matte, from each part's
    reference frame outward. Where a part shows too few reliable pixels, the skeleton's own
    bone carries it for that frame."""
    tr = sc.tr
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(0)
    rng = np.random.default_rng(0)
    h, w = sc.shape
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    D = {p: {} for p in BODY}
    n_fb = {p: 0 for p in BODY}
    R = tr.head_r
    er = np.ones((3, 3), np.uint8)
    for i in range(fa, fb):
        rgb = plates.rgb(i)
        M = sc.mattes(i, rgb, srgb_to_lin(rgb))
        g0, g1 = plates.gray(i), plates.gray(i + 1)
        f = dis.calc(g0, g1, None)
        bw = dis.calc(g1, g0, None)
        bwa = cv2.remap(
            bw,
            gx + f[..., 0],
            gy + f[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        fbe = np.linalg.norm(f + bwa, axis=-1)
        vis = M["m"]
        masks = {
            HEAD: (vis[HEAD] > 0.5) & (disk(sc.shape, sc.head_c[i], 1.25 * R) > 0.5),
            RHAND: cv2.erode((vis[RHAND] > 0.5).astype(np.uint8), er) > 0,
            RFORE: cv2.erode((vis[RFORE] > 0.5).astype(np.uint8), er) > 0,
            LHAND: cv2.erode((vis[LHAND] > 0.5).astype(np.uint8), er) > 0,
        }
        e0, w0, t0 = tr.hand(i)
        e1, w1, t1 = tr.hand(i + 1)
        _le0, lw0, _lt0 = sc.left_hand(i)
        _le1, lw1, _lt1 = sc.left_hand(i + 1)
        dc = sc.head_c[i + 1] - sc.head_c[i]
        fall = {
            HEAD: np.float32([[1, 0, dc[0]], [0, 1, dc[1]]]),
            RHAND: sim_from_segments(w0, t0, w1, t1),
            RFORE: sim_from_segments(e0, w0, e1, w1),
            LHAND: np.float32([[1, 0, lw1[0] - lw0[0]], [0, 1, lw1[1] - lw0[1]]]),
        }
        for p in BODY:
            A, n = fit_increment(f, fbe, masks[p], fall[p], rng)
            D[p][i] = A
            n_fb[p] += n == 0
    tracks = {}
    for p in BODY:
        ref = int(np.clip(refs[p], fa, fb))
        T = {ref: I23.copy()}
        for i in range(ref, fb):
            T[i + 1] = compose(D[p][i], T[i])
        for i in range(ref, fa, -1):
            T[i - 1] = compose(cv2.invertAffineTransform(D[p][i - 1]), T[i])
        tr_ = PartTrack(T, ref)
        tracks[p] = tr_
    log(
        "part tracks: frames using the skeleton fallback "
        + ", ".join(f"{PARTS[p]} {n_fb[p]}" for p in BODY)
    )
    return tracks


def derive_wound(tr: Tracks, sc: Scene, head: PartTrack, fps: float, window=(35.0, 36.0)):
    """The wound is where the hand lands. Of the two wrists, take the one that sits closest
    to the head over the press window; the press point is 90% of the way from that wrist to
    the tip of the hand (under the fingertips, which is where the temple hairline is while
    the palm cups the cheek), carried into the head's own coordinates."""
    frames = [i for i in range(round(window[0] * fps), round(window[1] * fps) + 1) if i in head.T]
    dl = np.mean([np.linalg.norm(tr.kp(i)[LWRI] - sc.head_c[i]) for i in frames])
    dr = np.mean([np.linalg.norm(tr.kp(i)[RWRI] - sc.head_c[i]) for i in frames])
    side = "right" if dr <= dl else "left"
    offs = []
    for i in frames:
        if side == "right":
            _e, w, tip = tr.hand(i)
        else:
            _e, w, tip = sc.left_hand(i)
        offs.append(head.inv(i, (w + 0.9 * (tip - w))[None])[0])
    return np.mean(offs, 0).astype(np.float32), side, float(dl), float(dr)


def encode(src: Source, out_path: Path, planes: dict, roi: Roi, log):
    ys, xs = roi.sl
    ys2, xs2 = roi.sl2
    oc = av.open(str(out_path), "w")
    st = oc.add_stream("libx264", rate=Fraction(src.fps).limit_denominator(1_000_000))
    st.width, st.height = src.width, src.height
    st.pix_fmt = "yuv420p"
    st.time_base = src.time_base
    st.codec_context.time_base = src.time_base
    # the camera's own GOP structure, at the camera's own bitrate, so the composite region
    # carries the same compression character as the plate around it
    st.bit_rate = src.bit_rate
    st.options = {
        "preset": "medium",
        "x264-params": "keyint=20:min-keyint=20:scenecut=0:bframes=0:ref=1",
    }
    n = 0
    for i, y, u, v in iter_yuv(src.path):
        if i in planes:
            Y, U, V = planes[i]
            y = y.copy()
            u = u.copy()
            v = v.copy()
            y[ys, xs] = Y
            u[ys2, xs2] = U
            v[ys2, xs2] = V
        buf = np.concatenate([y.reshape(-1), u.reshape(-1), v.reshape(-1)]).reshape(-1, src.width)
        vf = av.VideoFrame.from_ndarray(buf, format="yuv420p")
        vf.pts = int(src.pts[i])
        vf.time_base = src.time_base
        for pkt in st.encode(vf):
            oc.mux(pkt)
        n += 1
    for pkt in st.encode():
        oc.mux(pkt)
    oc.close()
    log(f"encoded {n} frames -> {out_path}")


def debug_overlay(path: Path, bl: Blood, M: dict, rgb: np.ndarray, t: float) -> None:
    """A 4x crop around the head: part supports as contours, beads as circles."""
    ov = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
    z = 4
    hc = M["head_c"]
    h, w = ov.shape[:2]
    cx0 = int(np.clip(hc[0] - 70, 0, w - 140))
    cy0 = int(np.clip(hc[1] - 50, 0, h - 140))
    crop = cv2.resize(
        ov[cy0 : cy0 + 140, cx0 : cx0 + 140],
        None,
        fx=z,
        fy=z,
        interpolation=cv2.INTER_NEAREST,
    )
    cols = {
        RHAND: (0, 0, 255),
        HEAD: (0, 255, 0),
        LHAND: (255, 255, 0),
        RFORE: (255, 0, 255),
    }
    for p, col in cols.items():
        mm = cv2.resize(M["sup"][p][cy0 : cy0 + 140, cx0 : cx0 + 140], None, fx=z, fy=z)
        cnts, _ = cv2.findContours(
            (mm > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(crop, cnts, -1, col, 1)
    for q, pp in zip(bl.b_pos, bl.b_part, strict=True):
        qi = bl.to_img(int(pp), q[None])[0]
        cv2.circle(
            crop,
            (int((qi[0] - cx0) * z), int((qi[1] - cy0) * z)),
            4,
            cols.get(int(pp), (255, 255, 255)),
            1,
        )
    if bl.wound is not None:
        wi = bl.to_img(HEAD, bl.wound[None])[0]
        cv2.circle(
            crop,
            (int((wi[0] - cx0) * z), int((wi[1] - cy0) * z)),
            6,
            (255, 255, 255),
            1,
        )
    cv2.putText(crop, f"{t:.2f}", (4, 16), 0, 0.5, (0, 255, 255), 1)
    cv2.imwrite(str(path), crop)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--onset", type=float, default=34.0)
    ap.add_argument("--end", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="pose cache .npz (reused if it covers the window)",
    )
    ap.add_argument("--clean-early", default="30.0:33.0")
    ap.add_argument("--clean-late", default="44.5:48.3")
    ap.add_argument("--roi", default="680,760,620,536", help="x,y,w,h of the effect region")
    ap.add_argument("--debug-dir", type=Path, default=None)
    ap.add_argument("--no-encode", action="store_true")
    ap.add_argument("--dump", default="", help="debug: comma-separated frames whose layers to save")
    ap.add_argument(
        "--no-persist",
        action="store_true",
        help="do not keep the seat/floor stains after --end (by default they stay: stains do not vanish)",
    )
    ap.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="override a Params field, e.g. --param initial_boost=2.5 (repeatable)",
    )
    a = ap.parse_args(argv)
    t_start = time.time()

    def log(msg):
        print(f"[bleed {time.time() - t_start:6.1f}s] {msg}", file=sys.stderr, flush=True)

    overrides = {}
    for kv in a.param:
        k, _, v = kv.partition("=")
        if k not in Params.__dataclass_fields__:
            raise SystemExit(f"--param: Params has no field {k!r}")
        cur = getattr(Params, k)
        overrides[k] = type(cur)(v) if not isinstance(cur, tuple) else tuple(float(x) for x in v.split(","))
    prm = Params(seed=a.seed, onset=a.onset, end=a.end, **overrides)
    if overrides:
        log(f"Params overrides: {overrides}")
    src = probe(a.src)
    fps = src.fps
    rx, ry, rw, rh = (int(v) for v in a.roi.split(","))
    roi = Roi(rx - rx % 2, ry - ry % 2, rw - rw % 2, rh - rh % 2)
    early = tuple(float(v) for v in a.clean_early.split(":"))
    late = tuple(float(v) for v in a.clean_late.split(":"))
    i_on = math.ceil(a.onset * fps - 1e-6)
    i_end = min(math.floor(a.end * fps + 1e-6), src.n_frames - 1)
    i0 = min(round(early[0] * fps), i_on - 2)
    i1 = min(max(i_end + 1, round(late[1] * fps)), src.n_frames - 1)
    log(
        f"{src.path.name}: {src.width}x{src.height}, {src.n_frames} frames at {fps:.4f} fps, "
        f"{src.bit_rate / 1e6:.2f} Mb/s; effect frames {i_on}..{i_end}"
    )
    cache = a.cache or (a.out.parent / f".{a.out.stem}.pose.npz")
    pose = run_pose(src, i0, i1, cache)
    tr = Tracks(src, pose, roi)
    log(
        f"head: ear-to-ear {tr.ear_px:.1f} px -> {tr.head_w_px:.0f} px wide -> "
        f"{tr.px_per_mm:.3f} px/mm; bead speed <= {prm.v_max_mm_s * tr.px_per_mm:.1f} px/s"
    )
    plates = Plates(src, roi, i0, i1)
    sc = Scene(plates, tr, early, late, log)
    sc.track_head(log)
    # plate statistics
    still = [i for i in range(round(late[0] * fps), round(late[1] * fps) + 1)][:40]
    sk = np.max(
        np.stack([skeleton_mask(tr, i, sc.shape, sc.head_c.get(i), pad=20) for i in still[::8]]),
        0,
    )
    noise = measure_noise(plates, still, sk < 0.5)
    blur = measure_blur(src, still[:6])
    door = srgb_to_lin(sc.bg[0:80, 480:600]).reshape(-1, 3)
    wb = np.median(door, 0)
    wb = (wb / luminance(wb[None])[0]).astype(np.float32)
    yblack = float(np.percentile(np.stack([plates.Y[i] for i in still[::4]]), 0.5))
    i_light = round(33.0 * fps)
    rgb_l = plates.rgb(i_light)
    M_l = sc.mattes(i_light, rgb_l, srgb_to_lin(rgb_l))
    light = estimate_light(rgb_l, M_l["sup"][HEAD], M_l["hair"], M_l["head_c"], tr.head_r)
    log(
        f"plate: noise Y sigma {np.round(noise['y_sigma'], 2).tolist()} (codes), "
        f"blur sigma {blur:.2f} px, wb {wb.round(3).tolist()}, black Y {yblack:.1f}, "
        f"key light {light.round(2).tolist()}"
    )
    gamut = measure_gamut(src, i_light)
    log(
        f"camera red envelope (Y: Cr max): "
        f"{', '.join(f'{a_:.0f}:{b_:.0f}' for a_, b_ in zip(gamut['y'], gamut['cr_max'], strict=True))}"
    )
    # ---- pass 1: part tracks
    fa, fb = i_on - 1, min(max(i_end + 1, round(36.5 * fps)), i1)  # (covers the press window)
    i_lref = int(np.clip(round(42.0 * fps), fa, fb))
    refs = {HEAD: i_on, RHAND: i_on, RFORE: i_on, LHAND: i_lref}
    tcache = a.out.parent / f".{a.out.stem}.tracks.npz"
    # keyed by the window, the references, the ROI and this file's own code: tracks come off
    # the mattes, so any edit here recomputes them
    code = zlib.crc32(Path(__file__).read_bytes())
    key = np.array([fa, fb, *[refs[p] for p in BODY], *roi.__dict__.values(), code], np.int64)
    tracks = None
    if tcache.exists():
        d = np.load(tcache)
        if np.array_equal(d["key"], key):
            tracks = {
                p: PartTrack({int(i): d[f"T{p}"][n] for n, i in enumerate(d["idx"])}, refs[p])
                for p in BODY
            }
            log(f"part tracks from {tcache}")
    if tracks is None:
        tracks = track_parts(src, sc, plates, fa, fb, refs, log)
        idx = np.arange(fa, fb + 1)
        np.savez(
            tcache,
            key=key,
            idx=idx,
            **{f"T{p}": np.stack([tracks[p].T[i] for i in idx]) for p in BODY},
        )
    wound, side, dl, dr = derive_wound(tr, sc, tracks[HEAD], fps)
    log(
        f"wound from the {side} wrist (mean wrist-head distance L {dl:.0f} px, R {dr:.0f} px): "
        f"({wound[0] + roi.x0:.1f}, {wound[1] + roi.y0:.1f}) in head coordinates at onset"
    )
    # canonical boxes of the part canvases
    R = tr.head_r
    hc = sc.head_c[i_on]
    e_, w_, t_ = tr.hand(i_on)
    _le, lw_, _lt = sc.left_hand(i_lref)

    def bb(pts, m):
        p = np.array(pts)
        return (
            p[:, 0].min() - m,
            p[:, 1].min() - m,
            p[:, 0].max() + m,
            p[:, 1].max() + m,
        )

    rim_pts = np.array(SEAT_RIM, np.float32) - [roi.x0, roi.y0]
    boxes = {
        HEAD: (hc[0] - 3.3 * R, hc[1] - 3.0 * R, hc[0] + 3.3 * R, hc[1] + 3.3 * R),
        RHAND: bb([w_, t_], 20),
        RFORE: bb([e_, w_], 18),
        LHAND: bb([lw_ - 46.0, lw_ + 46.0], 0),
        RIM: bb(rim_pts, 6),
        FLOOR: (930 - roi.x0, FLOOR_TOP_Y + 40 - roi.y0, 1150 - roi.x0, roi.h),
    }
    bl = Blood(prm, tr, sc, tracks, boxes, fps)
    tang = np.array([0.5, -0.87], np.float32) * (28.0 * tr.px_per_mm)  # ~2.8 cm laceration
    bl.init_wound(wound - 0.5 * tang, tang)
    rd = Renderer(prm, sc, blur, noise, light, wb, yblack, gamut, fps)
    # ---- pass 2: simulate and composite the effect window
    planes = {}
    per = {}
    for i in range(i_on, i_end + 1):
        t = i / fps
        rgb = plates.rgb(i)
        lin = srgb_to_lin(rgb)
        M = sc.mattes(i, rgb, lin)
        bl.step(i, t, M)
        Y, U, V, visible, parts, motion = rd.composite(
            i, plates.Y[i], plates.U[i], plates.V[i], rgb, M, bl
        )
        planes[i] = (Y, U, V)
        if visible.any():
            ys_, xs_ = np.nonzero(visible)
            per[i] = {
                "bbox": [int(xs_.min() + roi.x0), int(ys_.min() + roi.y0),
                         int(xs_.max() + roi.x0 + 1), int(ys_.max() + roi.y0 + 1)],
                "area": int(visible.sum()),
                "parts": [k for k, v in parts.items() if v >= 3],
            }  # fmt: skip
        if a.debug_dir is not None and str(i) in a.dump.split(","):
            a.debug_dir.mkdir(parents=True, exist_ok=True)
            win_ = (0, 0, roi.w, roi.h)
            np.savez_compressed(
                a.debug_dir / f"layers{i:05d}.npz",
                **{f"p{p}": bl.cv[p].push(bl.T(p, i), None, win_) for p in BODY},
                **{f"v{p}": M["m"][p] for p in range(NPART)},
                **{f"s{p}": M["sup"][p] for p in range(NPART)},
                Y=Y,
                U=U,
                V=V,
                Y0=plates.Y[i],
                U0=plates.U[i],
                V0=plates.V[i],
            )
        if a.debug_dir is not None and (i % 5 == 0):
            a.debug_dir.mkdir(parents=True, exist_ok=True)
            debug_overlay(a.debug_dir / f"o{i:05d}.png", bl, M, rgb, t)
        if (i - i_on) % 40 == 0:
            log(
                f"t={t:6.2f}s beads {len(bl.b_V)} drops {len(bl.drops)} spots {len(bl.spots)} "
                f"emitted {bl.emitted / 1000:.2f} ml seat {bl.seat.volume:.0f} mm3 "
                f"motion {motion:.1f} px visible {per.get(i, {}).get('area', 0)} px"
            )
    for ev in bl.events:
        log("event " + ev)
    for ent in bl.seat_log[:: max(len(bl.seat_log) // 30, 1)]:
        log(f"seat arrival {ent}")
    # ---- pass 3: after --end, the stains on the chair and the floor stay where they are
    if not a.no_persist and i_end + 1 < src.n_frames:
        ys, xs = roi.sl
        ys2, xs2 = roi.sl2
        off = sc.chair_off.get(i_end, np.zeros(2, np.float32)).copy()
        bl.shown_drops = []
        bl.drops = []
        for c in bl.cv.values():
            c.bead[:] = 0.0
        Ms = None
        for i, y, u, v in iter_yuv(src.path):
            if i <= i_end:
                continue
            Yr, Ur, Vr = y[ys, xs].copy(), u[ys2, xs2].copy(), v[ys2, xs2].copy()
            h_, w_ = Yr.shape
            rgb = yuv_to_rgb(Yr, cv2.resize(Ur, (w_, h_)), cv2.resize(Vr, (w_, h_)))
            lin = srgb_to_lin(rgb)
            occ = sc.occluder(rgb, lin, off)
            off = (0.8 * off + 0.2 * sc.chair_drift(np.ascontiguousarray(Yr), occ < 0.3)).astype(
                np.float32
            )
            bl.off = off
            sc.chair_off[i] = off
            fabm = poly_mask(sc.shape, SEAT_FABRIC, off, roi)
            rimm = poly_mask(sc.shape, SEAT_RIM, off, roi)
            seat_full = guided(rgb, np.clip(fabm + rimm, 0, 1), 2, 1e-3)
            s_fab = seat_full * fabm / np.maximum(fabm + rimm, 1e-3)
            s_rim = seat_full - s_fab
            vis = np.zeros((NPART, *sc.shape), np.float32)
            vis[SEAT] = s_fab * (1 - occ)
            vis[RIM] = s_rim * (1 - occ)
            vis[FLOOR] = sc.floor_static * (1 - seat_full) * (1 - occ)
            Ms = {"m": vis, "seat_full": seat_full, "hair": np.zeros(sc.shape, np.float32),
                  "navy": np.zeros(sc.shape, np.float32), "off": off}  # fmt: skip
            bl.i = i
            Y, U, V, visible, parts, _ = rd.composite(i, Yr, Ur, Vr, rgb, Ms, bl, body=False)
            planes[i] = (Y, U, V)
            if visible.any():
                ys_, xs_ = np.nonzero(visible)
                per[i] = {
                    "bbox": [int(xs_.min() + roi.x0), int(ys_.min() + roi.y0),
                             int(xs_.max() + roi.x0 + 1), int(ys_.max() + roi.y0 + 1)],
                    "area": int(visible.sum()),
                    "parts": [k for k, v in parts.items() if v >= 3],
                }  # fmt: skip
        log(f"stains kept on the chair and floor through frame {src.n_frames - 1}")
    if not a.no_encode:
        encode(src, a.out, planes, roi, log)
    onset_vis = min(per) if per else None
    frames_json = []
    for i in range(src.n_frames):
        rec = {
            "i": i,
            "t": round(i / fps, 4),
            "pts_s": round(float(src.pts[i] * src.time_base), 4),
        }
        if i in per:
            rec.update(
                {
                    "blood_bbox_xyxy": per[i]["bbox"],
                    "area_px": per[i]["area"],
                    "parts": per[i]["parts"],
                }
            )
        else:
            rec.update({"blood_bbox_xyxy": None, "area_px": 0, "parts": []})
        frames_json.append(rec)
    doc = {
        "schema": "vividly.bleed_synth/2",
        "source": str(a.src),
        "output": str(a.out),
        "fps": fps,
        "clock": "t = frame_index / fps (the repo's presentation clock); pts_s is the container PTS",
        "onset_s": round(a.onset, 4),
        "first_visible_s": None if onset_vis is None else round(onset_vis / fps, 4),
        "effect_end_s": round(i_end / fps, 4),
        "stains_persist_after_end": not a.no_persist,
        "wound_xy_at_onset": [
            round(float(apply(tracks[HEAD].M(i_on), wound[None])[0][0] + roi.x0), 1),
            round(float(apply(tracks[HEAD].M(i_on), wound[None])[0][1] + roi.y0), 1),
        ],
        "wound_from": f"{side} wrist",
        "per_frame": frames_json,
        "events": bl.events,
        "params": {
            **{k: v for k, v in asdict(prm).items() if k != "extra"},
            "roi_xywh": [roi.x0, roi.y0, roi.w, roi.h],
            "px_per_mm_head": round(tr.px_per_mm, 4),
            "head_width_px": round(tr.head_w_px, 1),
            "plate_blur_sigma_px": round(blur, 3),
            "plate_noise": noise,
            "white_balance_rgb": wb.round(4).tolist(),
            "black_level_y": yblack,
            "key_light_dir": light.round(3).tolist(),
            "camera_red_envelope": gamut,
            "emitted_ml": round(bl.emitted / 1000, 3),
            "to_seat_ml": round(bl.seat.volume / 1000, 3),
            "into_shirt_ml": round(bl.to_cloth / 1000, 3),
            "under_palm_ml": round(getattr(bl, "under_vol", 0.0) / 1000, 3),
            "floor_ml": round(sum(s_["V"] for s_ in bl.spots) / 1000, 3),
            "floor_drops": int(sum(s_["n"] for s_ in bl.spots)),
            "seat_stain_area_cm2": round(
                float((bl.seat.c > 0.3 * bl.seat.cap).sum()) * bl.seat.CELL**2 / 100.0, 1
            ),
            "encode": {
                "codec": "libx264",
                "bit_rate": src.bit_rate,
                "keyint": 20,
                "bframes": 0,
                "pix_fmt": "yuv420p",
            },
            "clean_plate_early_s": early,
            "clean_plate_late_s": late,
            "hand_tip_keys": HAND_TIP_KEYS,
            "seat_fabric_poly": SEAT_FABRIC,
            "seat_rim_poly": SEAT_RIM,
            "seat_quad": SEAT_QUAD,
        },
    }
    a.json.write_text(json.dumps(doc, indent=1))
    log(
        f"sidecar -> {a.json}; first visible blood at "
        f"{doc['first_visible_s']} s; done in {time.time() - t_start:.0f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
