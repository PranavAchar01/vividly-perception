"""Per-frame foreground mattes for a green-screen version of a replay render.

    uv run python tools/greenscreen_mattes.py --in take.mp4 --start 70 --end 79.5 \\
        --out mattes/ --vision tools/vision_matte [--static chairs.png] [--delta-from original.mp4]

For every source frame in [start, end) this writes <out>/<frame_index:06d>.png, an 8-bit
matte at the replay's display size (1280 wide), which replay_render.py --matte-dir uses to
lay the frame over green before it draws the overlays. Frame indices and the resize are the
replay's own (frame_index = round(t * fps), INTER_AREA to the display width), so a matte
lands on exactly the frame it was made from.

The matte is the union of:
  * the person: Apple Vision's foreground-subject model (tools/vision_matte.swift, on device)
    run on a crop around every detected person, upscaled so the person fills the model's
    input. On the full 1280 x 720 frame the model returns nothing for a person this small.
    A frame where the detector finds no one reuses the last crop, so a patient the detector
    loses on the floor keeps their matte.
  * --static: a fixed matte for props that stay put (the chairs), made once from an empty
    moment of the take.
  * --delta-from: every pixel where this video differs from another by more than a
    threshold. For the head-wound take it is the composited blood, so the stains and the
    drips on the floor stay in the picture when the floor around them goes green.
Then a short temporal blend (the model is run frame by frame and flickers at the edges) and
a guided filter against the frame, so the edge follows hair and fabric instead of the
model's low-resolution mask.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vividly.models.factory import build_detector  # noqa: E402

DISP_W = 1280


def frames(path: str, start: float, end: float):
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    f0, f1 = round(start * fps), round(end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    for idx in range(f0, f1):
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        yield (
            idx,
            cv2.resize(f, (DISP_W, round(h * DISP_W / w / 2) * 2), interpolation=cv2.INTER_AREA),
        )
    cap.release()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--vision", required=True, help="compiled tools/vision_matte")
    ap.add_argument("--static", default=None, help="fixed matte PNG at display size (props)")
    ap.add_argument("--delta-from", default=None, help="keep pixels that differ from this video")
    ap.add_argument("--delta-thr", type=float, default=14.0)
    ap.add_argument("--config", default="configs/eval/live_final.yaml")
    ap.add_argument("--temporal", type=float, default=0.35, help="weight of the previous matte")
    a = ap.parse_args()

    det = build_detector(dict(yaml.safe_load((REPO / a.config).read_text())["detector"]))
    tmp = Path(tempfile.mkdtemp(prefix="gs_"))
    (tmp / "crops").mkdir()
    crops: dict[int, tuple[int, int, int, int]] = {}
    last = None
    n = 0
    for idx, f in frames(a.inp, a.start, a.end):
        h, w = f.shape[:2]
        d = det.detect_one(f).filter(0.3)
        # Crop around the person being followed, not around every box: a union with the
        # detector's small false boxes (a shelf, the frame edge) made the crop the whole frame
        # and the person too small for either Vision model again (take1 99 s: nothing back).
        # Person-sized boxes only; the one nearest the previous crop; plus any box touching
        # it (the detector often puts two boxes on one body, a torso and a whole).
        cand = [b for b in d.xyxy if max(b[2] - b[0], b[3] - b[1]) >= 0.15 * h]
        if cand:
            if last is None:
                main = cand[int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in cand]))]
            else:
                lc = np.array([(last[0] + last[2]) / 2, (last[1] + last[3]) / 2])
                main = min(cand, key=lambda b: float(np.hypot(*((b[:2] + b[2:]) / 2 - lc))))
            g = np.array(main, np.float32)
            mw, mh = g[2] - g[0], g[3] - g[1]
            grow = np.array([g[0] - 0.3 * mw, g[1] - 0.3 * mh, g[2] + 0.3 * mw, g[3] + 0.3 * mh])
            for bb in cand:
                if bb[0] < grow[2] and bb[2] > grow[0] and bb[1] < grow[3] and bb[3] > grow[1]:
                    g = np.array([min(g[0], bb[0]), min(g[1], bb[1]), max(g[2], bb[2]), max(g[3], bb[3])])
            cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
            s = max(g[2] - g[0], g[3] - g[1]) * 1.5
            last = (int(max(0, cx - s / 2)), int(max(0, cy - s / 2)),
                    int(min(w, cx + s / 2)), int(min(h, cy + s / 2)))  # fmt: skip
        if last is None:
            continue
        x1, y1, x2, y2 = last
        crops[idx] = last
        k = 1200.0 / max(x2 - x1, y2 - y1)
        cv2.imwrite(str(tmp / "crops" / f"{idx:06d}.png"),
                    cv2.resize(f[y1:y2, x1:x2], None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC))  # fmt: skip
        n += 1
    print(f"{n} person crops; running Vision ...", flush=True)
    # Both Vision models, unioned. The subject model gives the solid, clean matte, but on a
    # crop holding a person lying across two chairs it can pick the chairs as "the subject"
    # and drop the person (measured: take1 99 s, the head-wound take 47.5 s). The person
    # model never drops the person but lets the backdrop through a dark shirt. Each covers
    # the other's failure.
    subprocess.run([a.vision, "subject", str(tmp / "crops"), str(tmp / "masks")], check=True)
    subprocess.run([a.vision, "person", str(tmp / "crops"), str(tmp / "pmasks")], check=True)

    static = None
    if a.static:
        static = cv2.imread(a.static, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    delta_src = frames(a.delta_from, a.start, a.end) if a.delta_from else None

    a.out.mkdir(parents=True, exist_ok=True)
    prev = None
    for idx, f in frames(a.inp, a.start, a.end):
        h, w = f.shape[:2]
        m = np.zeros((h, w), np.float32)
        if idx in crops:
            x1, y1, x2, y2 = crops[idx]
            for mp in (tmp / "masks" / f"{idx:06d}.png", tmp / "pmasks" / f"{idx:06d}.png"):
                if mp.exists():
                    part = cv2.resize(cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE), (x2 - x1, y2 - y1),
                                      interpolation=cv2.INTER_AREA) / 255.0  # fmt: skip
                    # A person lying down gets the right SHAPE at low confidence (measured:
                    # ~0.2 on the body, 0 on the floor around it), which composited as a
                    # see-through patient. Rescale each crop's mask by its own peak, then cut
                    # the faint halo, so a weak-but-correct mask is solid and a confident one
                    # is unchanged.
                    peak = float(np.percentile(part, 99.5))
                    if peak > 0.05:
                        part = np.clip((part / max(peak, 0.2) - 0.15) / 0.6, 0.0, 1.0)
                    m[y1:y2, x1:x2] = np.maximum(m[y1:y2, x1:x2], part)
        if delta_src is not None:
            _, g = next(delta_src)
            diff = np.abs(f.astype(np.int16) - g.astype(np.int16)).max(-1).astype(np.float32)
            m = np.maximum(m, np.clip((diff - a.delta_thr) / 10.0, 0, 1))
        if prev is not None and a.temporal > 0:
            m = (1 - a.temporal) * m + a.temporal * prev
        prev = m
        if static is not None:
            m = np.maximum(m, static)
        guide = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        m = np.clip(cv2.ximgproc.guidedFilter(guide, m, 3, 1e-3), 0, 1)
        cv2.imwrite(str(a.out / f"{idx:06d}.png"), (m * 255).astype(np.uint8))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"mattes -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
