"""A monitoring-wall montage: one feed, then 4, then 9, each tile lighting red when its
own detection fires.

    uv run python scripts/camera_wall.py --spec wall.json --out wall.mp4

The spec is JSON:

    {
      "duration": 10.0, "fps": 30, "width": 1280, "height": 720,
      "phases": [{"t": 0.0, "grid": 1}, {"t": 2.2, "grid": 2}, {"t": 4.8, "grid": 3}],
      "transition": 0.5,
      "dim_at": 8.6, "black_at": 9.2, "wordmark": "VIVIDLY",
      "crop": [40, 90, 1120, 630],
      "tiles": [
        {"file": "F2.mp4", "src_start": 89.0, "json": "F2.json", "kind": "fall",
         "label": "FALL", "appear": 0.0, "fire_at": 1.6, "slots": {"1": [0, 0], "2": [0, 0]}},
        {"file": "idle1.mp4", "appear": 2.2, "offset": 0.5, "slots": {"2": [1, 1]}}
      ]
    }

Every clip is a replay_render output, so the boxes, skeletons and the alarm are the real
detector's, not drawn on afterwards. For an event tile the alarm time is read from the
render's own event JSON (the first `banner_t` of `kind`, the moment it reached the screen),
and the clip is offset so that moment lands at `fire_at` on the wall: the red border and
label follow the detector, never a hand-placed cue. `crop` (x, y, w, h in the render's
pixels) drops the HUD bar and banner, which are unreadable at tile size; the wall draws
its own border and label instead. A clip that runs out holds its last frame.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

_GROUND = (18, 15, 11)  # near-black, blue-biased, like a control room
_IDLE_EDGE = (58, 52, 44)
_ALARM = (60, 60, 245)  # the appliance's alarm red
_INK = (255, 255, 255)
_ACCENT = (242, 157, 169)  # the appliance's lavender
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GAP = 6


def _ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 4 * x**3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


class Clip:
    """Sequential reader: montage time only moves forward, so each tile reads forward."""

    def __init__(self, path: str, crop) -> None:
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 20.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.crop = crop
        self.idx = -1
        self.frame = None

    def at(self, t: float) -> np.ndarray:
        want = min(max(0, round(t * self.fps)), self.n - 1)
        if want < self.idx or (self.frame is None and want > 0):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, want)
            self.idx = want - 1
        while self.idx < want:
            ok, f = self.cap.read()
            if not ok:
                break
            self.idx += 1
            x, y, w, h = self.crop
            self.frame = f[y : y + h, x : x + w]
        return self.frame


def _alarm_t(tile: dict) -> float:
    """File time at which this tile's alarm first reached the screen."""
    events = json.loads(Path(tile["json"]).read_text())
    kind = tile["kind"]
    hits = [e for e in events if e["kind"] == kind and e.get("banner_t") is not None]
    if not hits:
        raise SystemExit(f"{tile['file']}: no on-screen {kind} event in {tile['json']}")
    return min(e["banner_t"] for e in hits) - float(tile.get("src_start", 0.0))


def _rect(grid: int, slot, W: int, H: int, top: int) -> tuple[float, float, float, float]:
    r, c = slot
    tw = (W - _GAP * (grid + 1)) / grid
    th = (H - top - _GAP * (grid + 1)) / grid
    return (_GAP + c * (tw + _GAP), top + _GAP + r * (th + _GAP), tw, th)


def _lerp(a, b, k):
    return tuple(x + (y - x) * k for x, y in zip(a, b, strict=True))


def build(spec: dict, out: Path) -> None:
    W, H, fps = int(spec["width"]), int(spec["height"]), float(spec["fps"])
    n_frames = round(float(spec["duration"]) * fps)
    top = 34  # a slim wall header in the appliance's style
    phases = spec["phases"]
    trans = float(spec.get("transition", 0.5))
    crop_default = spec.get("crop", [0, 0, W, H])
    tiles = spec["tiles"]
    clips = [Clip(t["file"], t.get("crop", crop_default)) for t in tiles]
    alarm_file_t = [(_alarm_t(t) if t.get("kind") else None) for t in tiles]

    cmd = [
        "ffmpeg", "-v", "error", "-y", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
        "-an", "-sws_flags", "accurate_rnd+full_chroma_int",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
    ]  # fmt: skip
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def layout_at(t: float, tile: dict):
        """(rect, alpha) for a tile at wall time t, easing between grid phases."""
        cur = max((p for p in phases if p["t"] <= t), key=lambda p: p["t"])
        i = phases.index(cur)
        prev = phases[i - 1] if i > 0 else None
        slots = tile["slots"]
        g = str(cur["grid"])
        if g not in slots:
            return None, 0.0
        target = _rect(cur["grid"], slots[g], W, H, top)
        k = _ease((t - cur["t"]) / trans) if prev else 1.0
        if prev and str(prev["grid"]) in slots:
            return _lerp(_rect(prev["grid"], slots[str(prev["grid"])], W, H, top), target, k), 1.0
        if not prev:
            return target, 1.0
        # A tile new to this phase waits until the shrinking tiles have cleared its slot,
        # then fades in; fading in under a tile still sliding past it read as a double image.
        return target, _ease((t - cur["t"] - 0.55 * trans) / (0.45 * trans))

    for fi in range(n_frames):
        t = fi / fps
        canvas = np.full((H, W, 3), _GROUND, np.uint8)
        fired = 0
        dim = _ease((t - spec["dim_at"]) / 0.4) if "dim_at" in spec else 0.0
        # Tiles that persist from the previous grid are drawn last, over any newcomer.
        def _is_new(tile, t=t):
            cur = max((p for p in phases if p["t"] <= t), key=lambda p: p["t"])
            i = phases.index(cur)
            return i > 0 and str(phases[i - 1]["grid"]) not in tile["slots"]

        order = sorted(
            zip(tiles, clips, alarm_file_t, strict=True), key=lambda z: not _is_new(z[0])
        )
        for tile, clip, a_t in order:
            if t < float(tile.get("appear", 0.0)):
                continue
            rect, alpha = layout_at(t, tile)
            if rect is None or alpha <= 0.01:
                continue
            if a_t is not None:
                ft = a_t + (t - float(tile["fire_at"]))
                is_red = t >= float(tile["fire_at"])
            else:
                ft = float(tile.get("offset", 0.0)) + (t - float(tile["appear"]))
                is_red = False
            fired += is_red
            src = clip.at(ft)
            if src is None:
                continue
            x, y, w, h = (round(v) for v in rect)
            if w < 8 or h < 8:
                continue
            img = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
            shade = alpha * (1.0 - (0.62 * dim if not is_red else 0.0))
            roi = canvas[y : y + h, x : x + w].astype(np.float32)
            canvas[y : y + h, x : x + w] = (img * shade + roi * (1 - shade)).astype(np.uint8)
            if is_red:
                since = t - float(tile["fire_at"])
                pulse = 1.0 + 1.5 * max(0.0, 1.0 - since / 0.6)  # a thick flash that settles
                th_px = max(2, round(3 * pulse * w / 640))
                cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), _ALARM, th_px, cv2.LINE_AA)
                label = tile.get("label", tile["kind"].upper())
                fs = max(0.38, 0.62 * w / 640)
                (lw, lh), _ = cv2.getTextSize(label, _FONT, fs, 1 if fs < 0.6 else 2)
                pad = max(4, round(6 * w / 640))
                cv2.rectangle(
                    canvas, (x, y + h - lh - 2 * pad), (x + lw + 2 * pad, y + h), _ALARM, -1
                )
                cv2.putText(
                    canvas,
                    label,
                    (x + pad, y + h - pad),
                    _FONT,
                    fs,
                    _INK,
                    1 if fs < 0.6 else 2,
                    cv2.LINE_AA,
                )
            else:
                cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), _IDLE_EDGE, 1, cv2.LINE_AA)
        # header, the appliance's own wordmark and a running alert count
        cv2.putText(canvas, "VIVIDLY", (12, 23), _FONT, 0.55, _ACCENT, 1, cv2.LINE_AA)
        cv2.putText(
            canvas, "|  live perception", (92, 23), _FONT, 0.5, (200, 196, 190), 1, cv2.LINE_AA
        )
        count = f"{fired} alert{'s' if fired != 1 else ''}"
        (cw, _), _ = cv2.getTextSize(count, _FONT, 0.5, 1)
        cv2.putText(
            canvas,
            count,
            (W - cw - 14, 23),
            _FONT,
            0.5,
            _ALARM if fired else (200, 196, 190),
            1,
            cv2.LINE_AA,
        )
        # outro: fade to black, then the wordmark alone
        if "black_at" in spec and t >= spec["black_at"]:
            k = _ease((t - spec["black_at"]) / 0.35)
            canvas = (canvas.astype(np.float32) * (1 - k)).astype(np.uint8)
            wk = _ease((t - spec["black_at"] - 0.3) / 0.35)
            if wk > 0 and spec.get("wordmark"):
                word = spec["wordmark"]
                fs = 2.2 * W / 1280
                thick = max(2, round(4 * W / 1280))
                (ww, wh), _ = cv2.getTextSize(word, _FONT, fs, thick)
                layer = canvas.copy()
                cv2.putText(
                    layer,
                    word,
                    ((W - ww) // 2, (H + wh) // 2),
                    _FONT,
                    fs,
                    _ACCENT,
                    thick,
                    cv2.LINE_AA,
                )
                canvas = cv2.addWeighted(layer, wk, canvas, 1 - wk, 0)
        enc.stdin.write(canvas.tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"{out}  {n_frames} frames  {n_frames / fps:.2f}s  {len(tiles)} tiles")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not on PATH")
    build(json.loads(a.spec.read_text()), a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
