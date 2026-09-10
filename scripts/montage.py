"""Assemble annotated segments into one short, calm, minimal film.

    uv run python scripts/montage.py --cuts cuts.json --out reel.mp4 [--width 1280] [--fps 30]

cuts.json is a list of {file, start, end, title, subtitle}, in order. Every segment is
trimmed exactly, normalised to --width by 16:9 at --fps, and the whole film is handed to a
single encoder at the end, so the output is one clean h264 / yuv420p / faststart file with
no audio track at all.

The title treatment is the appliance's own: HERSHEY_SIMPLEX, its lavender accent, drawn as
a lower third. It is simply there at the head of the segment, holds for about a second and
fades. Nothing moves, nothing slides, nothing is boxed.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# The appliance's palette (BGR), lifted from scripts/live_stream.py so the film and the
# screen it films are drawn in one hand.
_ACCENT = (242, 157, 169)  # lavender
_SUBTLE = (225, 225, 225)  # the HUD's secondary ink
_DIP_TO = 0.0  # the dip blinks toward black; --paper blinks toward white
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Title timing, in seconds. Hold, then fade. No entrance.
_HOLD_S = 1.0
_FADE_S = 0.4

# A dip is a blink, not a gap: it darkens frames that already exist at the seam rather
# than inserting new ones, so a dip and a hard cut have exactly the same duration. It also
# never reaches full black, which would read as a hole in the film.
_DIP_FRAMES = 8


@dataclass(frozen=True)
class Cut:
    file: Path
    start: float
    end: float
    title: str
    subtitle: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def load_cuts(path: Path) -> list[Cut]:
    """Read the cut list. Relative segment paths resolve against the cuts file itself."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, list) or not raw:
        raise SystemExit(f"{path}: expected a non-empty list of cuts")
    cuts: list[Cut] = []
    for i, item in enumerate(raw):
        missing = {"file", "start", "end"} - set(item)
        if missing:
            raise SystemExit(f"{path}: cut {i} is missing {sorted(missing)}")
        f = Path(item["file"])
        if not f.is_absolute():
            f = (path.parent / f).resolve()
        if not f.exists():
            raise SystemExit(f"{path}: cut {i} points at a file that does not exist: {f}")
        start, end = float(item["start"]), float(item["end"])
        if end <= start:
            raise SystemExit(f"{path}: cut {i} has end ({end}) <= start ({start})")
        cuts.append(Cut(f, start, end, str(item.get("title", "")), str(item.get("subtitle", ""))))
    return cuts


def _title_alpha(t: float) -> float:
    """Opacity of the lower third t seconds into a segment: hold, then a soft fade out."""
    if t <= _HOLD_S:
        return 1.0
    x = (t - _HOLD_S) / _FADE_S
    if x >= 1.0:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * x))  # ease out, no linear edge to catch the eye


def draw_lower_third(
    frame: np.ndarray, title: str, subtitle: str, alpha: float, scrim: float
) -> None:
    """Draw the title as a lower third, in place, at the given opacity.

    `scrim` (0..1) is how strongly the very bottom of the picture is darkened behind the
    type. It exists only so the lavender survives a blown-out floor; it is a gradient with
    no edge of its own, and it fades out with the title.
    """
    if alpha <= 0.002 or not title:
        return
    h, w = frame.shape[:2]
    left = round(0.055 * w)
    base_y = round(0.815 * h)
    t_scale = 1.15 * w / 1280.0
    t_thick = max(2, round(3 * w / 1280.0))
    s_scale = 0.60 * w / 1280.0
    s_thick = max(1, round(1 * w / 1280.0))

    (_, t_h), _ = cv2.getTextSize(title, _FONT, t_scale, t_thick)
    band_top = max(0, base_y - t_h - round(0.055 * h))
    band = frame[band_top:h]
    over = band.copy()

    if scrim > 0.0:
        # A one-sided gradient to black. It has to be at full strength by the time it
        # reaches the type, or a floor highlight sits right behind the lavender, so it
        # ramps in over the empty space above the title and then holds. Smoothstep, so
        # both ends have zero slope and no edge is ever visible.
        y_rel = np.arange(band.shape[0], dtype=np.float32)
        x = np.clip(y_rel / max(1.0, float(base_y - band_top - t_h)), 0.0, 1.0)
        ramp = (scrim * x * x * (3.0 - 2.0 * x))[:, None, None]
        over = (over.astype(np.float32) * (1.0 - ramp)).astype(np.uint8)

    y = base_y - band_top
    cv2.putText(over, title, (left, y), _FONT, t_scale, _ACCENT, t_thick, cv2.LINE_AA)
    if subtitle:
        cv2.putText(
            over,
            subtitle,
            (left, y + round(0.052 * h)),
            _FONT,
            s_scale,
            _SUBTLE,
            s_thick,
            cv2.LINE_AA,
        )
    cv2.addWeighted(over, alpha, band, 1.0 - alpha, 0.0, dst=band)


def decode_segment(cut: Cut, w: int, h: int, fps: float):
    """Yield exactly-trimmed BGR frames for one cut, normalised to w x h at fps."""
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=bgr24"
    )
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin",
        "-ss", f"{cut.start:.6f}", "-i", str(cut.file), "-t", f"{cut.duration:.6f}",
        "-an", "-vf", vf, "-f", "rawvideo", "-",
    ]  # fmt: skip
    n = w * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=n * 4)
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        proc.wait()
    if proc.returncode not in (0, None):
        raise SystemExit(f"ffmpeg failed decoding {cut.file} (exit {proc.returncode})")


def open_encoder(out: Path, w: int, h: int, fps: float, crf: int) -> subprocess.Popen:
    """One encode, at the end, for the whole film. No audio stream is ever created."""
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
        "-an",
        # Default swscale rounding stores a flat white as Y=234, Cb=Cr=127, which every
        # player shows as 252,254,251: a green cast on a white backdrop. Accurate rounding
        # stores 235/128/128 and decodes to exactly 255 (measured).
        "-sws_flags", "accurate_rnd+full_chroma_int",
        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out),
    ]  # fmt: skip
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def build(
    cuts: list[Cut],
    out: Path,
    width: int,
    fps: float,
    transition: str,
    tail: float,
    scrim: float,
    crf: int,
) -> tuple[int, float]:
    w = width - (width % 2)
    h = round(w * 9 / 16)
    h -= h % 2
    enc = open_encoder(out, w, h, fps, crf)
    assert enc.stdin is not None

    total = 0
    last: np.ndarray | None = None
    try:
        for idx, cut in enumerate(cuts):
            frames = list(decode_segment(cut, w, h, fps))
            if not frames:
                raise SystemExit(f"no frames decoded from {cut.file} at {cut.start}s")
            n = len(frames)
            head_dip = transition == "dip" and idx > 0
            tail_dip = transition == "dip" and idx < len(cuts) - 1
            for i, frame in enumerate(frames):
                draw_lower_third(frame, cut.title, cut.subtitle, _title_alpha(i / fps), scrim)
                if head_dip and i < _DIP_FRAMES // 2:
                    k = (i + 1) / (_DIP_FRAMES // 2 + 1)
                    frame = (frame.astype(np.float32) * k + _DIP_TO * (1.0 - k)).astype(np.uint8)
                elif tail_dip and i >= n - _DIP_FRAMES // 2:
                    k = (n - i) / (_DIP_FRAMES // 2 + 1)
                    frame = (frame.astype(np.float32) * k + _DIP_TO * (1.0 - k)).astype(np.uint8)
                if _DIP_TO == 255.0:
                    # Every lossy hop drifts a flat white a few levels toward green (measured
                    # 255 -> 251,254,251 after the renderer's writer -> 247,253,250 after this
                    # encode). Near-white, near-neutral pixels are the backdrop: snap them back.
                    lo, hi = frame.min(axis=2), frame.max(axis=2)
                    frame[(lo >= 236) & (hi.astype(np.int16) - lo <= 12)] = 255
                enc.stdin.write(frame.tobytes())
                total += 1
                last = frame
        if tail > 0 and last is not None:
            for _ in range(round(tail * fps)):
                enc.stdin.write(last.tobytes())
                total += 1
    finally:
        enc.stdin.close()
        enc.wait()
    if enc.returncode != 0:
        raise SystemExit(f"ffmpeg encode failed (exit {enc.returncode})")
    return total, total / fps


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--cuts", required=True, type=Path, help="JSON list of {file,start,end,title,subtitle}"
    )
    ap.add_argument("--out", required=True, type=Path, help="output mp4")
    ap.add_argument("--width", type=int, default=1280, help="output width; height is 16:9")
    ap.add_argument("--fps", type=float, default=30.0, help="output frame rate")
    ap.add_argument(
        "--transition",
        choices=("cut", "dip"),
        default="dip",
        help="seam between segments: 'dip' blinks 8 frames, which is right when the segments "
        "share one locked-off camera and a hard cut would read as a jump cut; 'cut' when the "
        "framing changes between segments",
    )
    ap.add_argument(
        "--tail", type=float, default=0.0, metavar="SECONDS", help="hold the last frame"
    )
    ap.add_argument(
        "--scrim", type=float, default=0.55, help="0..1 darkening under the lower third"
    )
    ap.add_argument(
        "--paper",
        action="store_true",
        help="segments on a white backdrop: titles in deep indigo (6.8:1 on white, where the "
        "lavender is 2.4:1), no scrim, and the dip blinks to white instead of black",
    )
    ap.add_argument("--crf", type=int, default=18, help="x264 quality (lower is better)")
    ap.add_argument("--max-seconds", type=float, default=None, help="warn if the film runs longer")
    a = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not on PATH")
    if a.paper:
        global _ACCENT, _SUBTLE, _DIP_TO
        _ACCENT, _SUBTLE, _DIP_TO = (190, 72, 88), (110, 110, 110), 255.0
        a.scrim = 0.0
    if not 0.0 <= a.scrim <= 1.0:
        raise SystemExit("--scrim must be between 0 and 1")

    cuts = load_cuts(a.cuts)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    frames, seconds = build(cuts, a.out, a.width, a.fps, a.transition, a.tail, a.scrim, a.crf)

    print(
        f"{a.out}  {frames} frames  {seconds:.2f}s  @{a.fps:g}fps  {len(cuts)} segments ({a.transition})"
    )
    if a.max_seconds is not None and seconds > a.max_seconds:
        print(
            f"warning: {seconds:.2f}s exceeds --max-seconds {a.max_seconds:g} "
            f"by {seconds - a.max_seconds:.2f}s",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
