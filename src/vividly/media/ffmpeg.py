"""ffmpeg discipline: argv lists only, reproducible runs.

Rules enforced here:
  - subprocess with argv lists; never shell=True, never an f-string command
  - binary via VIVIDLY_FFMPEG else PATH; RuntimeError names the env var
  - every call carries -hide_banner -nostdin -y -loglevel error
  - the exact argv is returned to the caller, so a run can be recorded and repeated
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

BASE_FLAGS = ["-hide_banner", "-nostdin", "-y", "-loglevel", "error"]


def ffmpeg_bin() -> str:
    exe = os.environ.get("VIVIDLY_FFMPEG") or shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found. brew install ffmpeg, or set VIVIDLY_FFMPEG.")
    return exe


def ffprobe_bin() -> str:
    exe = os.environ.get("VIVIDLY_FFPROBE") or shutil.which("ffprobe")
    if not exe:
        raise RuntimeError("ffprobe not found. brew install ffmpeg, or set VIVIDLY_FFPROBE.")
    return exe


def ffmpeg_version() -> str:
    """First line of `ffmpeg -version`, recorded beside every run."""
    out = subprocess.run([ffmpeg_bin(), "-version"], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()[0].strip()


def probe_video(path: str | Path) -> dict:
    """width/height/fps/nb_frames/codec_name/duration_s via ffprobe JSON."""
    argv = [
        ffprobe_bin(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,codec_name,avg_frame_rate,nb_read_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.run(argv, capture_output=True, text=True, check=True)
    j = json.loads(out.stdout)
    s = j["streams"][0]
    num, den = s["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": round(fps, 4),
        "nb_frames": int(s.get("nb_read_frames") or 0),
        "codec_name": s["codec_name"],
        "duration_s": round(float(j.get("format", {}).get("duration") or 0.0), 3),
    }


def run_argv(argv: list[str], stdin_bytes: bytes | None = None) -> dict:
    """Run one ffmpeg argv; returns the recorded fields for the run. Raises on failure
    after deleting the partial output (CLI exit code 3 semantics)."""
    t0 = time.perf_counter()
    proc = subprocess.run(argv, input=stdin_bytes, capture_output=True)
    wall = round(time.perf_counter() - t0, 3)
    if proc.returncode != 0:
        dst = Path(argv[-1])
        if dst.exists():
            dst.unlink()
        raise RuntimeError(
            f"ffmpeg failed rc={proc.returncode}: {proc.stderr.decode(errors='replace')[-2000:]}"
        )
    return {"argv": list(argv), "returncode": 0, "wall_seconds": wall}
