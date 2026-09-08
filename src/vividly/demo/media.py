"""Demo-platform media plumbing: upload names, phone-video normalization, annotated H.264.

ffmpeg discipline is inherited from src/vividly/media/ffmpeg.py: argv lists only, never
shell=True, binary via ffmpeg_bin(). The demo adds its own long-lived Popen writer because
run_argv() feeds stdin in one shot: at this module's caps (NORMALIZE_FPS 15, 1280 on the
long side) a 60 s annotated clip is 900 frames of 1280x720x3, about 2.5 GB of raw BGR
frames, which must stream, not sit in memory.
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
from pathlib import Path

from vividly.media.ffmpeg import BASE_FLAGS, ffmpeg_bin, run_argv

ALLOWED_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"})

# 15 fps is the waiting-room rate the rest of the repo assumes (tracker frame_rate,
# tracks.fps, ring sizing) and halves processing versus 30 fps phone video.
NORMALIZE_FPS = 15.0
NORMALIZE_MAX_SIDE = 1280

# Where the zone-tool preview frame is grabbed. 1 s in dodges the black/blurry opening
# frame most phone clips start on; a clip shorter than that falls back to frame 0.
PREVIEW_SEEK_S = 1.0


def sanitize_filename(name: str) -> str:
    """A safe basename for an uploaded video: allowlisted suffix, tamed stem."""
    p = Path(name)
    suffix = p.suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_VIDEO_SUFFIXES))
        raise ValueError(f"unsupported video type {suffix or name!r}; allowed: {allowed}")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", p.stem).strip("._") or "upload"
    return f"{stem[:80]}{suffix}"


def _scale_vf(max_side: int) -> str:
    """The one scale filter that defines the demo's processing pixel space.

    Shared by normalize_argv and preview_frame_argv so the zone-drawing preview can
    never disagree with the frame geometry the pipeline actually runs on.
    """
    return (
        f"scale='min({max_side},iw)':'min({max_side},ih)'"
        f":force_original_aspect_ratio=decrease:force_divisible_by=2"
    )


def normalize_argv(
    src: Path, dst: Path, *, max_side: int = NORMALIZE_MAX_SIDE, fps: float = NORMALIZE_FPS
) -> list[str]:
    """One transcode that makes any phone video pipeline-safe.

    HEVC / 10-bit HDR -> H.264 8-bit yuv420p; rotation metadata baked in (ffmpeg
    autorotates before user filters, so a portrait iPhone clip decodes upright and iw/ih
    below are post-rotation); VFR -> CFR at ``fps`` so t_source = index/fps is exact;
    longest side capped at ``max_side`` without ever upscaling; audio dropped.
    """
    vf = f"{_scale_vf(max_side)},fps={fps:g}"
    return [
        ffmpeg_bin(),
        *BASE_FLAGS,
        "-i",
        str(src),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(dst),
    ]


def annotated_argv(dst: Path, width: int, height: int, fps: float) -> list[str]:
    """Encoder argv for the annotated output: raw BGR frames on stdin -> browser-playable
    H.264 (yuv420p + faststart; cv2's mp4v is not web-playable). The input side is the
    rawvideo pipe shape: raw BGR frames on stdin at the declared size and rate."""
    return [
        ffmpeg_bin(),
        *BASE_FLAGS,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:g}",
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(dst),
    ]


def preview_frame_argv(
    src: Path, dst: Path, *, seek_s: float, max_side: int = NORMALIZE_MAX_SIDE
) -> list[str]:
    """One JPEG frame of ``src`` in the normalized pixel space.

    Same autorotation (applied by ffmpeg before user filters) and the same scale filter
    as normalize_argv, so the JPEG's dimensions are exactly the dimensions the pipeline
    will process, which is the pixel space exit-zone coordinates are drawn and stored in.
    The fps resample is deliberately absent: it moves no pixel, and a single grabbed
    frame has no rate.
    """
    argv = [ffmpeg_bin(), *BASE_FLAGS]
    if seek_s > 0:
        argv += ["-ss", f"{seek_s:g}"]
    argv += [
        "-i",
        str(src),
        "-vf",
        _scale_vf(max_side),
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(dst),
    ]
    return argv


def jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """(width, height) from a JPEG's SOF header, stdlib-only.

    The preview cache needs the frame's dimensions and the demo package may not import
    cv2 at module level (import-discipline rule); parsing the two SOF bytes beats a
    second ffprobe subprocess for a file this code just wrote.
    """
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        raise ValueError("not a JPEG (no SOI marker)")
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:  # markers with no payload
            i += 2
            continue
        seg_len = int.from_bytes(data[i + 2 : i + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):  # SOFn
            if i + 9 > len(data):
                break
            height = int.from_bytes(data[i + 5 : i + 7], "big")
            width = int.from_bytes(data[i + 7 : i + 9], "big")
            if width > 0 and height > 0:
                return (width, height)
            raise ValueError("JPEG SOF carries zero dimensions")
        i += 2 + seg_len
    raise ValueError("no SOF marker found (truncated or not a JPEG)")


def ensure_preview(src: Path, dest_dir: Path) -> dict[str, int]:
    """Extract-and-cache one normalized-space frame for the zone-drawing tool.

    Writes ``preview.jpg`` and ``preview.json`` ({"width", "height"}) into ``dest_dir``
    and returns the dimensions. Idempotent: the cache is the contract, so job creation
    validates zone bounds against these dimensions, the server's own record of the
    frame the user drew on, never against client-sent numbers.
    """
    jpg = Path(dest_dir) / "preview.jpg"
    meta = Path(dest_dir) / "preview.json"
    if jpg.is_file() and meta.is_file():
        try:
            row = json.loads(meta.read_text())
            return {"width": int(row["width"]), "height": int(row["height"])}
        except (ValueError, KeyError, TypeError):
            pass  # corrupt cache: fall through and rebuild it
    for seek_s in (PREVIEW_SEEK_S, 0.0):  # short clips have no frame at 1 s; retry at 0
        try:
            run_argv(preview_frame_argv(src, jpg, seek_s=seek_s))
        except RuntimeError:
            continue
        if jpg.is_file() and jpg.stat().st_size > 0:
            break
        jpg.unlink(missing_ok=True)  # ffmpeg exited 0 but produced nothing
    if not jpg.is_file() or jpg.stat().st_size == 0:
        raise RuntimeError("no decodable video frame for a preview")
    width, height = jpeg_dimensions(jpg.read_bytes())
    meta.write_text(json.dumps({"width": width, "height": height}) + "\n")
    return {"width": width, "height": height}


class AnnotatedWriter:
    """Long-lived ffmpeg encode fed one BGR frame at a time.

    stderr goes to a sidecar log file rather than a PIPE so a chatty encoder can never
    deadlock the writer; the log tail is folded into any raised error and the file is
    deleted on a clean close.
    """

    def __init__(self, dst: Path, width: int, height: int, fps: float) -> None:
        self.dst = Path(dst)
        self.width = int(width)
        self.height = int(height)
        self.frames_written = 0
        self._log_path = self.dst.with_name(self.dst.name + ".ffmpeg.log")
        self._log = self._log_path.open("wb")
        self._proc = subprocess.Popen(
            annotated_argv(self.dst, self.width, self.height, fps),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._log,
        )
        self._closed = False

    def _stderr_tail(self) -> str:
        try:
            return self._log_path.read_text(errors="replace")[-2000:]
        except OSError:
            return "<no stderr captured>"

    def write(self, frame) -> None:  # frame: HxWx3 uint8 BGR (numpy; annotated here only)
        """rawvideo has no framing: a size mismatch would silently corrupt every later
        frame, so it raises here instead."""
        if tuple(frame.shape) != (self.height, self.width, 3):
            raise ValueError(
                f"frame shape {frame.shape} != declared {(self.height, self.width, 3)}"
            )
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(f"ffmpeg encoder died: {self._stderr_tail()}") from exc
        self.frames_written += 1

    def close(self) -> None:
        """Finalize the MP4. Idempotent; raises if the encoder failed."""
        if self._closed:
            return
        self._closed = True
        if self._proc.stdin is not None:
            with contextlib.suppress(BrokenPipeError):  # encoder gone; wait() reports it
                self._proc.stdin.close()
        try:
            rc = self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._log.close()
            raise RuntimeError(f"ffmpeg encoder hung: {self._stderr_tail()}") from None
        self._log.close()
        if rc != 0:
            raise RuntimeError(f"ffmpeg encoder failed rc={rc}: {self._stderr_tail()}")
        self._log_path.unlink(missing_ok=True)


_UPLOADS_README = """# Demo uploads

Demo material only. **Nothing in this tree may enter a training corpus or a dataset
registry entry.** That holds under any framing: a live demonstration is never a dataset,
and POLICY.md, rule 4 governs.
Everything under outputs/ is gitignored; these files stay on this machine.
"""

_RUNS_README = """# Demo runs

Each directory is one demo job: normalized input, annotated output, alerts.sqlite,
latency.jsonl, job.json. Demo material only, never training data, and rule firings in
here are engineering-demonstration output, never quotable performance numbers.
"""


def ensure_demo_dirs(root: Path) -> dict[str, Path]:
    """Create (idempotently) the gitignored demo trees, with the do-not-train READMEs.

    `root` is the directory the `outputs/` tree hangs under: the repository root by
    default, or whatever outputs root the caller was configured with.
    """
    uploads = Path(root) / "outputs" / "demo_uploads"
    runs = Path(root) / "outputs" / "demo_runs"
    (uploads / "incoming").mkdir(parents=True, exist_ok=True)
    (uploads / "models").mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    for tree, text in ((uploads, _UPLOADS_README), (runs, _RUNS_README)):
        readme = tree / "README.md"
        if not readme.exists():
            readme.write_text(text)
    return {
        "uploads": uploads,
        "incoming": uploads / "incoming",
        "models": uploads / "models",
        "runs": runs,
    }
