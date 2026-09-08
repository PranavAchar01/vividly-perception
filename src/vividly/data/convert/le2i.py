"""Le2i / ImViA Fall Detection Dataset annotation parser (registry entry le2i_fdd).

The interim training set for the fall verifier (the first fall set admitted for training,
alongside caucafall and gmdcsa24; ur_fall and multicam_fall are the held-out tests). This
module turns the upstream annotation files into a machine-readable index the
feature-extraction step keys on.

Upstream format, verified against the bytes (upstream README.txt plus a geometric check:
reading the four numbers as height/width/center overflows the 240 px frame, reading them
as corners does not):

    line 1: frame number where the fall starts   (1-based; "0" = no fall)
    line 2: frame number where the fall ends
    then per-frame rows:  frame,label,x1,y1,x2,y2

Realities of the distribution this parser absorbs rather than hides:
  * Lecture_room and Office ship no annotation files at all (60 of 190 videos).
  * Three files (Coffee_room_01/video_026, Coffee_room_02/video_050 + _052) carry the
    header pair misplaced mid-file (e.g. "197"/"227" spliced between frame rows 218 and
    219) instead of at the top. The parser recovers the interval and flags it
    header_misplaced. A file with no bare pair anywhere is a no-fall file.
  * A "0,0" header means the video contains no fall (31 files).
  * "-1,-1,-1,-1" box rows mark frames where the person is absent.
  * The label column runs 1..8; upstream documents no semantics for it. It is carried
    through untouched and never interpreted here.
  * Home_02/video_051 writes 25 rows (frames 241..265) with the corners in reverse
    order (x2<x1 and y2<y1). Fully-reversed corners are swap-normalized and counted
    (n_swapped); a box inverted on one axis only is still refused as malformed.
  * 13 files have more annotation rows than the container has frames (video_035 in
    Coffee_room_01 has 471 rows for 231 frames). The index records both counts; the
    fall interval is kept even when it lands beyond the last decodable frame, flagged
    via fall_beyond_container.
  * Container fps is per video (24.0003… and 25.0 are both present, not split cleanly
    by location), so every index record carries its own fps and time alignment must
    use it: t = (frame - 1) / fps of that video.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INDEX_FILENAME = "index.json"


class Le2iParseError(ValueError):
    """An annotation file that does not match the upstream format."""


@dataclass(frozen=True)
class Le2iBox:
    """One per-frame row. Coordinates are raw upstream values (may stray a few pixels
    outside the 320x240 frame; consumers clamp)."""

    frame: int
    label: int
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class Le2iAnnotation:
    """One parsed annotation file.

    fall_start/fall_end are 1-based frame numbers, or None when the video has no fall
    (an explicit "0,0" header or no header pair anywhere). Absent-person rows
    (-1,-1,-1,-1) are counted, not kept as boxes. header_misplaced marks the three
    upstream files whose header pair sits between frame rows instead of at the top.
    """

    fall_start: int | None
    fall_end: int | None
    has_header: bool
    boxes: tuple[Le2iBox, ...]
    n_absent: int
    header_misplaced: bool = False
    n_swapped: int = 0

    @property
    def has_fall(self) -> bool:
        return self.fall_start is not None


def parse_annotation_text(text: str, *, name: str = "<annotation>") -> Le2iAnnotation:
    """Parse one annotation file's text. Pure; raises Le2iParseError on malformed rows."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    fall_start: int | None = None
    fall_end: int | None = None
    has_header = False
    i = 0
    if len(lines) >= 2 and "," not in lines[0] and "," not in lines[1]:
        try:
            start, end = int(lines[0]), int(lines[1])
        except ValueError as exc:
            raise Le2iParseError(f"{name}: unreadable header lines {lines[:2]!r}") from exc
        has_header = True
        i = 2
        if start == 0 and end == 0:
            pass  # explicit no-fall
        elif 0 < start <= end:
            fall_start, fall_end = start, end
        else:
            raise Le2iParseError(f"{name}: nonsensical fall interval {start},{end}")

    boxes: list[Le2iBox] = []
    n_absent = 0
    n_swapped = 0
    header_misplaced = False
    rows = lines[i:]
    j = 0
    while j < len(rows):
        ln = rows[j]
        parts = ln.split(",")
        if len(parts) == 1:
            # A bare integer mid-stream: three upstream files splice the fall-interval
            # pair between frame rows. Recover it once; anything else is malformed.
            nxt = rows[j + 1] if j + 1 < len(rows) else ""
            try:
                start, end = int(ln), int(nxt)
            except ValueError as exc:
                raise Le2iParseError(f"{name}: stray non-row line {ln!r}") from exc
            if fall_start is not None or has_header or not 0 < start <= end:
                raise Le2iParseError(f"{name}: unexpected bare pair {start},{end} mid-file")
            fall_start, fall_end = start, end
            has_header = True
            header_misplaced = True
            j += 2
            continue
        if len(parts) != 6:
            raise Le2iParseError(f"{name}: expected 6 comma fields, got {ln!r}")
        try:
            frame, label, x1, y1, x2, y2 = (int(p) for p in parts)
        except ValueError as exc:
            raise Le2iParseError(f"{name}: non-integer field in {ln!r}") from exc
        j += 1
        if (x1, y1, x2, y2) == (-1, -1, -1, -1):
            n_absent += 1
            continue
        if x2 < x1 and y2 < y1:
            # Home_02/video_051 frames 241..265: corners written in reverse order.
            x1, y1, x2, y2 = x2, y2, x1, y1
            n_swapped += 1
        elif x2 < x1 or y2 < y1:
            raise Le2iParseError(f"{name}: inverted box in {ln!r}")
        boxes.append(Le2iBox(frame=frame, label=label, x1=x1, y1=y1, x2=x2, y2=y2))
    return Le2iAnnotation(
        fall_start=fall_start,
        fall_end=fall_end,
        has_header=has_header,
        boxes=tuple(boxes),
        n_absent=n_absent,
        header_misplaced=header_misplaced,
        n_swapped=n_swapped,
    )


def parse_annotation_file(path: Path) -> Le2iAnnotation:
    return parse_annotation_text(path.read_text(encoding="utf-8", errors="replace"), name=str(path))


def _probe_video_cv2(path: Path) -> tuple[float, int]:
    """(fps, frame_count) from the container. cv2 stays a function-local import so the
    module (and the pure parser above) loads on cv2-less environments."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise Le2iParseError(f"cannot open video {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if not fps > 0 or n <= 0:
        raise Le2iParseError(f"container reports fps={fps} frames={n} for {path}")
    return fps, n


def build_index(
    root: Path, *, probe: Callable[[Path], tuple[float, int]] | None = None
) -> dict[str, Any]:
    """Scan <root>/videos/<Location>/*.avi + <root>/annotations/ into one index dict.

    `probe` returns (fps, n_frames) for a video path; the default reads the container
    via cv2. Records are sorted by (location, video) so the index is deterministic.
    """
    if probe is None:
        probe = _probe_video_cv2
    vids_root = root / "videos"
    if not vids_root.is_dir():
        raise Le2iParseError(f"no videos/ directory under {root}")
    records: list[dict[str, Any]] = []
    for avi in sorted(vids_root.glob("*/*.avi")):
        location = avi.parent.name
        fps, n_frames = probe(avi)
        rec: dict[str, Any] = {
            "location": location,
            "video": str(avi.relative_to(root)),
            "fps": fps,
            "n_frames_container": n_frames,
            "annotation": None,
            "has_header": None,
            "header_misplaced": None,
            "has_fall": None,
            "fall_start_frame": None,
            "fall_end_frame": None,
            "fall_beyond_container": None,
            "n_boxes": None,
            "n_absent": None,
            "n_swapped": None,
        }
        ann_path = root / "annotations" / location / (avi.stem + ".txt")
        if ann_path.is_file():
            ann = parse_annotation_file(ann_path)
            rec.update(
                annotation=str(ann_path.relative_to(root)),
                has_header=ann.has_header,
                header_misplaced=ann.header_misplaced,
                has_fall=ann.has_fall,
                fall_start_frame=ann.fall_start,
                fall_end_frame=ann.fall_end,
                fall_beyond_container=(ann.fall_end is not None and ann.fall_end > n_frames),
                n_boxes=len(ann.boxes),
                n_absent=ann.n_absent,
                n_swapped=ann.n_swapped,
            )
        records.append(rec)
    annotated = [r for r in records if r["annotation"] is not None]
    summary = {
        "n_videos": len(records),
        "n_annotated": len(annotated),
        "n_fall": sum(1 for r in annotated if r["has_fall"]),
        "n_no_fall": sum(1 for r in annotated if not r["has_fall"]),
        "n_header_misplaced": sum(1 for r in annotated if r["header_misplaced"]),
        "n_fall_beyond_container": sum(1 for r in annotated if r["fall_beyond_container"]),
    }
    return {"dataset": "le2i_fdd", "summary": summary, "videos": records}


def write_index(root: Path, index: Mapping[str, Any] | None = None) -> Path:
    """Materialize annotations/index.json under the dataset root. Returns the path."""
    if index is None:
        index = build_index(root)
    out = root / "annotations" / INDEX_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out
