"""UR Fall Detection Dataset label parser (registry entry ur_fall, held-out test set).

ur_fall is one of the two registry-enforced test_only fall sets (event_grant role
test_only): it never trains, and the fall verifier's real numbers come from here and
from multicam_fall. This module turns the upstream per-frame label CSVs into the same
annotations/index.json schema vividly.data.convert.le2i writes, so
scripts/events/extract_track_sequences.py runs on this dataset unchanged.

Upstream format (urfall-cam0-falls.csv / urfall-cam0-adls.csv): rows
`sequence,frame,label,<8 depth-derived feature columns>`; label is -1 (not lying),
0 (falling, the dynamic transition), 1 (lying). Frames are 1-based.

Verified against the bytes:
  * fall-NN sequences: the label-0 span is the fall; index records it as the event.
  * adl-NN sequences also contain label-0 spans (819 rows across the 40 ADLs): those
    are controlled lying-down transitions (getting into a bed or couch), not falls. They
    stay has_fall=False, so a verifier must call them negative, and the span is kept in
    lie_transition_start/end_frame so evals can probe exactly that confounder.
  * Only cam0 carries labels. cam1 views exist for the falls and are frame-synced,
    but this index covers cam0 only; the second view is deferred rather than assumed.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

INDEX_FILENAME = "index.json"

FALLS_CSV = "urfall-cam0-falls.csv"
ADLS_CSV = "urfall-cam0-adls.csv"


class UrFallParseError(ValueError):
    """Label CSVs that do not match the upstream format."""


def _label_spans(path: Path) -> dict[str, dict[str, Any]]:
    """Per sequence: frame range and the min/max frame carrying label 0."""
    out: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or len(row) < 3:
                continue
            seq, frame_s, label_s = row[0].strip(), row[1].strip(), row[2].strip()
            try:
                frame, label = int(frame_s), int(label_s)
            except ValueError as exc:
                raise UrFallParseError(f"{path.name}: non-integer row {row[:3]}") from exc
            if label not in (-1, 0, 1):
                raise UrFallParseError(f"{path.name}: label {label} outside -1/0/1")
            rec = out.setdefault(
                seq, {"first": frame, "last": frame, "zero_first": None, "zero_last": None}
            )
            rec["first"] = min(rec["first"], frame)
            rec["last"] = max(rec["last"], frame)
            if label == 0:
                rec["zero_first"] = (
                    frame if rec["zero_first"] is None else min(rec["zero_first"], frame)
                )
                rec["zero_last"] = (
                    frame if rec["zero_last"] is None else max(rec["zero_last"], frame)
                )
    if not out:
        raise UrFallParseError(f"{path.name}: no label rows")
    return out


def _probe_video_cv2(path: Path) -> tuple[float, int]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise UrFallParseError(f"cannot open video {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if not fps > 0 or n <= 0:
        raise UrFallParseError(f"container reports fps={fps} frames={n} for {path}")
    return fps, n


def build_index(
    root: Path, *, probe: Callable[[Path], tuple[float, int]] | None = None
) -> dict[str, Any]:
    """Scan <root>/videos/*-cam0.mp4 against the two label CSVs under <root>/labels."""
    if probe is None:
        probe = _probe_video_cv2
    labels_dir = root / "labels"
    falls = _label_spans(labels_dir / FALLS_CSV)
    adls = _label_spans(labels_dir / ADLS_CSV)
    records: list[dict[str, Any]] = []
    for mp4 in sorted((root / "videos").glob("*-cam0.mp4")):
        seq = mp4.name.removesuffix("-cam0.mp4")  # fall-01 / adl-07
        is_fall_seq = seq.startswith("fall")
        spans = falls if is_fall_seq else adls
        if seq not in spans:
            raise UrFallParseError(
                f"video {mp4.name} has no rows in the {'falls' if is_fall_seq else 'adls'} CSV"
            )
        span = spans[seq]
        fps, n_frames = probe(mp4)
        if is_fall_seq and span["zero_first"] is None:
            raise UrFallParseError(f"{seq}: fall sequence with no label-0 frames")
        rec: dict[str, Any] = {
            "location": "cam0",
            "video": str(mp4.relative_to(root)),
            "fps": fps,
            "n_frames_container": n_frames,
            "annotation": f"labels/{FALLS_CSV if is_fall_seq else ADLS_CSV}",
            "has_fall": is_fall_seq,
            "fall_start_frame": span["zero_first"] if is_fall_seq else None,
            "fall_end_frame": span["zero_last"] if is_fall_seq else None,
            # ADLs: the controlled lying-down transition (same label 0, not a fall).
            "lie_transition_start_frame": None if is_fall_seq else span["zero_first"],
            "lie_transition_end_frame": None if is_fall_seq else span["zero_last"],
            "n_label_frames": span["last"] - span["first"] + 1,
        }
        records.append(rec)
    summary = {
        "n_videos": len(records),
        "n_fall": sum(1 for r in records if r["has_fall"]),
        "n_adl": sum(1 for r in records if not r["has_fall"]),
        "n_adl_with_lie_transition": sum(
            1 for r in records if r["lie_transition_start_frame"] is not None
        ),
    }
    return {"dataset": "ur_fall", "summary": summary, "videos": records}


def write_index(root: Path, index: Mapping[str, Any] | None = None) -> Path:
    if index is None:
        index = build_index(root)
    out = root / "annotations" / INDEX_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out
