"""GMDCSA-24 annotation parser (registry entry gmdcsa24, interim train set).

Second interim fall-verifier training set (with caucafall), admitted 2026-08-27
under the prototype weight class for positive-side angle/subject/scene diversity after fall
v2's split verdict. Turns the upstream layout into the same annotations/index.json
schema vividly.data.convert.le2i writes, so scripts/events/extract_track_sequences.py
runs on this dataset unchanged.

Upstream layout (Zenodo v2.0 zip = git archive `ekramalam-GMDCSA24-...-d3edb5d/`):

    Subject N/Fall/NN.mp4 + Subject N/Fall.csv     (N in 1..4)
    Subject N/ADL/NN.mp4  + Subject N/ADL.csv

Version skew, verified against both sources 2026-08-27: the Zenodo v2.0 zip's CSVs are
the older `File Name,Length,Description` format with no time intervals. GitHub commit
5abac7693229900cf80f722e878fbb119211fc1c (2024-08-21, one commit past the zip) rewrote
exactly those 8 CSVs, and nothing else, into the interval format below, so the zip's
videos are byte-identical to master's. The interval CSVs are downloaded at that pinned
commit into <root>/annotations_upstream/Subject N/{Fall,ADL}.csv, which this parser
prefers over the zip's own CSVs when present.

Each CSV row: `File Name,Length (seconds),Time of Recording,Attire,Description,Classes`
where Classes is a semicolon-separated list of `Label[start to end]` segments with the
times in seconds (e.g. `Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]`). The fall event is
the union of `Falling*` segments; ADL segments (Sitting/Sleeping/Walking/...) are the
quiet-footage negatives recorded on 2026-08-27.

Realities of the distribution this parser absorbs rather than hides (every one below
was surfaced by the completeness guard on the real bytes, then checked against them):
  * The CSVs are unquoted, so fields are not safely splittable on commas; segments are
    scanned from the whole line, every `[` must end up inside a parsed bracket, and
    every item inside a bracket must parse as an interval (nothing drops silently).
  * Formatting wobbles between rows: `Falling (SW)[1.8 to 5]:;` (stray colon),
    `Falling (BW) [0.3 to 5]` (space before bracket), `[ 2.8 to 6]` (space inside).
  * Two rows omit the `to`: `Eating[0.9 4.6]` (S2 Fall 19), `Falling (BW)[2.3 6]`
    (S4 Fall 09); the latter is the fall interval for that clip.
  * One bracket can stage multiple intervals: `Standing[2.4 to 3.6; 6.6 to 7.9]`
    (context labels only; no Falling bracket does this today, handled anyway).
  * S3 ADL 04 drops a `]`: `Standing[2.4 to 3.6; 6.6 to 7.9; Walking[3.6 to 5.2; ...]`
    A `Label[` opening inside an open bracket implicitly closes the previous one.
  * S3 Fall 01 labels its event `Fall (FW)`, not `Falling (FW)`: fall segments are
    matched on the `fall` prefix, and the Fall/ADL cross-check guards both classes.
  * S2 ADL 06 stages `Sitting[9.7 to 3]`, reversed, and contradicting its own
    Standing/Walking spans under either repair, so the intended reading is unknowable.
    Reversed context intervals are dropped and counted (n_reversed_dropped); a
    reversed fall interval is still refused outright, because it feeds events_t.
  * S4 Fall 15 writes `Falling (FW); Sitting[0 to 1.7]`, so the fall segment has no
    interval bracket at all. The clip is excluded from the index (counted and listed
    by key in summary.fall_unlabeled_excluded) rather than given an invented interval
    or allowed to train as a negative. The ADL-with-fall check stays a hard refusal.
  * Segment ends are authored as whole-second video lengths (`[3.4 to 6]` on a "06"
    video) while containers decode fractionally shorter; ends past the container get
    flagged fall_beyond_container (0.75 s tolerance), never clamped.
  * A video whose CSV row stages multiple Falling segments indexes as one event
    spanning their union (n_fall_segments records the count): the meta contract fills
    fall_start_t/fall_end_t only for single-event videos, and a 4-12 s clip whose
    union span went unfilled would silently train as a negative.

videos/<S{N}_{Fall,ADL}>/NN.mp4 are relative symlinks into upstream/ (rsync -a keeps
them valid inside the tree), giving space-free paths and stable extraction keys like
`S1_Fall__01`.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

INDEX_FILENAME = "index.json"

# `Label[` opener and `start to end` / `start end` interval items (wobbles above).
_LABEL_OPEN_RE = re.compile(r"([A-Za-z][A-Za-z ()]*?)\s*\[")
_INTERVAL_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s+(?:to\s+)?([0-9]+(?:\.[0-9]+)?)$")
# Segment ends are integer-second nominal lengths; containers decode slightly shorter.
_CONTAINER_TOLERANCE_S = 0.75

CLASS_DIRS = ("Fall", "ADL")


class Gmdcsa24ParseError(ValueError):
    """Upstream files that do not match the documented format."""


def parse_classes_line(
    line: str, *, name: str = "<row>"
) -> tuple[list[tuple[str, float, float]], int]:
    """All `Label[intervals]` segments on one CSV line as ((label, start_s, end_s), …),
    plus the count of dropped reversed context intervals (the S2 ADL 06 row).

    A bracket holds `;`-separated intervals, each `start to end` or `start end`; a
    multi-interval bracket yields one tuple per interval. A `Label[` opening inside an
    open bracket implicitly closes it (the S3 ADL 04 missing-`]` row). Completeness
    guards refuse the line if any `[` lands outside a parsed bracket or any bracket
    item does not parse, because a silently dropped Falling segment would poison training.
    """
    segments: list[tuple[str, float, float]] = []
    n_opened = 0
    n_reversed_dropped = 0
    m = _LABEL_OPEN_RE.search(line)
    while m:
        label = " ".join(m.group(1).split())
        n_opened += 1
        close = line.find("]", m.end())
        nested = _LABEL_OPEN_RE.search(line, m.end())
        if close < 0 and nested is None:
            raise Gmdcsa24ParseError(f"{name}: unclosed bracket for {label!r} in {line!r}")
        if close >= 0 and (nested is None or close < nested.start()):
            content, m = line[m.end() : close], _LABEL_OPEN_RE.search(line, close + 1)
        else:  # implicit close: the next `Label[` starts before any `]`
            assert nested is not None
            content, m = line[m.end() : nested.start()].rstrip("; \t"), nested
        for item in content.split(";"):
            iv = _INTERVAL_RE.match(item.strip())
            if iv is None:
                raise Gmdcsa24ParseError(f"{name}: unparseable interval {item!r} in {line!r}")
            start, end = float(iv.group(1)), float(iv.group(2))
            if start > end and not label.lower().startswith("fall"):
                n_reversed_dropped += 1
                continue
            if not 0 <= start <= end:
                raise Gmdcsa24ParseError(f"{name}: nonsensical interval [{start} to {end}]")
            segments.append((label, start, end))
    if line.count("[") != n_opened:
        raise Gmdcsa24ParseError(
            f"{name}: {line.count('[')} '[' but {n_opened} parsed brackets in {line!r}"
        )
    return segments, n_reversed_dropped


def parse_subject_csv(path: Path) -> dict[str, dict[str, Any]]:
    """Per video stem (`01`): segments + time-of-recording, keyed for cross-checks."""
    rows: dict[str, dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(",")
        filename = parts[0].strip()
        if not filename.lower().endswith(".mp4"):
            continue  # header row
        stem = filename[: -len(".mp4")]
        if stem in rows:
            raise Gmdcsa24ParseError(f"{path.name}: duplicate row for {filename}")
        segments, n_reversed_dropped = parse_classes_line(
            line, name=f"{path.parent.name}/{path.name}:{filename}"
        )
        rows[stem] = {
            "segments": segments,
            "n_reversed_dropped": n_reversed_dropped,
            "time_of_recording": parts[2].strip() if len(parts) > 2 else None,
        }
    if not rows:
        raise Gmdcsa24ParseError(f"{path.name}: no video rows")
    return rows


def _find_subject_dirs(root: Path) -> list[Path]:
    """`Subject N` dirs under upstream/ (git-archive top dir or unzipped flat)."""
    upstream = root / "upstream"
    for base in (upstream, *sorted(p for p in upstream.glob("*") if p.is_dir())):
        subjects = sorted(p for p in base.glob("Subject *") if p.is_dir())
        if subjects:
            return subjects
    raise Gmdcsa24ParseError(f"no 'Subject *' directories under {upstream}")


def _probe_video_cv2(path: Path) -> tuple[float, int]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise Gmdcsa24ParseError(f"cannot open video {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if not fps > 0 or n <= 0:
        raise Gmdcsa24ParseError(f"container reports fps={fps} frames={n} for {path}")
    return fps, n


def build_index(
    root: Path, *, probe: Callable[[Path], tuple[float, int]] | None = None
) -> dict[str, Any]:
    """Scan upstream/, refresh the videos/ symlink tree, and index every clip."""
    if probe is None:
        probe = _probe_video_cv2
    records: list[dict[str, Any]] = []
    fall_unlabeled_excluded: list[str] = []
    for subject_dir in _find_subject_dirs(root):
        subject_n = int(subject_dir.name.split()[-1])
        for cls in CLASS_DIRS:
            # Interval CSVs (pinned GitHub commit) override the zip's interval-less ones.
            override = root / "annotations_upstream" / subject_dir.name / f"{cls}.csv"
            csv_path = override if override.is_file() else subject_dir / f"{cls}.csv"
            video_dir = subject_dir / cls
            if not csv_path.is_file() or not video_dir.is_dir():
                raise Gmdcsa24ParseError(f"{subject_dir.name}: missing {cls}.csv or {cls}/")
            rows = parse_subject_csv(csv_path)
            mp4s = sorted(video_dir.glob("*.mp4"))
            video_stems = {p.stem for p in mp4s}
            if video_stems != rows.keys():
                raise Gmdcsa24ParseError(
                    f"{subject_dir.name}/{cls}: CSV/video mismatch: "
                    f"no row: {sorted(video_stems - rows.keys())}, "
                    f"no file: {sorted(rows.keys() - video_stems)}"
                )
            location = f"S{subject_n}_{cls}"
            link_dir = root / "videos" / location
            link_dir.mkdir(parents=True, exist_ok=True)
            for mp4 in mp4s:
                link = link_dir / mp4.name
                if link.is_symlink() or link.exists():
                    link.unlink()
                row = rows[mp4.stem]
                # `fall` prefix, not `falling`: S3 Fall 01 labels its event `Fall (FW)`.
                falling = [s for s in row["segments"] if s[0].lower().startswith("fall")]
                if cls == "Fall" and not falling:
                    # S4 Fall 15: fall labeled but no interval bracket. Excluded and
                    # listed, never an invented interval and never a silent negative.
                    fall_unlabeled_excluded.append(f"{location}/{mp4.name}")
                    continue
                if cls == "ADL" and falling:
                    raise Gmdcsa24ParseError(f"{location}/{mp4.name}: ADL clip with {falling}")
                link.symlink_to(os.path.relpath(mp4, link_dir))
                fps, n_frames = probe(mp4)
                events_t = (
                    [
                        [
                            round(min(s[1] for s in falling), 4),
                            round(max(s[2] for s in falling), 4),
                        ]
                    ]
                    if falling
                    else []
                )
                records.append(
                    {
                        "location": location,
                        "video": str(link.relative_to(root)),
                        "fps": fps,
                        "n_frames_container": n_frames,
                        "annotation": str(csv_path.relative_to(root)),
                        "has_fall": bool(falling),
                        "events_t": events_t,
                        "n_fall_segments": len(falling),
                        "fall_beyond_container": bool(falling)
                        and events_t[0][1] > n_frames / fps + _CONTAINER_TOLERANCE_S,
                        "segments": [[label, s, e] for label, s, e in row["segments"]],
                        "n_reversed_dropped": row["n_reversed_dropped"],
                        "time_of_recording": row["time_of_recording"],
                    }
                )
    records.sort(key=lambda r: (r["location"], r["video"]))
    summary = {
        "n_videos": len(records),
        "n_fall": sum(1 for r in records if r["has_fall"]),
        "n_adl": sum(1 for r in records if not r["has_fall"]),
        "n_subjects": len({r["location"].split("_")[0] for r in records}),
        "n_multi_fall_segment": sum(1 for r in records if r["n_fall_segments"] > 1),
        "n_fall_beyond_container": sum(1 for r in records if r["fall_beyond_container"]),
        "n_reversed_dropped": sum(r["n_reversed_dropped"] for r in records),
        "fall_unlabeled_excluded": sorted(fall_unlabeled_excluded),
    }
    return {"dataset": "gmdcsa24", "summary": summary, "videos": records}


def write_index(root: Path, index: Mapping[str, Any] | None = None) -> Path:
    if index is None:
        index = build_index(root)
    out = root / "annotations" / INDEX_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out
