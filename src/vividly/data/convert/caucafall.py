"""CAUCAFall annotation parser (registry entry caucafall, interim train set).

Second interim fall-verifier training set (with gmdcsa24), admitted 2026-08-27
under the prototype weight class. Turns the upstream layout into the same
annotations/index.json schema vividly.data.convert.le2i writes, so
scripts/events/extract_track_sequences.py runs on this dataset unchanged.

Upstream layout (Mendeley 7w7fccy7ky v4 bulk zip):

    Dataset CAUCAFall/CAUCAFall/Subject.N/<Activity>/   (N in 1..10)
        <ActivityName>S<N>.avi        one real video container per sequence
        <prefix><num>.png + .txt      per-frame stills with YOLO label rows
        classes.txt                   class-id names: `nofall` then `fall`

10 activities per subject: 5 falls (Fall backwards/forward/left/right/sitting) and
5 ADLs (Hop, Kneel, Pick up object, Sit down, Walk). Each frame txt row is
`<class> <cx> <cy> <w> <h>` (normalized); class 1 == falling. The fall event is the
[first, last] class-1 frame span, in frame numbers relative to the sequence's first
still (rel = trailing-digit number - min number + 1), the same span reading ur_fall
uses for its label-0 run.

The AVI is the source of record and the stills are only the label layer: the container
carries its own measured fps, which is how the paper's declared "23 fps" was caught
being wrong (every probed container reports 20.00), and the record page's
"1080x960" too (every container is 720x480, matching the stills).

Realities of the distribution, measured over all 100 sequences on landing
(2026-08-27):
  * Exactly one .avi per sequence directory (verified 100/100).
  * Still-name prefixes vary by activity and subject (`cas`, `cfs1-`, `cis`,
    `sals`, `cams`, `ars`, ...); some embed the subject digit, some restart at
    00001. Frame numbers are the trailing digit run; a mixed-prefix directory
    would make numbering ambiguous and is refused.
  * The still set systematically holds one more frame than the container reports
    (n_label_frames == n_frames_container + 1), with one exception: the truncated
    avi of S6 Fall backwards runs 23 frames short of its stills. So a span touching
    the last still may end past the container; the per-record
    label_frames_minus_container carries the measured difference, and the summary's
    label_minus_container_values is a set for that reason.
  * Subject.2/Fall backwards holds a stray `cas200091 - copia.png` duplicate
    ("copia" = copy). Stems without a trailing digit run are excluded and listed.
  * 4 directories have small holes in the numbering (stills missing for frames the
    container still plays); spans are read on numbering, not list position, so
    holes shift nothing. n_gap_frames records each hole size.
  * 6 stills across the set have no label txt (n_unlabeled); an unlabeled frame
    inside the fall span is still covered by the [first, last] reading.
  * Label quality at class level is perfect: every Fall directory has class-1
    frames, no ADL directory has any (both directions are still guarded). One fall
    flickers 1->0->1 mid-fall (n_class1_runs == 2); the span covers it.
  * ADL directories' classes.txt is the one-class `nofall` variant. Any other
    present content is refused, because a reordered classes.txt would invert semantics.

videos/S{n}/<activity_slug>.avi are relative symlinks to the upstream containers
(space-free paths, stable extraction keys like `S1__fall_backwards`).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

INDEX_FILENAME = "index.json"

_TRAILING_DIGITS_RE = re.compile(r"(\d+)$")

FALL_DIR_PREFIX = "Fall"
_CLASSES_FULL = ["nofall", "fall"]
_CLASSES_ADL = ["nofall"]


class CaucafallParseError(ValueError):
    """Upstream files that do not match the documented format."""


def _probe_video_cv2(path: Path) -> tuple[float, int, int, int]:
    """(fps, n_frames, width, height) from the container. cv2 stays function-local."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise CaucafallParseError(f"cannot open video {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if not fps > 0 or n <= 0:
        raise CaucafallParseError(f"container reports fps={fps} frames={n} for {path}")
    return fps, n, w, h


def _frame_class(txt: Path, *, name: str) -> int | None:
    """1 if any YOLO label row on the frame is class 1, else 0; None when no txt exists."""
    if not txt.is_file():
        return None
    got_fall = False
    for ln in txt.read_text(encoding="utf-8", errors="replace").splitlines():
        tok = ln.split()
        if not tok:
            continue
        if tok[0] not in ("0", "1"):
            raise CaucafallParseError(f"{name}/{txt.name}: class token {tok[0]!r} not 0/1")
        got_fall = got_fall or tok[0] == "1"
    return 1 if got_fall else 0


def _find_subject_dirs(root: Path) -> list[Path]:
    """Subject.* dirs under upstream/ across the known zip nestings."""
    upstream = root / "upstream"
    for base in (
        upstream / "Dataset CAUCAFall" / "CAUCAFall",
        upstream / "CAUCAFall",
        upstream,
    ):
        subjects = [p for p in base.glob("Subject.*") if p.is_dir()]
        if subjects:
            # Subject.10 must sort after Subject.9, not between .1 and .2
            return sorted(subjects, key=lambda p: int(p.name.split(".")[-1]))
    raise CaucafallParseError(f"no 'Subject.*' directories under {upstream}")


def build_index(
    root: Path, *, probe: Callable[[Path], tuple[float, int, int, int]] | None = None
) -> dict[str, Any]:
    """Scan upstream/, refresh the videos/ symlink tree, and index every sequence."""
    if probe is None:
        probe = _probe_video_cv2
    records: list[dict[str, Any]] = []
    excluded_odd: list[str] = []
    n_missing_classes_txt = 0
    for subject_dir in _find_subject_dirs(root):
        location = f"S{int(subject_dir.name.split('.')[-1])}"
        link_dir = root / "videos" / location
        link_dir.mkdir(parents=True, exist_ok=True)
        for act in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            seq_name = f"{subject_dir.name}/{act.name}"
            is_fall = act.name.startswith(FALL_DIR_PREFIX)
            avis = sorted(act.glob("*.avi"))
            if len(avis) != 1:
                raise CaucafallParseError(f"{seq_name}: expected exactly 1 avi, got {len(avis)}")
            classes_txt = act / "classes.txt"
            if classes_txt.is_file():
                got = classes_txt.read_text(encoding="utf-8", errors="replace").split()
                if got not in (_CLASSES_FULL, _CLASSES_ADL):
                    raise CaucafallParseError(
                        f"{seq_name}: classes.txt {got} would remap class ids"
                    )
            else:
                n_missing_classes_txt += 1

            nums: list[int] = []
            classes: dict[int, int | None] = {}
            prefixes: set[str] = set()
            for png in sorted(act.glob("*.png")):
                m = _TRAILING_DIGITS_RE.search(png.stem)
                if m is None:
                    excluded_odd.append(f"{seq_name}/{png.name}")
                    continue
                num = int(m.group(1))
                nums.append(num)
                prefixes.add(png.stem[: m.start()])
                classes[num] = _frame_class(png.with_suffix(".txt"), name=seq_name)
            if not nums:
                raise CaucafallParseError(f"{seq_name}: no numbered .png stills")
            if len(prefixes) > 1:
                raise CaucafallParseError(
                    f"{seq_name}: mixed still prefixes {sorted(prefixes)} make numbering ambiguous"
                )
            nums.sort()
            base_num = nums[0]
            ones_rel = sorted(n - base_num + 1 for n, c in classes.items() if c == 1)
            if is_fall and not ones_rel:
                raise CaucafallParseError(f"{seq_name}: Fall sequence with no class-1 frame")
            if not is_fall and ones_rel:
                raise CaucafallParseError(f"{seq_name}: ADL sequence with class-1 frames")
            n_runs = sum(1 for i, r in enumerate(ones_rel) if i == 0 or r != ones_rel[i - 1] + 1)
            fps, n_frames, width, height = probe(avis[0])

            slug = act.name.lower().replace(" ", "_")
            link = link_dir / f"{slug}.avi"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(os.path.relpath(avis[0], link_dir))
            records.append(
                {
                    "location": location,
                    "video": str(link.relative_to(root)),
                    "fps": fps,
                    "n_frames_container": n_frames,
                    "annotation": str(act.relative_to(root)),
                    "activity": act.name,
                    "has_fall": is_fall,
                    "fall_start_frame": ones_rel[0] if is_fall else None,
                    "fall_end_frame": ones_rel[-1] if is_fall else None,
                    "n_class1_frames": len(ones_rel),
                    "n_class1_runs": n_runs,
                    "n_label_frames": len(nums),
                    "label_frames_minus_container": len(nums) - n_frames,
                    "n_unlabeled": sum(1 for c in classes.values() if c is None),
                    "n_gap_frames": (nums[-1] - nums[0] + 1) - len(nums),
                    "frame_wh": [width, height],
                }
            )
    records.sort(key=lambda r: (int(r["location"][1:]), r["video"]))
    summary = {
        "n_sequences": len(records),
        "n_fall": sum(1 for r in records if r["has_fall"]),
        "n_adl": sum(1 for r in records if not r["has_fall"]),
        "n_subjects": len({r["location"] for r in records}),
        "n_class1_flicker": sum(1 for r in records if r["n_class1_runs"] > 1),
        "n_unlabeled_frames": sum(r["n_unlabeled"] for r in records),
        "n_gap_dirs": sum(1 for r in records if r["n_gap_frames"] > 0),
        "n_missing_classes_txt": n_missing_classes_txt,
        "excluded_odd_frames": sorted(excluded_odd),
        "fps_values": sorted({round(r["fps"], 3) for r in records}),
        "frame_whs": sorted({tuple(r["frame_wh"]) for r in records}),
        "label_minus_container_values": sorted(
            {r["label_frames_minus_container"] for r in records}
        ),
    }
    return {"dataset": "caucafall", "summary": summary, "videos": records}


def write_index(root: Path, index: Mapping[str, Any] | None = None) -> Path:
    if index is None:
        index = build_index(root)
    out = root / "annotations" / INDEX_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out
