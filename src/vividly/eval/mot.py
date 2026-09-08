"""MOTChallenge-format IO and the TrackEval driver.

MOTChallenge text is the interchange format between the tracking process and TrackEval.
Frame indices and track ids are 1-based in both ground truth and tracker output; that is
the single easiest way to get a tracking number silently wrong, so `MotRow` refuses a
zero or negative index at construction rather than shifting it later.

The layout TrackEval expects is rigid, and `build_trackeval_tree` is the only place that
knows it:

    <root>/gt/<BENCHMARK>-<SPLIT>/<SEQ>/gt/gt.txt
    <root>/gt/<BENCHMARK>-<SPLIT>/<SEQ>/seqinfo.ini
    <root>/gt/seqmaps/<BENCHMARK>-<SPLIT>.txt          # header line "name", then one seq per line
    <root>/trackers/<BENCHMARK>-<SPLIT>/<TRACKER>/data/<SEQ>.txt

Import discipline (this bit is load-bearing, not stylistic): trackeval and numpy are
imported lazily, inside `run_trackeval`, and `Detections` is imported lazily inside
`rows_from_detections`. trackeval 1.3.0 is declared in the `mac` and `linux-harness` groups
and in the `curate` extra, and is absent from the `cuda` extra entirely, which CI asserts
(the "Tracking metrics stay out of the cuda fork" step in .github/workflows/ci.yml), so a
module-level import would make `import vividly.eval.mot` fail on the GPU box. Everything
above the driver is pure text and needs neither.
"""

from __future__ import annotations

import configparser
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from vividly.models.base import Detections

# The 9 ground-truth columns, in the order published for MOT16, MOT17 and MOT20 and
# enumerated in full below. `conf` is the consider/ignore flag
# (1 = evaluate this box, 0 = ignore it), not a detector score.
GT_COLUMNS: tuple[str, ...] = (
    "frame",
    "id",
    "bb_left",
    "bb_top",
    "bb_width",
    "bb_height",
    "conf",
    "class",
    "visibility",
)
# The 10 tracker-output columns. The last three are the unused 3D world coordinates and
# are always -1 for 2D benchmarks. TrackEval reads column 7 as the class, so the first
# -1 also keeps every tracker row out of the distractor classes.
DT_COLUMNS: tuple[str, ...] = (
    "frame",
    "id",
    "bb_left",
    "bb_top",
    "bb_width",
    "bb_height",
    "conf",
    "x",
    "y",
    "z",
)
DT_TAIL: tuple[int, int, int] = (-1, -1, -1)

PEDESTRIAN_CLASS_ID = 1
GT_CONSIDER = 1
GT_IGNORE = 0

# DO_PREPROC drops MOTChallenge distractor classes using MOT-specific GT columns. On a
# custom benchmark that either deletes everything or silently changes the denominator, so
# it is True for exactly the two benchmarks whose GT carries those classes and False for
# everything else. MOT20 was skipped on merit, its geometry being street-level oblique
# rather than the target (DATASETS.md, "Considered and not used"), and no MOT17-format or
# MOT20-format set is in the registry, so every benchmark the code sees takes the False
# branch and the True branch stays unverified until such a set actually lands.
PREPROC_BENCHMARKS: frozenset[str] = frozenset({"MOT17", "MOT20"})

FileKind = Literal["gt", "tracker"]


class MotFormatError(ValueError):
    """A MOT text file or row violates the interchange format."""


# ---------------------------------------------------------------- rows
@dataclass(frozen=True, slots=True)
class MotRow:
    """One box in one frame of one sequence. `frame` is 1-based; `track_id` is
    non-negative: the MOT interchange format does not reserve 0, and a file written in it
    may count identities from 0 (verified on a fetched DanceTrack val split, 2026-08-08)
    or from 1, so 0 is a legitimate identity here, not corruption."""

    frame: int
    track_id: int
    left: float
    top: float
    width: float
    height: float
    conf: float = 1.0
    class_id: int = PEDESTRIAN_CLASS_ID
    visibility: float = 1.0

    def __post_init__(self) -> None:
        if self.frame < 1:
            raise MotFormatError(f"frame must be 1-based, got {self.frame}")
        if self.track_id < 0:
            raise MotFormatError(f"track_id must be non-negative, got {self.track_id}")
        if self.width <= 0 or self.height <= 0:
            raise MotFormatError(
                f"frame {self.frame} id {self.track_id}: box must have positive width and "
                f"height, got {self.width}x{self.height}"
            )

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.left, self.top, self.left + self.width, self.top + self.height)


def _check_unique_ids(rows: Sequence[MotRow], where: str) -> None:
    """TrackEval raises deep inside preprocessing on a repeated id; fail here instead."""
    seen: set[tuple[int, int]] = set()
    for row in rows:
        key = (row.frame, row.track_id)
        if key in seen:
            raise MotFormatError(
                f"{where}: track id {row.track_id} appears twice in frame {row.frame}; "
                "ids must be unique within a frame"
            )
        seen.add(key)


def _format_gt_row(row: MotRow) -> str:
    return (
        f"{row.frame},{row.track_id},"
        f"{row.left:.2f},{row.top:.2f},{row.width:.2f},{row.height:.2f},"
        f"{row.conf:g},{row.class_id:d},{row.visibility:g}"
    )


def _format_tracker_row(row: MotRow) -> str:
    tail = ",".join(str(v) for v in DT_TAIL)
    return (
        f"{row.frame},{row.track_id},"
        f"{row.left:.2f},{row.top:.2f},{row.width:.2f},{row.height:.2f},"
        f"{row.conf:.4f},{tail}"
    )


def _write_rows(path: Path, rows: Sequence[MotRow], kind: FileKind) -> int:
    """Write `rows` sorted by (frame, track_id). Returns the number of lines written.

    A frame with no rows writes no lines. Frame numbers are explicit in column 0, so an
    empty frame never shifts the indices of the frames after it.
    """
    _check_unique_ids(rows, str(path))
    fmt = _format_gt_row if kind == "gt" else _format_tracker_row
    ordered = sorted(rows, key=lambda r: (r.frame, r.track_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{fmt(r)}\n" for r in ordered)
    path.write_text(text, encoding="utf-8")
    return len(ordered)


def write_gt_file(path: Path, rows: Sequence[MotRow]) -> int:
    """Write 9-column MOTChallenge ground truth. Returns the number of lines written."""
    return _write_rows(Path(path), rows, "gt")


def write_tracker_file(path: Path, rows: Sequence[MotRow]) -> int:
    """Write 10-column MOTChallenge tracker output. Returns the number of lines written.

    An empty `rows` writes a zero-byte file on purpose: that is a legitimate tracker
    result (the tracker emitted nothing) and it must score 0, not crash. See
    `run_trackeval` and the empty-tracker regression test.
    """
    return _write_rows(Path(path), rows, "tracker")


def parse_mot_file(path: Path, *, kind: FileKind = "gt") -> list[MotRow]:
    """Parse a MOTChallenge text file back into rows. The inverse of the writers.

    `kind="tracker"` asserts the trailing -1,-1,-1 and restores the class/visibility
    defaults, so `parse_mot_file(p, kind="tracker") == rows` round-trips exactly.
    """
    path = Path(path)
    rows: list[MotRow] = []
    expected = len(GT_COLUMNS) if kind == "gt" else len(DT_COLUMNS)
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != expected:
            raise MotFormatError(
                f"{path}:{lineno}: expected {expected} columns for a {kind} file, got {len(parts)}"
            )
        if kind == "tracker":
            tail = tuple(int(float(p)) for p in parts[7:10])
            if tail != DT_TAIL:
                raise MotFormatError(
                    f"{path}:{lineno}: tracker rows must end in {DT_TAIL}, got {tail}"
                )
            class_id, visibility = PEDESTRIAN_CLASS_ID, 1.0
        else:
            class_id, visibility = int(float(parts[7])), float(parts[8])
        rows.append(
            MotRow(
                frame=int(float(parts[0])),
                track_id=int(float(parts[1])),
                left=float(parts[2]),
                top=float(parts[3]),
                width=float(parts[4]),
                height=float(parts[5]),
                conf=float(parts[6]),
                class_id=class_id,
                visibility=visibility,
            )
        )
    return rows


def rows_from_detections(per_frame: Sequence[Detections], *, first_frame: int = 1) -> list[MotRow]:
    """`per_frame[i]` are the tracked detections of frame i (0-based); rows are 1-based.

    numpy arrives with `Detections`, so the import is deferred to here: the writers above
    stay usable in an environment that has no numpy at all.
    """
    rows: list[MotRow] = []
    for idx, det in enumerate(per_frame):
        if len(det) == 0:
            continue
        if det.track_ids is None:
            raise MotFormatError(
                f"frame {idx}: tracker output has no track_ids; only a Tracker may write "
                "a MOT tracker file"
            )
        for (x1, y1, x2, y2), score, tid in zip(det.xyxy, det.scores, det.track_ids, strict=True):
            rows.append(
                MotRow(
                    frame=idx + first_frame,
                    track_id=int(tid),
                    left=float(x1),
                    top=float(y1),
                    width=float(x2) - float(x1),
                    height=float(y2) - float(y1),
                    conf=float(score),
                )
            )
    return rows


# ---------------------------------------------------------------- seqinfo / seqmap
@dataclass(frozen=True, slots=True)
class SeqInfo:
    """The contents of a MOTChallenge seqinfo.ini."""

    name: str
    seq_length: int
    im_width: int
    im_height: int
    frame_rate: int = 15
    im_dir: str = "img1"
    im_ext: str = ".jpg"


def write_seqinfo(seq_dir: Path, info: SeqInfo) -> Path:
    """Write <seq_dir>/seqinfo.ini. TrackEval reads seqLength from it."""
    seq_dir = Path(seq_dir)
    seq_dir.mkdir(parents=True, exist_ok=True)
    path = seq_dir / "seqinfo.ini"
    path.write_text(
        "[Sequence]\n"
        f"name={info.name}\n"
        f"imDir={info.im_dir}\n"
        f"frameRate={info.frame_rate}\n"
        f"seqLength={info.seq_length}\n"
        f"imWidth={info.im_width}\n"
        f"imHeight={info.im_height}\n"
        f"imExt={info.im_ext}\n",
        encoding="utf-8",
    )
    return path


def read_seqinfo(path: Path) -> SeqInfo:
    """Parse a seqinfo.ini written by this module or shipped with a MOTChallenge sequence."""
    path = Path(path)
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    if "Sequence" not in parser:
        raise MotFormatError(f"{path}: missing a [Sequence] section")
    sec = parser["Sequence"]
    try:
        return SeqInfo(
            name=sec["name"],
            seq_length=int(sec["seqLength"]),
            im_width=int(sec["imWidth"]),
            im_height=int(sec["imHeight"]),
            frame_rate=int(sec.get("frameRate", "15")),
            im_dir=sec.get("imDir", "img1"),
            im_ext=sec.get("imExt", ".jpg"),
        )
    except KeyError as exc:  # pragma: no cover - defensive, message matters more than cover
        raise MotFormatError(f"{path}: seqinfo.ini missing key {exc}") from None


def write_seqmap(path: Path, seqs: Sequence[str]) -> Path:
    """Write the seqmap: a literal "name" header line, then one sequence per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("name\n" + "".join(f"{s}\n" for s in seqs), encoding="utf-8")
    return path


def read_seqmap(path: Path) -> list[str]:
    """Read a seqmap, dropping the header line and any blanks."""
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()]
    if not lines or lines[0] != "name":
        raise MotFormatError(f"{path}: seqmap must start with a literal 'name' header line")
    return [ln for ln in lines[1:] if ln]


# ---------------------------------------------------------------- directory tree
@dataclass(frozen=True, slots=True)
class TrackEvalTree:
    """A materialised TrackEval directory tree, plus the identifiers it was built for."""

    root: Path
    benchmark: str
    split: str
    tracker: str
    seqs: tuple[str, ...]

    @property
    def gt_folder(self) -> Path:
        return self.root / "gt"

    @property
    def trackers_folder(self) -> Path:
        return self.root / "trackers"

    @property
    def output_folder(self) -> Path:
        return self.root / "results"

    @property
    def gt_set(self) -> str:
        return f"{self.benchmark}-{self.split}"

    @property
    def seqmap_file(self) -> Path:
        return self.gt_folder / "seqmaps" / f"{self.gt_set}.txt"

    @property
    def do_preproc(self) -> bool:
        return self.benchmark in PREPROC_BENCHMARKS

    def gt_file(self, seq: str) -> Path:
        return self.gt_folder / self.gt_set / seq / "gt" / "gt.txt"

    def seqinfo_file(self, seq: str) -> Path:
        return self.gt_folder / self.gt_set / seq / "seqinfo.ini"

    def tracker_file(self, seq: str) -> Path:
        return self.trackers_folder / self.gt_set / self.tracker / "data" / f"{seq}.txt"


def _place(src: Path, dst: Path, *, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link:
        dst.symlink_to(Path(src).resolve())
    else:
        shutil.copyfile(src, dst)


def build_trackeval_tree(
    root: Path,
    *,
    benchmark: str,
    split: str,
    tracker: str,
    gt_files: Mapping[str, Path],
    tracker_files: Mapping[str, Path],
    seqinfo: Mapping[str, SeqInfo],
    link: bool = False,
) -> TrackEvalTree:
    """Materialise the TrackEval tree under `root` and return a handle to it.

    `link=True` symlinks the gt and tracker text instead of copying it, which is what a
    multi-GB registered benchmark wants; the default copies, so a tmp tree is
    self-contained. Every mapping is keyed by sequence name and all three must agree.
    """
    seqs = tuple(gt_files)
    missing_tracker = [s for s in seqs if s not in tracker_files]
    missing_info = [s for s in seqs if s not in seqinfo]
    extra_tracker = [s for s in tracker_files if s not in gt_files]
    if missing_tracker or missing_info or extra_tracker:
        raise MotFormatError(
            "gt_files, tracker_files and seqinfo must cover the same sequences; "
            f"missing tracker files {missing_tracker}, missing seqinfo {missing_info}, "
            f"tracker files with no ground truth {extra_tracker}"
        )
    if not seqs:
        raise MotFormatError("no sequences given; TrackEval refuses an empty seqmap")

    tree = TrackEvalTree(
        root=Path(root), benchmark=benchmark, split=split, tracker=tracker, seqs=seqs
    )
    for seq in seqs:
        _place(Path(gt_files[seq]), tree.gt_file(seq), link=link)
        _place(Path(tracker_files[seq]), tree.tracker_file(seq), link=link)
        write_seqinfo(tree.seqinfo_file(seq).parent, seqinfo[seq])
    write_seqmap(tree.seqmap_file, seqs)
    tree.output_folder.mkdir(parents=True, exist_ok=True)
    return tree


# ---------------------------------------------------------------- TrackEval driver
#: The metrics lifted out of TrackEval's nested result dict, in report order.
REPORTED_METRICS: tuple[str, ...] = (
    "HOTA",
    "DetA",
    "AssA",
    "MOTA",
    "MOTP",
    "IDF1",
    "IDSW",
)


def run_trackeval(
    tree: TrackEvalTree,
    *,
    num_cores: int = 1,
    class_name: str = "pedestrian",
) -> dict[str, dict[str, float]]:
    """HOTA / MOTA / IDF1 per sequence, as plain floats.

    Returns `{sequence_name: {"HOTA": ..., "MOTA": ..., "IDF1": ..., ...}}` with the extra
    key "COMBINED_SEQ" holding TrackEval's own combination over sequences. Every value is
    a built-in float, so the result serialises to JSON with no numpy encoder.

    HOTA is an array over TrackEval's alpha sweep (0.05..0.95); the scalar is its mean.
    That is the published definition and it is stated here because taking element 0 by
    accident is a plausible and silent error.

    A sequence whose tracker file is empty scores 0.0 and does not raise. That branch is
    why trackeval is pinned to the PyPI fork rather than the upstream git URL, whose
    `np.float` on the empty path was removed in NumPy 1.24.
    """
    import numpy as np
    import trackeval

    ev_cfg = trackeval.Evaluator.get_default_eval_config()
    ev_cfg.update(
        {
            "USE_PARALLEL": num_cores > 1,
            "NUM_PARALLEL_CORES": max(1, num_cores),
            "PRINT_RESULTS": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": True,
            "PLOT_CURVES": False,
            # The default writes error_log.txt into the installed package directory.
            "LOG_ON_ERROR": None,
        }
    )
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update(
        {
            "GT_FOLDER": str(tree.gt_folder),
            "TRACKERS_FOLDER": str(tree.trackers_folder),
            "OUTPUT_FOLDER": str(tree.output_folder),
            "BENCHMARK": tree.benchmark,
            "SPLIT_TO_EVAL": tree.split,
            "TRACKERS_TO_EVAL": [tree.tracker],
            "SEQ_INFO": dict.fromkeys(tree.seqs),
            "SEQMAP_FILE": str(tree.seqmap_file),
            "DO_PREPROC": tree.do_preproc,
            "CLASSES_TO_EVAL": [class_name],
            "PRINT_CONFIG": False,
        }
    )
    # Each metric carries its own PRINT_CONFIG, which is not covered by the eval config.
    quiet = {"PRINT_CONFIG": False}

    res, msg = trackeval.Evaluator(ev_cfg).evaluate(
        [trackeval.datasets.MotChallenge2DBox(ds_cfg)],
        [
            trackeval.metrics.HOTA(quiet),
            trackeval.metrics.CLEAR(quiet),
            trackeval.metrics.Identity(quiet),
        ],
    )
    per_tracker = res["MotChallenge2DBox"][tree.tracker]
    if per_tracker is None:
        raise RuntimeError(
            f"TrackEval could not evaluate tracker '{tree.tracker}': "
            f"{msg['MotChallenge2DBox'][tree.tracker]}"
        )

    def _scalars(block: Mapping[str, object]) -> dict[str, float]:
        hota, clear, ident = block["HOTA"], block["CLEAR"], block["Identity"]
        return {
            "HOTA": float(np.mean(hota["HOTA"])),
            "DetA": float(np.mean(hota["DetA"])),
            "AssA": float(np.mean(hota["AssA"])),
            "MOTA": float(clear["MOTA"]),
            "MOTP": float(clear["MOTP"]),
            "IDF1": float(ident["IDF1"]),
            "IDSW": float(clear["IDSW"]),
        }

    return {key: _scalars(value[class_name]) for key, value in per_tracker.items()}
