"""MOT adapter tests: the writers, the TrackEval tree, and metric ordering.

This file covers vividly.eval.mot, the format layer. It builds and scores a committed
synthetic fixture and never reads a registered dataset, so no acquisition or counsel gate
is in scope here; those are pinned in tests/test_registry.py.

Two import rules hold throughout, because breaking either one turns a Linux install without
the linux-harness group red while the Mac stays green:

  * trackeval, numpy and cv2 are darwin-marked in the dev fork. Every test that needs one
    calls pytest.importorskip first.
  * vividly.eval.mot itself must import with none of them present, which
    `test_mot_module_has_no_top_level_darwin_only_imports` pins.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from vividly.eval.mot import (
    DT_COLUMNS,
    DT_TAIL,
    GT_COLUMNS,
    PREPROC_BENCHMARKS,
    MotFormatError,
    MotRow,
    SeqInfo,
    build_trackeval_tree,
    parse_mot_file,
    read_seqinfo,
    read_seqmap,
    rows_from_detections,
    run_trackeval,
    write_gt_file,
    write_seqinfo,
    write_seqmap,
    write_tracker_file,
)

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "mot_synth"
SEQS = ("SEQ01", "SEQ02")
BENCHMARK = "VIVIDLYSYNTH"
SPLIT = "train"


# --------------------------------------------------------------- helpers
def _rows(frames: int = 3, tracks: int = 2) -> list[MotRow]:
    """Deterministic straight-line rows. No RNG anywhere in this file."""
    return [
        MotRow(
            frame=f,
            track_id=t,
            left=10.0 * t + 5.0 * (f - 1),
            top=100.0 * t,
            width=40.0,
            height=90.0,
            conf=1.0,
        )
        for f in range(1, frames + 1)
        for t in range(1, tracks + 1)
    ]


def _fixture_inputs(tracker: str) -> tuple[dict[str, Path], dict[str, Path], dict[str, SeqInfo]]:
    gt = {s: FIXTURE / "seqs" / s / "gt" / "gt.txt" for s in SEQS}
    tf = {s: FIXTURE / "trackers" / tracker / f"{s}.txt" for s in SEQS}
    info = {s: read_seqinfo(FIXTURE / "seqs" / s / "seqinfo.ini") for s in SEQS}
    return gt, tf, info


def _score(tmp_path: Path, tracker: str) -> dict[str, dict[str, float]]:
    gt, tf, info = _fixture_inputs(tracker)
    tree = build_trackeval_tree(
        tmp_path / tracker,
        benchmark=BENCHMARK,
        split=SPLIT,
        tracker=tracker,
        gt_files=gt,
        tracker_files=tf,
        seqinfo=info,
    )
    return run_trackeval(tree)


# --------------------------------------------------------------- writers, round-trip
def test_gt_file_round_trips_exactly(tmp_path: Path) -> None:
    rows = _rows()
    n = write_gt_file(tmp_path / "gt.txt", rows)
    assert n == len(rows)
    assert parse_mot_file(tmp_path / "gt.txt") == sorted(rows, key=lambda r: (r.frame, r.track_id))


def test_tracker_file_round_trips_exactly(tmp_path: Path) -> None:
    rows = _rows()
    write_tracker_file(tmp_path / "SEQ.txt", rows)
    parsed = parse_mot_file(tmp_path / "SEQ.txt", kind="tracker")
    assert parsed == sorted(rows, key=lambda r: (r.frame, r.track_id))


def test_column_counts_and_tracker_tail(tmp_path: Path) -> None:
    write_gt_file(tmp_path / "gt.txt", _rows(frames=2, tracks=1))
    write_tracker_file(tmp_path / "dt.txt", _rows(frames=2, tracks=1))
    gt_lines = (tmp_path / "gt.txt").read_text().splitlines()
    dt_lines = (tmp_path / "dt.txt").read_text().splitlines()
    assert all(len(ln.split(",")) == len(GT_COLUMNS) == 9 for ln in gt_lines)
    assert all(len(ln.split(",")) == len(DT_COLUMNS) == 10 for ln in dt_lines)
    assert all(tuple(int(v) for v in ln.split(",")[7:]) == DT_TAIL for ln in dt_lines)


def test_frames_are_one_based_and_ids_non_negative() -> None:
    with pytest.raises(MotFormatError, match="frame must be 1-based"):
        MotRow(frame=0, track_id=1, left=0, top=0, width=10, height=10)
    # track_id 0 is legal: MOT-format ground truth is not guaranteed to count identities
    # from 1. DanceTrack, read during the retired re-identification v0 work and not a
    # registry entry now, starts every validation sequence at id 0.
    MotRow(frame=1, track_id=0, left=0, top=0, width=10, height=10)
    with pytest.raises(MotFormatError, match="track_id must be non-negative"):
        MotRow(frame=1, track_id=-1, left=0, top=0, width=10, height=10)


def test_degenerate_box_is_rejected() -> None:
    with pytest.raises(MotFormatError, match="positive width and height"):
        MotRow(frame=1, track_id=1, left=0, top=0, width=0, height=10)


def test_empty_frame_writes_no_lines_and_does_not_shift_indices(tmp_path: Path) -> None:
    """Frame 2 has no detections. Frame 3's rows must still say 3."""
    rows = [r for r in _rows(frames=3, tracks=1) if r.frame != 2]
    write_tracker_file(tmp_path / "dt.txt", rows)
    parsed = parse_mot_file(tmp_path / "dt.txt", kind="tracker")
    assert [r.frame for r in parsed] == [1, 3]


def test_duplicate_track_id_within_a_frame_is_refused(tmp_path: Path) -> None:
    row = MotRow(frame=4, track_id=2, left=0, top=0, width=10, height=10)
    with pytest.raises(MotFormatError, match="appears twice in frame 4"):
        write_gt_file(tmp_path / "gt.txt", [row, row])


def test_rows_are_sorted_by_frame_then_id(tmp_path: Path) -> None:
    rows = list(reversed(_rows(frames=3, tracks=2)))
    write_gt_file(tmp_path / "gt.txt", rows)
    parsed = parse_mot_file(tmp_path / "gt.txt")
    assert [(r.frame, r.track_id) for r in parsed] == sorted((r.frame, r.track_id) for r in rows)


def test_empty_tracker_write_produces_a_zero_byte_file(tmp_path: Path) -> None:
    assert write_tracker_file(tmp_path / "dt.txt", []) == 0
    assert (tmp_path / "dt.txt").read_bytes() == b""
    assert parse_mot_file(tmp_path / "dt.txt", kind="tracker") == []


def test_parse_rejects_wrong_column_count(tmp_path: Path) -> None:
    (tmp_path / "bad.txt").write_text("1,1,0,0,10,10,1\n")
    with pytest.raises(MotFormatError, match="expected 9 columns"):
        parse_mot_file(tmp_path / "bad.txt")


def test_parse_rejects_a_tracker_row_without_the_minus_one_tail(tmp_path: Path) -> None:
    (tmp_path / "bad.txt").write_text("1,1,0.00,0.00,10.00,10.00,1.0000,-1,-1,7\n")
    with pytest.raises(MotFormatError, match=r"must end in"):
        parse_mot_file(tmp_path / "bad.txt", kind="tracker")


# --------------------------------------------------------------- Detections adapter
def test_rows_from_detections_is_one_based_and_xywh() -> None:
    np = pytest.importorskip("numpy")
    from vividly.models.base import Detections

    det = Detections(
        xyxy=np.array([[10.0, 20.0, 50.0, 110.0]], dtype=np.float32),
        scores=np.array([0.75], dtype=np.float32),
        class_ids=np.zeros(1, dtype=np.int32),
        track_ids=np.array([7], dtype=np.int32),
    )
    empty = Detections(
        xyxy=np.zeros((0, 4), dtype=np.float32),
        scores=np.zeros(0, dtype=np.float32),
        class_ids=np.zeros(0, dtype=np.int32),
    )
    rows = rows_from_detections([empty, det])
    assert len(rows) == 1
    row = rows[0]
    assert (row.frame, row.track_id) == (2, 7)  # index 1 -> frame 2, and the empty
    assert (row.left, row.top, row.width, row.height) == (10.0, 20.0, 40.0, 90.0)
    assert row.conf == pytest.approx(0.75)


def test_rows_from_detections_refuses_untracked_output() -> None:
    np = pytest.importorskip("numpy")
    from vividly.models.base import Detections

    det = Detections(
        xyxy=np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
        scores=np.array([0.5], dtype=np.float32),
        class_ids=np.zeros(1, dtype=np.int32),
    )
    with pytest.raises(MotFormatError, match="no track_ids"):
        rows_from_detections([det])


# --------------------------------------------------------------- seqinfo and seqmap
def test_seqinfo_round_trips(tmp_path: Path) -> None:
    info = SeqInfo(name="SEQ09", seq_length=12, im_width=1280, im_height=720, frame_rate=10)
    write_seqinfo(tmp_path, info)
    assert read_seqinfo(tmp_path / "seqinfo.ini") == info


def test_seqmap_has_a_literal_name_header(tmp_path: Path) -> None:
    write_seqmap(tmp_path / "map.txt", ["SEQ01", "SEQ02"])
    assert (tmp_path / "map.txt").read_text().splitlines()[0] == "name"
    assert read_seqmap(tmp_path / "map.txt") == ["SEQ01", "SEQ02"]


def test_seqmap_without_the_header_is_refused(tmp_path: Path) -> None:
    (tmp_path / "map.txt").write_text("SEQ01\n")
    with pytest.raises(MotFormatError, match="'name' header"):
        read_seqmap(tmp_path / "map.txt")


# --------------------------------------------------------------- tree builder
def test_tree_layout_is_exactly_what_trackeval_looks_for(tmp_path: Path) -> None:
    gt, tf, info = _fixture_inputs("perfect")
    tree = build_trackeval_tree(
        tmp_path / "root",
        benchmark=BENCHMARK,
        split=SPLIT,
        tracker="t1",
        gt_files=gt,
        tracker_files=tf,
        seqinfo=info,
    )
    root = tmp_path / "root"
    gt_set = f"{BENCHMARK}-{SPLIT}"
    for seq in SEQS:
        assert (root / "gt" / gt_set / seq / "gt" / "gt.txt").is_file()
        assert (root / "gt" / gt_set / seq / "seqinfo.ini").is_file()
        assert (root / "trackers" / gt_set / "t1" / "data" / f"{seq}.txt").is_file()
    assert (root / "gt" / "seqmaps" / f"{gt_set}.txt").is_file()
    assert read_seqmap(tree.seqmap_file) == list(SEQS)
    # The copies are real copies, byte-identical to the committed fixture.
    assert tree.gt_file("SEQ01").read_bytes() == gt["SEQ01"].read_bytes()
    assert not tree.gt_file("SEQ01").is_symlink()


def test_tree_can_symlink_instead_of_copy(tmp_path: Path) -> None:
    gt, tf, info = _fixture_inputs("perfect")
    tree = build_trackeval_tree(
        tmp_path / "root",
        benchmark=BENCHMARK,
        split=SPLIT,
        tracker="t1",
        gt_files=gt,
        tracker_files=tf,
        seqinfo=info,
        link=True,
    )
    assert tree.gt_file("SEQ01").is_symlink()
    assert tree.gt_file("SEQ01").resolve() == gt["SEQ01"].resolve()


def test_tree_is_rebuildable_in_place(tmp_path: Path) -> None:
    """Two builds into the same root must not fail on the existing files."""
    gt, tf, info = _fixture_inputs("perfect")
    kwargs = dict(benchmark=BENCHMARK, split=SPLIT, tracker="t1", gt_files=gt, seqinfo=info)
    build_trackeval_tree(tmp_path / "root", tracker_files=tf, **kwargs)
    _, corrupt, _ = _fixture_inputs("corrupted")
    tree = build_trackeval_tree(tmp_path / "root", tracker_files=corrupt, **kwargs)
    assert tree.tracker_file("SEQ02").read_bytes() == corrupt["SEQ02"].read_bytes()


def test_tree_refuses_mismatched_sequence_sets(tmp_path: Path) -> None:
    gt, tf, info = _fixture_inputs("perfect")
    with pytest.raises(MotFormatError, match="missing tracker files"):
        build_trackeval_tree(
            tmp_path / "root",
            benchmark=BENCHMARK,
            split=SPLIT,
            tracker="t1",
            gt_files=gt,
            tracker_files={"SEQ01": tf["SEQ01"]},
            seqinfo=info,
        )


def test_tree_refuses_an_empty_seqmap(tmp_path: Path) -> None:
    with pytest.raises(MotFormatError, match="empty seqmap"):
        build_trackeval_tree(
            tmp_path / "root",
            benchmark=BENCHMARK,
            split=SPLIT,
            tracker="t1",
            gt_files={},
            tracker_files={},
            seqinfo={},
        )


def test_do_preproc_is_on_only_for_the_two_mot_benchmarks(tmp_path: Path) -> None:
    """DO_PREPROC on a custom benchmark either deletes everything or silently changes
    the denominator. It must follow the benchmark name, never a config knob."""
    assert {"MOT17", "MOT20"} == PREPROC_BENCHMARKS
    gt, tf, info = _fixture_inputs("perfect")
    for benchmark, expected in (("MOT20", True), ("MOT17", True), (BENCHMARK, False)):
        tree = build_trackeval_tree(
            tmp_path / benchmark,
            benchmark=benchmark,
            split=SPLIT,
            tracker="t1",
            gt_files=gt,
            tracker_files=tf,
            seqinfo=info,
        )
        assert tree.do_preproc is expected


# --------------------------------------------------------------- the committed fixture
def test_fixture_files_are_committed_and_parse() -> None:
    for seq in SEQS:
        gt_rows = parse_mot_file(FIXTURE / "seqs" / seq / "gt" / "gt.txt")
        assert gt_rows, f"{seq}: ground truth is empty"
        assert min(r.frame for r in gt_rows) == 1
        info = read_seqinfo(FIXTURE / "seqs" / seq / "seqinfo.ini")
        assert max(r.frame for r in gt_rows) == info.seq_length
        for tracker in ("perfect", "corrupted"):
            parse_mot_file(FIXTURE / "trackers" / tracker / f"{seq}.txt", kind="tracker")


def test_perfect_fixture_tracker_is_an_exact_copy_of_ground_truth() -> None:
    for seq in SEQS:
        gt_rows = parse_mot_file(FIXTURE / "seqs" / seq / "gt" / "gt.txt")
        dt_rows = parse_mot_file(FIXTURE / "trackers" / "perfect" / f"{seq}.txt", kind="tracker")
        assert [(r.frame, r.track_id, r.xyxy) for r in dt_rows] == [
            (r.frame, r.track_id, r.xyxy) for r in gt_rows
        ]


def test_corrupted_fixture_has_the_documented_corruptions() -> None:
    """An id swap from frame 7 of SEQ01, and frames 4-6 dropped from SEQ02."""
    gt1 = parse_mot_file(FIXTURE / "seqs" / "SEQ01" / "gt" / "gt.txt")
    dt1 = parse_mot_file(FIXTURE / "trackers" / "corrupted" / "SEQ01.txt", kind="tracker")
    assert len(dt1) == len(gt1)  # nothing dropped in SEQ01, only relabelled
    boxes = {(r.frame, r.track_id): r.xyxy for r in gt1}
    swapped = [r for r in dt1 if r.frame >= 7 and boxes[(r.frame, r.track_id)] != r.xyxy]
    assert {r.track_id for r in swapped} == {1, 2}

    dt2 = parse_mot_file(FIXTURE / "trackers" / "corrupted" / "SEQ02.txt", kind="tracker")
    assert {r.frame for r in dt2}.isdisjoint({4, 5, 6})

    for seq in SEQS:
        assert (FIXTURE / "trackers" / "empty" / f"{seq}.txt").stat().st_size == 0


# --------------------------------------------------------------- metrics (need trackeval)
def test_perfect_tracker_scores_one(tmp_path: Path) -> None:
    pytest.importorskip("trackeval")
    res = _score(tmp_path, "perfect")
    for key in (*SEQS, "COMBINED_SEQ"):
        assert res[key]["HOTA"] == pytest.approx(1.0)
        assert res[key]["MOTA"] == pytest.approx(1.0)
        assert res[key]["IDF1"] == pytest.approx(1.0)
        assert res[key]["IDSW"] == 0.0


def test_corrupted_tracker_scores_strictly_below_perfect(tmp_path: Path) -> None:
    pytest.importorskip("trackeval")
    perfect = _score(tmp_path, "perfect")
    corrupted = _score(tmp_path, "corrupted")
    assert corrupted["COMBINED_SEQ"]["HOTA"] < perfect["COMBINED_SEQ"]["HOTA"]
    assert corrupted["COMBINED_SEQ"]["IDF1"] < 1.0
    for seq in SEQS:
        assert corrupted[seq]["HOTA"] < perfect[seq]["HOTA"]
        assert corrupted[seq]["IDF1"] < 1.0
    # The two corruptions are visible in different metrics, which is the point of having
    # both: an id swap costs association, a dropped frame costs detection.
    assert corrupted["SEQ01"]["IDSW"] > 0
    assert corrupted["SEQ01"]["DetA"] == pytest.approx(1.0)
    assert corrupted["SEQ02"]["IDSW"] == 0.0
    assert corrupted["SEQ02"]["DetA"] < 1.0


def test_empty_tracker_file_scores_zero_and_does_not_crash(tmp_path: Path) -> None:
    """The latent np.float branch that broke upstream TrackEval. This test is the reason
    trackeval is pinned to the maintained distribution on the index rather than the git
    source: a normal happy-path run never reaches it."""
    pytest.importorskip("trackeval")
    res = _score(tmp_path, "empty")
    for key in (*SEQS, "COMBINED_SEQ"):
        assert res[key]["HOTA"] == 0.0
        assert res[key]["IDF1"] == 0.0
        assert isinstance(res[key]["HOTA"], float)


def test_metrics_are_plain_floats(tmp_path: Path) -> None:
    """They go straight into results.json, so no numpy scalars may survive."""
    pytest.importorskip("trackeval")
    res = _score(tmp_path, "perfect")
    for row in res.values():
        for value in row.values():
            assert type(value) is float
    json.dumps(res)  # would raise on a numpy scalar


# --------------------------------------------------------------- import discipline
def test_mot_module_has_no_top_level_darwin_only_imports() -> None:
    """trackeval, numpy and cv2 are darwin-marked in the dev fork, and trackeval is absent
    from the cuda fork entirely. A module-level import of any of them here would make
    `import vividly.eval.mot` fail on the GPU box and on any Linux install without the
    linux-harness group. AST, not grep, because a comment mentioning trackeval is fine and
    an import is not."""
    import vividly.eval.mot as mot

    tree = ast.parse(Path(mot.__file__).read_text())
    banned = {"trackeval", "numpy", "cv2"}
    offenders = []
    for node in tree.body:  # module level only; function bodies are not walked
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if a.name.split(".")[0] in banned]
        elif (
            isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in banned
        ):
            offenders.append(node.module)
    assert not offenders, f"imported at module level: {offenders}"
