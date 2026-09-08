"""Le2i annotation parser tests (registry entry le2i_fdd, an interim prototype training set).

Pure-parser cases cover every distribution reality recorded in the module docstring:
the 2-line fall header, the explicit "0,0" no-fall header, the three files whose header
pair is spliced mid-file, absent-person -1 rows, and malformed input. The index test
injects a fake probe so no test needs cv2 or real dataset bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vividly.data.convert import le2i

FALL_TEXT = "144\n164\n1,1,205,70,259,170\n2,1,205,70,259,170\n3,7,77,129,200,194\n"


def test_parses_fall_header_and_rows() -> None:
    ann = le2i.parse_annotation_text(FALL_TEXT)
    assert ann.has_header and ann.has_fall
    assert (ann.fall_start, ann.fall_end) == (144, 164)
    assert len(ann.boxes) == 3 and ann.n_absent == 0
    first = ann.boxes[0]
    assert (first.frame, first.label) == (1, 1)
    assert (first.x1, first.y1, first.x2, first.y2) == (205, 70, 259, 170)


def test_zero_zero_header_means_no_fall() -> None:
    ann = le2i.parse_annotation_text("0\n0\n1,1,10,10,20,40\n")
    assert ann.has_header and not ann.has_fall
    assert ann.fall_start is None and ann.fall_end is None


def test_headerless_file_parses_as_no_fall() -> None:
    ann = le2i.parse_annotation_text("1,1,72,58,132,170\n2,1,72,58,132,170\n")
    assert not ann.has_header and not ann.has_fall and not ann.header_misplaced
    assert len(ann.boxes) == 2


def test_misplaced_mid_file_header_pair_is_recovered() -> None:
    # Coffee_room_01/video_026 + Coffee_room_02/video_050/052 splice the fall-interval
    # pair between frame rows instead of putting it at the top.
    text = "218,2,144,137,234,201\n197\n227\n219,2,137,137,238,197\n"
    ann = le2i.parse_annotation_text(text)
    assert ann.has_header and ann.header_misplaced and ann.has_fall
    assert (ann.fall_start, ann.fall_end) == (197, 227)
    assert [b.frame for b in ann.boxes] == [218, 219]


def test_second_bare_pair_is_refused() -> None:
    text = "0\n0\n1,1,10,10,20,40\n5\n9\n"
    with pytest.raises(le2i.Le2iParseError, match="bare pair"):
        le2i.parse_annotation_text(text)


def test_absent_person_rows_are_counted_not_kept() -> None:
    ann = le2i.parse_annotation_text("0\n0\n1,1,-1,-1,-1,-1\n2,1,10,10,20,40\n")
    assert ann.n_absent == 1
    assert [b.frame for b in ann.boxes] == [2]


@pytest.mark.parametrize(
    "text",
    [
        "5\n2\n1,1,10,10,20,40\n",  # inverted interval
        "0\n0\n1,1,10,10\n",  # wrong field count
        "0\n0\n1,1,a,b,c,d\n",  # non-integer
    ],
)
def test_malformed_input_raises(text: str) -> None:
    with pytest.raises(le2i.Le2iParseError):
        le2i.parse_annotation_text(text)


def _fake_tree(root: Path) -> None:
    (root / "videos" / "Home_01").mkdir(parents=True)
    (root / "videos" / "Office").mkdir(parents=True)
    (root / "annotations" / "Home_01").mkdir(parents=True)
    (root / "videos" / "Home_01" / "video_001.avi").write_bytes(b"")
    (root / "videos" / "Home_01" / "video_002.avi").write_bytes(b"")
    (root / "videos" / "Office" / "video_001.avi").write_bytes(b"")  # unannotated subset
    (root / "annotations" / "Home_01" / "video_001.txt").write_text(FALL_TEXT)
    (root / "annotations" / "Home_01" / "video_002.txt").write_text("0\n0\n1,1,1,1,2,2\n")


def test_index_records_per_video_fps_and_fall_flags(tmp_path: Path) -> None:
    _fake_tree(tmp_path)
    # fps differs per video in the real distribution (24.0003 and 25.0 mixed), so the
    # index must carry each container's own clock, not one dataset-wide constant.
    fps_by_name = {"video_001.avi": 25.0, "video_002.avi": 24.0003840061441}
    index = le2i.build_index(tmp_path, probe=lambda p: (fps_by_name.get(p.name, 25.0), 150))
    assert index["summary"] == {
        "n_videos": 3,
        "n_annotated": 2,
        "n_fall": 1,
        "n_no_fall": 1,
        "n_header_misplaced": 0,
        "n_fall_beyond_container": 1,  # fall_end 164 > 150 container frames
    }
    by_video = {r["video"]: r for r in index["videos"]}
    fall = by_video["videos/Home_01/video_001.avi"]
    assert fall["has_fall"] and fall["fall_beyond_container"]
    assert (fall["fall_start_frame"], fall["fall_end_frame"]) == (144, 164)
    assert fall["fps"] == 25.0
    assert by_video["videos/Home_01/video_002.avi"]["fps"] == pytest.approx(24.0003840)
    office = by_video["videos/Office/video_001.avi"]
    assert office["annotation"] is None and office["has_fall"] is None


def test_write_index_round_trips(tmp_path: Path) -> None:
    _fake_tree(tmp_path)
    out = le2i.write_index(tmp_path, le2i.build_index(tmp_path, probe=lambda p: (25.0, 300)))
    assert out == tmp_path / "annotations" / "index.json"
    loaded = json.loads(out.read_text())
    assert loaded["dataset"] == "le2i_fdd"
    assert loaded["summary"]["n_videos"] == 3


def test_fully_reversed_corners_are_swap_normalized() -> None:
    # Home_02/video_051 frames 241..265 write the corners in reverse order.
    ann = le2i.parse_annotation_text("0\n0\n241,1,117,133,50,38\n")
    assert ann.n_swapped == 1
    box = ann.boxes[0]
    assert (box.x1, box.y1, box.x2, box.y2) == (50, 38, 117, 133)


def test_single_axis_inversion_still_refused() -> None:
    with pytest.raises(le2i.Le2iParseError, match="inverted box"):
        le2i.parse_annotation_text("0\n0\n1,1,50,10,20,40\n")
