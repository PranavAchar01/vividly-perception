"""UR Fall label parser tests (registry entry ur_fall, a held-out test set).

The distribution reality that matters most: ADL sequences also carry label-0 frames
(controlled lying-down transitions). Those must surface as lie_transition_* metadata
with has_fall False, never as fall events, because they are the confounder class
the held-out eval breaks out. Probe injection keeps cv2 and real bytes out of CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vividly.data.convert import ur_fall

FALLS = "fall-01,1,-1,0\nfall-01,2,0,0\nfall-01,3,0,0\nfall-01,4,1,0\n"
ADLS = (
    "adl-01,1,-1,0\nadl-01,2,-1,0\n"
    "adl-02,1,-1,0\nadl-02,2,0,0\nadl-02,3,1,0\n"  # controlled lie-down transition
)


def _tree(root: Path) -> None:
    (root / "videos").mkdir(parents=True)
    (root / "labels").mkdir()
    for name in ("fall-01-cam0.mp4", "adl-01-cam0.mp4", "adl-02-cam0.mp4"):
        (root / "videos" / name).write_bytes(b"")
    (root / "labels" / ur_fall.FALLS_CSV).write_text(FALLS)
    (root / "labels" / ur_fall.ADLS_CSV).write_text(ADLS)


def test_index_falls_and_adl_transitions(tmp_path: Path) -> None:
    _tree(tmp_path)
    index = ur_fall.build_index(tmp_path, probe=lambda p: (30.0, 120))
    assert index["summary"] == {
        "n_videos": 3,
        "n_fall": 1,
        "n_adl": 2,
        "n_adl_with_lie_transition": 1,
    }
    by = {r["video"]: r for r in index["videos"]}
    fall = by["videos/fall-01-cam0.mp4"]
    assert fall["has_fall"] and (fall["fall_start_frame"], fall["fall_end_frame"]) == (2, 3)
    assert fall["lie_transition_start_frame"] is None
    plain = by["videos/adl-01-cam0.mp4"]
    assert not plain["has_fall"] and plain["lie_transition_start_frame"] is None
    lied = by["videos/adl-02-cam0.mp4"]
    # label-0 in an ADL is a controlled transition, never a fall event
    assert not lied["has_fall"] and lied["fall_start_frame"] is None
    assert (lied["lie_transition_start_frame"], lied["lie_transition_end_frame"]) == (2, 2)


def test_fall_sequence_without_zero_frames_is_refused(tmp_path: Path) -> None:
    _tree(tmp_path)
    (tmp_path / "labels" / ur_fall.FALLS_CSV).write_text("fall-01,1,-1,0\nfall-01,2,1,0\n")
    with pytest.raises(ur_fall.UrFallParseError, match="no label-0"):
        ur_fall.build_index(tmp_path, probe=lambda p: (30.0, 120))


def test_video_missing_from_csv_is_refused(tmp_path: Path) -> None:
    _tree(tmp_path)
    (tmp_path / "videos" / "fall-99-cam0.mp4").write_bytes(b"")
    with pytest.raises(ur_fall.UrFallParseError, match="no rows"):
        ur_fall.build_index(tmp_path, probe=lambda p: (30.0, 120))


def test_bad_label_value_is_refused(tmp_path: Path) -> None:
    _tree(tmp_path)
    (tmp_path / "labels" / ur_fall.ADLS_CSV).write_text("adl-01,1,7,0\n")
    with pytest.raises(ur_fall.UrFallParseError, match="outside"):
        ur_fall.build_index(tmp_path, probe=lambda p: (30.0, 120))


def test_write_index_lands_in_annotations(tmp_path: Path) -> None:
    _tree(tmp_path)
    out = ur_fall.write_index(tmp_path, ur_fall.build_index(tmp_path, probe=lambda p: (30.0, 60)))
    assert out == tmp_path / "annotations" / "index.json"
