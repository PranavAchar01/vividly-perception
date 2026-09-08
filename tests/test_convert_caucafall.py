"""CAUCAFall parser tests (registry entry caucafall, an interim prototype fall training set).

The distribution realities that matter: the avi is the source of record (measured
fps caught the paper's declared 23 fps being wrong; containers report 20), the
stills are only the label layer, spans are read on trailing-digit numbering (not
list position) so numbering holes shift nothing, and the real `- copia` stray is
excluded by name shape. classes.txt with any content other than the two known forms
is refused, because a reordered file would invert class semantics. Probe injection keeps
cv2 out of CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vividly.data.convert import caucafall

PROBE = lambda p: (20.0, 5, 720, 480)  # noqa: E731  (fps, n_frames, w, h)


def _seq(
    act_dir: Path,
    labels: list[int | None],
    *,
    prefix: str = "cas",
    start: int = 100001,
    avi: str | None = None,
) -> None:
    """One sequence dir: avi + numbered stills; label None = png without txt."""
    act_dir.mkdir(parents=True)
    (act_dir / (avi or f"{act_dir.name.replace(' ', '')}S1.avi")).write_bytes(b"")
    for i, label in enumerate(labels):
        stem = f"{prefix}{start + i:05d}"
        (act_dir / f"{stem}.png").write_bytes(b"")
        if label is not None:
            (act_dir / f"{stem}.txt").write_text(f"{label} 0.5 0.5 0.2 0.4\n")
    (act_dir / "classes.txt").write_text(
        "nofall\nfall\n" if any(v == 1 for v in labels) else "nofall\n"
    )


def _tree(root: Path) -> Path:
    base = root / "upstream" / "Dataset CAUCAFall" / "CAUCAFall"
    _seq(base / "Subject.1" / "Fall backwards", [0, 0, 1, 1, 1, 0])
    # dash-prefixed numbering restarting at 1, the real `cfs1-00001` shape
    _seq(base / "Subject.1" / "Walk", [0, 0, 0, 0], prefix="cfs1-", start=1)
    return base


def test_index_span_symlinks_and_summary(tmp_path: Path) -> None:
    _tree(tmp_path)
    index = caucafall.build_index(tmp_path, probe=PROBE)
    assert index["summary"] == {
        "n_sequences": 2,
        "n_fall": 1,
        "n_adl": 1,
        "n_subjects": 1,
        "n_class1_flicker": 0,
        "n_unlabeled_frames": 0,
        "n_gap_dirs": 0,
        "n_missing_classes_txt": 0,
        "excluded_odd_frames": [],
        "fps_values": [20.0],
        "frame_whs": [(720, 480)],
        "label_minus_container_values": [-1, 1],  # 6 and 4 stills vs 5-frame probe
    }
    by = {r["video"]: r for r in index["videos"]}
    fall = by["videos/S1/fall_backwards.avi"]
    assert fall["has_fall"] and (fall["fall_start_frame"], fall["fall_end_frame"]) == (3, 5)
    assert fall["fps"] == 20.0 and fall["n_frames_container"] == 5
    assert fall["n_class1_runs"] == 1 and fall["n_label_frames"] == 6
    walk = by["videos/S1/walk.avi"]
    assert not walk["has_fall"] and walk["fall_start_frame"] is None
    link = tmp_path / "videos" / "S1" / "fall_backwards.avi"
    assert link.is_symlink() and link.resolve().is_file()
    # re-runs refresh the link cleanly
    caucafall.build_index(tmp_path, probe=PROBE)
    assert link.resolve().is_file()


def test_copia_stray_is_excluded_and_spans_read_on_numbering(tmp_path: Path) -> None:
    base = _tree(tmp_path)
    fall_dir = base / "Subject.1" / "Fall backwards"
    (fall_dir / "cas100003 - copia.png").write_bytes(b"")
    # numbering hole before the fall: remove still 100002 entirely
    (fall_dir / "cas100002.png").unlink()
    (fall_dir / "cas100002.txt").unlink()
    index = caucafall.build_index(tmp_path, probe=PROBE)
    assert index["summary"]["excluded_odd_frames"] == [
        "Subject.1/Fall backwards/cas100003 - copia.png"
    ]
    rec = {r["video"]: r for r in index["videos"]}["videos/S1/fall_backwards.avi"]
    # span stays (3, 5) because it is read on numbering, not list position
    assert (rec["fall_start_frame"], rec["fall_end_frame"]) == (3, 5)
    assert rec["n_gap_frames"] == 1


def test_flicker_and_unlabeled_are_counted(tmp_path: Path) -> None:
    base = _tree(tmp_path)
    _seq(
        base / "Subject.2" / "Fall left", [0, 1, 0, 1, None, 0], start=200010, avi="FallLeftS2.avi"
    )
    index = caucafall.build_index(tmp_path, probe=PROBE)
    rec = {r["video"]: r for r in index["videos"]}["videos/S2/fall_left.avi"]
    assert rec["n_class1_runs"] == 2 and rec["n_unlabeled"] == 1
    assert (rec["fall_start_frame"], rec["fall_end_frame"]) == (2, 4)
    assert index["summary"]["n_class1_flicker"] == 1


def test_adl_with_class1_is_refused(tmp_path: Path) -> None:
    base = _tree(tmp_path)
    _seq(base / "Subject.3" / "Hop", [0, 1, 0], start=300001, avi="HopS3.avi")
    with pytest.raises(caucafall.CaucafallParseError, match="ADL sequence with class-1"):
        caucafall.build_index(tmp_path, probe=PROBE)


def test_fall_without_class1_is_refused(tmp_path: Path) -> None:
    base = _tree(tmp_path)
    _seq(base / "Subject.3" / "Fall right", [0, 0, 0], start=300001, avi="FallRightS3.avi")
    with pytest.raises(caucafall.CaucafallParseError, match="no class-1"):
        caucafall.build_index(tmp_path, probe=PROBE)


def test_remapped_classes_txt_is_refused(tmp_path: Path) -> None:
    base = _tree(tmp_path)
    (base / "Subject.1" / "Fall backwards" / "classes.txt").write_text("fall\nnofall\n")
    with pytest.raises(caucafall.CaucafallParseError, match="remap class ids"):
        caucafall.build_index(tmp_path, probe=PROBE)


def test_bad_class_token_is_refused(tmp_path: Path) -> None:
    base = _tree(tmp_path)
    (base / "Subject.1" / "Walk" / "cfs1-00002.txt").write_text("7 0.5 0.5 0.2 0.4\n")
    with pytest.raises(caucafall.CaucafallParseError, match="not 0/1"):
        caucafall.build_index(tmp_path, probe=PROBE)


def test_mixed_prefixes_are_refused(tmp_path: Path) -> None:
    base = _tree(tmp_path)
    (base / "Subject.1" / "Fall backwards" / "zzz900001.png").write_bytes(b"")
    with pytest.raises(caucafall.CaucafallParseError, match="mixed still prefixes"):
        caucafall.build_index(tmp_path, probe=PROBE)


def test_missing_or_extra_avi_is_refused(tmp_path: Path) -> None:
    base = _tree(tmp_path)
    (base / "Subject.1" / "Fall backwards" / "FallbackwardsS1-extra.avi").write_bytes(b"")
    with pytest.raises(caucafall.CaucafallParseError, match="exactly 1 avi"):
        caucafall.build_index(tmp_path, probe=PROBE)


def test_subject_10_sorts_after_subject_9(tmp_path: Path) -> None:
    base = _tree(tmp_path)
    _seq(base / "Subject.10" / "Walk", [0], start=990001, avi="WalkS10.avi")
    _seq(base / "Subject.9" / "Walk", [0], start=880001, avi="WalkS9.avi")
    index = caucafall.build_index(tmp_path, probe=PROBE)
    assert [r["location"] for r in index["videos"]] == ["S1", "S1", "S9", "S10"]


def test_write_index_lands_in_annotations(tmp_path: Path) -> None:
    _tree(tmp_path)
    out = caucafall.write_index(tmp_path, caucafall.build_index(tmp_path, probe=PROBE))
    assert out == tmp_path / "annotations" / "index.json"
