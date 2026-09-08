"""GMDCSA-24 parser tests (registry entry gmdcsa24, an interim prototype fall training set).

The distribution realities that matter: unquoted CSVs (segments regex-extracted with a
bracket-count guard so nothing drops silently), formatting wobbles verified against the
real Subject 1 files (stray colon, spaces around brackets), and the Fall/ADL cross
checks: a Fall clip with no Falling segment or an ADL clip with one is refused, since
either would poison a training class. Probe injection keeps cv2 out of CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vividly.data.convert import gmdcsa24

# Quirks copied exactly from the real Subject 1 CSVs: stray `:` after a segment, space
# before `[`, space inside `[ `, integer and decimal times, multi-segment rows.
FALL_CSV = (
    "File Name,Length (seconds),Time of Recording,Attire,Description, Classes\n"
    "01.mp4,06, Day (Light On),Full T-shirt & Trousers, Walking to falling, "
    "Falling (SW)[1.8 to 5]:; Walking[0 to 1.8]\n"
    "02.mp4,05, Night (Light Off),T-shirt & Trousers, Sitting to left side fall, "
    "Falling (SW) [ 2.8 to 6]; Sitting [0 to 2.9]\n"
)
ADL_CSV = (
    "File Name,Length (seconds),Time of Recording,Attire,Description, Classes\n"
    "01.mp4,08, Day (Light On), Full T-shirt & Trousers, Sitting to sleeping, "
    "Sitting[0 to 1]; Sleeping[2 to 8]\n"
)


def _tree(root: Path) -> Path:
    subject = root / "upstream" / "repo-d3edb5d" / "Subject 1"
    for cls, csv_text, names in (
        ("Fall", FALL_CSV, ("01.mp4", "02.mp4")),
        ("ADL", ADL_CSV, ("01.mp4",)),
    ):
        (subject / cls).mkdir(parents=True)
        (subject / f"{cls}.csv").write_text(csv_text)
        for name in names:
            (subject / cls / name).write_bytes(b"")
    return subject


def test_classes_line_survives_upstream_wobbles() -> None:
    segs, dropped = gmdcsa24.parse_classes_line(
        "05.mp4,05, Day, attire, desc, Falling (SW)[1.8 to 5]:; Walking[0 to 1.8]"
    )
    assert segs == [("Falling (SW)", 1.8, 5.0), ("Walking", 0.0, 1.8)] and dropped == 0


def test_classes_line_missing_to_and_multi_interval() -> None:
    # Copied exactly from S2 Fall 19 (`Eating[0.9 4.6]`, no `to`); S4 Fall 09 does it in the
    # Falling bracket itself; plus a multi-interval bracket (one tuple per interval).
    segs, dropped = gmdcsa24.parse_classes_line(
        "19.mp4,15,Day,T-Shirt & Pants 2,Eating in standing position then falling forward,"
        "Falling (FW)[8.2 to 15]; Eating[0.9 4.6]; Standing[0 to 4; 6.8 to 8.1]"
    )
    assert (
        segs
        == [
            ("Falling (FW)", 8.2, 15.0),
            ("Eating", 0.9, 4.6),
            ("Standing", 0.0, 4.0),
            ("Standing", 6.8, 8.1),
        ]
        and dropped == 0
    )


def test_classes_line_recovers_missing_close_bracket() -> None:
    # Copied exactly from S3 ADL 04: `Standing[...` never closes; `Walking[` implicitly closes it.
    segs, dropped = gmdcsa24.parse_classes_line(
        "04.mp4,10,Day,Full T-shirt & Trousers,Picking a book from the floor and putting it "
        "on the bed; sitting on the bed, Sitting[9.2 to 10]; "
        "Standing[2.4 to 3.6; 6.6 to 7.9; Walking[3.6 to 5.2; 7.9 to 9.2] "
    )
    assert (
        segs
        == [
            ("Sitting", 9.2, 10.0),
            ("Standing", 2.4, 3.6),
            ("Standing", 6.6, 7.9),
            ("Walking", 3.6, 5.2),
            ("Walking", 7.9, 9.2),
        ]
        and dropped == 0
    )


def test_reversed_context_interval_dropped_but_reversed_fall_refused() -> None:
    # Copied exactly from S2 ADL 06: `Sitting[9.7 to 3]` contradicts its own Standing/Walking
    # spans under either repair, so it is dropped and counted. A reversed FALL interval feeds
    # events_t and is refused outright.
    segs, dropped = gmdcsa24.parse_classes_line(
        "06.mp4,10,Day,Full Shirt & Pant,Picking up the mobile phone, "
        "Sitting[9.7 to 3]; Standing[0 to 0.5; 4.2 to 6.5]; Walking[0.5 to 1.8; 6 to 8.7]"
    )
    assert dropped == 1 and ("Sitting", 9.7, 3.0) not in segs and len(segs) == 4
    with pytest.raises(gmdcsa24.Gmdcsa24ParseError, match="nonsensical"):
        gmdcsa24.parse_classes_line("01.mp4,05,d,a,desc, Falling (SW)[5 to 1.8]")


def test_unmatched_bracket_is_refused() -> None:
    with pytest.raises(gmdcsa24.Gmdcsa24ParseError, match="unparseable interval"):
        gmdcsa24.parse_classes_line("01.mp4,05,d,a,desc, Falling [3.4 - 6]")
    with pytest.raises(gmdcsa24.Gmdcsa24ParseError, match="unclosed bracket"):
        gmdcsa24.parse_classes_line("01.mp4,05,d,a,desc, Falling [3.4 to 6")


def test_index_events_symlinks_and_summary(tmp_path: Path) -> None:
    _tree(tmp_path)
    index = gmdcsa24.build_index(tmp_path, probe=lambda p: (30.0, 150))
    assert index["summary"] == {
        "n_videos": 3,
        "n_fall": 2,
        "n_adl": 1,
        "n_subjects": 1,
        "n_multi_fall_segment": 0,
        "n_fall_beyond_container": 1,  # 02.mp4 ends at 6 s but the container holds 5 s
        "n_reversed_dropped": 0,
        "fall_unlabeled_excluded": [],
    }
    by = {r["video"]: r for r in index["videos"]}
    fall = by["videos/S1_Fall/01.mp4"]
    assert fall["has_fall"] and fall["events_t"] == [[1.8, 5.0]]
    assert fall["time_of_recording"] == "Day (Light On)"
    assert not fall["fall_beyond_container"]
    assert by["videos/S1_Fall/02.mp4"]["fall_beyond_container"]
    adl = by["videos/S1_ADL/01.mp4"]
    assert not adl["has_fall"] and adl["events_t"] == []
    assert ["Sleeping", 2.0, 8.0] in adl["segments"]
    # relative symlink resolves to the upstream bytes and survives re-runs
    link = tmp_path / "videos" / "S1_Fall" / "01.mp4"
    assert link.is_symlink() and link.resolve().is_file()
    gmdcsa24.build_index(tmp_path, probe=lambda p: (30.0, 150))
    assert link.resolve().is_file()


def test_fall_clip_without_fall_interval_is_excluded_and_listed(tmp_path: Path) -> None:
    # The exact S4 Fall 15 shape: `Falling (FW); Sitting[0 to 1.7]`, fall labeled but
    # no interval bracket. Excluded and listed, never indexed as a silent negative.
    subject = _tree(tmp_path)
    (subject / "Fall.csv").write_text(FALL_CSV.replace("Falling (SW)[1.8 to 5]:", "Falling (FW)"))
    index = gmdcsa24.build_index(tmp_path, probe=lambda p: (30.0, 150))
    assert index["summary"]["fall_unlabeled_excluded"] == ["S1_Fall/01.mp4"]
    assert index["summary"]["n_videos"] == 2 and index["summary"]["n_fall"] == 1
    assert "videos/S1_Fall/01.mp4" not in {r["video"] for r in index["videos"]}
    assert not (tmp_path / "videos" / "S1_Fall" / "01.mp4").exists()


def test_adl_clip_with_falling_segment_is_refused(tmp_path: Path) -> None:
    subject = _tree(tmp_path)
    (subject / "ADL.csv").write_text(ADL_CSV.replace("Sleeping[2 to 8]", "Falling[2 to 8]"))
    with pytest.raises(gmdcsa24.Gmdcsa24ParseError, match="ADL clip with"):
        gmdcsa24.build_index(tmp_path, probe=lambda p: (30.0, 150))


def test_csv_video_mismatch_is_refused(tmp_path: Path) -> None:
    subject = _tree(tmp_path)
    (subject / "Fall" / "99.mp4").write_bytes(b"")
    with pytest.raises(gmdcsa24.Gmdcsa24ParseError, match="no row: \\['99'\\]"):
        gmdcsa24.build_index(tmp_path, probe=lambda p: (30.0, 150))


def test_multi_falling_segments_union_to_one_event(tmp_path: Path) -> None:
    subject = _tree(tmp_path)
    (subject / "Fall.csv").write_text(
        FALL_CSV.replace("Walking[0 to 1.8]", "Falling (FW)[0.2 to 1.5]")
    )
    index = gmdcsa24.build_index(tmp_path, probe=lambda p: (30.0, 150))
    rec = {r["video"]: r for r in index["videos"]}["videos/S1_Fall/01.mp4"]
    # union span, never two half-labelled events (an unfilled fall trains as negative)
    assert rec["events_t"] == [[0.2, 5.0]] and rec["n_fall_segments"] == 2
    assert index["summary"]["n_multi_fall_segment"] == 1


def test_fall_prefix_label_counts_as_fall(tmp_path: Path) -> None:
    # Copied exactly from S3 Fall 01, which labels its event `Fall (FW)`, not `Falling (FW)`.
    subject = _tree(tmp_path)
    (subject / "Fall.csv").write_text(
        FALL_CSV.replace("Falling (SW)[1.8 to 5]:", "Fall (FW)[0 to 3]")
    )
    index = gmdcsa24.build_index(tmp_path, probe=lambda p: (30.0, 150))
    rec = {r["video"]: r for r in index["videos"]}["videos/S1_Fall/01.mp4"]
    assert rec["has_fall"] and rec["events_t"] == [[0.0, 3.0]]


def test_annotations_upstream_overrides_zip_csv(tmp_path: Path) -> None:
    # The Zenodo v2.0 zip ships interval-less CSVs; the pinned-commit download wins.
    _tree(tmp_path)
    override_dir = tmp_path / "annotations_upstream" / "Subject 1"
    override_dir.mkdir(parents=True)
    (override_dir / "Fall.csv").write_text(FALL_CSV.replace("[1.8 to 5]", "[2.5 to 5]"))
    index = gmdcsa24.build_index(tmp_path, probe=lambda p: (30.0, 150))
    rec = {r["video"]: r for r in index["videos"]}["videos/S1_Fall/01.mp4"]
    assert rec["events_t"] == [[2.5, 5.0]]
    assert rec["annotation"] == "annotations_upstream/Subject 1/Fall.csv"


def test_write_index_lands_in_annotations(tmp_path: Path) -> None:
    _tree(tmp_path)
    out = gmdcsa24.write_index(
        tmp_path, gmdcsa24.build_index(tmp_path, probe=lambda p: (30.0, 150))
    )
    assert out == tmp_path / "annotations" / "index.json"
