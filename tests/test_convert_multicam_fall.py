"""Montreal Multi-Camera Fall table tests (registry entry multicam_fall, a held-out test set).

The annotation here is a transcription of the technical report's tables with the
erratum applied, so the tests lean on mechanical invariants: the report's periods are
strictly contiguous per scenario (validate_table), the erratum's corrected values and
camera remaps are pinned, and the fall census matches the table (25 fall events over
23 scenarios; chute04 and chute23 stage two falls; chute24 stages none).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vividly.data.convert import multicam_fall as mc


def test_table_passes_its_own_validation() -> None:
    mc.validate_table()


def test_validation_catches_broken_contiguity() -> None:
    broken = tuple(
        (s, cam, (((874, 1011, 1), (1013, 1079, 6)) if s == 1 else periods))
        for s, cam, periods in mc.SCENARIOS
    )
    with pytest.raises(mc.MulticamParseError, match="not contiguous"):
        mc.validate_table(broken)


def test_fall_census_matches_the_report() -> None:
    by_scenario = {s: mc.fall_periods(periods) for s, _, periods in mc.SCENARIOS}
    assert sum(len(v) for v in by_scenario.values()) == 25
    assert len(by_scenario[4]) == 2 and len(by_scenario[23]) == 2
    assert by_scenario[24] == []
    assert by_scenario[1] == [(1080, 1108)]


def test_erratum_values_are_applied() -> None:
    by_scenario = {s: (cam, periods) for s, cam, periods in mc.SCENARIOS}
    # camera remaps: report "11" -> 8, "10" -> 7
    assert by_scenario[1][0] == 8 and by_scenario[19][0] == 7
    # corrected period lists for scenarios 2, 7, 8, 14, 18
    assert mc.fall_periods(by_scenario[2][1]) == [(395, 425)]
    assert mc.fall_periods(by_scenario[7][1]) == [(491, 520)]
    assert mc.fall_periods(by_scenario[8][1]) == [(351, 380)]
    assert mc.fall_periods(by_scenario[14][1]) == [(1020, 1060)]
    assert mc.fall_periods(by_scenario[18][1]) == [(685, 710)]


def _tree(root: Path) -> None:
    for s, cam, _ in mc.SCENARIOS:
        d = root / mc.REMUX_DIRNAME / f"chute{s:02d}"
        d.mkdir(parents=True)
        (d / f"cam{cam}.avi").write_bytes(b"")


def test_index_records_reference_camera_and_event_lists(tmp_path: Path) -> None:
    _tree(tmp_path)
    index = mc.build_index(tmp_path, probe=lambda p: (30.0, 6000))
    assert index["summary"] == {
        "n_videos": 24,
        "n_fall_videos": 23,
        "n_fall_events": 25,
        "n_no_fall": 1,
    }
    by = {r["location"]: r for r in index["videos"]}
    single = by["chute01"]
    assert single["video"] == "remuxed/chute01/cam8.avi"
    assert single["video_upstream"] == "extracted/dataset/chute01/cam8.avi"
    assert (single["fall_start_frame"], single["fall_end_frame"]) == (1080, 1108)
    assert single["event_frames"] == [[1080, 1108]]
    double = by["chute04"]
    assert double["has_fall"] and double["fall_start_frame"] is None
    assert double["event_frames"] == [[288, 314], [601, 638]]
    none = by["chute24"]
    assert not none["has_fall"] and none["event_frames"] == []
    assert none["confounder_codes"] == [4, 5]


def test_missing_reference_video_names_the_remux_step(tmp_path: Path) -> None:
    _tree(tmp_path)
    (tmp_path / mc.REMUX_DIRNAME / "chute06" / "cam1.avi").unlink()
    with pytest.raises(mc.MulticamParseError, match="remux_reference_videos"):
        mc.build_index(tmp_path, probe=lambda p: (30.0, 6000))


def test_upstream_120fps_header_is_refused(tmp_path: Path) -> None:
    # The guard against indexing files that skipped the rate correction.
    _tree(tmp_path)
    with pytest.raises(mc.MulticamParseError, match="120 fps header"):
        mc.build_index(tmp_path, probe=lambda p: (mc.UPSTREAM_HEADER_FPS, 6000))


def test_remux_argv_is_stream_copy_at_the_corrected_rate() -> None:
    argv = mc._remux_argv("ffmpeg", Path("in.avi"), Path("out.avi"), mc.REFERENCE_FPS)
    assert argv == [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-r",
        "30",
        "-i",
        "in.avi",
        "-c:v",
        "copy",
        "-an",
        "out.avi",
    ]
    # -r must precede -i: it is an input option re-stamping the wrong header.
    assert argv.index("-r") < argv.index("-i")


# ---- all-cameras surface (delays from report fig. 3, probe-validated) ----------


def test_camera_delays_table_shape() -> None:
    assert sorted(mc.CAMERA_DELAYS) == list(range(1, 25))
    for s, delays in mc.CAMERA_DELAYS.items():
        assert len(delays) == 8
        assert min(delays) == 0, f"scenario {s}: some camera must be the shortest"


def test_shift_periods_maps_chute06_fall_into_cam2() -> None:
    periods = next(p for s, _, p in mc.SCENARIOS if s == 6)
    d = mc.CAMERA_DELAYS[6]
    shifted = mc.shift_periods(periods, d[0], d[1], 6000)  # ref cam1 -> cam2, +100
    assert (683, 729, mc.FALL_CODE) in shifted
    prev_end = None
    for s0, e0, _ in shifted:
        if prev_end is not None:
            assert s0 == prev_end + 1
        prev_end = e0


def test_shift_periods_clips_nonfall_and_drops_outside() -> None:
    periods = ((1, 50, 1), (51, 80, 3))
    out = mc.shift_periods(periods, 40, 0, 6000)  # shift -40
    assert out == [(1, 10, 1), (11, 40, 3)]
    out2 = mc.shift_periods(((1, 30, 1),), 40, 0, 6000)  # wholly before frame 1
    assert out2 == []


def test_shift_periods_refuses_clipping_a_fall() -> None:
    with pytest.raises(mc.MulticamParseError, match="refusing to shrink"):
        mc.shift_periods(((100, 160, 2),), 0, 0, 150)
    with pytest.raises(mc.MulticamParseError, match="leaves the container"):
        mc.shift_periods(((100, 160, 2),), 200, 0, 6000)


def _fake_counts(scenario: int) -> list[int]:
    tab = mc.CAMERA_DELAYS[scenario]
    counts = [2000 + d for d in tab]
    if scenario == 24:
        counts[7] -= mc.CHUTE24_CAM8_ANOMALY_FRAMES
    return counts


def test_validate_delays_accepts_probed_tree_with_known_anomaly(tmp_path: Path) -> None:
    def probe(p: Path) -> tuple[float, int]:
        scenario = int(p.parent.name[5:7])
        cam = int(p.stem[3:])
        return 30.0, _fake_counts(scenario)[cam - 1]

    mc.validate_delays(tmp_path, probe=probe)


def test_validate_delays_refuses_a_new_mismatch(tmp_path: Path) -> None:
    def probe(p: Path) -> tuple[float, int]:
        scenario = int(p.parent.name[5:7])
        cam = int(p.stem[3:])
        n = _fake_counts(scenario)[cam - 1]
        return 30.0, n + (1 if (scenario, cam) == (3, 2) else 0)

    with pytest.raises(mc.MulticamParseError, match="chute03"):
        mc.validate_delays(tmp_path, probe=probe)


def _tree_allcams(root: Path) -> None:
    for s in range(1, 25):
        d = root / mc.REMUX_DIRNAME / f"chute{s:02d}"
        d.mkdir(parents=True, exist_ok=True)
        for cam in range(1, 9):
            (d / f"cam{cam}.avi").write_bytes(b"")


def test_build_index_all_cameras_full_surface(tmp_path: Path) -> None:
    _tree_allcams(tmp_path)
    index = mc.build_index_all_cameras(tmp_path, probe=lambda p: (30.0, 6000))
    assert index["summary"] == {
        "n_videos": 192,
        "n_fall_videos": 184,
        "n_fall_events": 200,
        "n_no_fall": 8,
    }
    recs = {(r["location"], r["camera"]): r for r in index["videos"]}
    assert len(recs) == 192
    ref = mc.build_index(tmp_path, probe=lambda p: (30.0, 6000))
    for r in ref["videos"]:
        cam = int(Path(r["video"]).stem[3:])
        twin = recs[(r["location"], cam)]
        assert twin["is_reference_camera"] and twin["event_frames"] == r["event_frames"]
    anomaly = recs[("chute24", 8)]
    assert anomaly["annotation_alignment"].startswith("APPROXIMATE")
    assert not anomaly["has_fall"]
