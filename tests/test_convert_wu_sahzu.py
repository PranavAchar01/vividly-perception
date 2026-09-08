"""vividly.data.convert.wu_sahzu: label parsing, cut planning, index building."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from vividly.data.convert.wu_sahzu import (
    CUT_POST_S,
    CUT_PRE_S,
    ICTAL_S,
    WHOLE_FILE_MAX_S,
    WuSahzuParseError,
    _cut_argv,
    build_index,
    event_video_rel,
    negative_videos,
    parse_label_xlsx,
    plan_cut,
    write_index,
)

_SHEET_HEADER = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    "<sheetData>"
)


def _xlsx(path: Path, rows: list[list[tuple[str, object]]]) -> Path:
    """Minimal xlsx: rows are [(col_letter, value)]; str values go to sharedStrings."""
    shared: list[str] = []
    body = []
    for i, row in enumerate(rows, start=1):
        cells = []
        for col, value in row:
            ref = f"{col}{i}"
            if isinstance(value, str):
                if value not in shared:
                    shared.append(value)
                cells.append(f'<c r="{ref}" t="s"><v>{shared.index(value)}</v></c>')
            else:
                cells.append(f'<c r="{ref}"><v>{value!r}</v></c>')
        body.append(f'<row r="{i}">{"".join(cells)}</row>')
    sheet = _SHEET_HEADER + "".join(body) + "</sheetData></worksheet>"
    strings = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        + "".join(f"<si><t>{s}</t></si>" for s in shared)
        + "</sst>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/sharedStrings.xml", strings)
        z.writestr("xl/worksheets/sheet1.xml", sheet)
    return path


HEADER = [
    ("A", "PatID"),
    ("B", "Seizure Type"),
    ("C", "#Seizure"),
    ("D", "EEG onset"),
    ("E", "Clinical Onset"),
]


def _label_rows() -> list[list[tuple[str, object]]]:
    """Two patients, forward-filled type, day-fraction onsets."""
    return [
        HEADER,
        # Pat01 (PG): first row carries PatID + type; second forward-fills both.
        [("A", "Pat01"), ("B", "PG"), ("C", "Sz1"), ("D", 0.0414351852), ("E", 0.0416666667)],
        [("C", "Sz2"), ("D", 0.001), ("E", 0.0015)],
        # Pat04 (P)
        [("A", "Pat04"), ("B", "P"), ("C", "Sz1"), ("D", 0.0005), ("E", 0.00075)],
    ]


class TestParseLabelXlsx:
    def test_day_fractions_and_forward_fill(self, tmp_path: Path) -> None:
        events = parse_label_xlsx(_xlsx(tmp_path / "Label.xlsx", _label_rows()))
        assert [e["patid"] for e in events] == ["Pat01", "Pat01", "Pat04"]
        assert [e["seizure_type"] for e in events] == ["PG", "PG", "P"]
        assert [e["seizure"] for e in events] == ["Sz1", "Sz2", "Sz1"]
        # 0.0414351852 days * 86400 = 3580.0 s (00:59:40)
        assert events[0]["eeg_onset_s"] == pytest.approx(3580.0, abs=0.01)
        assert events[0]["clinical_onset_s"] == pytest.approx(3600.0, abs=0.01)

    def test_clinical_before_eeg_rejected(self, tmp_path: Path) -> None:
        rows = [HEADER, [("A", "Pat01"), ("B", "PG"), ("C", "Sz1"), ("D", 0.002), ("E", 0.001)]]
        with pytest.raises(WuSahzuParseError, match="before"):
            parse_label_xlsx(_xlsx(tmp_path / "Label.xlsx", rows))

    def test_unknown_type_rejected(self, tmp_path: Path) -> None:
        rows = [HEADER, [("A", "Pat01"), ("B", "XX"), ("C", "Sz1"), ("D", 0.001), ("E", 0.002)]]
        with pytest.raises(WuSahzuParseError, match="unknown seizure type"):
            parse_label_xlsx(_xlsx(tmp_path / "Label.xlsx", rows))


class TestPlanCut:
    def test_short_file_stays_whole(self) -> None:
        assert plan_cut(WHOLE_FILE_MAX_S, 100.0) is None

    def test_long_file_cut_window(self) -> None:
        start, dur = plan_cut(3654.0, 3600.0)
        assert start == 3600.0 - CUT_PRE_S
        assert dur == pytest.approx(CUT_PRE_S + 54.0)  # post clamps at file end

    def test_early_onset_clamps_to_zero(self) -> None:
        start, dur = plan_cut(1000.0, 60.0)
        assert start == 0.0
        assert dur == pytest.approx(60.0 + CUT_POST_S)

    def test_cut_argv_reencodes(self) -> None:
        argv = _cut_argv("ffmpeg", Path("in.mp4"), Path("out.mp4"), 3420.0, 420.0)
        assert argv == [
            "ffmpeg", "-y", "-v", "error",
            "-ss", "3420.000", "-i", "in.mp4", "-t", "420.000",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-an", "out.mp4",
        ]  # fmt: skip
        # input-seek (-ss before -i) + re-encode = frame accurate; -an drops audio
        assert argv.index("-ss") < argv.index("-i")


def _fake_tree(root: Path) -> None:
    _xlsx(root / "dataset" / "Label.xlsx", _label_rows())
    for rel in ("pat01/Sz1PG.mp4", "pat01/Sz2PG.mp4", "pat04/Sz1P.mp4", "pat04/free.mp4"):
        p = root / "dataset" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x00")


#: durations chosen so pat01/Sz1 (onset 3600 s) needs a cut, the rest stay whole.
_DURATIONS = {
    "Sz1PG.mp4": 3654.0,
    "Sz2PG.mp4": 200.0,
    "Sz1P.mp4": 120.0,
    "free.mp4": 300.0,
    # the clip the cut would have produced (probed when cut=False finds it on disk)
    "clips": 420.0,
}


def _fake_probe(path: Path) -> tuple[float, float, int]:
    dur = _DURATIONS["clips"] if "clips" in path.parts else _DURATIONS[path.name]
    return dur, 30.0, int(dur * 30)


class TestBuildIndex:
    def test_records(self, tmp_path: Path) -> None:
        _fake_tree(tmp_path)
        clip = tmp_path / "clips" / "pat01" / "Sz1PG.mp4"
        clip.parent.mkdir(parents=True)
        clip.write_bytes(b"\x00")
        index = build_index(tmp_path, probe=_fake_probe, cut=False)
        assert index["summary"] == {
            "n_videos": 4,
            "n_seizure": 3,
            "n_convulsive": 2,
            "n_focal": 1,
            "n_negative": 1,
            "n_patients": 2,
        }
        by_video = {r["video"]: r for r in index["videos"]}

        cut = by_video["clips/pat01/Sz1PG.mp4"]
        assert cut["cut_start_s"] == 3600.0 - CUT_PRE_S
        assert cut["clinical_onset_t"] == pytest.approx(CUT_PRE_S, abs=0.01)
        assert cut["eeg_onset_t"] == pytest.approx(CUT_PRE_S - 20.0, abs=0.01)
        assert cut["events_t"][0][0] == pytest.approx(CUT_PRE_S, abs=0.01)
        # onset + ICTAL_S = 300 s fits inside the 420 s clip, so no clamp here
        assert cut["events_t"][0][1] == pytest.approx(CUT_PRE_S + ICTAL_S)
        assert cut["convulsive"] is True
        assert cut["video_upstream"] == "dataset/pat01/Sz1PG.mp4"

        whole = by_video["dataset/pat01/Sz2PG.mp4"]
        assert whole["cut_start_s"] == 0.0
        assert whole["events_t"][0][0] == pytest.approx(0.0015 * 86400, abs=0.01)
        assert whole["events_t"][0][1] == pytest.approx(_DURATIONS["Sz2PG.mp4"])

        focal = by_video["dataset/pat04/Sz1P.mp4"]
        assert focal["convulsive"] is False
        assert focal["has_seizure"] is True
        end = focal["events_t"][0][1]
        assert end == pytest.approx(min(0.00075 * 86400 + ICTAL_S, 120.0))

        neg = by_video["dataset/pat04/free.mp4"]
        assert neg["has_seizure"] is False
        assert neg["events_t"] == []
        assert neg["annotation"]  # extractor's annotated_only filter must keep it

    def test_onset_beyond_duration_rejected(self, tmp_path: Path) -> None:
        _fake_tree(tmp_path)

        def probe(path: Path) -> tuple[float, float, int]:
            return 10.0, 30.0, 300  # every file 10 s, all onsets beyond it

        with pytest.raises(WuSahzuParseError, match="beyond duration"):
            build_index(tmp_path, probe=probe, cut=False)

    def test_missing_video_rejected(self, tmp_path: Path) -> None:
        _fake_tree(tmp_path)
        clip = tmp_path / "clips" / "pat01" / "Sz1PG.mp4"
        clip.parent.mkdir(parents=True)
        clip.write_bytes(b"\x00")
        (tmp_path / "dataset" / "pat01" / "Sz2PG.mp4").unlink()
        with pytest.raises(WuSahzuParseError, match="missing event video"):
            build_index(tmp_path, probe=_fake_probe, cut=False)

    def test_write_index(self, tmp_path: Path) -> None:
        _fake_tree(tmp_path)
        clip = tmp_path / "clips" / "pat01" / "Sz1PG.mp4"
        clip.parent.mkdir(parents=True)
        clip.write_bytes(b"\x00")
        out = write_index(tmp_path, build_index(tmp_path, probe=_fake_probe, cut=False))
        assert out == tmp_path / "annotations" / "index.json"
        assert out.is_file()


def test_event_video_rel() -> None:
    rel = event_video_rel({"patid": "Pat01", "seizure": "Sz1", "seizure_type": "PG"})
    assert rel == Path("dataset/pat01/Sz1PG.mp4")


def test_negative_videos(tmp_path: Path) -> None:
    _fake_tree(tmp_path)
    assert negative_videos(tmp_path) == [Path("dataset/pat04/free.mp4")]
