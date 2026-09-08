"""Evidence packs + false-alarm report (vividly.eval.evidence, scripts/eval/extract_evidence.py).

Pure window/manifest/report logic first; then an end-to-end on a frame directory written by
the test and an EventStore filled by the test. No model, no network, no committed footage.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from vividly.eval.evidence import (
    NOTE,
    SCHEMA_VERSION,
    EvidenceError,
    EvidenceWindow,
    false_alarm_report,
    pack_id,
    pack_manifest,
    read_pack,
    render_false_alarm_report,
    window_for,
)
from vividly.events import EventStore
from vividly.schemas.events import AlertRecord, make_alert_id, utc_now_iso

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "eval" / "extract_evidence.py"


def _alert(
    rule_id: str, track_id: int, onset: float, detected: float, feedback: str | None = None
) -> dict[str, Any]:
    return {
        "alert_id": make_alert_id(rule_id, track_id, onset),
        "rule_id": rule_id,
        "rule_version": "0.1.0",
        "track_id": track_id,
        "onset_ts": onset,
        "detected_ts": detected,
        "details": {"down_s": 31.0},
        "baseline": None,
        "feedback": feedback,
    }


# ------------------------------------------------------------------ windows
def test_window_brackets_onset_and_detection_and_clamps_at_zero() -> None:
    w = window_for(_alert("fall_then_still_v0", 1, 100.0, 130.0))
    assert w == EvidenceWindow(onset_t=100.0, detected_t=130.0, start_t=90.0, end_t=135.0)
    assert w.span_s == 45.0
    early = window_for(_alert("fall_then_still_v0", 1, 3.0, 4.0), pre_s=10.0, post_s=0.0)
    assert early.start_t == 0.0 and early.end_t == 4.0
    assert window_for(_alert("r", 1, 5.0, 5.0), pre_s=0.0, post_s=0.0).span_s == 0.0
    with pytest.raises(EvidenceError, match="precedes"):
        window_for(_alert("r", 1, 10.0, 9.0))
    with pytest.raises(EvidenceError, match="pre_s"):
        window_for(_alert("r", 1, 1.0, 2.0), pre_s=-1.0)
    with pytest.raises(EvidenceError, match="numeric"):
        window_for({"rule_id": "r"})


def test_frame_range_is_inclusive_and_clamped_to_the_source() -> None:
    w = EvidenceWindow(onset_t=10.0, detected_t=12.0, start_t=9.5, end_t=12.5)
    assert list(w.frame_range(1.0, None)) == [9, 10, 11, 12, 13]
    assert list(w.frame_range(2.0, None)) == list(range(19, 26))
    assert list(w.frame_range(1.0, 11)) == [9, 10]
    assert list(w.frame_range(1.0, 5)) == []
    with pytest.raises(EvidenceError):
        w.frame_range(0.0, None)


def test_pack_id_is_filesystem_safe_and_derived() -> None:
    a = _alert("fall_then_still_v0", 7, 12.5, 40.0)
    assert pack_id(a) == a["alert_id"].replace("/", "_")
    assert pack_id({"alert_id": "a/b c:d"}) == "a_b_c_d"
    assert pack_id({"rule_id": "r", "track_id": 2, "onset_ts": 1.5}) == "r-2-1.5"


def test_manifest_carries_the_label_window_frames_and_the_review_note() -> None:
    a = _alert("seizure_candidate_v0", 3, 20.0, 28.0, feedback="false_alarm")
    w = window_for(a, pre_s=2.0, post_s=1.0)
    frames = [{"index": 18, "t_source": 18.0, "file": "frames/0000018.jpg"}]
    m = pack_manifest(
        a, w, frames, fps=1.0, source={"images": "/x"}, run={"run_dir": "/r"}, extracted_utc="t"
    )
    assert m["schema_version"] == SCHEMA_VERSION and m["label"] == "false_alarm"
    assert m["window"] == {"onset_t": 20.0, "detected_t": 28.0, "start_t": 18.0, "end_t": 29.0}
    assert m["n_frames"] == 1 and m["frames"] == frames and m["note"] == NOTE
    assert m["alert"]["alert_id"] == a["alert_id"]
    with pytest.raises(EvidenceError, match="unknown feedback label"):
        pack_manifest(
            {**a, "feedback": "maybe"}, w, frames, fps=1.0, source={}, run={}, extracted_utc="t"
        )


# ------------------------------------------------------------------ report
def test_false_alarm_report_counts_labels_and_the_negative_control_stance() -> None:
    alerts = [
        _alert("fall_then_still_v0", 1, 10.0, 40.0, "false_alarm"),
        _alert("fall_then_still_v0", 2, 50.0, 80.0, "agree"),
        _alert("fall_then_still_v0", 3, 90.0, 120.0),
        _alert("seizure_candidate_v0", 4, 5.0, 13.0, "false_alarm"),
        _alert("egress_without_discharge_v0", 5, 1.0, 601.0),
    ]
    r = false_alarm_report(alerts, camera_s=7200.0)
    assert r["rule_ids"] == [
        "fall_then_still_v0",
        "seizure_candidate_v0",
        "egress_without_discharge_v0",
    ]
    fall = r["per_rule"]["fall_then_still_v0"]
    assert fall == {
        "firings": 3,
        "labelled": 2,
        "agree": 1,
        "false_alarm": 1,
        "unlabelled": 1,
        "false_alarm_frac_of_labelled": 0.5,
        "false_positives_counted": 1,
        "false_positives_per_camera_hour": 0.5,
    }
    assert r["per_rule"]["egress_without_discharge_v0"]["false_alarm_frac_of_labelled"] is None
    assert "unlabelled firings are neither" in r["status"]
    neg = false_alarm_report(alerts, camera_s=7200.0, all_false_positives=True)
    assert neg["per_rule"]["fall_then_still_v0"]["false_positives_counted"] == 3
    assert neg["per_rule"]["fall_then_still_v0"]["false_positives_per_camera_hour"] == 1.5
    assert "negative control" in neg["status"]
    no_time = false_alarm_report(alerts)
    assert no_time["per_rule"]["fall_then_still_v0"]["false_positives_per_camera_hour"] is None
    md = render_false_alarm_report(r, title="T")
    assert "| `fall_then_still_v0` | 3 | 2 | 1 | 1 | 1 | 0.50 | 1 | 0.50 |" in md
    assert "PROVISIONAL" in md and "not a rate" in md


# -------------------------------------------------------------- end to end
FPS = 2.0
N = 40


@pytest.fixture
def run_with_frames(tmp_path: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(N):
        assert cv2.imwrite(
            str(frames / f"seq_{i + 1:06d}.jpg"), np.full((16, 24, 3), 5 * i, dtype=np.uint8)
        )
        (frames / f"seq_{i + 1:06d}.txt").write_text("x\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    store = EventStore(run_dir / "alerts.sqlite")
    alerts = [
        _alert("fall_then_still_v0", 1, 10.0, 12.0),
        _alert("seizure_candidate_v0", 2, 1.0, 3.0),
    ]
    try:
        for a in alerts:
            store.insert_alert(
                AlertRecord(
                    alert_id=a["alert_id"],
                    track_id=a["track_id"],
                    rule_id=a["rule_id"],
                    rule_version=a["rule_version"],
                    onset_ts=a["onset_ts"],
                    detected_ts=a["detected_ts"],
                    created_utc=utc_now_iso(),
                    baseline=None,
                    details=a["details"],
                )
            )
        store.set_feedback(alerts[0]["alert_id"], "false_alarm")
    finally:
        store.close()
    return run_dir, frames, alerts


def _load_script() -> Any:
    name = "vividly_extract_evidence_script"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_extracts_a_pack_per_alert_from_an_image_sequence(
    run_with_frames: tuple[Path, Path, list[dict[str, Any]]], capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, frames, alerts = run_with_frames
    script = _load_script()
    rc = script.main(
        [
            "--run-dir",
            str(run_dir),
            "--images",
            str(frames),
            "--fps",
            str(FPS),
            "--pre-s",
            "2",
            "--post-s",
            "1",
        ]
    )
    assert rc == 0
    out = run_dir / "evidence"
    index = json.loads((out / "index.json").read_text())
    assert sorted(p["rule_id"] for p in index["packs"]) == [
        "fall_then_still_v0",
        "seizure_candidate_v0",
    ]
    assert all(p["status"] == "written" for p in index["packs"])
    fall_pack = out / pack_id(alerts[0])
    m = read_pack(fall_pack)
    # onset 10, detected 12, pre 2, post 1 -> [8, 13] s at 2 Hz -> frames 16..26
    assert [f["index"] for f in m["frames"]] == list(range(16, 27))
    assert m["label"] == "false_alarm" and m["fps"] == FPS
    assert m["source"]["images"] == str(frames) and m["run"]["run_dir"] == str(run_dir)
    files = sorted((fall_pack / "frames").iterdir())
    assert [f.name for f in files] == [f"{i:07d}.jpg" for i in range(16, 27)]
    # no .txt sneaked in, and the pack's first frame is the source's frame 16
    assert (frames / "seq_000017.jpg").read_bytes() == files[0].read_bytes()
    seizure_pack = out / pack_id(alerts[1])
    assert [f["index"] for f in read_pack(seizure_pack)["frames"]] == list(
        range(0, 9)
    )  # clamped at 0
    report = json.loads((out / "false_alarms.json").read_text())
    assert report["per_rule"]["fall_then_still_v0"]["false_alarm"] == 1
    assert report["per_rule"]["seizure_candidate_v0"]["unlabelled"] == 1
    assert (out / "false_alarms.md").exists()
    assert "2 pack(s) (2 written, 0 cached)" in capsys.readouterr().out
    # Idempotent: a second run caches; --force rewrites.
    assert script.main(["--run-dir", str(run_dir), "--images", str(frames), "--fps", str(FPS)]) == 0
    assert all(
        p["status"] == "cached" for p in json.loads((out / "index.json").read_text())["packs"]
    )
    assert (
        script.main(
            ["--run-dir", str(run_dir), "--images", str(frames), "--fps", str(FPS), "--force"]
        )
        == 0
    )
    assert all(
        p["status"] == "written" for p in json.loads((out / "index.json").read_text())["packs"]
    )


def test_script_refuses_incoherent_arguments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _load_script()
    assert script.main([]) == script.EXIT_CONFIG
    assert script.main(["--run-dir", str(tmp_path)]) == script.EXIT_CONFIG
    assert (
        script.main(["--run-dir", str(tmp_path), "--images", str(tmp_path)]) == script.EXIT_CONFIG
    )  # no fps
    assert (
        script.main(
            ["--run-dir", str(tmp_path), "--images", str(tmp_path), "--fps", "2", "--clip", "x.mp4"]
        )
        == script.EXIT_CONFIG
    )
    assert (
        script.main(["--run-dir", str(tmp_path), "--images", str(tmp_path), "--fps", "2"])
        == script.EXIT_CONFIG
    )  # no store
    assert "config error" in capsys.readouterr().err
