"""ClipEmitter tests (CLIP-LIBRARY-DESIGN.md P3): the spool the Clip Library reads.

The load-bearing assertions: an injected `seizure_candidate_v0` record produces no file
and no side effect; the floor firing of the committed synthetic scene produces exactly one
file whose name carries no rule id; every `details` key outside the allowlist and every
`baseline` key except `n_samples` and `complete` is gone; the basis names only the fired
rule's own thresholds and its own file hash; the runner refuses every profile that would
break the non-device rules. Deterministic, network-free, model-free.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from vividly.clips_emitter import (
    BASELINE_KEYS,
    CLIP_TRIGGER_RULES,
    DETAILS_ALLOWLIST,
    BasisFacts,
    ClipEmitter,
    ClipsConfig,
    SourceClock,
    spool_id_for,
)
from vividly.models.checkpoints import sha256_of
from vividly.rules import RULE_KEYS, RuleConfig
from vividly.rules.fixtures import PATIENT_TRACK_ID, collapse_and_stay_down, torso_keypoints
from vividly.schemas.events import AlertRecord, make_alert_id
from vividly.sources import SyntheticDetectionSource
from vividly.tracks import TrackBook, TrackConfig, TrackObservation

REPO = Path(__file__).resolve().parents[1]
RUN_PIPELINE = REPO / "scripts" / "run_pipeline.py"
SYNTH_PROFILE = REPO / "configs" / "pipeline" / "clip_library_synth.yaml"
LIVE_PROFILE = REPO / "configs" / "pipeline" / "clip_library.yaml"
SCENE = REPO / "tests" / "fixtures" / "pipeline_synth" / "scene_60s.jsonl"
EGRESS_SCENE = REPO / "tests" / "fixtures" / "pipeline_synth" / "egress_scene.jsonl"
RULES_DIR = REPO / "configs" / "pipeline" / "rules"
RULE_ID_RE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*_v\d+")
EPOCH = 1_800_000_000.0


# ------------------------------------------------------------------- helpers
def _basis() -> BasisFacts:
    cfg = RuleConfig()
    return BasisFacts(
        rule_versions={rid: "0.1.0" for rid in RULE_KEYS if rid != "engine"},
        rule_thresholds={rid: cfg.rule_thresholds(rid) for rid in RULE_KEYS if rid != "engine"},
        rule_file_sha256={
            "fall_then_still_v0": sha256_of(RULES_DIR / "fall_then_still_v0.yaml"),
            "egress_without_discharge_v0": sha256_of(
                RULES_DIR / "egress_without_discharge_v0.yaml"
            ),
        },
        pipeline_config_sha256="00" * 32,
        model_ids=("synthetic-fixture",),
    )


def _emitter(tmp_path: Path, book: TrackBook, **overrides: Any) -> ClipEmitter:
    cfg = ClipsConfig.from_mapping(
        {
            "spool_dir": str(tmp_path / "spool"),
            "camera_id": "cam-test-01",
            "zone_id": "seating_a",
            "exit_zone_id": "door_b",
            "demo_mode": True,
            "source_media_path": str(tmp_path / "render.mp4"),
            **overrides,
        },
        default_source_kind="synthetic_jsonl",
    )
    return ClipEmitter(cfg, book=book, run_id="test-run", basis=_basis(), clock=SourceClock(EPOCH))


def _record(
    rule_id: str, *, track_id: int = 1, onset: float = 20.4, detected: float = 52.0
) -> AlertRecord:
    return AlertRecord(
        alert_id=make_alert_id(rule_id, track_id, onset),
        track_id=track_id,
        rule_id=rule_id,
        rule_version="0.1.0",
        onset_ts=onset,
        detected_ts=detected,
        created_utc="2026-09-09T14:00:52.000Z",
        baseline={
            "movement_var": 1.2,
            "posture_deg_mean": 5.0,
            "posture_deg_std": 1.0,
            "cy_mean": 540.0,
            "n_samples": 75,
            "complete": True,
        },
        details={
            "fall_s": 1.5,
            "fall_drop_px": 120.0,
            "down_s": 30.2,
            "n_down_samples": 151.0,
            "down_stale_s": 0.2,
            "upright_deg": 6.0,
            "down_deg_newest": 78.0,
            "down_mean_step_px": 0.1,
            "threshold_fall_max_s": 3.0,
            "threshold_status": "PROVISIONAL",
            "mrn_binding": "MOCK-1",
            "last_cx": 960.0,
        },
        mrn_binding="MOCK-1",
    )


def _load_runner() -> Any:
    name = "vividly_run_pipeline_script_clips"
    spec = importlib.util.spec_from_file_location(name, RUN_PIPELINE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _profile(tmp_path: Path, *, fixture: Path = SCENE, **overrides: Any) -> Path:
    cfg = yaml.safe_load(SYNTH_PROFILE.read_text(encoding="utf-8"))
    cfg["run_name"] = "clip_library_test"
    cfg["detections_from"] = str(fixture)
    cfg["output"] = {"dir": str(tmp_path / "data" / "pipeline")}
    cfg["clips"]["spool_dir"] = str(tmp_path / "data" / "spool")
    cfg["clips"]["source_media_path"] = str(tmp_path / "render.mp4")
    for key, value in overrides.items():
        if value is None:
            cfg.pop(key, None)
        else:
            cfg[key] = value
    # the temporary paths win over anything an override copied from the committed profile
    if isinstance(cfg.get("clips"), dict):
        cfg["clips"]["spool_dir"] = str(tmp_path / "data" / "spool")
        cfg["clips"]["source_media_path"] = str(tmp_path / "render.mp4")
    if isinstance(cfg.get("rules"), dict) and cfg["rules"].get("include"):
        cfg["rules"]["include"] = [str(RULES_DIR / Path(p).name) for p in cfg["rules"]["include"]]
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


# --------------------------------------------------------------- the refusal
def test_a_non_allowlisted_firing_produces_no_file(tmp_path: Path) -> None:
    """The load-bearing assertion: the third head can never reach the spool."""
    book = TrackBook(TrackConfig())
    emitter = _emitter(tmp_path, book)
    record = _record("seizure_candidate_v0", onset=41.2, detected=53.0)
    assert record.rule_id not in CLIP_TRIGGER_RULES
    emitter.on_alert(record)
    emitter.tick(60.0)
    emitter.tick(400.0)
    emitter.flush()
    assert list((tmp_path / "spool").iterdir()) == []
    assert emitter.n_written == 0 and emitter.n_dropped == 1
    assert emitter.stats()["n_open_windows"] == 0


def test_allowlist_is_exactly_the_two_triggers() -> None:
    assert frozenset({"fall_then_still_v0", "egress_without_discharge_v0"}) == CLIP_TRIGGER_RULES
    assert set(DETAILS_ALLOWLIST) == CLIP_TRIGGER_RULES
    assert BASELINE_KEYS == ("n_samples", "complete")


# ------------------------------------------------------------ the floor record
def test_floor_firing_writes_one_reduced_record_after_the_window(tmp_path: Path) -> None:
    batches = collapse_and_stay_down(
        lead_in_minutes=20.0 / 60.0, down_minutes=2.5, fall_s=1.5, hz=5.0, staff_track_id=None
    )
    book = TrackBook(TrackConfig.from_mapping({"fps": 5.0, "ring_window_s": 300.0}))
    emitter = _emitter(tmp_path, book, post_roll_max_s=40.0)
    record = _record("fall_then_still_v0")
    fired = False
    for batch in batches:
        book.ingest(batch)
        t = batch[0].t
        if not fired and t >= 52.0:
            emitter.on_alert(record)
            fired = True
        emitter.tick(t)
        if fired and t < 92.0:
            assert list((tmp_path / "spool").iterdir()) == [], "written before the window closed"
    files = list((tmp_path / "spool").iterdir())
    assert len(files) == 1
    path = files[0]
    assert path.name == f"{spool_id_for(record.alert_id)}.json"
    assert not RULE_ID_RE.search(path.name) and len(path.stem) == 16
    assert path.stat().st_mode & 0o777 == 0o600
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["kind"] == "clip" and doc["schema_version"] == 1
    assert doc["spool_id"] == hashlib.sha256(record.alert_id.encode()).hexdigest()[:16]
    fired_doc = doc["alert"]
    assert set(fired_doc["details"]) <= DETAILS_ALLOWLIST["fall_then_still_v0"]
    assert "mrn_binding" not in fired_doc["details"] and "last_cx" not in fired_doc["details"]
    assert fired_doc["mrn_binding"] is None and fired_doc["feedback"] is None
    assert set(fired_doc["baseline"]) == set(BASELINE_KEYS)
    assert doc["window"]["start_t"] == pytest.approx(20.4 - 30.0)
    assert doc["window"]["end_t"] == pytest.approx(92.0)  # detected + post_roll_max_s
    assert doc["window"]["pre_s"] == 30.0 and doc["window"]["post_max_s"] == 40.0
    assert doc["onset_utc"] == "2027-01-15T08:00:20.400Z"
    assert doc["source_epoch_utc"] == "2027-01-15T08:00:00.000Z"
    assert doc["first_staff_contact_source"] == "none" and doc["first_staff_contact_ts"] is None
    assert doc["contact_utc"] is None
    assert doc["source_kind"] == "synthetic_jsonl" and doc["demo_mode"] is True
    assert doc["camera_id"] == "cam-test-01" and doc["zone_id"] == "seating_a"
    basis = doc["basis"]
    assert basis["rule_id"] == "fall_then_still_v0"
    assert set(basis["thresholds"]) == set(RULE_KEYS["fall_then_still_v0"])
    assert basis["rule_file_sha256"] == sha256_of(RULES_DIR / "fall_then_still_v0.yaml")
    assert basis["baseline"] == {"n_samples": 75, "complete": True}
    assert "score" not in json.dumps(basis) and "verif" not in json.dumps(basis)
    boxes = doc["subject_track"]
    assert boxes and all(set(b) == {"t", "x", "y", "w", "h"} for b in boxes)
    assert all(doc["window"]["start_t"] <= b["t"] <= doc["window"]["end_t"] for b in boxes)
    assert "keypoints" not in json.dumps(doc)


def test_first_contact_is_measured_and_its_source_is_the_boundary_only_when_crossed(
    tmp_path: Path,
) -> None:
    def run(boundary: list[list[float]] | None) -> dict[str, Any]:
        batches = collapse_and_stay_down(
            lead_in_minutes=20.0 / 60.0, down_minutes=2.0, fall_s=1.5, hz=5.0, staff_track_id=None
        )
        book = TrackBook(TrackConfig.from_mapping({"fps": 5.0, "ring_window_s": 300.0}))
        emitter = _emitter(tmp_path, book, staff_boundary=boundary)
        record = _record("fall_then_still_v0")
        fired = False
        for batch in batches:
            t = batch[0].t
            extra = list(batch)
            if t >= 60.0:  # a second track walks in from the right and stays beside the person
                progress = min((t - 60.0) / 6.0, 1.0)
                cx = 1600.0 - (1600.0 - 1030.0) * progress
                cy = 300.0 + (640.0 - 300.0) * progress
                extra.append(
                    TrackObservation(
                        track_id=7,
                        t=t,
                        cx=cx,
                        cy=cy,
                        w=76.0,
                        h=104.0,
                        keypoints=torso_keypoints(cx, cy, 8.0),
                    )
                )
            book.ingest(extra)
            if not fired and t >= 52.0:
                emitter.on_alert(record)
                fired = True
            emitter.tick(t)
        emitter.flush()
        files = list((tmp_path / "spool").iterdir())
        assert len(files) == 1
        doc = json.loads(files[0].read_text(encoding="utf-8"))
        files[0].unlink()
        return doc

    doc = run(None)
    assert doc["first_staff_contact_source"] == "another_person"
    contact = doc["first_staff_contact_ts"]
    assert contact is not None and 63.0 <= contact <= 68.0
    assert doc["window"]["end_t"] == pytest.approx(contact + 15.0)
    assert doc["contact_utc"] is not None
    assert doc["staff_absent_s"] == pytest.approx(contact - doc["window"]["start_t"])

    doc = run([[1500.0, 250.0], [1700.0, 250.0], [1700.0, 400.0], [1500.0, 400.0]])
    assert doc["first_staff_contact_source"] == "staff_boundary"


def test_floor_window_left_open_at_close_is_flushed_at_the_last_tick(tmp_path: Path) -> None:
    book = TrackBook(TrackConfig.from_mapping({"fps": 5.0, "ring_window_s": 300.0}))
    emitter = _emitter(tmp_path, book)
    batches = collapse_and_stay_down(
        lead_in_minutes=20.0 / 60.0, down_minutes=0.5, fall_s=1.5, hz=5.0, staff_track_id=None
    )
    for batch in batches:
        book.ingest(batch)
        emitter.tick(batch[0].t)
    emitter.on_alert(_record("fall_then_still_v0"))
    emitter.tick(batches[-1][0].t)
    assert list((tmp_path / "spool").iterdir()) == []
    emitter.flush()
    files = list((tmp_path / "spool").iterdir())
    assert len(files) == 1
    doc = json.loads(files[0].read_text(encoding="utf-8"))
    assert doc["window"]["end_t"] == pytest.approx(batches[-1][0].t)


# ---------------------------------------------------------------- the runner
def test_headless_synthetic_profile_writes_exactly_one_spool_file(tmp_path: Path) -> None:
    runner = _load_runner()
    profile = _profile(tmp_path)
    assert runner.main(["--config", str(profile), "--headless"]) == 0
    spool = tmp_path / "data" / "spool"
    files = sorted(spool.iterdir())
    assert len(files) == 1
    assert not RULE_ID_RE.search(files[0].name)
    doc = json.loads(files[0].read_text(encoding="utf-8"))
    assert doc["alert"]["rule_id"] == "fall_then_still_v0"
    assert doc["alert"]["onset_ts"] == pytest.approx(20.4, abs=0.3)
    assert doc["run_id"].startswith("clip_library_test-")
    assert doc["basis"]["pipeline_config_sha256"] == sha256_of(profile)
    assert doc["basis"]["rule_file_sha256"] == sha256_of(RULES_DIR / "fall_then_still_v0.yaml")
    assert doc["basis"]["trace_ref"].startswith("latency.jsonl:")
    assert set(doc["basis"]["thresholds"]) == set(RULE_KEYS["fall_then_still_v0"])
    # the pipeline wrote nothing under the repository's outputs/ tree and no store file
    assert not (REPO / "outputs" / "pipeline" / "clip_library_test").exists()
    run_dir = tmp_path / "data" / "pipeline" / "clip_library_test"
    assert (run_dir / "latency.jsonl").exists() and not (run_dir / "alerts.sqlite").exists()
    summary = json.loads((run_dir / "latency.jsonl").read_text().splitlines()[-1])
    assert summary["clips"]["n_written"] == 1 and summary["clips"]["n_dropped"] == 0


def test_headless_egress_profile_writes_the_exit_record_and_the_stage_two_update(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    cfg = yaml.safe_load(SYNTH_PROFILE.read_text(encoding="utf-8"))
    tracks = {**cfg["tracks"], "reap_after_s": 8.0, "fps": 2.0, "baseline_min_samples": 10}
    rules = {**cfg["rules"], "egress_absent_min_s": 90.0, "egress_min_presence_s": 45.0}
    egress = {
        "enabled": True,
        "zones": [
            {"name": "door_b", "polygon": [[1640, 380], [1920, 380], [1920, 700], [1640, 700]]}
        ],
    }
    clips = {**cfg["clips"], "no_return_s": 30.0}
    profile = _profile(
        tmp_path, fixture=EGRESS_SCENE, tracks=tracks, rules=rules, egress=egress, clips=clips
    )
    assert runner.main(["--config", str(profile), "--headless"]) == 0
    spool = tmp_path / "data" / "spool"
    names = sorted(p.name for p in spool.iterdir())
    assert len(names) == 2
    clip_file = next(p for p in spool.iterdir() if p.suffix == ".json" and ".update." not in p.name)
    update_file = next(p for p in spool.iterdir() if ".update." in p.name)
    doc = json.loads(clip_file.read_text(encoding="utf-8"))
    assert doc["alert"]["rule_id"] == "egress_without_discharge_v0"
    assert doc["zone_id"] == "door_b"
    assert doc["window"]["end_t"] == pytest.approx(doc["alert"]["onset_ts"] + 15.0, abs=0.6)
    assert doc["window"]["end_t"] - doc["window"]["start_t"] <= 90.0
    assert set(doc["alert"]["details"]) <= DETAILS_ALLOWLIST["egress_without_discharge_v0"]
    assert doc["alert"]["details"]["exit_zone"] == "door_b"
    assert doc["subject_track"], "the departed track's boxes come from the reap snapshot"
    upd = json.loads(update_file.read_text(encoding="utf-8"))
    assert upd == {
        "schema_version": 1,
        "kind": "update",
        "spool_id": doc["spool_id"],
        "absent_s": 30.0,
        "returned": False,
        "updated_utc": upd["updated_utc"],
    }
    assert update_file.name == f"{doc['spool_id']}.update.json"


@pytest.mark.parametrize(
    ("label", "overrides", "match"),
    [
        ("verify", {"verify": {"policy": "annotate", "heads": {}}}, "verify"),
        (
            "third head",
            {"rules": {"enabled": ["fall_then_still_v0", "seizure_candidate_v0"], "include": []}},
            "may not run",
        ),
        ("no allowlist", {"rules": {"include": []}}, "rules.enabled"),
        ("reassoc", {"reassoc": {"persist": True}}, "reassoc.persist"),
        ("store on disk", {"store": {"path": "/tmp/elsewhere/alerts.sqlite"}}, "store.path"),
    ],
)
def test_runner_refuses_a_spool_profile_that_breaks_a_rule(
    tmp_path: Path, label: str, overrides: dict[str, Any], match: str
) -> None:
    runner = _load_runner()
    profile = _profile(tmp_path, **overrides)
    cfg = yaml.safe_load(profile.read_text(encoding="utf-8"))
    with pytest.raises(runner.ConfigError, match=match):
        runner.Pipeline(cfg, REPO, config_path=profile)


def test_runner_refuses_a_spool_profile_that_writes_under_outputs(tmp_path: Path) -> None:
    runner = _load_runner()
    profile = _profile(tmp_path, output={"dir": "outputs/pipeline"})
    cfg = yaml.safe_load(profile.read_text(encoding="utf-8"))
    with pytest.raises(runner.ConfigError, match="outputs/"):
        runner.Pipeline(cfg, REPO, config_path=profile)


def test_live_source_is_reserved_for_the_two_profile_families(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setenv("TEST_RTSP_URL", "rtsp://cam.invalid/s")
    cfg = yaml.safe_load(LIVE_PROFILE.read_text(encoding="utf-8"))
    cfg["run_name"] = "replay_demo"
    cfg["output"] = {"dir": str(tmp_path / "data" / "pipeline")}
    cfg["clips"]["spool_dir"] = str(tmp_path / "data" / "spool")
    cfg["source"]["rtsp_env"] = "TEST_RTSP_URL"
    cfg["rules"]["include"] = [str(RULES_DIR / Path(p).name) for p in cfg["rules"]["include"]]
    pipeline = runner.Pipeline(cfg, REPO, config_path=LIVE_PROFILE)
    try:
        assert str(pipeline.source_path) == "rtsp-env:TEST_RTSP_URL"
        with pytest.raises(runner.ConfigError, match="profile families"):
            pipeline._footage_source()
    finally:
        pipeline.close()
    assert "cam.invalid" not in pipeline.summary()


def test_an_unset_data_variable_is_a_config_error(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.delenv("VIVIDLY_CLIP_DATA", raising=False)
    cfg = yaml.safe_load(SYNTH_PROFILE.read_text(encoding="utf-8"))
    with pytest.raises(runner.ConfigError, match="not set"):
        runner.Pipeline(cfg, REPO, config_path=SYNTH_PROFILE)
    assert not (REPO / "${VIVIDLY_CLIP_DATA}").exists()


def test_masks_are_parsed_and_validated() -> None:
    runner = _load_runner()
    assert runner.build_masks({"masks": [{"x": 0, "y": 0, "w": 10, "h": 5}]}) == ((0, 0, 10, 5),)
    assert runner.build_masks({"path": "x.mp4"}) == ()
    with pytest.raises(runner.ConfigError):
        runner.build_masks({"masks": [{"x": 0, "y": 0, "w": 0, "h": 5}]})
    with pytest.raises(runner.ConfigError):
        runner.build_masks({"masks": [{"x": 0, "y": 0}]})


def test_committed_profiles_pass_the_guard_and_name_only_the_two_heads() -> None:
    for profile in (SYNTH_PROFILE, LIVE_PROFILE):
        cfg = yaml.safe_load(profile.read_text(encoding="utf-8"))
        assert cfg["rules"]["enabled"] == ["fall_then_still_v0", "egress_without_discharge_v0"]
        assert "verify" not in cfg
        assert cfg["store"]["path"] == ":memory:"
        assert cfg["reassoc"]["persist"] is False
        text = profile.read_text(encoding="utf-8").lower()
        assert "convuls" not in text and "hf_ratio" not in text
    scene = list(SyntheticDetectionSource(SCENE).observations())
    assert any(o.track_id == PATIENT_TRACK_ID for o in scene[0])
