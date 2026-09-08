"""Extract per-track observation sequences from fall-dataset videos, the interim event
verifier data admitted by POLICY.md, rule 4.

Replays each annotated video through the same detector, pose and tracker stack the live
pipeline runs (scripts/run_pipeline.py, burn-in operating point), taps the frame stream,
and writes one JSONL of per-frame track observations per video plus a meta record with
the video's ground-truth fall interval. The fall verifier trains and evaluates on these
sequences, so its training features are exactly what vividly.rules.verify hands a
verifier at inference: track positions and posture angles on the pipeline's own clock,
never the dataset's ground-truth boxes, which the live system does not have.

    uv run python scripts/events/extract_track_sequences.py --config configs/events/extract_le2i_v2c.yaml
    ... --only Home_01/video_001 --max-frames 50    (smoke)
    ... --dry-run                                   (list what would run)

Resumable: a video whose sequence file already exists is skipped, so a killed run
continues where it stopped. Output layout under out_dir:

    sequences/<Location>__<video>.jsonl   one row per (frame, track) observation
    meta/<Location>__<video>.json         GT interval, fps, counts, rule firings
    runs/<key>/                           the underlying pipeline run (alerts.sqlite, latency)
    extraction.json                       models provenance, config digest, totals

Row schema (JSONL): {"f": frame_index_0based, "t": seconds, "track": id, "cx", "cy",
"w", "h", "angle": degrees_from_vertical_or_null, "kpc": mean_keypoint_conf_or_null},
plus "kp" under the config's `emit_keypoints: true`: [[x, y, conf], ...] in COCO-17
order, or null when the estimator returned nothing for that observation.
t is frame_index / container_fps, the ReplaySource clock, and the meta record carries
the same fps, so ground-truth frame numbers (1-based) align as t = (frame - 1) / fps.

`emit_keypoints` is off by default and every existing extraction keeps the frozen
9-field row. It exists because the reduction to angle+kpc was measured on 2026-08-28
to destroy the limb motion a convulsion consists of: rhythm computed from every
surviving signal scores ROC 0.35-0.53 against chance 0.5 on real seizures, while the
pose estimator had the joints in hand all along; research/event-verifier-v1-record.md
carries that measurement. Rows cost about 5 times more with it on.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vividly.data.convert.le2i import INDEX_FILENAME  # noqa: E402
from vividly.data.registry import Registry, default_registry  # noqa: E402
from vividly.tracks.state import posture_angle_deg  # noqa: E402

EXIT_OK = 0
EXIT_CONFIG = 3
RUN_PIPELINE = REPO_ROOT / "scripts" / "run_pipeline.py"


def _load_runner() -> Any:
    """scripts/ is not a package, so run_pipeline.py is loaded from its path.

    src/vividly/demo/jobs.py loads it the same way, through
    importlib.util.spec_from_file_location on the same script.
    """
    name = "vividly_run_pipeline_script_events"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RUN_PIPELINE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def _model_provenance(model: Any) -> dict[str, Any]:
    """label + weights + sha256 of the cached ONNX, when the backend exposes it.

    Provenance is recorded from the loaded model,
    never assumed from the config, as the checkpoint-provenance rule in POLICY.md requires.
    """
    if model is None:
        return {}
    from vividly.models.checkpoints import sha256_of

    prov = getattr(model, "provenance", None)
    out: dict[str, Any] = {}
    if prov is not None:
        out["label"] = prov.label()
        out["weights_uri"] = prov.weights_uri
        out["input_size"] = list(prov.input_size)
        out["sha256"] = prov.sha256
    inner = getattr(model, "_m", None)
    onnx_path = getattr(inner, "onnx_model", None)
    if isinstance(onnx_path, str) and Path(onnx_path).is_file():
        out["onnx_file"] = onnx_path
        out["sha256"] = sha256_of(Path(onnx_path))
    return out


def _key(record: dict[str, Any]) -> str:
    """`videos/Home_01/video_001.avi` -> `Home_01__video_001` (location + stem, so a
    flat-layout dataset like ur_fall keys as `cam0__fall-01-cam0`)."""
    return f"{record['location']}__{Path(record['video']).stem}"


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit(f"config {path} is not a mapping")
    for k in ("dataset", "out_dir", "pipeline"):
        if k not in cfg:
            raise SystemExit(f"config {path} is missing `{k}`")
    return cfg


def select_videos(
    index: dict[str, Any], *, only: str | None, annotated_only: bool
) -> list[dict[str, Any]]:
    records = list(index["videos"])
    if annotated_only:
        # The unannotated Lecture_room/Office subsets contain unlabelled falls (the
        # record page counts 143 falls over 191 videos, while the distribution measured
        # against the bytes in configs/datasets/le2i_fdd.yaml is 190 videos, of which
        # 130 are annotated and hold 99 falls), so they are poison as negatives and are
        # excluded, not just unlabelled.
        records = [r for r in records if r.get("annotation")]
    if only:
        needle = only.replace("/", "__")
        records = [r for r in records if needle in _key(r)]
    return records


def run_config_for(
    record: dict[str, Any], cfg: dict[str, Any], video_path: Path, runs_dir: Path
) -> dict[str, Any]:
    """The scripts/run_pipeline.py config for one video: template plus its own clock."""
    pipe = cfg["pipeline"]
    fps = float(record["fps"])
    if video_path.is_dir():
        # Frame directories: the runner's ImageSequenceSource shape, with fps mandatory
        # because a directory of frames records no rate of its own. frame_pattern rides
        # in from the index record when the frames are not the source's *.jpg default
        # (caucafall: *.png).
        source: dict[str, Any] = {"images": str(video_path), "fps": fps, "rate": None}
        if record.get("frame_pattern"):
            source["pattern"] = str(record["frame_pattern"])
    else:
        source = {"path": str(video_path), "rate": None}
    return {
        "schema_version": 1,
        "run_name": _key(record),
        "headless": True,
        "source": source,
        "detector": dict(pipe["detector"]),
        "pose": dict(pipe["pose"]),
        "tracker": {**dict(pipe["tracker"]), "frame_rate": fps},
        "tracks": {**dict(pipe["tracks"]), "fps": fps},
        "rules_config": str(pipe["rules_config"]),
        "output": {"dir": str(runs_dir)},
        "dashboard": {"jpeg_quality": int(cfg.get("jpeg_quality", 30))},
    }


def extract_one(
    record: dict[str, Any],
    cfg: dict[str, Any],
    dataset_root: Path,
    out_dir: Path,
    *,
    max_frames: int | None,
) -> dict[str, Any]:
    """One video through the pipeline; returns the meta record it wrote."""
    runner = _load_runner()
    key = _key(record)
    video_path = dataset_root / record["video"]
    run_cfg = run_config_for(record, cfg, video_path, out_dir / "runs")
    kp_conf_min = float(run_cfg["tracks"]["kp_conf_min"])
    # Opt-in per config: existing extractions and every model trained on them keep the
    # frozen 9-field row exactly as it was.
    emit_keypoints = bool(cfg.get("emit_keypoints", False))

    seq_path = out_dir / "sequences" / f"{key}.jsonl"
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    frame_counter = {"i": -1}

    pipeline = runner.Pipeline(run_cfg, REPO_ROOT)
    original_rules_step = pipeline._rules_step

    def tapped_rules_step(observations: list[Any], t: float) -> list[Any]:
        frame_counter["i"] += 1
        f = frame_counter["i"]
        for obs in observations:
            angle = posture_angle_deg(obs.keypoints, kp_conf_min)
            kpc = None
            if obs.keypoints:
                confs = [kp[2] for kp in obs.keypoints]
                kpc = round(sum(confs) / len(confs), 4)
            row = {
                "f": f,
                "t": round(obs.t, 4),
                "track": obs.track_id,
                "cx": round(obs.cx, 2),
                "cy": round(obs.cy, 2),
                "w": round(obs.w, 2),
                "h": round(obs.h, 2),
                "angle": None if angle is None else round(angle, 2),
                "kpc": kpc,
            }
            if emit_keypoints:
                # "kp" is [[x, y, conf], ...] in COCO-17 order, or null when the
                # estimator returned nothing for this observation. The module docstring
                # records why the reduction to angle+kpc is not enough for convulsions.
                row["kp"] = (
                    [
                        [round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                        for x, y, c in obs.keypoints
                    ]
                    if obs.keypoints
                    else None
                )
            rows.append(json.dumps(row, separators=(",", ":")))
        return original_rules_step(observations, t)

    pipeline._rules_step = tapped_rules_step
    t0 = time.perf_counter()
    try:
        n_frames = pipeline.run(max_frames)
        provenance = {
            "detector": _model_provenance(pipeline._detector),
            "pose": _model_provenance(pipeline._pose),
            "tracker": _model_provenance(pipeline._tracker),
        }
        n_alerts = pipeline.n_alerts
    finally:
        pipeline.close()
    elapsed = time.perf_counter() - t0

    seq_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    fps = float(record["fps"])
    # 1-based GT frame N is shown at t = (N - 1) / fps on the ReplaySource clock.
    # events_t generalizes the single-interval fields: multicam_fall scenarios 4 and
    # 23 stage two falls each (index record key `event_frames`).
    if record.get("events_t"):
        # wu_sahzu_emu: GT is authored in seconds (expert onset timestamps), so the
        # index record carries events_t directly instead of frame intervals.
        events_t = [[round(float(s), 4), round(float(e), 4)] for s, e in record["events_t"]]
    elif record.get("event_frames"):
        events_t = [
            [round((s - 1) / fps, 4), round((e - 1) / fps, 4)] for s, e in record["event_frames"]
        ]
    elif record.get("fall_start_frame") is not None:
        events_t = [
            [
                round((record["fall_start_frame"] - 1) / fps, 4),
                round((record["fall_end_frame"] - 1) / fps, 4),
            ]
        ]
    else:
        events_t = []
    meta = {
        "key": key,
        "video": record["video"],
        "location": record["location"],
        "fps": fps,
        "n_frames_processed": int(n_frames),
        "n_frames_container": record["n_frames_container"],
        "has_fall": record["has_fall"],
        "fall_start_frame": record.get("fall_start_frame"),
        "fall_end_frame": record.get("fall_end_frame"),
        "fall_start_t": events_t[0][0] if len(events_t) == 1 else None,
        "fall_end_t": events_t[0][1] if len(events_t) == 1 else None,
        "events_t": events_t,
        "lie_transition_start_frame": record.get("lie_transition_start_frame"),
        "n_rows": len(rows),
        "n_tracks": len({json.loads(r)["track"] for r in rows}),
        "n_rule_firings": int(n_alerts),
        "elapsed_s": round(elapsed, 1),
        "provenance": provenance,
    }
    # Seizure-dataset context rides along for per-type eval breakouts (wu_sahzu_emu:
    # convulsive PG vs focal P, clip-time offsets, EEG-vs-clinical onset latency).
    for extra in (
        "has_seizure",
        "seizure_type",
        "convulsive",
        "seizure_id",
        "cut_start_s",
        "eeg_onset_t",
        "clinical_onset_t",
        "video_upstream",
    ):
        if extra in record:
            meta[extra] = record[extra]
    meta_path = out_dir / "meta" / f"{key}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--only", default=None, help="substring of Location/video to run")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    registry: Registry = default_registry()
    dataset_root = registry.dataset_root(str(cfg["dataset"]))
    # Optional alternate index file within the registered dataset root (for example the
    # multicam all-cameras surface index_allcams.json). The registry still resolves
    # the root, so the licence-bucket enforcement is untouched; only the record list
    # changes. Path traversal out of annotations/ is refused.
    index_name = str(cfg.get("index", INDEX_FILENAME))
    if "/" in index_name or index_name != Path(index_name).name:
        print(f"index must be a bare filename, got {index_name!r}", file=sys.stderr)
        return EXIT_CONFIG
    index_path = dataset_root / "annotations" / index_name
    if not index_path.is_file():
        print(
            f"no {index_path}: run vividly.data.convert.le2i.write_index over the "
            f"dataset root first",
            file=sys.stderr,
        )
        return EXIT_CONFIG
    index = json.loads(index_path.read_text(encoding="utf-8"))
    records = select_videos(
        index, only=args.only, annotated_only=bool(cfg.get("annotated_only", True))
    )
    out_dir = REPO_ROOT / str(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        total = sum(int(r["n_frames_container"]) for r in records)
        for r in records:
            print(f"{_key(r)}  frames={r['n_frames_container']}  fall={r['has_fall']}")
        print(f"-- {len(records)} videos, {total} frames")
        return EXIT_OK

    done = skipped = 0
    metas: list[dict[str, Any]] = []
    for i, record in enumerate(records, 1):
        key = _key(record)
        seq_path = out_dir / "sequences" / f"{key}.jsonl"
        meta_path = out_dir / "meta" / f"{key}.json"
        if seq_path.exists() and meta_path.exists():
            skipped += 1
            metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
            continue
        meta = extract_one(record, cfg, dataset_root, out_dir, max_frames=args.max_frames)
        metas.append(meta)
        done += 1
        print(
            f"[{i}/{len(records)}] {key}: {meta['n_rows']} rows / "
            f"{meta['n_tracks']} tracks / {meta['n_frames_processed']} frames "
            f"in {meta['elapsed_s']}s",
            flush=True,
        )

    config_digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()
    summary = {
        "config": str(args.config),
        "config_sha256": config_digest,
        "dataset": cfg["dataset"],
        "git_commit": _git_commit(),
        "n_videos": len(records),
        "n_extracted": done,
        "n_skipped_existing": skipped,
        "n_fall": sum(1 for m in metas if m.get("has_fall")),
        "n_rows_total": sum(int(m.get("n_rows", 0)) for m in metas),
        "provenance": metas[-1]["provenance"] if metas else {},
        "partial": bool(args.only or args.max_frames),
    }
    (out_dir / "extraction.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=1))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
