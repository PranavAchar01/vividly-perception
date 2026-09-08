#!/usr/bin/env python3
"""Evidence packs for every alert of a run, plus the per-rule false-alarm report.

    # a run over an image sequence
    uv run python scripts/eval/extract_evidence.py --run-dir outputs/pipeline/<run> --images <frames dir> --fps 10
    # a run over a clip
    uv run python scripts/eval/extract_evidence.py --run-dir outputs/pipeline/replay_demo --clip tests/fixtures/sample_clip.mp4

Each alert becomes <out>/<alert_id>/frames/*.jpg + manifest.json (window, alert record, label,
provenance); <out>/index.json lists the packs and <out>/false_alarms.md is the per-rule
report from the dashboard's Agree / False-alarm labels (or, with --negative-control, every
firing counted as a false positive). Idempotent: a pack with a manifest is skipped unless
--force. Evidence packs exist for a person to review; nothing here trains anything.
Exit codes: 0 ok, 3 config error.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vividly.eval.evidence import (  # noqa: E402
    MANIFEST_NAME,
    EvidenceError,
    extract_from_clip,
    extract_from_files,
    false_alarm_report,
    pack_id,
    pack_manifest,
    render_false_alarm_report,
    window_for,
    write_pack,
)
from vividly.events import EventStore  # noqa: E402
from vividly.sources.image_sequence import list_frames  # noqa: E402

EXIT_OK = 0
EXIT_CONFIG = 3


def _alerts(run_dir: Path) -> list[dict[str, Any]]:
    db = run_dir / "alerts.sqlite"
    if not db.exists():
        raise EvidenceError(f"no alerts.sqlite in {run_dir}")
    store = EventStore(db)
    try:
        rows = [a.model_dump(mode="json") for a in store.alerts(limit=1_000_000)]
    finally:
        store.close()
    rows.sort(key=lambda r: (float(r["onset_ts"]), str(r["rule_id"]), int(r["track_id"])))
    return rows


def extract_run(
    *,
    run_dir: Path,
    out_dir: Path,
    source: dict[str, Any],
    pre_s: float,
    post_s: float,
    force: bool,
    run_label: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """All packs for one run. Returns (index rows, alert rows)."""
    alerts = _alerts(run_dir)
    extracted = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    files: Sequence[Path] | None = None
    clip: Any = None
    fps: float
    if "images" in source:
        files = list_frames(source["images"], source.get("pattern", "*.jpg"))
        fps = float(source["fps"])
    else:
        from vividly.sources.replay import ReplaySource

        clip = ReplaySource(source["clip"])
        fps = float(clip.open().fps)
    index_rows: list[dict[str, Any]] = []
    try:
        for alert in alerts:
            pid = pack_id(alert)
            pack_dir = out_dir / pid
            if (pack_dir / MANIFEST_NAME).exists() and not force:
                index_rows.append(
                    {
                        "pack_id": pid,
                        "alert_id": alert["alert_id"],
                        "rule_id": alert["rule_id"],
                        "status": "cached",
                        "dir": str(pack_dir),
                    }
                )
                continue
            window = window_for(alert, pre_s=pre_s, post_s=post_s)
            if files is not None:
                frames = extract_from_files(files, fps, window, pack_dir)
            else:
                frames = extract_from_clip(clip, window, pack_dir)
            manifest = pack_manifest(
                alert,
                window,
                frames,
                fps=fps,
                source=source,
                run={"run_dir": str(run_dir), **run_label},
                extracted_utc=extracted,
            )
            write_pack(pack_dir, manifest)
            index_rows.append(
                {
                    "pack_id": pid,
                    "alert_id": alert["alert_id"],
                    "rule_id": alert["rule_id"],
                    "status": "written",
                    "n_frames": len(frames),
                    "dir": str(pack_dir),
                }
            )
    finally:
        if clip is not None:
            clip.close()
    return index_rows, alerts


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--run-dir", type=Path, help="a scripts/run_pipeline.py output directory (alerts.sqlite)"
    )
    ap.add_argument("--images", type=Path, help="the image sequence the run played")
    ap.add_argument("--fps", type=float, help="that sequence's declared frame rate")
    ap.add_argument("--pattern", default="*.jpg")
    ap.add_argument("--clip", type=Path, help="the clip the run played")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="pack root (default <run-dir>/evidence)",
    )
    ap.add_argument("--pre-s", type=float, default=10.0)
    ap.add_argument("--post-s", type=float, default=5.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--negative-control",
        action="store_true",
        help="count every firing as a false positive (footage with no events)",
    )
    args = ap.parse_args(argv)

    try:
        if args.run_dir is None:
            raise EvidenceError("give --run-dir with --images/--fps or --clip")
        if (args.images is None) == (args.clip is None):
            raise EvidenceError("give exactly one of --images (with --fps) or --clip")
        if args.images is not None and (args.fps is None or args.fps <= 0):
            raise EvidenceError(
                "--images needs a positive --fps (no frame directory records its rate)"
            )
        source = (
            {"images": str(args.images), "fps": float(args.fps), "pattern": args.pattern}
            if args.images is not None
            else {"clip": str(args.clip)}
        )
        out_root = args.out or args.run_dir / "evidence"
        all_index, all_alerts = extract_run(
            run_dir=args.run_dir,
            out_dir=out_root,
            source=source,
            pre_s=args.pre_s,
            post_s=args.post_s,
            force=args.force,
            run_label={},
        )
        report = false_alarm_report(all_alerts, all_false_positives=args.negative_control)
        title = f"False-alarm report: {args.run_dir.name}"
    except (EvidenceError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "index.json").write_text(
        json.dumps({"packs": all_index, "report": report}, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    (out_root / "false_alarms.md").write_text(
        render_false_alarm_report(report, title=title), encoding="utf-8"
    )
    (out_root / "false_alarms.json").write_text(
        json.dumps(report, indent=1, sort_keys=True), encoding="utf-8"
    )
    written = sum(1 for r in all_index if r["status"] == "written")
    print(
        f"{len(all_index)} pack(s) ({written} written, {len(all_index) - written} cached) under {out_root}"
    )
    for rid in report["rule_ids"]:
        r = report["per_rule"][rid]
        print(
            f"  {rid:<24} firings {r['firings']:>4}  labelled {r['labelled']:>4}  false alarms {r['false_alarm']:>4}"
        )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
