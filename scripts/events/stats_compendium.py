"""Regenerate research/stats-compendium.md from the run artefacts on disk.

Walks two output trees and emits one document. From `outputs/verifier`: the complete
statistics for every trained event verifier (training data counts, val numbers,
checkpoint and onnx hashes) and every eval re-run of each (window ROC and PR,
threshold sweeps, per-class video rollups, per-event catches and detection latency,
negative-footage false-alarm rates, WU-SAHZU class breakouts). From `outputs/eval`:
the detector and pose instrument runs on the COCO person-500 subset. The eyes, joints
and reid models of record are not walked; the lane pages under `final_models/` are
their record.

    uv run python scripts/events/stats_compendium.py

Recorded requirement of 2026-08-26: for every event verifier present, the complete
data statistics are present for it and for each of its re-runs. The document is
regenerated in full on every invocation, so after any new eval or retrain, one command
brings the compendium current. Sources of truth stay the per-run JSON artefacts, so
wherever those run directories are available the page is regenerated rather than edited
by hand. Event-verifier numbers travel with the checkpoint and graph hashes required by
POLICY.md, rule 10, the instrument rows carry the model provenance the eval runner
recorded, and the prototype disclosures travel in the header.

The run artefacts live under `outputs/`, which is not tracked, so a fresh clone holds
none of them. When no run artefact is found the script writes nothing and exits
non-zero, because a header-only document would replace the committed record with an
empty one.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "research" / "stats-compendium.md"
SWEEP_THRESHOLDS = (0.3, 0.5, 0.8, 0.9)


def _load(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _f(v: Any, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def verifier_sections(lines: list[str]) -> None:
    root = REPO_ROOT / "outputs" / "verifier"
    if not root.is_dir():
        return
    for run_dir in sorted(p for p in root.iterdir() if (p / "report.json").exists()):
        tr = _load(run_dir / "report.json")
        if tr is None or "val_auc_best" not in tr:
            continue
        onnx8 = tr["onnx"]["sha256"][:8]
        lines += [
            f"## {run_dir.name}, task `{tr['task']}`, onnx `{onnx8}`",
            "",
            f"- weight class **{tr['vividly_weight_class']}**, git `{tr['git_commit']}`, "
            f"checkpoint `{tr['checkpoint']['sha256'][:8]}`",
            f"- training data: {tr['n_windows']} windows "
            f"({tr['n_positive']} pos / {tr['n_negative']} neg) from {tr['n_videos']} videos",
            f"- best in-training val AUC: **{_f(tr['val_auc_best'], 5)}** "
            "(same-domain holdout split, not a held-out-test number)",
            "",
        ]
        for eval_dir in sorted(
            p for p in run_dir.iterdir() if p.is_dir() and (p / "report.json").exists()
        ):
            ev = _load(eval_dir / "report.json")
            if ev is None:
                continue
            lines.append(f"### eval re-run: `{eval_dir.name}`")
            for res in ev.get("results", []):
                lines.append(
                    f"- source `{res['dir']}`: **window AUC {_f(res['window_auc'], 5)}** / "
                    f"video AUC {_f(res['video_auc'], 5)} "
                    f"({res['n_windows']} windows, {res['n_window_pos']} pos; "
                    f"{res['n_videos']} videos, {res['n_video_pos']} with events)"
                )
                ops = {op["threshold"]: op for op in res.get("video_operating_points", [])}
                for thr in (0.5, 0.8, 0.9):
                    if thr in ops:
                        o = ops[thr]
                        lines.append(
                            f"  - video ops @{thr}: TP {o['video_tp']} FN {o['video_fn']} "
                            f"FP {o['video_fp']} TN {o['video_tn']}"
                            + (
                                f" (lie-transition FP {o['lie_transition_fp']}/{o['lie_transition_n']})"
                                if o.get("lie_transition_n")
                                else ""
                            )
                        )
            md = _load(eval_dir / "metrics_deep.json")
            if md is not None:
                w = md["window"]
                lines.append(
                    f"- deep: **ROC {_f(w['roc_auc'], 5)} / PR {_f(w['pr_auc'], 5)}** "
                    f"({w['n_pos']}+/{w['n_neg']}- windows); pos score median "
                    f"{_f((w['pos_score_dist'] or {}).get('median'))} vs neg "
                    f"{_f((w['neg_score_dist'] or {}).get('median'))}"
                )
                rows = {r["threshold"]: r for r in w["threshold_sweep"]}
                cells = [
                    f"@{t}: sens {_f(rows[t]['sensitivity'])} spec {_f(rows[t]['specificity'])} "
                    f"prec {_f(rows[t]['precision'])} F1 {_f(rows[t]['f1'])}"
                    for t in SWEEP_THRESHOLDS
                    if t in rows
                ]
                lines.append("  - sweep: " + "; ".join(cells))
                if w.get("best_f1_point"):
                    b = w["best_f1_point"]
                    lines.append(
                        f"  - best F1 {_f(b['f1'])} at thr {b['threshold']} "
                        f"(sens {_f(b['sensitivity'])}, spec {_f(b['specificity'])})"
                    )
                for thr, ls in sorted(md["events"]["latency_summary"].items()):
                    lat = ls.get("latency_s") or {}
                    lines.append(
                        f"  - events @{thr}: {ls['events_caught']}/{ls['events_total']} caught"
                        + (
                            f", latency median {_f(lat.get('median'), 1)} s "
                            f"(mean {_f(lat.get('mean'), 1)}, max {_f(lat.get('max'), 1)})"
                            if lat
                            else ""
                        )
                    )
                neg = md["negatives"]
                if neg["n_videos"]:
                    fp = neg["fp_windows_per_hour"]
                    lines.append(
                        f"  - negatives: {neg['n_videos']} videos / {neg['recorded_hours']} h; "
                        f"FP windows/hr @0.5 {_f(fp.get('0.5'), 1)}, @0.8 {_f(fp.get('0.8'), 1)}"
                        f", @0.9 {_f(fp.get('0.9'), 1)}"
                    )
                classes = md["video"]["per_class"]
                cells = [
                    f"{cls}: {v['videos_over_thr']['0.5']}/{v['n_videos']} over 0.5"
                    for cls, v in sorted(classes.items())
                ]
                lines.append("  - per-class videos: " + "; ".join(cells))
                cov = md.get("coverage")
                if cov is not None:
                    dark = cov["videos_zero_windows"]
                    lines.append(
                        f"  - coverage: {cov['videos_with_windows']}/{cov['videos_in_extraction']} "
                        f"videos produced windows; footage covered "
                        f"{_f(cov['footage_covered_frac'])} of {cov['footage_hours']} h; "
                        f"{len(cov['events_no_overlapping_window'])} events with no overlapping "
                        f"window"
                        + (
                            f"; zero-window videos: {len(dark)} ({cov['zero_window_hours']} h)"
                            if dark
                            else ""
                        )
                    )
            st = _load(eval_dir / "decision_layer_study.json")
            if st is not None:
                singles = {
                    r["rule"]: r for r in st.get("rules", []) if r["rule"].startswith("single:")
                }
                cells = [
                    f"@{r['rule'].split(':')[1]}: eps/h {_f(r['fp_episodes_per_h'], 1)}"
                    + (
                        f", cam/h {_f(r['fp_episodes_per_h_camera'], 1)}"
                        if "fp_episodes_per_h_camera" in r
                        else ""
                    )
                    + f", in-alarm {_f(100 * r['neg_time_in_alarm'], 1)}%"
                    for r in singles.values()
                ]
                if cells:
                    lines.append("  - alarm experience (negatives): " + "; ".join(cells))
            bo = _load(eval_dir / "breakout.json")
            if bo is not None:
                lines.append(
                    "  - WU-SAHZU breakout: see `"
                    + str(eval_dir.relative_to(REPO_ROOT))
                    + "/breakout.json`"
                )
            lines.append("")
        lines.append("")


def detector_sections(lines: list[str]) -> None:
    root = REPO_ROOT / "outputs" / "eval"
    if not root.is_dir():
        return
    runs = sorted(p for p in root.iterdir() if (p / "results.json").exists())
    coco = [p for p in runs if p.name.startswith(("coco500_eyes_", "cpu_proof_coco500"))]
    # pose_ladder_ runs come from scripts/pose/eval_pose_ladder.py and belong to the
    # joints lane, so they get their own heading instead of an eyes one.
    ladder = [p for p in runs if p.name.startswith("pose_ladder_")]
    for heading, group in (
        ("## Detector (eyes): COCO person-500 perspective controls", coco),
        ("## Pose (joints): COCO person-500 ladder runs", ladder),
    ):
        if not group:
            continue
        lines += [heading, ""]
        for p in group:
            r = _load(p / "results.json")
            if r is None:
                continue
            for row in r.get("rows", []):
                lines.append(
                    f"- `{p.name}` [{row.get('condition', 'clean')}]: "
                    f"det AP {row['det_AP']:.3f}"
                    + (f", kpt AP {row['kpt_AP']:.3f}" if row.get("kpt_AP") is not None else "")
                    + f"; models {row.get('det_provenance', 'not recorded')}"
                    + f" / {row.get('pose_provenance', 'not recorded')}"
                )
        lines.append("")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help=f"destination document (default {OUT.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args(argv)

    body: list[str] = []
    verifier_sections(body)
    detector_sections(body)
    if not body:
        print(
            "no run artefacts found under outputs/verifier or outputs/eval, so nothing was "
            f"written and {args.out} keeps its committed content",
            file=sys.stderr,
        )
        return 1

    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "# Vividly model statistics compendium",
        "",
        "This is the numeric record of the event-verifier and detector evaluation runs",
        f"that were present when it was generated, {stamp}, produced by",
        "`scripts/events/stats_compendium.py` from the run artefacts under `outputs/`,",
        "which are not in version control. Regenerating it needs those run directories, so",
        "a clone of this repository cannot reproduce it; where those directories are",
        "available, regenerate this page rather than editing it by hand. The lane pages",
        "under `final_models/` are the record for the models of record.",
        "",
        "All event-verifier weights are interim prototypes under the carve-out in",
        "POLICY.md, rule 4: internal demonstration only, never quoted as clinical",
        "detection performance. Event-verifier numbers travel with their checkpoint and",
        "graph hashes (POLICY.md, rule 10), and the instrument rows carry the model",
        "provenance the eval runner recorded. Narrative, verdicts and caveats live in",
        "`research/event-verifier-v1-record.md`; this file is the numeric record.",
        "",
        "# Event verifier models",
        "",
    ]
    lines += body
    args.out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
