"""Deep-metrics instrument over an event-verifier eval dir (interim prototype reporting).

Reads the report.json + windows.jsonl an eval run leaves behind (see
scripts/events/eval_event_verifier.py) plus the extraction metas, and computes the
full statistics battery the aggregate report doesn't carry:

  * window-level: ROC AUC, PR AUC (average precision), a threshold sweep with
    sensitivity/specificity/precision/F1/Youden J, best-F1 and best-J points, and
    score distributions for the positive and negative window populations;
  * video-level: catch rates per class (convulsive_PG / focal_P / non_seizure for
    the EMU set; fall / adl_lie_transition / adl_other for the fall sets);
  * event-level: per labelled event, caught-or-missed at each threshold and the
    detection latency (first firing window's t_end minus the event onset);
  * negative behaviour: false-positive windows per recorded hour on event-free
    videos, and how many distinct videos ever cross each threshold;
  * coverage: videos in the extraction that produced zero windows, the fraction
    of footage seconds any window covered, and events no window overlapped, which
    is the accounting that proves every video and every event was tested (events
    in zero-window videos count as missed in every event denominator);
  * per-location (patient / camera) rollup.

    uv run python scripts/events/metrics_deep.py \
        --eval-dir outputs/verifier/convulsion_v2_real/eval_wusahzu_testsplit

Every number inherits the eval report's prototype disclosure and model sha256, which
POLICY.md, rule 10 requires; the output repeats both. Nothing here is clinical detection
performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
LATENCY_THRESHOLDS = (0.5, 0.8)


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def roc_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pr_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    """Average precision: sum of precision at each positive, in score-desc order."""
    n_pos = int((y == 1).sum())
    if n_pos == 0 or n_pos == len(y):
        return None
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, len(y_sorted) + 1)
    return float(precision[y_sorted == 1].sum() / n_pos)


def _dist(v: np.ndarray) -> dict[str, float] | None:
    if len(v) == 0:
        return None
    return {
        "n": len(v),
        "mean": round(float(v.mean()), 4),
        "median": round(float(np.median(v)), 4),
        "p10": round(float(np.percentile(v, 10)), 4),
        "p90": round(float(np.percentile(v, 90)), 4),
        "max": round(float(v.max()), 4),
    }


def sweep(y: np.ndarray, s: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    for thr in THRESHOLDS:
        det = s >= thr
        tp = int((det & (y == 1)).sum())
        fp = int((det & (y == 0)).sum())
        fn = n_pos - tp
        tn = n_neg - fp
        sens = tp / n_pos if n_pos else None
        spec = tn / n_neg if n_neg else None
        prec = tp / (tp + fp) if (tp + fp) else None
        f1 = 2 * prec * sens / (prec + sens) if prec and sens else None
        rows.append(
            {
                "threshold": thr,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "sensitivity": None if sens is None else round(sens, 4),
                "specificity": None if spec is None else round(spec, 4),
                "precision": None if prec is None else round(prec, 4),
                "f1": None if f1 is None else round(f1, 4),
                "youden_j": None if sens is None or spec is None else round(sens + spec - 1, 4),
            }
        )
    return rows


def video_class(meta: dict[str, Any]) -> str:
    if "has_seizure" in meta:
        if meta.get("convulsive"):
            return "convulsive_PG"
        if meta.get("has_seizure"):
            return "focal_P"
        return "non_seizure"
    if meta.get("events_t") or meta.get("fall_start_t") is not None:
        return "fall"
    if meta.get("lie_transition_start_frame") is not None:
        return "adl_lie_transition"
    return "adl_other"


def events_of(meta: dict[str, Any]) -> list[tuple[float, float]]:
    ev = meta.get("events_t")
    if ev:
        return [(float(a), float(b)) for a, b in ev]
    if meta.get("fall_start_t") is not None:
        return [(float(meta["fall_start_t"]), float(meta["fall_end_t"]))]
    return []


def analyze(eval_dir: Path) -> dict[str, Any]:
    report = json.loads((eval_dir / "report.json").read_text(encoding="utf-8"))
    win_path = eval_dir / "windows.jsonl"
    if not win_path.exists():
        raise SystemExit(
            f"{win_path} missing; re-run the eval, which writes windows.jsonl beside report.json"
        )
    windows = [
        json.loads(line) for line in win_path.read_text(encoding="utf-8").splitlines() if line
    ]
    window_s = float(report["config"]["window"]["window_s"])

    metas: dict[str, dict[str, Any]] = {}
    for source in report["config"]["sources"]:
        meta_dir = _resolve(source["dir"]) / "meta"
        src_locations = set(source.get("locations") or [])
        for p in sorted(meta_dir.glob("*.json")):
            m = json.loads(p.read_text(encoding="utf-8"))
            if src_locations and m.get("location") not in src_locations:
                continue
            metas[m["key"]] = m

    y = np.array([w["label"] for w in windows], dtype=np.int64)
    s = np.array([w["score"] for w in windows], dtype=np.float64)

    # ---- video rollup ----------------------------------------------------------
    by_video: dict[str, list[dict[str, Any]]] = {}
    for w in windows:
        by_video.setdefault(w["key"], []).append(w)
    video_rows = []
    for key, ws in sorted(by_video.items()):
        meta = metas[key]
        video_rows.append(
            {
                "key": key,
                "class": video_class(meta),
                "location": meta.get("location"),
                "max_score": round(max(w["score"] for w in ws), 4),
                "n_windows": len(ws),
                "duration_s": round(meta["n_frames_container"] / meta["fps"], 1),
            }
        )

    classes = sorted({r["class"] for r in video_rows})
    per_class: dict[str, Any] = {}
    for cls in classes:
        rows = [r for r in video_rows if r["class"] == cls]
        scores = np.array([r["max_score"] for r in rows])
        per_class[cls] = {
            "n_videos": len(rows),
            "max_score_dist": _dist(scores),
            "videos_over_thr": {str(t): int((scores >= t).sum()) for t in THRESHOLDS},
        }

    # ---- event-level: caught + latency ----------------------------------------
    # Iterate the metas, not the scored videos: an event in a video whose
    # extraction produced zero windows must count as missed in every denominator,
    # not silently vanish from it (the hole that voided the first v4 run).
    event_rows = []
    for key, meta in sorted(metas.items()):
        ws = by_video.get(key, [])
        for idx, (e0, e1) in enumerate(events_of(meta)):
            row: dict[str, Any] = {
                "key": key,
                "event": idx,
                "class": video_class(meta),
                "onset_s": round(e0, 2),
                "end_s": round(e1, 2),
                "video_has_windows": bool(ws),
            }
            # windows overlapping the event interval at all
            overlapping = [w for w in ws if w["t_end"] >= e0 and w["t_end"] - window_s <= e1]
            for thr in LATENCY_THRESHOLDS:
                firing = sorted(
                    w["t_end"] for w in overlapping if w["score"] >= thr and w["t_end"] >= e0
                )
                row[f"caught@{thr}"] = bool(firing)
                row[f"latency_s@{thr}"] = round(firing[0] - e0, 1) if firing else None
            row["event_max_score"] = (
                round(max(w["score"] for w in overlapping), 4) if overlapping else None
            )
            event_rows.append(row)

    latency_summary = {}
    for thr in LATENCY_THRESHOLDS:
        lat = np.array(
            [r[f"latency_s@{thr}"] for r in event_rows if r[f"latency_s@{thr}"] is not None]
        )
        latency_summary[str(thr)] = {
            "events_caught": int(sum(1 for r in event_rows if r[f"caught@{thr}"])),
            "events_total": len(event_rows),
            "latency_s": _dist(lat),
        }

    # ---- negative behaviour ----------------------------------------------------
    neg_rows = [
        r for r in video_rows if r["class"] in ("non_seizure", "adl_lie_transition", "adl_other")
    ]
    neg_hours = sum(r["duration_s"] for r in neg_rows) / 3600.0
    neg_keys = {r["key"] for r in neg_rows}
    neg_windows = [w for w in windows if w["key"] in neg_keys]
    negative_behaviour = {
        "n_videos": len(neg_rows),
        "recorded_hours": round(neg_hours, 3),
        "fp_windows_per_hour": {
            str(t): (
                round(sum(1 for w in neg_windows if w["score"] >= t) / neg_hours, 2)
                if neg_hours
                else None
            )
            for t in THRESHOLDS
        },
        "videos_ever_over_thr": {
            str(t): sum(1 for r in neg_rows if r["max_score"] >= t) for t in THRESHOLDS
        },
    }

    # ---- per-location ----------------------------------------------------------
    per_location: dict[str, Any] = {}
    for loc in sorted({r["location"] for r in video_rows if r["location"]}):
        rows = [r for r in video_rows if r["location"] == loc]
        per_location[loc] = {
            "n_videos": len(rows),
            "classes": sorted({r["class"] for r in rows}),
            "max_scores": {r["key"]: r["max_score"] for r in rows},
        }

    # ---- coverage: is every video and every event actually being judged? -------
    # (recorded requirement of 2026-08-28: every video and every event is tested.) Reports the
    # holes the sections above cannot see: videos whose extraction produced zero
    # windows, footage seconds no window ever covered, and events with no
    # overlapping window at all. Coverage = union of the scored windows'
    # [t_end - window_s, t_end] spans clipped to the container duration.
    per_video_cov: dict[str, float | None] = {}
    tot_s = cov_s = 0.0
    for key, meta in sorted(metas.items()):
        dur = meta["n_frames_container"] / meta["fps"]
        ivs = sorted(
            (max(0.0, w["t_end"] - window_s), min(dur, w["t_end"])) for w in by_video.get(key, [])
        )
        covered = 0.0
        end: float | None = None
        for s0, e0 in ivs:
            if e0 <= s0:
                continue
            if end is None or s0 > end:
                covered += e0 - s0
                end = e0
            elif e0 > end:
                covered += e0 - end
                end = e0
        tot_s += dur
        cov_s += covered
        per_video_cov[key] = round(covered / dur, 4) if dur else None
    zero_window = sorted(set(metas) - set(by_video))
    coverage = {
        "videos_in_extraction": len(metas),
        "videos_with_windows": len(set(metas) & set(by_video)),
        "videos_zero_windows": zero_window,
        "zero_window_hours": round(
            sum(metas[k]["n_frames_container"] / metas[k]["fps"] for k in zero_window) / 3600.0, 3
        ),
        "footage_hours": round(tot_s / 3600.0, 3),
        "footage_covered_frac": round(cov_s / tot_s, 4) if tot_s else None,
        "events_total": len(event_rows),
        "events_no_overlapping_window": [
            {"key": r["key"], "event": r["event"]}
            for r in event_rows
            if r["event_max_score"] is None
        ],
        "per_video_covered_frac": per_video_cov,
    }

    sweep_rows = sweep(y, s)
    best_f1 = max(
        (r for r in sweep_rows if r["f1"] is not None), key=lambda r: r["f1"], default=None
    )
    best_j = max(
        (r for r in sweep_rows if r["youden_j"] is not None),
        key=lambda r: r["youden_j"],
        default=None,
    )

    return {
        "eval_dir": str(eval_dir),
        "task": report["task"],
        "model": report["model"],
        "vividly_weight_class": report["vividly_weight_class"],
        "disclosure": report["disclosure"],
        "window": {
            "n_windows": len(windows),
            "n_pos": int((y == 1).sum()),
            "n_neg": int((y == 0).sum()),
            "roc_auc": None if (a := roc_auc(y, s)) is None else round(a, 5),
            "pr_auc": None if (a := pr_auc(y, s)) is None else round(a, 5),
            "pos_score_dist": _dist(s[y == 1]),
            "neg_score_dist": _dist(s[y == 0]),
            "threshold_sweep": sweep_rows,
            "best_f1_point": best_f1,
            "best_youden_point": best_j,
        },
        "video": {
            "n_videos": len(video_rows),
            "per_class": per_class,
            "videos": video_rows,
        },
        "events": {"rows": event_rows, "latency_summary": latency_summary},
        "negatives": negative_behaviour,
        "coverage": coverage,
        "per_location": per_location,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-dir", required=True, nargs="+", type=Path)
    args = ap.parse_args(argv)
    for eval_dir in args.eval_dir:
        out = analyze(_resolve(eval_dir))
        dest = _resolve(eval_dir) / "metrics_deep.json"
        dest.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        w = out["window"]
        print(
            f"{eval_dir}: ROC {w['roc_auc']} PR {w['pr_auc']} "
            f"({w['n_pos']}+/{w['n_neg']}- windows) -> {dest}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
