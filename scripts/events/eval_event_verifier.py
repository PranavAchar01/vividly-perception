"""Evaluate a trained event verifier on a held-out extraction (the acted-event test sets
held under POLICY.md, rule 4).

Scores every window of an extraction directory with the exported ONNX (onnxruntime, no
torch needed, runs in the main Mac environment) and reports both granularities:

  * window-level: AUC + operating points over labelled windows;
  * video-level: max window score per video against the video's event label, which is
    the product question of whether any window of this clip would have confirmed. For
    ur_fall, the ADL subset with a controlled lying-down transition is broken out
    separately: it is the exact confounder class (label-0 frames that are not falls),
    and a verifier that fires on it has learned to detect going down rather than
    falling.

    uv run python scripts/events/eval_event_verifier.py --config configs/events/eval_fall_urfall_v1.yaml

The report carries the prototype disclosure, the model sha256 that POLICY.md, rule 10
requires, and the window spec, which must equal the training spec because the graph's
input geometry pins n_steps and the time scale pins window_s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vividly.verifier.features import (  # noqa: E402
    WindowSpec,
    channel_names,
    windows_from_extraction,
)

DISCLOSURE = (
    "Interim prototype event verifier, admitted under the acted-event carve-out of "
    "POLICY.md, rule 4: internal demonstration only; no number here is clinical "
    "detection performance. The verifier and this instrument measure a motion "
    "signature on public acted-event footage."
)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def roc_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for key in ("task", "window", "sources", "model", "out_dir"):
        if key not in cfg:
            raise SystemExit(f"config {args.config} is missing `{key}`")

    import onnxruntime as ort

    onnx_path = _resolve(cfg["model"]["onnx"])
    onnx_sha = sha256_file(onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # Trained models may carry a temperature (fitted on their own train-side val,
    # scripts/events/train_event_verifier.py); scoring applies it so eval scores are
    # the deployed scores. The rescaling is monotonic, so AUC and ranking metrics are
    # unaffected.
    temperature = 1.0
    train_report_path = onnx_path.parent / "report.json"
    if train_report_path.exists():
        try:
            train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
            temperature = float((train_report.get("calibration") or {}).get("temperature", 1.0))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            # Falling back to 1.0 would rescale every published score and every
            # threshold of a calibrated model, and the report written below would
            # record temperature 1.0 as though the model were uncalibrated.
            # src/vividly/models/event_verifier.py refuses in the same situation, so
            # the two paths cannot disagree about the score scale.
            raise SystemExit(
                f"{train_report_path} exists but cannot be read for the calibration "
                f"temperature ({exc}); refusing to score on an assumed scale"
            ) from exc

    spec = WindowSpec(**cfg["window"])
    per_source: list[dict[str, Any]] = []
    # Every scored window (key, track, t_end, label, score) is dumped alongside the
    # report so downstream instruments (scripts/events/metrics_deep.py) can compute
    # threshold sweeps and per-event detection latency without re-running inference.
    window_rows: list[dict[str, Any]] = []
    for source in cfg["sources"]:
        src_dir = _resolve(source["dir"])
        event_keys = tuple(source.get("event_keys", ("fall_start_t", "fall_end_t")))
        locations = source.get("locations")
        wins = windows_from_extraction(
            src_dir,
            spec,
            event_keys=event_keys,
            locations=tuple(locations) if locations else None,
        )
        if not wins:
            raise SystemExit(f"no windows from {src_dir}")
        X = np.stack([w.x for w in wins]).astype(np.float32)
        logits = session.run(None, {"windows": X})[0].reshape(-1)
        scores = 1.0 / (1.0 + np.exp(-logits / temperature))
        y = np.array([w.label for w in wins], dtype=np.int64)

        # Per-video: max window score vs the video's own event label from its meta.
        metas = {
            m["key"]: m
            for m in (
                json.loads(p.read_text(encoding="utf-8"))
                for p in sorted((src_dir / "meta").glob("*.json"))
            )
        }
        by_video: dict[str, dict[str, Any]] = {}
        for w, s in zip(wins, scores, strict=True):
            rec = by_video.setdefault(
                w.key,
                {
                    # events_t first: multi-event videos (multicam chute04/chute23)
                    # null the single-interval fields.
                    "has_event": bool(metas[w.key].get("events_t"))
                    or metas[w.key].get(event_keys[0]) is not None,
                    "lie_transition": metas[w.key].get("lie_transition_start_frame") is not None,
                    "max_score": 0.0,
                    "n_windows": 0,
                },
            )
            rec["max_score"] = max(rec["max_score"], float(s))
            rec["n_windows"] += 1
        video_rows = [
            {"key": k, **v, "max_score": round(v["max_score"], 4)}
            for k, v in sorted(by_video.items())
        ]
        vid_y = np.array([1 if r["has_event"] else 0 for r in video_rows])
        vid_s = np.array([r["max_score"] for r in video_rows])
        thresholds = [0.5, 0.8, 0.9]
        video_ops = []
        for thr in thresholds:
            det = vid_s >= thr
            tp = int((det & (vid_y == 1)).sum())
            fn = int((~det & (vid_y == 1)).sum())
            fp = int((det & (vid_y == 0)).sum())
            tn = int((~det & (vid_y == 0)).sum())
            lie_fp = sum(1 for r in video_rows if r["lie_transition"] and r["max_score"] >= thr)
            lie_n = sum(1 for r in video_rows if r["lie_transition"])
            video_ops.append(
                {
                    "threshold": thr,
                    "video_tp": tp,
                    "video_fn": fn,
                    "video_fp": fp,
                    "video_tn": tn,
                    "lie_transition_fp": lie_fp,
                    "lie_transition_n": lie_n,
                }
            )
        window_auc = roc_auc(y, scores)
        video_auc = roc_auc(vid_y, vid_s)
        window_rows.extend(
            {
                "source": source["dir"],
                "key": w.key,
                "track_id": w.track_id,
                "t_end": round(float(w.t_end), 3),
                "label": int(w.label),
                # full precision: rounding collapses the sigmoid's saturated tail
                # into ties and visibly shifts downstream AUC recomputation
                "score": float(s),
            }
            for w, s in zip(wins, scores, strict=True)
        )
        per_source.append(
            {
                "dir": source["dir"],
                "n_windows": len(wins),
                "n_window_pos": int((y == 1).sum()),
                "n_videos": len(video_rows),
                "n_video_pos": int(vid_y.sum()),
                "window_auc": None if window_auc is None else round(window_auc, 5),
                "video_auc": None if video_auc is None else round(video_auc, 5),
                "video_operating_points": video_ops,
                "videos": video_rows,
            }
        )
        print(
            f"{source['dir']}: window AUC "
            f"{per_source[-1]['window_auc']} / video AUC {per_source[-1]['video_auc']} "
            f"({len(video_rows)} videos)",
            flush=True,
        )

    out_dir = _resolve(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "task": cfg["task"],
        "vividly_weight_class": "prototype",
        "disclosure": DISCLOSURE,
        "git_commit": _git_commit(),
        "config": cfg,
        # From the spec, never the bare CHANNELS tuple: a spec with extra channel
        # blocks on (scene/rhythm/limb/kp) must name every column it fed the model.
        "channels": list(channel_names(spec)),
        "model": {"onnx": str(onnx_path), "sha256": onnx_sha, "temperature": temperature},
        "results": per_source,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (out_dir / "windows.jsonl").open("w", encoding="utf-8") as fh:
        for row in window_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out_dir / "report.json"), "model_sha256": onnx_sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
