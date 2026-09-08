#!/usr/bin/env python3
r"""Deployed-path pose ladder with a per-image bootstrap CI (2026-09-05).

    uv run python scripts/pose/eval_pose_ladder.py \
        --config configs/eval/coco500_pose_body7_e48f03d0.yaml \
        --config configs/eval/coco500_pose_v2_6ba96285_noflip.yaml \
        --config configs/eval/coco500_pose_v2_6ba96285_flip.yaml \
        --out-md outputs/pose/coco500_v2pose.table.md

That is the reproduction command recorded in TRAINING.md.

The rendered table for the run of record is carried in
`research/pose-deployed-path-cpu-proof.md`.

Why not scripts/eval/eval_runner.py. The stock runner produces exactly the rows this
ladder needs and its configs are interchangeable with these; running any of them through
it returns the same AP. What it does not produce is an interval. Three rows that differ
by 0.01 mean nothing without one, and the two v2 rows are expected to differ by less than
that. So this runner keeps the per-image COCOeval intermediates the stock one discards and
resamples over them.

What the interval is. A per-image bootstrap: resample the 500 evaluated images with
replacement `--bootstrap` times (default 1000), recompute AP over each resample, and
report the 2.5/97.5 percentiles. Images are the independent unit here; detections are not
(they are matched within an image, and a resample over detections would break the
matching). It measures one thing: how much of a row's AP is the particular 500 images.
It does not cover model seed variance, detector variance, or domain shift, and two rows'
intervals overlapping is not a significance test, because the rows share images and share
a detector, so their errors are correlated and the honest read is the paired delta, which
this also reports.

How the resample is computed. COCOeval splits into `evaluate()` (per image x category x
area: match detections to ground truth) and `accumulate()` (sort all detections by score,
sweep the PR curve). Only the second depends on which images are in the set, so
`evaluate()` runs once and each resample re-runs a re-implementation of `accumulate` over
a resampled index list. `pycocotools`' own accumulate cannot be reused: it derives its
image list through `set()`, which silently drops the duplicates that are the bootstrap.
The re-implementation is checked against the real thing on every run: resampling the
identity index list must reproduce `COCOeval.stats[0]` to 1e-9, and the runner aborts if
it does not.

The gate. A config may carry a `gate:` block naming a metric, an expected value and a
tolerance. It is checked the moment that row finishes, and a miss stops the ladder before
the next config is built. Row 1 of this ladder is a reproduction of an already-recorded
number, and rows 2 and 3 are only meaningful if it lands.

Provenance. Every row records the pose ONNX sha256, the detector that produced the boxes
with its own declared digest, the subset fingerprint and the host CPU. The full host block,
including the onnxruntime version and its execution providers, is recorded once per run in
`results.json`, on a completed ladder and on one the gate stops alike. A local graph is
hashed when it is resolved and a run refuses to start when the bytes disagree with the
declared digest; a graph named by URL is fetched and cached by rtmlib, so its digest stays a
declared value. Rows are labelled as eval-smoke provenance, not a Vividly accuracy claim,
wherever they are printed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXIT_OK = 0
EXIT_GATE = 5

# COCOeval.stats[0] is AP averaged over IoU thresholds at area='all' and the last maxDets
# entry (100 for bbox, 20 for keypoints). Named once; the bootstrap has to use the same
# cell or its interval would describe a different statistic than the point estimate.
AREA_ALL = 0


# ----------------------------------------------------------------- bootstrap machinery
def _accumulate_ap(coco_eval: Any, img_positions: np.ndarray) -> float:
    """AP over a (possibly repeating) list of image positions in `_paramsEval.imgIds`.

    A faithful re-implementation of the one cell of `COCOeval.accumulate` that
    `stats[0]` reads: category 0 (person), area 'all', last maxDets, mean of the 101-point
    interpolated precision over all IoU thresholds. Returns nan when the resample contains
    no non-ignored ground truth, which is what accumulate's -1 sentinel means.
    """
    p = coco_eval.params
    pe = coco_eval._paramsEval
    n_img = len(pe.imgIds)
    n_area = len(pe.areaRng)
    max_det = p.maxDets[-1]
    # evalImgs is flat: [category][area][image]
    base = 0 * n_area * n_img + AREA_ALL * n_img
    entries = [coco_eval.evalImgs[base + int(i)] for i in img_positions]
    entries = [e for e in entries if e is not None]
    if not entries:
        return float("nan")

    gt_ignore = np.concatenate([e["gtIgnore"] for e in entries])
    n_pos = int(np.count_nonzero(gt_ignore == 0))
    if n_pos == 0:
        return float("nan")

    scores = np.concatenate([e["dtScores"][0:max_det] for e in entries])
    order = np.argsort(-scores, kind="mergesort")
    matches = np.concatenate([e["dtMatches"][:, 0:max_det] for e in entries], axis=1)[:, order]
    ignored = np.concatenate([e["dtIgnore"][:, 0:max_det] for e in entries], axis=1)[:, order]

    # "Matched" is spelled differently by the two forks and getting it wrong is silent:
    # pycocotools leaves an unmatched detection's dtMatches at 0 and tests truthiness,
    # while xtcocotools (cocoapi PR #332, which the keypoint path needs) initialises the
    # row to -1 and tests `>= 0`. Read -1 as truthy and every false positive becomes a
    # true positive, measured here as a keypoint AP of exactly 1.0. Detect the
    # convention from the data rather than from the class name; when nothing is unmatched
    # the two agree anyway.
    unmatched_is_negative = bool(matches.size) and bool((matches < 0).any())
    matched = matches >= 0 if unmatched_is_negative else matches.astype(bool)
    tps = np.logical_and(matched, np.logical_not(ignored))
    fps = np.logical_and(np.logical_not(matched), np.logical_not(ignored))
    tp_sum = np.cumsum(tps, axis=1).astype(float)
    fp_sum = np.cumsum(fps, axis=1).astype(float)

    rec_thrs = np.asarray(p.recThrs)
    out = np.zeros((len(p.iouThrs), len(rec_thrs)), float)
    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum, strict=True)):
        rc = tp / n_pos
        pr = tp / (fp + tp + np.spacing(1))
        # monotone envelope, in place and backwards, exactly as pycocotools does it
        for i in range(len(pr) - 1, 0, -1):
            if pr[i] > pr[i - 1]:
                pr[i - 1] = pr[i]
        idx = np.searchsorted(rc, rec_thrs, side="left")
        keep = idx < len(pr)
        q = np.zeros(len(rec_thrs), float)
        q[keep] = pr[idx[keep]]
        out[t] = q
    return float(out.mean())


def bootstrap_ci(
    coco_eval: Any,
    official_ap: float,
    n_resamples: int,
    seed: int,
    label: str,
) -> dict[str, Any]:
    """Per-image bootstrap around one COCOeval AP. Self-checks against the official value.

    Returns the interval plus `per_resample`, so a paired delta between two rows can be
    formed later from the same resample indices (identical seed -> identical draws).
    """
    n_img = len(coco_eval._paramsEval.imgIds)
    identity = np.arange(n_img)
    recomputed = _accumulate_ap(coco_eval, identity)
    if not np.isfinite(recomputed) or abs(recomputed - official_ap) > 1e-9:
        raise RuntimeError(
            f"{label}: the bootstrap's accumulate re-implementation disagrees with "
            f"COCOeval ({recomputed!r} vs {official_ap!r}). Refusing to publish an "
            "interval around a statistic it cannot reproduce."
        )
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, float)
    for b in range(n_resamples):
        draws[b] = _accumulate_ap(coco_eval, rng.integers(0, n_img, n_img))
    finite = draws[np.isfinite(draws)]
    return {
        "metric": label,
        "point": float(official_ap),
        "n_images": n_img,
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "n_finite": int(finite.size),
        "mean": float(finite.mean()),
        "std_err": float(finite.std(ddof=1)),
        "ci95_lo": float(np.percentile(finite, 2.5)),
        "ci95_hi": float(np.percentile(finite, 97.5)),
        "per_resample": draws.tolist(),
    }


def _build_eval(kind: str, gt_file: Path, preds: list[dict], img_ids: list[int]) -> Any:
    """A COCOeval with `evaluate()` + `accumulate()` + `summarize()` already run."""
    if kind == "bbox":
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        gt = COCO(str(gt_file))
        with contextlib.redirect_stdout(io.StringIO()):
            e = COCOeval(gt, gt.loadRes(preds), "bbox")
            e.params.imgIds = list(img_ids)
            e.params.catIds = [1]
            e.evaluate()
            e.accumulate()
            e.summarize()
        return e
    from xtcocotools.coco import COCO as XCOCO
    from xtcocotools.cocoeval import COCOeval as XCOCOeval

    gt = XCOCO(str(gt_file))
    with contextlib.redirect_stdout(io.StringIO()):
        e = XCOCOeval(gt, gt.loadRes(preds), "keypoints", sigmas=None, use_area=True)
        e.params.imgIds = list(img_ids)
        e.params.catIds = [1]
        e.evaluate()
        e.accumulate()
        e.summarize()
    return e


def _resolve_onnx(spec: dict) -> dict:
    """Repo-relative local ONNX paths -> absolute, so the run does not depend on the CWD.

    rtmlib decides "is this a local file or a URL to fetch" with a bare
    `os.path.exists(onnx_model)`, which is CWD-relative; a config naming
    `final_models/joints/model.onnx` would otherwise try to download that string from
    anywhere but the repo root. URLs and already-absolute paths pass through untouched.

    A local graph is hashed here and checked against the digest the config declares, because
    the backend carries that digest as metadata and never opens the file to confirm it. The
    table below republishes the declared value as the row's identity, so an unchecked digest
    would let a row name bytes it did not run.
    """
    from vividly.models.checkpoints import sha256_of

    onnx = str(spec.get("onnx", ""))
    if onnx.startswith(("http://", "https://")) or Path(onnx).is_absolute():
        return spec
    candidate = REPO_ROOT / onnx
    if not candidate.is_file():
        raise FileNotFoundError(
            f"{onnx} is neither a URL nor a file under {REPO_ROOT}. The v2 pose ONNX is "
            "committed at final_models/joints/model.onnx, a Git LFS object that "
            "`git lfs pull` fetches."
        )
    declared = spec.get("sha256")
    if declared:
        actual = sha256_of(candidate)
        if actual != declared:
            raise ValueError(
                f"{onnx}: sha256 on disk is {actual}, the configuration declares {declared}. "
                "Refusing to publish a row whose graph digest is not the graph it ran. On a "
                "fresh clone the committed graphs are Git LFS pointers until `git lfs pull`."
            )
    return {**spec, "onnx": str(candidate)}


# ------------------------------------------------------------------------- the one row
def run_row(cfg: dict, limit: int | None, n_resamples: int, seed: int) -> dict[str, Any]:
    """Build the models, sweep the frozen subset once, score, bootstrap. One clean row."""
    import cv2

    from vividly.eval.runner import (
        HOST,
        assert_row_complete,
        latency_stats,
        load_subset,
        materialize_condition,
        subset_fingerprint,
        to_coco_detections,
        to_coco_keypoints,
    )
    from vividly.models.base import Timing
    from vividly.models.factory import build_detector, build_pose

    conditions = cfg.get("conditions") or []
    if [c["name"] for c in conditions] != ["clean"]:
        raise ValueError(
            f"{cfg['run_name']}: this ladder measures the clean condition only; got "
            f"{[c['name'] for c in conditions]}. Degradation ladders belong in "
            "scripts/eval/eval_runner.py."
        )

    subset_file = REPO_ROOT / cfg["dataset"]["subset_file"]
    images = load_subset(subset_file)
    fingerprint = subset_fingerprint(subset_file)
    if limit:
        images = images[:limit]

    # Diagnostic knob, off by default. rtmlib's pose preprocessing never converts colour:
    # it affine-crops the cv2 image (BGR) and normalises it. mmpose's RTMPose
    # data_preprocessor is `bgr_to_rgb=True`, and neither ONNX graph carries the swap
    # (the mmdeploy SDK does it in pipeline.json, which rtmlib does not implement), so
    # the deployed path feeds every RTMPose graph channel-swapped input. Setting
    # `pose.color_order: rgb` converts the image handed to the pose stage only; the
    # detector keeps its own convention, so this changes exactly one thing. It is popped
    # here because it is a runner concern, not a backend kwarg.
    pose_spec = dict(_resolve_onnx(cfg["pose"]))
    color_order = str(pose_spec.pop("color_order", "bgr")).lower()
    if color_order not in ("bgr", "rgb"):
        raise ValueError(f"pose.color_order must be bgr or rgb, got {color_order!r}")

    det = build_detector(_resolve_onnx(cfg["detector"]))
    pose = build_pose(pose_spec)
    det.warmup()
    pose.warmup()

    paths, gt_file = materialize_condition(conditions[0], images, cfg, REPO_ROOT)
    det.reset_timings()
    pose.reset_timings()
    bbox_preds: list[dict] = []
    kpt_preds: list[dict] = []
    e2e: list[Timing] = []
    t_start = time.perf_counter()
    for n, (ref, path) in enumerate(zip(images, paths, strict=True), start=1):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        t0 = time.perf_counter()
        d = det.detect_one(img)
        pose_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if color_order == "rgb" else img
        p = pose.estimate(pose_img, d.xyxy)
        e2e.append(Timing("e2e", (time.perf_counter() - t0) * 1000.0, 1, "runner"))
        bbox_preds += to_coco_detections(ref.image_id, d)
        kpt_preds += to_coco_keypoints(ref.image_id, d, p)
        if n % 50 == 0 or n == len(images):
            done = time.perf_counter() - t_start
            print(
                f"    {n}/{len(images)} images  {done:.0f}s elapsed  "
                f"~{done / n * (len(images) - n):.0f}s left",
                flush=True,
            )

    img_ids = [r.image_id for r in images]
    e_box = _build_eval("bbox", gt_file, bbox_preds, img_ids)
    e_kpt = _build_eval("keypoints", gt_file, kpt_preds, img_ids)

    row: dict[str, Any] = {
        "run_name": cfg["run_name"],
        "condition": "clean",
        "kind": "passthrough",
        "n_images": len(images),
        "n_det_predictions": len(bbox_preds),
        "n_kpt_predictions": len(kpt_preds),
        "det_AP": float(e_box.stats[0]),
        "det_AP50": float(e_box.stats[1]),
        "det_AR100": float(e_box.stats[8]),
        "kpt_AP": float(e_kpt.stats[0]),
        "kpt_AP50": float(e_kpt.stats[1]),
        "kpt_AR": float(e_kpt.stats[5]),
        **latency_stats(det.timings + pose.timings + e2e),
        "host_cpu": HOST["cpu"],
        "det_provenance": det.provenance.label(),
        "pose_provenance": pose.provenance.label(),
        "pose_backend": cfg["pose"]["backend"],
        "pose_color_order": color_order,
        "pose_onnx": cfg["pose"]["onnx"],
        "pose_onnx_sha256": cfg["pose"].get("sha256"),
        "det_onnx": cfg["detector"]["onnx"],
        "det_onnx_sha256": cfg["detector"].get("sha256"),
        "det_name": cfg["detector"].get("name"),
        "det_score_thr": cfg["detector"].get("score_thr"),
        "gt_file": str(gt_file),
        "subset_fingerprint": fingerprint,
        "wall_seconds": round(time.perf_counter() - t_start, 1),
    }
    assert_row_complete(row)

    print(f"    bootstrapping {n_resamples} resamples ...", flush=True)
    row["bootstrap"] = {
        "det_AP": bootstrap_ci(e_box, row["det_AP"], n_resamples, seed, "det_AP"),
        "kpt_AP": bootstrap_ci(e_kpt, row["kpt_AP"], n_resamples, seed, "kpt_AP"),
    }
    return row


def check_gate(cfg: dict, row: dict, limit: int | None) -> str | None:
    """Return a failure message, or None when the row passes (or carries no gate)."""
    gate = cfg.get("gate")
    if not gate:
        return None
    if limit:
        # A gate is a claim about a frozen subset. Under --limit the row is a different
        # measurement on a different set of images, so the gate is inapplicable rather
        # than passed. State that plainly instead of quietly comparing incomparables.
        print(
            f"  Gate skipped (--limit {limit}): {gate['metric']} = {row[gate['metric']]:.4f} "
            f"on {row['n_images']} images is not the frozen-subset statistic the gate pins."
        )
        return None
    got = row[gate["metric"]]
    lo, hi = gate["expected"] - gate["tol"], gate["expected"] + gate["tol"]
    if lo <= got <= hi:
        print(
            f"  Gate pass: {gate['metric']} {got:.4f} within {gate['expected']} +/- {gate['tol']}"
        )
        return None
    return (
        f"Gate fail on {cfg['run_name']}: {gate['metric']} = {got:.4f}, expected "
        f"{gate['expected']} +/- {gate['tol']} (recorded at: {gate.get('source', '?')}). "
        "The instrument does not reproduce its own history, so nothing measured after "
        "this row would be interpretable. Ladder stopped."
    )


def _boxes_cell(row: dict) -> str:
    """The boxes column, read off the row so a different detector cannot be mislabelled."""
    name = row.get("det_name") or Path(str(row.get("det_onnx", ""))).name or "-"
    sha = (row.get("det_onnx_sha256") or "")[:8] or "-"
    thr = row.get("det_score_thr")
    thr_text = "" if thr is None else f" thr{thr}"
    return f"{name} `{sha}`{thr_text}"


def rows_markdown(rows: list[dict], deltas: list[dict] | None = None) -> str:
    n_res = rows[0]["bootstrap"]["kpt_AP"]["n_resamples"]
    head = (
        "| row | pose onnx sha8 | boxes | det AP | kpt AP | kpt AP 95% CI | "
        "det p50 ms | pose p50 ms | e2e p50 ms |\n"
        "|---|---|---|---:|---:|---|---:|---:|---:|\n"
    )
    lines = []
    for r in rows:
        sha = (r.get("pose_onnx_sha256") or "")[:8] or "-"
        ci = r["bootstrap"]["kpt_AP"]
        lines.append(
            f"| {r['run_name']} | `{sha}` | {_boxes_cell(r)} | "
            f"{r['det_AP']:.4f} | {r['kpt_AP']:.4f} | "
            f"[{ci['ci95_lo']:.4f}, {ci['ci95_hi']:.4f}] | "
            f"{r['detect_p50_ms']:.1f} | {r['pose_p50_ms']:.1f} | {r['e2e_p50_ms']:.1f} |"
        )
    foot = (
        f"\n\nNote: eval-smoke provenance, not a Vividly accuracy claim. CIs are a "
        f"{n_res}-resample per-image bootstrap over the frozen coco_val_person500 subset; "
        "rows share images and share a detector, so overlapping intervals are not a "
        "significance test, so read the paired deltas.\n"
    )
    if deltas:
        foot += (
            "\n| paired delta (same resample draws) | metric | point | 95% CI | P(delta>0) |\n"
            "|---|---|---:|---|---:|\n"
        )
        foot += (
            "\n".join(
                f"| {d['from']} -> {d['to']} | {d['metric']} | {d['point']:+.4f} | "
                f"[{d['ci95_lo']:+.4f}, {d['ci95_hi']:+.4f}] | {d['frac_positive']:.3f} |"
                for d in deltas
            )
            + "\n"
        )
    return head + "\n".join(lines) + foot


def paired_deltas(rows: list[dict]) -> list[dict]:
    """Paired bootstrap deltas between consecutive rows, on the same resample draws."""
    out = []
    for a, b in itertools.pairwise(rows):
        for metric in ("det_AP", "kpt_AP"):
            da = np.asarray(a["bootstrap"][metric]["per_resample"], float)
            db = np.asarray(b["bootstrap"][metric]["per_resample"], float)
            n = min(da.size, db.size)
            diff = db[:n] - da[:n]
            diff = diff[np.isfinite(diff)]
            out.append(
                {
                    "metric": metric,
                    "from": a["run_name"],
                    "to": b["run_name"],
                    "point": b[metric] - a[metric],
                    "ci95_lo": float(np.percentile(diff, 2.5)),
                    "ci95_hi": float(np.percentile(diff, 97.5)),
                    "frac_positive": float((diff > 0).mean()),
                }
            )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", action="append", required=True, type=Path, help="repeat per row")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N subset images")
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument(
        "--out-root", type=Path, default=REPO_ROOT / "outputs/eval", help="run directory root"
    )
    args = ap.parse_args(argv)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"pose_ladder_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for path in args.config:
        cfg = yaml.safe_load(path.read_text())
        print(f"\n=== {cfg['run_name']}  ({path})", flush=True)
        row = run_row(cfg, args.limit, args.bootstrap, args.seed)
        row["config_path"] = str(path)
        rows.append(row)
        ci = row["bootstrap"]["kpt_AP"]
        print(
            f"  det AP {row['det_AP']:.4f}  kpt AP {row['kpt_AP']:.4f} "
            f"[{ci['ci95_lo']:.4f}, {ci['ci95_hi']:.4f}]  "
            f"pose p50 {row['pose_p50_ms']:.1f} ms  ({row['wall_seconds']:.0f}s)",
            flush=True,
        )
        failure = check_gate(cfg, row, args.limit)
        if failure:
            from vividly.eval.runner import HOST

            (out_dir / "results.json").write_text(
                json.dumps({"host": HOST, "rows": rows, "stopped": failure}, indent=1)
            )
            print(f"\n{failure}", file=sys.stderr)
            return EXIT_GATE

    from vividly.eval.runner import HOST

    deltas = paired_deltas(rows)
    payload = {
        "ladder": "cpu_proof_coco500_v2pose",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": HOST,
        "bootstrap": {"n_resamples": args.bootstrap, "seed": args.seed, "unit": "image"},
        "limit": args.limit,
        "rows": rows,
        "paired_deltas": deltas,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=1))
    table = rows_markdown(rows, deltas)
    (out_dir / "results.md").write_text(table)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(table)
    print("\n" + table)
    print(f"results: {out_dir}/results.json")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
