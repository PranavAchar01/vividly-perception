#!/usr/bin/env python
r"""I2 Mac grader: run an exported v2 detector ONNX on CPU over a COCO surface.

    uv run python scripts/eval/eval_detector_v2_onnx.py \
        --onnx   outputs/eyes_b1/b1_best_stg2_345c30a5.onnx \
        --gt     data/eval_only/mvor/annotations/mvor_person_v2_coco.json \
        --images data/eval_only/mvor/camma_mvor_dataset \
        --out-det  outputs/eyes_b1/i2/dets_mvor_345c30a5.json \
        --out-json outputs/eyes_b1/i2/mvor_345c30a5.json \
        --cluster-depth 2 --bootstrap 200 --label mvor-internal

The paths under `data/` and `outputs/` are not in version control.

Declared by research/eyes-instrument-protocol.md as instrument I2: a Mac run of
scripts/eval/eval_detector_v2_onnx.py with onnxruntime CPU, pycocotools and a
per-camera clustered bootstrap.

Writes a pycocotools results list (so scripts/eval/detector_precision_at_thr.py can
grade the same detections at fixed score cuts) plus a sidecar carrying the ONNX
sha256, the GT sha256 and the host, and prints COCO AP.

Preprocessing is the trainer's own val pipeline at pin 55fefaae, not a guess:
Resize to a fixed 640x640 in PIL space (torchvision v2 Resize over a PIL input,
aspect ratio not preserved) then `pil_to_tensor -> float -> /255`, with no
mean/std normalisation. `orig_target_sizes` is [width, height] in original pixels
and the postprocessor scales boxes by it, so detections come back in original
pixels. Full citations in the header of scripts/eval/onnx_parity_v2.py.

Device is CPU and only CPU: POLICY.md, section 6, records that the CoreML execution
provider crashes on both detector graphs at the pinned runtime version (onnxruntime
1.28.0; docs/PIPELINE.md).

`--bootstrap N` resamples clusters (not images) with replacement N times and
re-runs COCOeval on each resample, reporting the 2.5/97.5 percentiles of AP. The
cluster key is the first `--cluster-depth` path segments of the GT `file_name`
(`Warehouse_005/Camera_0000`, `day1/cam2`), because frames from one camera are not
independent samples and an image-level bootstrap would report a CI several times
too narrow. It is off by default because it costs one full COCOeval per resample.

`--label` is stamped into the sidecar and printed on every line. An EVAL_ONLY or
NC-licensed surface must carry its own label (e.g. `mvor-internal`); numbers from
such a surface are internal-only and are never quoted externally.

Exit codes: 0 ok, 3 config/prerequisite problem.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

EXIT_OK = 0
EXIT_CONFIG = 3

INPUT_SIZE = 640


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def cpu_brand() -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def cluster_key(file_name: str, depth: int) -> str:
    parts = [p for p in Path(file_name).as_posix().split("/") if p not in ("", ".")]
    if len(parts) <= 1:
        return "(unknown)"
    return "/".join(parts[: min(depth, len(parts) - 1)])


def preprocess(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        rgb = im.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        arr = np.asarray(rgb, dtype=np.uint8)
    return (arr.transpose(2, 0, 1).astype(np.float32) / 255.0)[None, ...]


def coco_ap(gt_path: Path, results: list[dict[str, Any]], img_ids: list[int]) -> list[float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(results)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.imgIds = img_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return [float(v) for v in ev.stats]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path, help="root the GT file_names hang off")
    ap.add_argument("--out-det", required=True, type=Path, help="COCO results list to write")
    ap.add_argument("--out-json", type=Path, default=None, help="metrics + provenance")
    ap.add_argument("--score-floor", type=float, default=0.0)
    ap.add_argument("--cluster-depth", type=int, default=2)
    ap.add_argument("--bootstrap", type=int, default=0, help="clustered resamples; 0 = off")
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--label", default="internal", help="surface label stamped on every number")
    ap.add_argument("--limit", type=int, default=0, help="smoke only: stop after N images")
    args = ap.parse_args(argv)

    for p in (args.onnx, args.gt):
        if not p.is_file():
            print(f"FATAL: no such file: {p}", file=sys.stderr)
            return EXIT_CONFIG
    if not args.images.is_dir():
        print(f"FATAL: not a directory: {args.images}", file=sys.stderr)
        return EXIT_CONFIG

    import onnxruntime as ort

    with args.gt.open() as fh:
        gt = json.load(fh)

    todo = []
    missing = 0
    for img in gt["images"]:
        path = args.images / img["file_name"]
        if path.is_file():
            todo.append({**img, "path": path})
        else:
            missing += 1
    if missing:
        print(f"WARNING: {missing} of {len(gt['images'])} GT file_names do not resolve")
    if not todo:
        print(f"FATAL: no GT image resolves under {args.images}", file=sys.stderr)
        return EXIT_CONFIG
    todo.sort(key=lambda r: r["id"])
    if args.limit:
        todo = todo[: args.limit]

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]

    results: list[dict[str, Any]] = []
    infer_ms: list[float] = []
    t_start = time.perf_counter()
    for row in todo:
        blob = preprocess(row["path"])
        sizes = np.array([[int(row["width"]), int(row["height"])]], dtype=np.int64)
        t0 = time.perf_counter()
        outs = sess.run(out_names, {"images": blob, "orig_target_sizes": sizes})
        infer_ms.append((time.perf_counter() - t0) * 1000.0)
        named = dict(zip(out_names, outs, strict=True))
        boxes = np.asarray(named["boxes"][0], dtype=np.float64)
        scores = np.asarray(named["scores"][0], dtype=np.float64)
        labels = np.asarray(named["labels"][0]).astype(int)
        keep = scores >= args.score_floor
        for box, score, label in zip(boxes[keep], scores[keep], labels[keep], strict=True):
            results.append(
                {
                    "image_id": int(row["id"]),
                    "category_id": int(label),
                    "bbox": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2] - box[0]),
                        float(box[3] - box[1]),
                    ],
                    "score": float(score),
                }
            )
    wall_s = time.perf_counter() - t_start

    args.out_det.parent.mkdir(parents=True, exist_ok=True)
    args.out_det.write_text(json.dumps(results))

    img_ids = [int(r["id"]) for r in todo]
    stats = coco_ap(args.gt, results, img_ids)
    stat_names = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    metrics = dict(zip(stat_names, stats, strict=True))

    clusters: dict[str, list[int]] = defaultdict(list)
    for row in todo:
        clusters[cluster_key(row["file_name"], args.cluster_depth)].append(int(row["id"]))

    boot: dict[str, Any] | None = None
    if args.bootstrap > 0:
        keys = sorted(clusters)
        rng = np.random.default_rng(args.bootstrap_seed)
        by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for rec in results:
            by_image[rec["image_id"]].append(rec)
        aps: list[float] = []
        for _ in range(args.bootstrap):
            pick = rng.integers(0, len(keys), size=len(keys))
            ids: list[int] = []
            for k in pick:
                ids.extend(clusters[keys[k]])
            uniq = sorted(set(ids))  # COCOeval keys on image id; duplicates collapse
            sub = [rec for i in uniq for rec in by_image.get(i, [])]
            if not sub:
                continue
            aps.append(coco_ap(args.gt, sub, uniq)[0])
        arr = np.array(aps)
        boot = {
            "n_resamples": int(args.bootstrap),
            "n_used": int(arr.size),
            "cluster_depth": args.cluster_depth,
            "n_clusters": len(keys),
            "clusters": keys,
            "AP_mean": float(arr.mean()) if arr.size else None,
            "AP_p2.5": float(np.percentile(arr, 2.5)) if arr.size else None,
            "AP_p97.5": float(np.percentile(arr, 97.5)) if arr.size else None,
            "note": (
                "clusters resampled with replacement; duplicate cluster draws collapse "
                "to one copy because COCOeval keys on image id, so the interval is "
                "conservative (slightly narrower than a true cluster bootstrap)"
            ),
        }

    payload = {
        "written_by": "scripts/eval/eval_detector_v2_onnx.py",
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": args.label,
        "onnx": str(args.onnx),
        "onnx_sha256": sha256_file(args.onnx),
        "gt": str(args.gt),
        "gt_sha256": sha256_file(args.gt),
        "images_root": str(args.images),
        "detections": str(args.out_det),
        "detections_sha256": sha256_file(args.out_det),
        "n_images": len(todo),
        "n_images_missing": missing,
        "n_detections": len(results),
        "score_floor": args.score_floor,
        "metrics": metrics,
        "bootstrap": boot,
        "timing": {
            "infer_mean_ms": float(np.mean(infer_ms)),
            "infer_p50_ms": float(np.percentile(infer_ms, 50)),
            "infer_p95_ms": float(np.percentile(infer_ms, 95)),
            "wall_s_including_preprocess": wall_s,
            "note": "infer_* are session.run only; wall includes PIL decode + resize",
        },
        "host": {
            "cpu": cpu_brand(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "ort_provider_used": "CPUExecutionProvider",
        },
    }
    print(json.dumps({k: payload[k] for k in ("label", "n_images", "metrics", "timing")}, indent=2))
    if boot:
        print(json.dumps({"bootstrap": {k: boot[k] for k in boot if k != "clusters"}}, indent=2))
    print(f"wrote {args.out_det}")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.out_json}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
