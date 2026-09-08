#!/usr/bin/env python
r"""I2: does the exported ONNX reproduce the checkpoint? Plus the Mac CPU ms/frame.

    uv run python scripts/eval/onnx_parity_v2.py \
        --onnx   outputs/eyes_b1/b1_best_stg2_345c30a5.onnx \
        --gt     data/train_ok/physicalai_smartspaces/annotations/person_v2_devval.json \
        --images outputs/eyes_b1/parity_sample \
        --det    outputs/eyes_b1/dumps/dets_b1_rtv4s_seed0_identity.json \
        --out-json outputs/eyes_b1/i2/parity_345c30a5.json

The paths under `data/` and `outputs/` are not in version control.

The identity dump is the checkpoint's reference output on dev-val: it was written by
`scripts/detector/dump_detections_v2.py run`, walking the trainer's own eval loop with
the trainer's EMA weights and postprocessor. The RT-DETRv4 exporter
`tools/deployment/export_onnx.py` loads
those same EMA tensors (`checkpoint['ema']['module']`) and wraps `model.deploy()` +
`postprocessor.deploy()`. So for one image the two must emit the same 300 boxes, in
the same score order, up to fp32 drift. This script measures that gap.

Preprocessing, taken from the trainer at pin 55fefaae and not invented here:
  RT-DETRv4 configs/base/dataloader.yml `val_dataloader` ops are exactly
      - {type: Resize, size: [640, 640]}
      - {type: ConvertPILImage, dtype: 'float32', scale: True}
  the `Resize` registration in engine/data/transforms/_transforms.py,
  `Resize = register()(T.Resize)`, is torchvision.transforms.v2.Resize, which on a
  PIL input calls PIL's own `Image.resize((w, h), BILINEAR)`, so the resize happens
  in PIL space, to a fixed 640x640, aspect ratio not preserved. `ConvertPILImage`,
  in that same transforms module, is `pil_to_tensor -> .float() -> / 255.` and
  nothing else: there is no ImageNet mean/std normalisation anywhere in this
  pipeline. `orig_size` is stored **[w, h]** by the dataset in
  engine/data/dataset/coco_dataset.py and the postprocessor multiplies normalised
  cxcywh by `orig_target_sizes.repeat(1, 2)` in the box-scaling step of
  engine/rtv4/postprocessor.py, so `orig_target_sizes = [[width, height]]`
  in original pixels and the boxes come back in original-image pixels.

The ONNX graph takes ['images', 'orig_target_sizes'] and emits
['labels', 'boxes', 'scores'] (boxes xyxy); the dump stores xywh, so boxes are
converted here, never the other way round.

Device is CPU and only CPU: POLICY.md, section 6, records that the CoreML execution
provider crashes on both detector graphs at the pinned runtime version (onnxruntime
1.28.0; docs/PIPELINE.md).

Two comparisons, because either alone can lie:
  - Rank-aligned: query k of the ONNX against row k of the dump. Both lists are the
    postprocessor's score-descending top-300 over the same queries, so this is the
    strict test, but two near-tied scores can swap ranks and blow up the box delta
    while nothing is actually wrong.
  - Best-IoU: every ONNX box matched to its highest-IoU dump box for the same image.
    Order-free. A structural export failure shows up here as a low worst-case IoU,
    and cannot be explained away as a tie swap.

Both are reported at several score floors, and the floor is not cosmetic. The
postprocessor emits a fixed top-300 per image whatever the image contains
(num_queries 300, num_classes 1 -> `topk(300)` returns every query). On a dev-val
frame only ~20 of those clear 0.3; the rest are the model's noise tail, scores down
to 1e-4, whose box regressions are arbitrary and whose rank order is decided by
float noise. At floor 0.0 the worst-case numbers are therefore dominated by
tail queries that mean nothing, and a rank swap between two 0.0002-scored boxes
prints a box delta of about 1900 px on a 1920 px-wide frame. Read the 0.3 / 0.4 /
0.5 rows, which are the operating points I1 grades; the 0.0 row is kept only so the
tail's behaviour is visible rather than quietly dropped.

`--bench N` additionally times N single-image inferences after `--warmup` untimed
ones, on the same preprocessed tensor, and reports mean / p50 / p95 ms per frame.

Exit codes: 0 ok, 3 config/prerequisite problem.
"""

from __future__ import annotations

import argparse
import hashlib
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

INPUT_SIZE = 640  # the exported graph's static spatial dims


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


def preprocess(path: Path) -> np.ndarray:
    """PIL RGB, then PIL bilinear resize to 640x640, then CHW float32 / 255. See header."""
    with Image.open(path) as im:
        rgb = im.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        arr = np.asarray(rgb, dtype=np.uint8)
    chw = arr.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw[None, ...]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """xyxy IoU, (len(a), len(b)). Plain numpy; no pycocotools needed for a sample."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path, help="COCO GT (image ids, sizes, names)")
    ap.add_argument("--images", required=True, type=Path, help="root the GT file_names hang off")
    ap.add_argument("--det", required=True, type=Path, help="the reference identity dump")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--bench", type=int, default=0, help="timed single-image inferences")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument(
        "--floors",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.3, 0.4, 0.5],
        help="score floors the deltas are reported at; see the header",
    )
    ap.add_argument("--threads", type=int, default=0, help="0 = onnxruntime default")
    args = ap.parse_args(argv)

    for p in (args.onnx, args.gt, args.det):
        if not p.is_file():
            print(f"FATAL: no such file: {p}", file=sys.stderr)
            return EXIT_CONFIG
    if not args.images.is_dir():
        print(f"FATAL: not a directory: {args.images}", file=sys.stderr)
        return EXIT_CONFIG

    import onnxruntime as ort

    with args.gt.open() as fh:
        gt = json.load(fh)
    images = {img["id"]: img for img in gt["images"]}

    present: list[dict[str, Any]] = []
    for img in gt["images"]:
        path = args.images / img["file_name"]
        if path.is_file():
            present.append({**img, "path": path})
    if not present:
        print(f"FATAL: none of the GT file_names resolve under {args.images}", file=sys.stderr)
        return EXIT_CONFIG
    present.sort(key=lambda r: r["id"])
    wanted = {r["id"] for r in present}

    ref: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with args.det.open() as fh:
        for rec in json.load(fh):
            if int(rec["image_id"]) in wanted:
                ref[int(rec["image_id"])].append(rec)
    for rows in ref.values():  # the dump is already score-descending; make it explicit
        rows.sort(key=lambda r: -float(r["score"]))

    opts = ort.SessionOptions()
    if args.threads > 0:
        opts.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(
        str(args.onnx), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    out_names = [o.name for o in sess.get_outputs()]

    per_image: list[dict[str, Any]] = []
    for row in present:
        image_id = int(row["id"])
        blob = preprocess(row["path"])
        sizes = np.array([[int(row["width"]), int(row["height"])]], dtype=np.int64)
        outs = sess.run(out_names, {"images": blob, "orig_target_sizes": sizes})
        named = dict(zip(out_names, outs, strict=True))
        onnx_boxes = np.asarray(named["boxes"][0], dtype=np.float64)  # xyxy
        onnx_scores = np.asarray(named["scores"][0], dtype=np.float64)
        onnx_labels = np.asarray(named["labels"][0]).astype(int)

        rows = ref.get(image_id, [])
        ref_boxes = np.array(
            [
                [
                    r["bbox"][0],
                    r["bbox"][1],
                    r["bbox"][0] + r["bbox"][2],
                    r["bbox"][1] + r["bbox"][3],
                ]
                for r in rows
            ],
            dtype=np.float64,
        ).reshape(-1, 4)
        ref_scores = np.array([float(r["score"]) for r in rows], dtype=np.float64)
        ref_labels = np.array([int(r["category_id"]) for r in rows], dtype=int)

        floors: dict[str, Any] = {}
        for thr in args.floors:
            ob = onnx_boxes[onnx_scores >= thr]
            os_ = onnx_scores[onnx_scores >= thr]
            ol = onnx_labels[onnx_scores >= thr]
            rb = ref_boxes[ref_scores >= thr]
            rs = ref_scores[ref_scores >= thr]
            rl = ref_labels[ref_scores >= thr]
            n = min(len(os_), len(rs))
            ious = iou_matrix(ob, rb)
            best_j = ious.argmax(axis=1) if ious.size else np.zeros(0, dtype=int)
            best_iou = ious.max(axis=1) if ious.size else np.zeros(0)
            floors[f"{thr:g}"] = {
                "n_onnx": len(os_),
                "n_ref": len(rs),
                "count_delta": int(len(os_) - len(rs)),
                "rank_aligned_max_abs_score_delta": (
                    float(np.max(np.abs(os_[:n] - rs[:n]))) if n else None
                ),
                "rank_aligned_max_abs_bbox_delta_px": (
                    float(np.max(np.abs(ob[:n] - rb[:n]))) if n else None
                ),
                "rank_aligned_label_mismatches": int(np.sum(ol[:n] != rl[:n])) if n else 0,
                "best_iou_min": float(best_iou.min()) if len(best_iou) else None,
                "best_iou_mean": float(best_iou.mean()) if len(best_iou) else None,
                "best_iou_matched_max_abs_score_delta": (
                    float(np.max(np.abs(os_ - rs[best_j]))) if len(best_j) else None
                ),
                "best_iou_matched_max_abs_bbox_delta_px": (
                    float(np.max(np.abs(ob - rb[best_j]))) if len(best_j) else None
                ),
                "_matched_score_deltas": (np.abs(os_ - rs[best_j]).tolist() if len(best_j) else []),
                "_matched_bbox_deltas_px": (
                    np.max(np.abs(ob - rb[best_j]), axis=1).tolist() if len(best_j) else []
                ),
            }

        per_image.append(
            {
                "image_id": image_id,
                "file_name": row["file_name"],
                "image_sha256": sha256_file(row["path"]),
                "n_onnx": len(onnx_scores),
                "n_ref": len(ref_scores),
                "onnx_top1_score": float(onnx_scores[0]) if len(onnx_scores) else None,
                "ref_top1_score": float(ref_scores[0]) if len(ref_scores) else None,
                "floors": floors,
            }
        )

    def agg(thr: str, key: str, how: str = "max") -> float | None:
        vals = [r["floors"][thr][key] for r in per_image if r["floors"][thr][key] is not None]
        if not vals:
            return None
        return float(max(vals)) if how == "max" else float(min(vals))

    summary: dict[str, Any] = {"n_images": len(per_image), "by_score_floor": {}}
    for thr in args.floors:
        key = f"{thr:g}"
        pooled_s = np.array(
            [v for r in per_image for v in r["floors"][key]["_matched_score_deltas"]]
        )
        pooled_b = np.array(
            [v for r in per_image for v in r["floors"][key]["_matched_bbox_deltas_px"]]
        )
        summary["by_score_floor"][key] = {
            "n_onnx_total": int(sum(r["floors"][key]["n_onnx"] for r in per_image)),
            "n_ref_total": int(sum(r["floors"][key]["n_ref"] for r in per_image)),
            "images_with_count_delta": int(
                sum(1 for r in per_image if r["floors"][key]["count_delta"] != 0)
            ),
            "max_rank_aligned_abs_score_delta": agg(key, "rank_aligned_max_abs_score_delta"),
            "max_rank_aligned_abs_bbox_delta_px": agg(key, "rank_aligned_max_abs_bbox_delta_px"),
            "total_rank_aligned_label_mismatches": int(
                sum(r["floors"][key]["rank_aligned_label_mismatches"] for r in per_image)
            ),
            "min_best_iou": agg(key, "best_iou_min", "min"),
            "max_best_iou_matched_abs_score_delta": agg(
                key, "best_iou_matched_max_abs_score_delta"
            ),
            "max_best_iou_matched_abs_bbox_delta_px": agg(
                key, "best_iou_matched_max_abs_bbox_delta_px"
            ),
            "matched_abs_score_delta_p50": (
                float(np.percentile(pooled_s, 50)) if pooled_s.size else None
            ),
            "matched_abs_score_delta_p99": (
                float(np.percentile(pooled_s, 99)) if pooled_s.size else None
            ),
            "matched_abs_score_delta_n_over_0.01": int((pooled_s > 0.01).sum()),
            "matched_abs_bbox_delta_px_p50": (
                float(np.percentile(pooled_b, 50)) if pooled_b.size else None
            ),
            "matched_abs_bbox_delta_px_p99": (
                float(np.percentile(pooled_b, 99)) if pooled_b.size else None
            ),
            "matched_n_pairs": int(pooled_s.size),
        }
    for r in per_image:  # drop the raw per-box arrays from the persisted payload
        for f in r["floors"].values():
            f.pop("_matched_score_deltas", None)
            f.pop("_matched_bbox_deltas_px", None)
    summary["max_abs_top1_score_delta"] = float(
        max(abs(r["onnx_top1_score"] - r["ref_top1_score"]) for r in per_image)
    )

    bench: dict[str, Any] | None = None
    if args.bench > 0:
        blob = preprocess(present[0]["path"])
        sizes = np.array(
            [[int(present[0]["width"]), int(present[0]["height"])]],
            dtype=np.int64,
        )
        feed = {"images": blob, "orig_target_sizes": sizes}
        for _ in range(max(0, args.warmup)):
            sess.run(out_names, feed)
        times_ms: list[float] = []
        for _ in range(args.bench):
            t0 = time.perf_counter()
            sess.run(out_names, feed)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        arr = np.array(times_ms)
        bench = {
            "n_warmup": int(args.warmup),
            "n_timed": int(args.bench),
            "input_size": [INPUT_SIZE, INPUT_SIZE],
            "batch": 1,
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "note": "model forward + postprocessor only; preprocessing not included",
        }

    det_side = args.det.with_suffix(args.det.suffix + ".sidecar.json")
    ckpt_sha = None
    if det_side.is_file():
        with det_side.open() as fh:
            ckpt_sha = json.load(fh).get("checkpoint_sha256")

    payload = {
        "written_by": "scripts/eval/onnx_parity_v2.py",
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "onnx": str(args.onnx),
        "onnx_sha256": sha256_file(args.onnx),
        "reference_dump": str(args.det),
        "reference_dump_sha256": sha256_file(args.det),
        "checkpoint_sha256": ckpt_sha,
        "gt": str(args.gt),
        "gt_sha256": sha256_file(args.gt),
        "images_root": str(args.images),
        "host": {
            "cpu": cpu_brand(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "ort_providers_available": ort.get_available_providers(),
            "ort_provider_used": "CPUExecutionProvider",
            "intra_op_num_threads": args.threads or "onnxruntime default",
        },
        "summary": summary,
        "bench": bench,
        "per_image": per_image,
    }
    print(json.dumps({"summary": summary, "bench": bench}, indent=2))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.out_json}")
    if len(images) and not per_image:
        return EXIT_CONFIG
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
