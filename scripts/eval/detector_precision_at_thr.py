#!/usr/bin/env python
r"""Precision instrument (I1): precision / recall / FP-per-image at fixed score cuts.

    uv run python scripts/eval/detector_precision_at_thr.py \
        --gt  data/train_ok/physicalai_smartspaces/annotations/person_v2_devval.json \
        --det outputs/eyes_b1/dumps/dets_b1_rtv4s_seed0_identity.json \
        --title "B1 RT-DETRv4-S seed0 (345c30a5) identity, dev-val" \
        --out-md   outputs/eyes_b1/i1/precision_b1_rtv4s_seed0_identity.md \
        --out-json research/eyes-precision-345c30a5-identity.json

That is the recorded invocation (research/eyes-instrument-grading.md). The paths under
`data/` and `outputs/` are not in version control.

AP hides the operating point. This reports what the demo actually shows: at score
0.3 / **0.4 (the declared demo point)** / 0.5, IoU 0.5, how many of the drawn boxes
are real (precision), how many people are missed (recall), and how many phantom
boxes land on an average frame (FP-per-image), overall and per camera.

Matching is COCOeval's own greedy rule at a single IoU threshold, re-implemented
here because COCOeval reports AP over a score sweep and never exposes a
fixed-score confusion count. IoU comes from pycocotools (`pycocotools.mask.iou`),
the same primitive COCOeval.computeIoU calls, so a box pair scores identically in
both. `iscrowd` ground truth marks ignore regions: a detection matched to one is neither TP
nor FP, and the region is not in the recall denominator.

Differences from a COCOeval run, on purpose:
- No maxDets cap by default (`--max-dets 0`). COCOeval's 100/image cap would
  silently delete FPs on crowded frames, which is the exact quantity this
  instrument exists to count.
- No area ranges. Small-object AP is already in the training log.

Camera is parsed from the GT `file_name` prefix. The v2 person manifests write
`Warehouse_00N/Camera_NNNN/FFFFFF.jpg` (configs/convert/person_v2_manifests.yaml),
so `--camera-depth 2` (default) yields `Warehouse_005/Camera_0000`.

Provenance label: every row carries `src`, defaulting to `synthetic-devval`, whose
long form reads: synthetic SmartSpaces dev-val; not deployment evidence. Change it
with `--label` / `--label-long` when grading a different surface. Quote no number from
this script without the detections file's checkpoint sha256, which the dump
sidecar written by scripts/detector/dump_detections_v2.py carries (POLICY.md, rule 10).

Exit codes: 0 ok, 3 config/prerequisite problem.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

EXIT_OK = 0
EXIT_CONFIG = 3

DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5)
DEFAULT_IOU = 0.5
DEFAULT_LABEL = "synthetic-devval"
DEFAULT_LABEL_LONG = "synthetic SmartSpaces dev-val; not deployment evidence."


def camera_key(file_name: str, depth: int = 2) -> str:
    """First `depth` path segments of a COCO file_name, e.g. Warehouse_005/Camera_0000.

    A file_name with fewer than `depth` segments returns everything above the
    basename; a bare basename returns "(unknown)" rather than inventing a camera.
    """
    parts = [p for p in Path(file_name).as_posix().split("/") if p not in ("", ".")]
    if len(parts) <= 1:
        return "(unknown)"
    return "/".join(parts[: min(depth, len(parts) - 1)])


@dataclass
class Counts:
    """TP / FP / FN plus the image and GT bookkeeping a rate needs."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    n_images: int = 0
    n_gt: int = 0  # non-ignore GT only
    n_det: int = 0  # detections at or above the score cut, ignore-matched included

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.n_gt if self.n_gt else 0.0

    @property
    def fp_per_image(self) -> float:
        return self.fp / self.n_images if self.n_images else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def add(self, other: Counts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.n_images += other.n_images
        self.n_gt += other.n_gt
        self.n_det += other.n_det

    def as_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "n_images": self.n_images,
            "n_gt": self.n_gt,
            "n_det": self.n_det,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "fp_per_image": self.fp_per_image,
        }


@dataclass
class ImageDets:
    """One image's GT and detections, already sorted the way matching wants them."""

    boxes_gt: list[list[float]] = field(default_factory=list)  # xywh
    ignore_gt: list[int] = field(default_factory=list)  # 1 = iscrowd/ignore
    boxes_dt: list[list[float]] = field(default_factory=list)  # xywh, score-desc
    scores_dt: list[float] = field(default_factory=list)


def box_ious(boxes_dt: list[list[float]], boxes_gt: list[list[float]], ignore_gt: list[int]):
    """pycocotools IoU matrix, shape (n_dt, n_gt). Crowd GT use the COCO IoA rule."""
    if not boxes_dt or not boxes_gt:
        return np.zeros((len(boxes_dt), len(boxes_gt)), dtype=np.float64)
    return mask_utils.iou(boxes_dt, boxes_gt, [int(v) for v in ignore_gt])


def match_image(img: ImageDets, iou_thr: float, score_thr: float, max_dets: int = 0) -> Counts:
    """COCOeval's greedy match at one IoU and one score cut. Returns this image's counts.

    Detections must already be score-descending. GT are internally reordered so
    non-ignore regions are offered first, which is what makes "prefer a real match
    over an ignore match" a simple early break (COCOeval.evaluateImg does the same).
    """
    keep = [i for i, s in enumerate(img.scores_dt) if s >= score_thr]
    if max_dets > 0:
        keep = keep[:max_dets]
    boxes_dt = [img.boxes_dt[i] for i in keep]

    # non-ignore GT first; `order` maps matching position -> original GT index
    order = sorted(range(len(img.boxes_gt)), key=lambda g: img.ignore_gt[g])
    boxes_gt = [img.boxes_gt[g] for g in order]
    ignore_gt = [img.ignore_gt[g] for g in order]

    ious = box_ious(boxes_dt, boxes_gt, ignore_gt)

    gt_matched = [False] * len(boxes_gt)
    counts = Counts(n_images=1, n_gt=sum(1 for v in ignore_gt if not v), n_det=len(boxes_dt))

    for d in range(len(boxes_dt)):
        best_iou = min(iou_thr, 1 - 1e-10)
        best_g = -1
        for g in range(len(boxes_gt)):
            if gt_matched[g] and not ignore_gt[g]:
                continue
            # already holding a real match and the remaining GT are all ignores
            if best_g > -1 and not ignore_gt[best_g] and ignore_gt[g]:
                break
            if ious[d, g] < best_iou:
                continue
            best_iou = float(ious[d, g])
            best_g = g
        if best_g == -1:
            counts.fp += 1
            continue
        gt_matched[best_g] = True
        if ignore_gt[best_g]:
            counts.n_det -= 1  # ignore-matched detections leave the tally entirely
        else:
            counts.tp += 1
    counts.fn = counts.n_gt - counts.tp
    return counts


def load_pair(gt_path: Path, det_path: Path, category_ids: list[int] | None = None):
    """Load GT + detections into per-image ImageDets, keyed by COCO image id."""
    coco_gt = COCO(str(gt_path))
    with det_path.open() as fh:
        raw = json.load(fh)
    # pycocotools' results-list format is a bare list; tolerate a wrapped
    # {"annotations": [...]} because that is what a naive dumper writes. A dict
    # with none of those keys is an error rather than an empty grade: silently
    # scoring zero detections would report precision 0.0 as if the model failed.
    if isinstance(raw, dict):
        for key in ("annotations", "detections", "results"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
        else:
            raise ValueError(
                f"{det_path}: expected a COCO results LIST (or a dict wrapping one under "
                f"'annotations'/'detections'/'results'); got a dict with keys {sorted(raw)[:8]}"
            )
    if not isinstance(raw, list):
        raise ValueError(f"{det_path}: expected a COCO results LIST, got {type(raw).__name__}")

    cats = set(category_ids) if category_ids else None
    per_image: dict[int, ImageDets] = {}
    file_names: dict[int, str] = {}
    for img_id, img in coco_gt.imgs.items():
        per_image[img_id] = ImageDets()
        file_names[img_id] = img.get("file_name", "")
    for ann in coco_gt.dataset.get("annotations", []):
        if cats and ann.get("category_id") not in cats:
            continue
        img = per_image.get(ann["image_id"])
        if img is None:
            continue
        img.boxes_gt.append([float(v) for v in ann["bbox"]])
        img.ignore_gt.append(int(ann.get("iscrowd", 0) or ann.get("ignore", 0)))

    unknown = 0
    for det in raw:
        if cats and det.get("category_id") not in cats:
            continue
        img = per_image.get(det["image_id"])
        if img is None:
            unknown += 1
            continue
        img.boxes_dt.append([float(v) for v in det["bbox"]])
        img.scores_dt.append(float(det.get("score", 1.0)))
    if unknown:
        print(f"WARNING: {unknown} detections reference image_ids absent from the GT; dropped")

    for img in per_image.values():
        if img.boxes_dt:
            idx = sorted(range(len(img.scores_dt)), key=lambda i: -img.scores_dt[i])
            img.boxes_dt = [img.boxes_dt[i] for i in idx]
            img.scores_dt = [img.scores_dt[i] for i in idx]
    return per_image, file_names


def grade(
    per_image: dict[int, ImageDets],
    file_names: dict[int, str],
    thresholds=DEFAULT_THRESHOLDS,
    iou_thr: float = DEFAULT_IOU,
    camera_depth: int = 2,
    max_dets: int = 0,
) -> dict[float, dict[str, Counts]]:
    """-> {score_threshold: {"OVERALL"|camera: Counts}}."""
    out: dict[float, dict[str, Counts]] = {}
    for thr in thresholds:
        buckets: dict[str, Counts] = defaultdict(Counts)
        overall = Counts()
        for img_id, img in per_image.items():
            c = match_image(img, iou_thr, thr, max_dets=max_dets)
            buckets[camera_key(file_names.get(img_id, ""), camera_depth)].add(c)
            overall.add(c)
        table: dict[str, Counts] = {"OVERALL": overall}
        for cam in sorted(buckets):
            table[cam] = buckets[cam]
        out[thr] = table
    return out


def _row(name: str, c: Counts, label: str) -> str:
    return (
        f"| {name} | {label} | {c.n_images} | {c.n_gt} | {c.n_det} | "
        f"{c.tp} | {c.fp} | {c.fn} | "
        f"{c.precision:.4f} | {c.recall:.4f} | {c.f1:.4f} | {c.fp_per_image:.3f} |"
    )


def to_markdown(
    tables: dict[float, dict[str, Counts]],
    meta: dict[str, Any],
    label: str = DEFAULT_LABEL,
    label_long: str = DEFAULT_LABEL_LONG,
) -> str:
    head = "| camera | src | imgs | GT | dets | TP | FP | FN | precision | recall | F1 | FP/img |"
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [
        f"# Detector precision at fixed score cuts: {meta.get('title', 'unnamed run')}",
        "",
        f"**`{label}` = {label_long}**",
        "",
        f"- detections: `{meta.get('det')}`",
        f"- checkpoint sha256: `{meta.get('checkpoint_sha256', 'unknown, no sidecar found')}`",
        f"- GT: `{meta.get('gt')}`",
        f"- IoU {meta.get('iou')} · maxDets {meta.get('max_dets') or 'unlimited'} "
        f"· camera-depth {meta.get('camera_depth')}",
        f"- graded {meta.get('date')} by `scripts/eval/detector_precision_at_thr.py`",
        "",
    ]
    for thr in sorted(tables):
        demo = ", the demo point" if abs(thr - 0.4) < 1e-9 else ""
        lines += [f"## score >= {thr:g}{demo}", "", head, sep]
        table = tables[thr]
        lines.append(_row("**OVERALL**", table["OVERALL"], label))
        for cam, c in table.items():
            if cam != "OVERALL":
                lines.append(_row(cam, c, label))
        lines.append("")
    lines += [
        "Precision = TP/(TP+FP); recall = TP/GT; FP/img = FP / images in the GT split "
        "(images with no detection still count in the denominator). Matching is "
        "COCOeval's greedy rule at one IoU, no maxDets cap, no area ranges; "
        "`iscrowd` GT are ignore regions.",
        "",
    ]
    return "\n".join(lines)


def _sidecar_sha(det_path: Path) -> str | None:
    side = det_path.with_suffix(det_path.suffix + ".sidecar.json")
    if not side.is_file():
        side = det_path.with_name(det_path.stem + "_sidecar.json")
    if not side.is_file():
        return None
    try:
        with side.open() as fh:
            return json.load(fh).get("checkpoint_sha256")
    except (OSError, json.JSONDecodeError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gt", required=True, type=Path, help="COCO ground-truth json")
    ap.add_argument("--det", required=True, type=Path, help="COCO detections json (results list)")
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    ap.add_argument("--iou", type=float, default=DEFAULT_IOU)
    ap.add_argument("--camera-depth", type=int, default=2)
    ap.add_argument("--max-dets", type=int, default=0, help="0 = no cap (default; see header)")
    ap.add_argument("--category-ids", type=int, nargs="*", default=None)
    ap.add_argument("--title", default="")
    ap.add_argument("--label", default=DEFAULT_LABEL)
    ap.add_argument("--label-long", default=DEFAULT_LABEL_LONG)
    ap.add_argument("--checkpoint-sha256", default=None, help="overrides the dump sidecar")
    args = ap.parse_args(argv)

    for p in (args.gt, args.det):
        if not p.is_file():
            print(f"FATAL: no such file: {p}", file=sys.stderr)
            return EXIT_CONFIG
    if not 0.0 < args.iou <= 1.0:
        print(f"FATAL: --iou must be in (0, 1], got {args.iou}", file=sys.stderr)
        return EXIT_CONFIG

    per_image, file_names = load_pair(args.gt, args.det, args.category_ids)
    tables = grade(
        per_image,
        file_names,
        thresholds=sorted(args.thresholds),
        iou_thr=args.iou,
        camera_depth=args.camera_depth,
        max_dets=args.max_dets,
    )

    meta = {
        "title": args.title or args.det.name,
        "gt": str(args.gt),
        "det": str(args.det),
        "iou": args.iou,
        "max_dets": args.max_dets,
        "camera_depth": args.camera_depth,
        "thresholds": sorted(args.thresholds),
        "label": args.label,
        "label_long": args.label_long,
        "checkpoint_sha256": args.checkpoint_sha256 or _sidecar_sha(args.det),
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    md = to_markdown(tables, meta, label=args.label, label_long=args.label_long)
    print(md)

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md)
        print(f"wrote {args.out_md}")
    if args.out_json:
        payload = {
            "meta": meta,
            "tables": {
                f"{thr:g}": {cam: c.as_dict() for cam, c in table.items()}
                for thr, table in tables.items()
            },
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.out_json}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
