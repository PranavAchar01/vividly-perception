"""Accuracy metrics. Pure functions over serialized predictions; no model objects.

Detection : pycocotools COCOeval(iouType='bbox')
Pose      : xtcocotools COCOeval(iouType='keypoints')  [OKS]
Tracking  : trackeval 1.3.0 (the harness groups and the curate extra, absent from the cuda
            extra; see mot.py)
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# COCOeval.stats index maps. These are positional and undocumented in the API; getting
# one wrong silently reports AR as AP. Written out once, used everywhere.
BBOX_STATS = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR@1",
    "AR@10",
    "AR@100",
    "AR_small",
    "AR_medium",
    "AR_large",
)  # 12 entries, iouType='bbox'
KPT_STATS = (
    "AP",
    "AP50",
    "AP75",
    "AP_medium",
    "AP_large",
    "AR",
    "AR50",
    "AR75",
    "AR_medium",
    "AR_large",
)  # 10 entries, iouType='keypoints'; note that AR is index 5, not 6


@dataclass(frozen=True)
class MetricBlock:
    """One metric family for one condition."""

    kind: str  # "bbox" | "keypoints"
    values: dict[str, float]
    n_images: int
    n_predictions: int
    n_gt: int

    def to_dict(self) -> dict:
        return asdict(self)


def _summarize(stats: np.ndarray, names: tuple[str, ...]) -> dict[str, float]:
    if len(stats) != len(names):
        raise ValueError(f"expected {len(names)} stats, got {len(stats)}")
    # COCOeval returns -1 for an undefined statistic (no GT or no dets in that bucket).
    # Normalise to nan so it is visibly missing rather than silently a negative score.
    return {n: (float(v) if v >= 0 else float("nan")) for n, v in zip(names, stats, strict=True)}


def coco_bbox_eval(
    gt_ann_file: Path | str,
    predictions: list[dict],
    img_ids: list[int],
    cat_ids: tuple[int, ...] = (1,),
) -> MetricBlock:
    """`predictions`: COCO detection results
    [{image_id, category_id, bbox=[x,y,w,h], score}, ...]"""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(str(gt_ann_file))
    n_gt = len(gt.getAnnIds(imgIds=img_ids, catIds=list(cat_ids), iscrowd=None))
    if not predictions:
        return MetricBlock("bbox", {n: 0.0 for n in BBOX_STATS}, len(img_ids), 0, n_gt)
    with contextlib.redirect_stdout(io.StringIO()):  # COCOeval is very chatty
        dt = gt.loadRes(predictions)
        e = COCOeval(gt, dt, "bbox")
        e.params.imgIds = list(img_ids)
        e.params.catIds = list(cat_ids)
        e.evaluate()
        e.accumulate()
        e.summarize()
    return MetricBlock(
        "bbox", _summarize(e.stats, BBOX_STATS), len(img_ids), len(predictions), n_gt
    )


def coco_keypoint_eval(
    gt_ann_file: Path | str,
    predictions: list[dict],
    img_ids: list[int],
    sigmas: np.ndarray | None = None,
    use_area: bool = True,
) -> MetricBlock:
    """OKS AP via xtcocotools. `predictions`:
    [{image_id, category_id, keypoints=[x,y,v]*17, score}, ...]

    `sigmas=None` uses the COCO 17-keypoint defaults. When domain keypoint variance is
    measured on real data under the fine-tuned models of record, pass domain sigmas here.
    That is the single seam for it, and it must be recorded in results.json because OKS is
    not comparable across sigma sets.
    """
    from xtcocotools.coco import COCO as XCOCO
    from xtcocotools.cocoeval import COCOeval as XCOCOeval

    gt = XCOCO(str(gt_ann_file))
    n_gt = len(gt.getAnnIds(imgIds=img_ids, catIds=[1], iscrowd=None))
    if not predictions:
        return MetricBlock("keypoints", {n: 0.0 for n in KPT_STATS}, len(img_ids), 0, n_gt)
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(predictions)
        e = XCOCOeval(gt, dt, "keypoints", sigmas=sigmas, use_area=use_area)
        e.params.imgIds = list(img_ids)
        e.params.catIds = [1]
        e.evaluate()
        e.accumulate()
        e.summarize()
    return MetricBlock(
        "keypoints", _summarize(e.stats, KPT_STATS), len(img_ids), len(predictions), n_gt
    )
