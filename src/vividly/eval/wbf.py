"""Weighted Boxes Fusion over several detection views of one frame (pure logic, no torch).

The one caller is `scripts/detector/fuse_tta_dumps.py`, the driver behind the I1 ensemble
and test-time-augmentation row of `research/eyes-instrument-protocol.md`: it fuses the
identity and horizontally flipped detection dumps of one checkpoint into one dump, at
`WBFParams()` exactly as declared there. Every input is a view of the same frame, and
each view is a `RawDetections` with a `source` name.

The load-bearing conventions, each pinned by `tests/test_wbf.py`:

- ensemble-boxes' WBF consumes normalized [0,1] xyxy and produces wrong output for
  pixel input without complaining; this module normalizes on the way in and
  denormalizes on the way out, so callers live entirely in pixels.
- A view's boxes are clipped to the frame before fusion, and a box that collapses to
  nothing under clipping is dropped together with its score.
- A view that emits no scores is fused only under a constant-score calibration; every
  other calibration mode refuses it rather than fabricating a per-box value.
- `conf_type='avg'` rescales a cluster's score by the weight of the views that
  contributed over the total weight, so a box seen by one of two views keeps half its
  score; a fixed cut on fused scores is therefore not the operating point the same cut
  gives on one view. A view with zero boxes still occupies its weighted slot, so that
  rescaling is not silently inflated when one view goes blind on a frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class FuseError(ValueError):
    """The views disagree with the declared calibration or weights."""


@dataclass
class RawDetections:
    """One view's output for one frame: pixel xyxy boxes, optional scores."""

    source: str
    boxes: list[list[float]]
    scores: list[float] | None = None  # None: the view emits no scores


@dataclass(frozen=True)
class ScoreCal:
    """Per-view score calibration.

    mode 'identity': scores pass through (refused if the view emits none);
    mode 'affine':   score * scale + bias, clamped to [0, 1];
    mode 'constant': every box gets `constant`, the only mode legal for a scoreless view.
    """

    mode: str = "identity"
    scale: float = 1.0
    bias: float = 0.0
    constant: float = 0.55


@dataclass(frozen=True)
class WBFParams:
    """Fusion knobs. iou_thr 0.55 is the WBF paper's default cluster threshold;
    skip_box_thr drops sub-threshold boxes before clustering; conf_type 'avg'
    is the reference behavior."""

    iou_thr: float = 0.55
    skip_box_thr: float = 0.05
    conf_type: str = "avg"
    weights: dict[str, float] = field(default_factory=dict)  # per-view, default 1.0


def clip_boxes(boxes: list[list[float]], width: int, height: int) -> list[list[float]]:
    """Clip pixel xyxy to the frame; drops boxes that collapse to nothing."""
    out: list[list[float]] = []
    for x1, y1, x2, y2 in boxes:
        cx1 = min(max(0.0, float(x1)), float(width))
        cy1 = min(max(0.0, float(y1)), float(height))
        cx2 = min(max(0.0, float(x2)), float(width))
        cy2 = min(max(0.0, float(y2)), float(height))
        if cx2 - cx1 > 0 and cy2 - cy1 > 0:
            out.append([cx1, cy1, cx2, cy2])
    return out


def calibrate_scores(det: RawDetections, cal: ScoreCal) -> list[float]:
    if cal.mode == "constant":
        return [cal.constant] * len(det.boxes)
    if det.scores is None:
        raise FuseError(
            f"{det.source}: the view emits no scores but calibration mode is "
            f"{cal.mode!r}; only 'constant' is legal for it"
        )
    if len(det.scores) != len(det.boxes):
        raise FuseError(f"{det.source}: {len(det.boxes)} boxes vs {len(det.scores)} scores")
    if cal.mode == "identity":
        return [float(s) for s in det.scores]
    if cal.mode == "affine":
        return [min(1.0, max(0.0, float(s) * cal.scale + cal.bias)) for s in det.scores]
    raise FuseError(f"unknown calibration mode {cal.mode!r}")


def fuse_wbf(
    detections: list[RawDetections],
    width: int,
    height: int,
    calibration: dict[str, ScoreCal],
    params: WBFParams,
) -> tuple[list[list[float]], list[float]]:
    """Fuse per-view pixel detections into consensus pixel boxes plus scores.

    Views with zero boxes still occupy an ensemble slot (an empty list with their
    weight), so the cluster rescaling is not silently inflated when one view goes
    blind on a frame.
    """
    if width <= 0 or height <= 0:
        raise FuseError(f"bad frame size {width}x{height}")
    boxes_list: list[list[list[float]]] = []
    scores_list: list[list[float]] = []
    labels_list: list[list[int]] = []
    weights: list[float] = []
    for det in detections:
        if det.source not in calibration:
            raise FuseError(f"{det.source}: no calibration entry")
        all_scores = calibrate_scores(det, calibration[det.source])
        clipped: list[list[float]] = []
        scores: list[float] = []
        for box, score in zip(det.boxes, all_scores, strict=True):
            cb = clip_boxes([box], width, height)
            if cb:  # clipping drops degenerate boxes; the score must follow its box
                clipped.append(cb[0])
                scores.append(score)
        boxes_list.append(
            [[x1 / width, y1 / height, x2 / width, y2 / height] for x1, y1, x2, y2 in clipped]
        )
        scores_list.append(scores)
        labels_list.append([0] * len(clipped))
        weights.append(float(params.weights.get(det.source, 1.0)))
    if not any(boxes_list):
        return [], []

    from ensemble_boxes import weighted_boxes_fusion  # numpy-only path; import here

    fused_boxes, fused_scores, _ = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights,
        iou_thr=params.iou_thr,
        skip_box_thr=params.skip_box_thr,
        conf_type=params.conf_type,
    )
    pixel = [
        [float(x1) * width, float(y1) * height, float(x2) * width, float(y2) * height]
        for x1, y1, x2, y2 in fused_boxes
    ]
    return pixel, [float(s) for s in fused_scores]
