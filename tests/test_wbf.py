"""Box-fusion pins for `vividly.eval.wbf`: the pixel to normalized boundary, the clip
law, the no-score law, and WBF's cluster rescaling (a box seen by one of two views
keeps half its score). Uses the real ensemble-boxes WBF, because the reference
implementation is part of what is under test."""

import pytest

from vividly.eval.wbf import (
    FuseError,
    RawDetections,
    ScoreCal,
    WBFParams,
    calibrate_scores,
    clip_boxes,
    fuse_wbf,
)

W, H = 320, 240
CAL2 = {"a": ScoreCal(), "b": ScoreCal()}


def test_clip_boxes_reels_in_overhang() -> None:
    # a box past the frame bottom is clipped, a box past the left edge likewise
    boxes = clip_boxes([[10, 10, 50, 300], [-5, 4, 20, 30]], W, H)
    assert boxes == [[10, 10, 50, 240], [0, 4, 20, 30]]
    # a box entirely outside collapses and is dropped
    assert clip_boxes([[400, 10, 500, 30]], W, H) == []


def test_scoreless_view_requires_constant_mode() -> None:
    det = RawDetections(source="scoreless", boxes=[[1, 1, 9, 9]], scores=None)
    assert calibrate_scores(det, ScoreCal(mode="constant", constant=0.55)) == [0.55]
    with pytest.raises(FuseError, match="constant"):
        calibrate_scores(det, ScoreCal(mode="identity"))


def test_affine_calibration_clamps_to_unit() -> None:
    det = RawDetections(source="x", boxes=[[0, 0, 1, 1]] * 3, scores=[0.2, 0.9, 0.5])
    assert calibrate_scores(det, ScoreCal(mode="affine", scale=2.0, bias=0.0)) == [0.4, 1.0, 1.0]


def test_wbf_agreeing_views_fuse_to_one_pixel_box() -> None:
    # Two views, nearly identical boxes in pixels. If fuse_wbf forgot to
    # normalize, ensemble-boxes would clamp everything to [0,1] and the fused
    # box would come back microscopic, so the pixel-scale assertion below is
    # the normalization pin.
    a = RawDetections("a", [[100, 80, 180, 200]], [0.9])
    b = RawDetections("b", [[104, 84, 184, 196]], [0.8])
    boxes, scores = fuse_wbf([a, b], W, H, CAL2, WBFParams())
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    assert 100 <= x1 <= 104 and 80 <= y1 <= 84 and 180 <= x2 <= 184 and 196 <= y2 <= 200
    assert scores[0] == pytest.approx(0.85, abs=0.02)


def test_wbf_cluster_rescaling_halves_a_box_seen_by_one_of_two_views() -> None:
    # A box only one of two views saw has its score rescaled by cluster size over
    # ensemble size (0.9 to about 0.45), while the box both views saw keeps its
    # average. This is why a fixed cut on fused scores is not the same operating
    # point as the same cut on one view's scores.
    a = RawDetections("a", [[100, 80, 180, 200]], [0.9])
    b = RawDetections("b", [[102, 82, 182, 198], [10, 10, 40, 60]], [0.85, 0.9])
    boxes, scores = fuse_wbf([a, b], W, H, CAL2, WBFParams())
    assert len(boxes) == 2
    by_x = sorted(zip(boxes, scores, strict=True), key=lambda bs: bs[0][0])
    loner, shared = by_x[0], by_x[1]
    assert loner[0][0] == pytest.approx(10, abs=1)
    assert loner[1] == pytest.approx(0.45, abs=0.02)
    assert shared[0][0] == pytest.approx(101, abs=2)
    assert shared[1] > 0.5


def test_wbf_keeps_ensemble_slot_for_blind_view() -> None:
    # A view with zero boxes still occupies a weighted slot; the rescaling
    # denominator must not shrink when one view goes blind.
    a = RawDetections("a", [[100, 80, 180, 200]], [0.9])
    b = RawDetections("b", [], [])
    boxes, scores = fuse_wbf([a, b], W, H, CAL2, WBFParams())
    assert len(boxes) == 1
    assert scores[0] < 0.9  # rescaled by 1-of-2 cluster membership


def test_wbf_all_empty_short_circuits() -> None:
    a = RawDetections("a", [], [])
    b = RawDetections("b", [], [])
    assert fuse_wbf([a, b], W, H, CAL2, WBFParams()) == ([], [])


def test_missing_calibration_refused() -> None:
    det = RawDetections("mystery", [[1, 1, 9, 9]], [0.5])
    with pytest.raises(FuseError, match="calibration"):
        fuse_wbf([det], W, H, {}, WBFParams())


def test_bad_frame_size_refused() -> None:
    det = RawDetections("a", [[1, 1, 9, 9]], [0.5])
    with pytest.raises(FuseError, match="frame size"):
        fuse_wbf([det], 0, H, {"a": ScoreCal()}, WBFParams())
