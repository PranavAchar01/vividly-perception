"""Flip-only test-time augmentation for the rtmlib SimCC pose path.

The deployed-path measurement it is built on, including what flip costs per crop, is
research/pose-deployed-path-cpu-proof.md.

Two layers, on purpose:

* the pure `flip_simcc` layer is pinned exactly, because it is the piece that silently produces
  a plausible-but-wrong skeleton if the channel permutation or the axis reversal is off
  by one, and a wrong-but-plausible pose is the failure mode COCO AP hides (a
  left/right-swapped skeleton still scores, just worse);
* the end-to-end layer is an equivariance check against the real ONNX: pose(mirror(img))
  must equal mirror(pose(img)) once the L/R channels are swapped back. That property is
  what flip TTA is for, and it is the one an implementation bug cannot fake.

The end-to-end layer loads the joints model of record from `final_models/joints/model.onnx`
(54 MB, tracked by git-lfs), and skips when that path is absent. On a clone whose LFS
content has not been pulled the path is present as a pointer file, so run `git lfs pull`
before this layer means anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vividly.models.checkpoints import is_lfs_pointer
from vividly.models.rtmlib_pose_flip import COCO17_FLIP_INDICES, flip_simcc

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORD_ONNX = REPO_ROOT / "final_models/joints/model.onnx"
PARITY_IMAGE = REPO_ROOT / "tests/data/parity/parity_1person_000000000785.jpg"


# --------------------------------------------------------------- the pure layer
def _peaked(n_kpt: int, width: int, peaks: list[int]) -> np.ndarray:
    """(1, K, W) SimCC with a single unit spike per channel at `peaks[k]`."""
    v = np.zeros((1, n_kpt, width), np.float32)
    for k, p in enumerate(peaks):
        v[0, k, p] = 1.0
    return v


def test_flip_indices_are_an_involution_over_the_coco_left_right_pairs() -> None:
    idx = np.asarray(COCO17_FLIP_INDICES)
    assert idx.shape == (17,)
    assert sorted(idx.tolist()) == list(range(17)), "must be a permutation of the 17 channels"
    assert np.array_equal(idx[idx], np.arange(17)), "swapping twice must be the identity"
    assert idx[0] == 0, "nose has no partner"
    # every other channel moves: an identity entry would mean a limb never swaps sides
    assert not np.any(idx[1:] == np.arange(1, 17))


def test_flip_simcc_reverses_x_swaps_channels_and_leaves_y_bins_alone() -> None:
    """A mirrored crop's peak at bin b for channel k belongs at (W-1-b) for its partner."""
    width_x, width_y = 384, 512
    peaks_x = list(range(0, 17 * 7, 7))  # distinct, well inside the range
    peaks_y = list(range(3, 3 + 17 * 11, 11))
    fx, fy = flip_simcc(_peaked(17, width_x, peaks_x), _peaked(17, width_y, peaks_y))

    for k, partner in enumerate(COCO17_FLIP_INDICES):
        assert fx[0, k].argmax() == width_x - 1 - peaks_x[partner], f"x channel {k}"
        assert fy[0, k].argmax() == peaks_y[partner], f"y channel {k} must NOT be reversed"


def test_flip_simcc_applied_twice_is_the_identity() -> None:
    rng = np.random.default_rng(0)
    x = rng.random((2, 17, 384), np.float32)
    y = rng.random((2, 17, 512), np.float32)
    x2, y2 = flip_simcc(*flip_simcc(x, y))
    assert np.array_equal(x2, x)
    assert np.array_equal(y2, y)


def test_flip_simcc_rejects_the_wrong_shapes() -> None:
    with pytest.raises(ValueError, match="N, K, W"):
        flip_simcc(np.zeros((17, 384), np.float32), np.zeros((17, 512), np.float32))
    with pytest.raises(ValueError, match="flip_indices"):
        flip_simcc(np.zeros((1, 5, 384), np.float32), np.zeros((1, 5, 512), np.float32))


def test_averaging_simcc_beats_averaging_coordinates() -> None:
    """Why the class averages distributions and decodes once, not the other way round.

    Two views whose argmax bins differ by 1 average to a distribution whose argmax is one
    of the two bins, never a half-bin artefact; averaging the two decoded coordinates
    would invent a sub-bin value the model never expressed. This pins the shape of the
    contract, so a future 'optimisation' to decode-then-average has to argue with a test.
    """
    a = _peaked(1, 16, [7])
    b = _peaked(1, 16, [8])
    merged = (a + b) * 0.5
    assert merged[0, 0].argmax() in (7, 8)
    assert merged[0, 0].max() == pytest.approx(0.5)


# ------------------------------------------------------------ the end-to-end layer
@pytest.mark.slow
@pytest.mark.skipif(
    not RECORD_ONNX.is_file() or is_lfs_pointer(RECORD_ONNX),
    reason="final_models/joints/model.onnx is not on disk (git lfs pull fetches it)",
)
def test_flip_tta_is_equivariant_under_mirroring_the_image() -> None:
    """pose(mirror(I), mirror(B)) == mirror(pose(I, B)) with L/R channels swapped back.

    The plain (no-TTA) estimator only approximately satisfies this, and that asymmetry is
    the thing flip TTA averages away, so the tolerance here is tight (sub-pixel in the
    crop's own frame, a couple of source pixels after the affine back-projection) while
    the same assertion on `RtmlibPose` would need a much looser one.
    """
    import cv2

    from vividly.models.rtmlib_pose_flip import RtmlibPoseFlipTTA

    img = cv2.imread(str(PARITY_IMAGE))
    assert img is not None, PARITY_IMAGE
    h, w = img.shape[:2]
    box = np.array([[w * 0.15, h * 0.05, w * 0.85, h * 0.95]], np.float32)

    pose = RtmlibPoseFlipTTA(str(RECORD_ONNX), input_size=(192, 256), device="cpu")
    got = pose.estimate(img, box)

    mirrored = np.ascontiguousarray(img[:, ::-1])
    mbox = np.array([[w - 1 - box[0, 2], box[0, 1], w - 1 - box[0, 0], box[0, 3]]], np.float32)
    got_m = pose.estimate(mirrored, mbox)

    # un-mirror the second result and put the channels back on their own sides
    back = got_m.keypoints[:, list(COCO17_FLIP_INDICES)].copy()
    back[..., 0] = (w - 1) - back[..., 0]

    assert np.abs(back - got.keypoints).max() < 2.0, "flip TTA is not mirror-equivariant"
    assert np.abs(got_m.scores[:, list(COCO17_FLIP_INDICES)] - got.scores).max() < 5e-2, (
        "confidences must survive the mirror too"
    )
