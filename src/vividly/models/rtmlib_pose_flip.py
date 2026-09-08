"""Flip-only test-time augmentation for the rtmlib SimCC pose path.

Why this exists. The joints model of record's headline number (COCO val2017 AP 0.774115,
checkpoint `a0a035fe`, exported ONNX `6ba96285`,
`research/pose-seed-floor-and-champion.md`) was measured trainer-side: the mmpose
evaluation loop, ground-truth boxes, and `flip_test=True`. The artefact the event lane
actually consumes is the exported ONNX driven through `RtmlibPose`, which has neither
ground-truth boxes nor flip TTA. Two of the three differences are the deployed path's own
(detector boxes, ONNX numerics); the third is a switch, and this class is that switch, so
a deployed-path measurement can separate a missing flip TTA from everything else.

What it does, and why it matches mmpose rather than inventing something. The mmpose
`FlipTestMixin` / `flip_vectors` for a SimCC head is exactly four steps, in this order
(mmpose 1.3.2, `mmpose/models/utils/tta.py` + `RTMCCHead.predict`):

    1. run the network on the horizontally mirrored crop;
    2. permute the keypoint channels by `flip_indices` (COCO L/R pairs);
    3. reverse the X SimCC vector along its bin axis, and only the X one;
       the Y vector is channel-permuted but never reversed;
    4. average the two SimCC pairs elementwise, then argmax-decode once.

Averaging the SimCC vectors and decoding once is not the same thing as decoding twice and
averaging coordinates, which is the tempting shortcut: the SimCC argmax is a hard
`argmax`, so averaging coordinates would average two quantised integers, while averaging
the distributions lets a confident view pull a bin and a half. mmpose does the former,
and so does this class.

No half-bin shift. Heatmap-based flip TTA in mmpose applies `shift_heatmap`, a roll by one
pixel, to correct the mirror's off-by-one; the SimCC path does not, since `flip_vectors` is
a bare `.flip(-1)`, because the SimCC bin grid is built with the same
`x -> (W - 1 - x)` symmetry the mirror uses. Adding a shift here would be a silent
half-pixel bias against the trainer-side number this row exists to compare with.

Decoding stays outside the graph. `simcc_split_ratio`, the affine back-projection and the
per-keypoint confidence all come from rtmlib's own `postprocess`, unmodified, so the
no-flip and flip rows differ in exactly one thing.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from vividly.models.base import Poses
from vividly.models.rtmlib_backend import RtmlibPose

# COCO-17 left/right channel pairs, as the index permutation mmpose calls `flip_indices`:
# 0 nose is self-paired, then (eye, ear, shoulder, elbow, wrist, hip, knee, ankle) swap.
COCO17_FLIP_INDICES: tuple[int, ...] = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)


def flip_simcc(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    flip_indices: tuple[int, ...] = COCO17_FLIP_INDICES,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a mirrored crop's SimCC vectors back into the original crop's frame.

    `simcc_x` (N, K, Wx) and `simcc_y` (N, K, Wy) come from a network run on the
    horizontally mirrored crop. Returns the same pair re-expressed for the unmirrored
    crop: channels permuted through `flip_indices`, and the X vector reversed along its
    bin axis. Pure function with no model state, which is what the unit test pins.
    """
    if simcc_x.ndim != 3 or simcc_y.ndim != 3:
        raise ValueError(f"expected (N, K, W) SimCC vectors, got {simcc_x.shape} / {simcc_y.shape}")
    k = simcc_x.shape[1]
    if k != len(flip_indices):
        raise ValueError(f"flip_indices has {len(flip_indices)} entries for {k} keypoint channels")
    idx = list(flip_indices)
    return (
        np.ascontiguousarray(simcc_x[:, idx][:, :, ::-1]),
        np.ascontiguousarray(simcc_y[:, idx]),
    )


class RtmlibPoseFlipTTA(RtmlibPose):
    """`RtmlibPose` plus mmpose-equivalent horizontal-flip TTA. Two forwards per crop.

    Costs ~2x the pose stage's wall clock, which is why it is a separate backend key
    (`pose.backend: rtmlib_flip`) and not a flag on the default path: the deployed
    pipeline's latency budget is measured on the no-flip class.
    """

    def __init__(self, *args, flip_indices: tuple[int, ...] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flip_indices = tuple(flip_indices or COCO17_FLIP_INDICES)
        # A row must not be able to claim the no-flip path's identity. ModelProvenance is
        # frozen, so the note is re-stamped by rebuilding it with dataclasses.replace.
        self.provenance = replace(
            self.provenance,
            notes=(
                f"{self.provenance.notes}; flip-TTA (mmpose flip_vectors equivalent), "
                "2 forwards/crop"
            ).lstrip("; "),
        )

    def _forward_one(self, image: np.ndarray, boxes: np.ndarray) -> Poses:
        if len(boxes) == 0:
            return Poses.empty()
        m = self._m
        kpts_out, score_out = [], []
        for bbox in boxes.tolist():
            crop, center, scale = m.preprocess(image, bbox)
            simcc_x, simcc_y = m.inference(crop)
            # The mirror is applied to the PREPROCESSED crop, after normalisation. That is
            # equivalent to mirroring the raw crop (mean/std are per-channel scalars) and
            # it guarantees the two forwards see the identical affine box.
            flipped = np.ascontiguousarray(crop[:, ::-1])
            fx, fy = m.inference(flipped)
            fx, fy = flip_simcc(np.asarray(fx), np.asarray(fy), self.flip_indices)
            simcc_x = (np.asarray(simcc_x) + fx) * 0.5
            simcc_y = (np.asarray(simcc_y) + fy) * 0.5
            kpts, scores = m.postprocess([simcc_x, simcc_y], center, scale)
            kpts_out.append(kpts)
            score_out.append(scores)
        kpts_arr = np.asarray(np.concatenate(kpts_out, axis=0), np.float32)
        score_arr = np.asarray(np.concatenate(score_out, axis=0), np.float32)
        if kpts_arr.shape[1] != 17:
            raise ValueError(f"expected 17 COCO keypoints, got {kpts_arr.shape[1]}")
        return Poses(keypoints=kpts_arr, scores=score_arr, boxes=boxes.astype(np.float32))
