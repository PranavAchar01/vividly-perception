"""ONNX Runtime backend via rtmlib (Apache-2.0). No training framework is imported here.

Runs on the Mac development environment and on any CPU box. It is the runtime shape the
shipped path uses: the locally exported .onnx of the lane, substituted for the rtmlib zoo
asset, which is body7-trained and never shippable.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from rtmlib import RTMDet, RTMPose

from vividly.models.base import (
    Detections,
    ModelProvenance,
    Poses,
    TimedModel,
)

# Filename tokens that mean "these weights saw datasets beyond COCO".
_NON_COCO_TOKENS = ("body7", "body8", "halpe", "humanart", "ucoco", "cocktail", "obj365")


def _shippability(uri: str) -> tuple[str, tuple[str, ...]]:
    """Derive (shippable, train_data) from the artefact name. Filename tokens are the
    only provenance signal upstream provides, and one average precision value does not
    identify a checkpoint, per POLICY.md, rule 10, so the name is what gets recorded."""
    stem = Path(uri).name.lower()
    hits = tuple(t for t in _NON_COCO_TOKENS if t in stem)
    if not hits:
        return "counsel", ("COCO",)
    if "body7" in hits or "body8" in hits:
        return "no", ("body7",)
    return "no", hits


class RtmlibDetector(TimedModel):
    """RTMDet person detector over ONNX Runtime.

    rtmlib's own RTMDet.__call__ returns bboxes only and discards the scores, which
    makes COCO mAP impossible, because mAP needs a score per box to build the PR curve.
    The 12-line call is re-implemented here over rtmlib's own preprocess and inference so
    that the scores survive. Verified against rtmlib 0.0.16: the detector head emits
    (1, N, 5) as
    [x1, y1, x2, y2, score] in letterboxed coordinates, divided by `ratio` to get back
    to source pixels.
    """

    def __init__(
        self,
        onnx: str,
        input_size: tuple[int, int] = (320, 320),
        device: str = "cpu",
        backend: str = "onnxruntime",
        score_thr: float = 0.05,
        name: str = "rtmdet-nano-person-onnx",
        sha256: str | None = None,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError(
                f"device={device!r} rejected. CoreML ('mps') creates a session and then "
                "fails inside .run() on both the YOLOX and RTMDet graphs "
                "(onnxruntime 1.28.0, macOS 26.5): 'CoreML static output shape "
                "({1,1,1,2100,2100}) and inferred shape ({1,2100}) have different ranks'. "
                "Use device='cpu'."
            )
        ship, train = _shippability(onnx)
        super().__init__(
            ModelProvenance(
                name=name,
                task="person_detection",
                weights_uri=onnx,
                sha256=sha256,
                train_data=train,
                pretrain_data=("ImageNet-1k",),
                code_license="Apache-2.0",
                shippable=ship,
                input_size=input_size,
                runtime=f"onnxruntime-{device}",
                notes="rtmlib zoo asset" if not Path(onnx).exists() else "local onnx",
            )
        )
        self.score_thr = float(score_thr)
        self._m = RTMDet(
            onnx,
            model_input_size=input_size,
            backend=backend,
            device=device,
        )

    # -- core ---------------------------------------------------------------
    def _forward_one(self, image: np.ndarray) -> Detections:
        img, ratio = self._m.preprocess(image)
        out = self._m.inference(img)[0]
        if out.ndim != 3 or out.shape[-1] != 5:
            raise NotImplementedError(
                f"unexpected RTMDet output shape {out.shape}; expected (1, N, 5)"
            )
        boxes = out[0, :, :4] / ratio
        scores = out[0, :, 4]
        keep = scores >= self.score_thr
        boxes, scores = boxes[keep], scores[keep]
        h, w = image.shape[:2]
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, w - 1)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, h - 1)
        return Detections(
            xyxy=boxes.astype(np.float32),
            scores=scores.astype(np.float32),
            class_ids=np.zeros(len(scores), np.int32),
        )

    def detect(self, images: Sequence[np.ndarray]) -> list[Detections]:
        # ONNX graph is static batch-1; loop, but bill one Timing for the whole call.
        return self._timed("detect", len(images), lambda: [self._forward_one(im) for im in images])

    def detect_one(self, image: np.ndarray) -> Detections:
        return self.detect([image])[0]

    def _warmup_once(self, image: np.ndarray) -> None:
        self._forward_one(image)


class RtmlibPose(TimedModel):
    """RTMPose (SimCC) over ONNX Runtime. Top-down: needs boxes."""

    def __init__(
        self,
        onnx: str,
        input_size: tuple[int, int] = (192, 256),  # (w, h), the rtmlib order
        device: str = "cpu",
        backend: str = "onnxruntime",
        name: str = "rtmpose-m-onnx",
        sha256: str | None = None,
    ) -> None:
        ship, train = _shippability(onnx)
        super().__init__(
            ModelProvenance(
                name=name,
                task="body_pose",
                weights_uri=onnx,
                sha256=sha256,
                train_data=train,
                # Every published RTMPose carries the pt-aic-coco backbone pretrain, which
                # configs/checkpoints/registry.yaml decodes as AI Challenger plus COCO.
                pretrain_data=("AIC", "COCO"),
                code_license="Apache-2.0",
                shippable=ship,
                input_size=input_size,
                runtime=f"onnxruntime-{device}",
                notes=(
                    "body7 assets are eval-smoke only, never shippable "
                    "(POLICY.md, rule 8; configs/eval/smoke_mac.yaml)"
                ),
            )
        )
        self._m = RTMPose(
            onnx,
            model_input_size=input_size,
            backend=backend,
            device=device,
        )

    def _forward_one(self, image: np.ndarray, boxes: np.ndarray) -> Poses:
        if len(boxes) == 0:
            return Poses.empty()
        kpts, scores = self._m(image, bboxes=boxes.tolist())
        kpts = np.asarray(kpts, np.float32)
        scores = np.asarray(scores, np.float32)
        if kpts.shape[1] != 17:
            raise ValueError(f"expected 17 COCO keypoints, got {kpts.shape[1]}")
        return Poses(keypoints=kpts, scores=scores, boxes=boxes.astype(np.float32))

    def estimate(self, image: np.ndarray, boxes: np.ndarray) -> Poses:
        return self._timed("pose", max(len(boxes), 1), lambda: self._forward_one(image, boxes))

    def estimate_batch(
        self, images: Sequence[np.ndarray], boxes: Sequence[np.ndarray]
    ) -> list[Poses]:
        if len(images) != len(boxes):
            raise ValueError("images and boxes must be index-aligned")
        n = sum(len(b) for b in boxes)
        return self._timed(
            "pose",
            max(n, 1),
            lambda: [self._forward_one(i, b) for i, b in zip(images, boxes, strict=True)],
        )

    def _warmup_once(self, image: np.ndarray) -> None:
        h, w = image.shape[:2]
        self._forward_one(image, np.array([[0, 0, w - 1, h - 1]], np.float32))
