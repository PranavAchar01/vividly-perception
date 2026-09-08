"""Config to model. The registry is closed: an unknown key is an error, never a
dynamic import of an arbitrary module path, which would be a hole straight through
the AGPL and face-biometric deny lists."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vividly.models.base import AppearanceEmbedder, PersonDetector, PoseEstimator, Tracker


def _rtmlib_det(**kw) -> PersonDetector:
    from vividly.models.rtmlib_backend import RtmlibDetector

    return RtmlibDetector(**kw)


def _rtdetr_onnx_det(**kw) -> PersonDetector:
    # The v2 eyes lane: a locally exported RT-DETRv4 .onnx (two graph inputs, three
    # outputs, PIL-space preprocessing). Separate key rather than a flag on `rtmlib`
    # because the pre/post is a different pipeline, not a different weight.
    from vividly.models.rtdetr_backend import RTDETRv4Detector

    return RTDETRv4Detector(**kw)


def _rtmlib_pose(**kw) -> PoseEstimator:
    from vividly.models.rtmlib_backend import RtmlibPose

    return RtmlibPose(**kw)


def _rtmlib_pose_flip(**kw) -> PoseEstimator:
    # Same ONNX, same decode, one extra mirrored forward per crop. Separate key rather
    # than a flag so a results row cannot silently change cost class (~2x pose latency).
    from vividly.models.rtmlib_pose_flip import RtmlibPoseFlipTTA

    return RtmlibPoseFlipTTA(**kw)


def _bytetrack(**kw) -> Tracker:
    from vividly.models.tracker import ByteTrack  # requires numpy>=2

    return ByteTrack(**kw)


def _botsort(**kw) -> Tracker:
    from vividly.models.tracker import BoTSORT  # requires numpy>=2

    return BoTSORT(**kw)


def _onnx_embedder(**kw) -> AppearanceEmbedder:
    from vividly.models.reid_backend import OnnxAppearanceEmbedder  # requires onnxruntime

    return OnnxAppearanceEmbedder(**kw)


DETECTORS: dict[str, Callable[..., PersonDetector]] = {
    "rtmlib": _rtmlib_det,
    "rtdetr_onnx": _rtdetr_onnx_det,
}
POSE: dict[str, Callable[..., PoseEstimator]] = {
    "rtmlib": _rtmlib_pose,
    "rtmlib_flip": _rtmlib_pose_flip,
}
# botsort is the shipped tracker for the top-view lane; bytetrack stays as the frozen
# baseline and the comparison arm in tracking evaluations.
TRACKERS: dict[str, Callable[..., Tracker]] = {"bytetrack": _bytetrack, "botsort": _botsort}
# Re-identification appearance embedders. The table is as closed as the others:
# the licence traps here are the reason a weight can never arrive by config string alone,
# and the backend refuses a LUPerson-derived lineage on top of that.
EMBEDDERS: dict[str, Callable[..., AppearanceEmbedder]] = {"onnx": _onnx_embedder}


def _build(table: dict[str, Callable[..., Any]], spec: dict, what: str):
    key = spec.get("backend")
    if key not in table:
        raise KeyError(
            f"unknown {what} backend {key!r}; available: {sorted(table)}. "
            "Backends are registered in src/vividly/models/factory.py. Adding one is "
            "a code review, not a config change."
        )
    kwargs = {k: v for k, v in spec.items() if k != "backend"}
    for k in ("input_size", "det_input_size", "pose_input_size"):
        if k in kwargs and isinstance(kwargs[k], list):
            kwargs[k] = tuple(kwargs[k])
    return table[key](**kwargs)


def build_detector(spec: dict) -> PersonDetector:
    return _build(DETECTORS, spec, "detector")


def build_pose(spec: dict) -> PoseEstimator:
    return _build(POSE, spec, "pose")


def build_tracker(spec: dict | None) -> Tracker | None:
    return None if not spec else _build(TRACKERS, spec, "tracker")


def build_embedder(spec: dict | None) -> AppearanceEmbedder | None:
    """None when no embedder is configured. Re-identification is optional in every
    pipeline, and nothing it produces enters the shipped path before counsel clears it."""
    return None if not spec else _build(EMBEDDERS, spec, "embedder")
