"""Env smoke tests for the development and curate forks.

These assert the three latent failure paths that a happy-path import test misses:
xtcocotools' numpy-ABI import crash, the ByteTrack wrapper against the trackers 2.6.0
API, and TrackEval's zero-detection branch that took down eval sweeps under the
bitrotten upstream. The fork layout the tests hold the environment to is in CLAUDE.md,
environments, and in POLICY.md, toolchain rules.
"""

import importlib.util

import numpy as np
import pytest


def test_no_mm_or_torch_in_dev_env() -> None:
    # The dev/mac fork must never contain the CUDA training stack: POLICY.md, rule 5 names
    # ultralytics and boxmot, and POLICY.md, section 6 makes the GPU extra a no-op on macOS.
    assert importlib.util.find_spec("torch") is None, "torch leaked into the dev env"
    assert importlib.util.find_spec("mmcv") is None, "mmcv leaked into the dev env"
    assert importlib.util.find_spec("ultralytics") is None, "AGPL dep leaked into the dev env"
    assert importlib.util.find_spec("boxmot") is None, "AGPL dep leaked into the dev env"


def test_xtcocotools_imports_under_numpy2() -> None:
    # The published wheel is numpy-1 ABI and dies at import under numpy 2; the
    # global no-binary source build is the fix. This import is the assertion.
    # (A bare `uv sync` is minimal by design; the analysis stack arrives with the mac group
    # on macOS and the linux-harness group on Linux.)
    pytest.importorskip("pycocotools")
    pytest.importorskip("xtcocotools")
    from pycocotools.coco import COCO  # noqa: F401
    from xtcocotools.cocoeval import COCOeval  # noqa: F401


def test_bytetrack_wrapper_matches_trackers_260_api() -> None:
    # The wrapper once passed sv.ByteTrack's `minimum_matching_threshold` to a library
    # that never had that parameter, and nothing constructed the wrapper until a live
    # pipeline run (the acceptance run of 2026-08-07) hit the TypeError. Constructing and
    # stepping it against the real trackers 2.6.0 API is the assertion.
    pytest.importorskip("trackers")
    pytest.importorskip("supervision")
    from vividly.models.base import Detections
    from vividly.models.tracker import ByteTrack

    tracker = ByteTrack(frame_rate=15.0)
    box = Detections(
        xyxy=np.array([[10.0, 10.0, 50.0, 80.0]], np.float32),
        scores=np.array([0.9], np.float32),
        class_ids=np.array([0], np.int32),
    )
    first = tracker.update(box, 0)
    assert len(first) == 0, (
        "frame 0 is unconfirmed (tracker_id -1 in trackers 2.6.0) and the wrapper must "
        "filter the sentinel, not hand it downstream as a real track id"
    )
    out = first
    for frame_idx in range(1, 3):
        out = tracker.update(box, frame_idx)
    assert out.track_ids is not None and len(out) == 1, (
        "a steady high-confidence box must hold one confirmed track by frame 3"
    )
    assert int(out.track_ids[0]) >= 0
    tracker.reset()
    assert len(tracker.update(Detections.empty(), 0)) == 0


def test_trackeval_empty_tracker_hota_path() -> None:
    # Upstream TrackEval crashes on np.float in the zero-detection early-return
    # branch; the pinned PyPI fork must return zeros instead.
    pytest.importorskip("trackeval")
    from trackeval.metrics import HOTA

    data = {
        "num_tracker_dets": 0,
        "num_gt_dets": 2,
        "num_gt_ids": 1,
        "num_tracker_ids": 0,
        "gt_ids": [np.array([0]), np.array([0])],
        "tracker_ids": [np.array([], dtype=int), np.array([], dtype=int)],
        "similarity_scores": [np.zeros((1, 0)), np.zeros((1, 0))],
    }
    hota = HOTA().eval_sequence(data)["HOTA"]
    assert np.allclose(hota, 0.0), f"empty-tracker HOTA path returned {hota[:3]}"
