"""Trackers via roboflow `trackers` 2.6.0 (Apache-2.0), the shippable trackers.

BoTSORT is the v2 shipped tracker (adopted in the 2026-08-31 review,
research/v2-training-stack.md section 7a); ByteTrack stays registered as the frozen
baseline and the comparison arm. The swap is not taste: the library's ByteTrack port has
no low-score floor in its second association pass (in `ByteTrackTracker.update`, the
low-confidence pass takes every detection under the high-confidence threshold, with no
floor beneath it), and with a raw DETR's 300-box tail that dragged a motionless occluded
person's track 515.6 px in the recorded measurement. The BoT-SORT port carries upstream's
`> 0.1` floor and only lets the low pass touch currently-tracked tracks.

Not sv.ByteTrack: deprecated in supervision 0.28.0, removed in 0.31.0.
Not boxmot: AGPL-3.0, so it is declared only in the eval-agpl extra, which the conflicts
table in pyproject.toml keeps apart from the cuda extra at any version.

Environment note: `trackers` requires numpy>=2.0.2, so this module is importable in the
development (mac) and `curate` environments and not in `cuda`. Tracking evaluation is
therefore a two-process job: the cuda environment dumps per-frame detections, and this
environment tracks and scores them.
"""

from __future__ import annotations

import numpy as np
import supervision as sv
from trackers import BoTSORTTracker, ByteTrackTracker
from trackers.core.botsort.tracklet import BoTSORTTracklet

from vividly.models.base import Detections, ModelProvenance, TimedModel


class ByteTrack(TimedModel):
    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_iou_threshold: float = 0.1,
        minimum_consecutive_frames: int = 2,
        high_conf_det_threshold: float = 0.6,
        frame_rate: float = 15.0,
    ) -> None:
        super().__init__(
            ModelProvenance(
                name="bytetrack-trackers-2.6.0",
                task="tracking",
                weights_uri="pkg:pypi/trackers@2.6.0",
                sha256=None,
                train_data=(),
                pretrain_data=(),
                code_license="Apache-2.0",
                shippable="yes",
                input_size=(0, 0),
                runtime="cpu-numpy",
                notes="frame_rate must match the ingest rate; waiting rooms are 15 fps",
            )
        )
        # These are trackers 2.6.0's own parameter names, passed through untranslated.
        # sv.ByteTrack's `minimum_matching_threshold` does not exist in this library; its
        # association gate is `minimum_iou_threshold` (accept a match when IoU exceeds it).
        self._kwargs = dict(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_iou_threshold=minimum_iou_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            high_conf_det_threshold=high_conf_det_threshold,
            frame_rate=frame_rate,
        )
        self._t = ByteTrackTracker(**self._kwargs)

    def update(self, detections: Detections, frame_idx: int) -> Detections:
        def _run() -> Detections:
            if len(detections) == 0:
                # Still step the tracker so lost-track ages advance on empty frames.
                self._t.update(sv.Detections.empty())
                return Detections.empty()
            dets = sv.Detections(
                xyxy=detections.xyxy.astype(np.float32),
                confidence=detections.scores.astype(np.float32),
                class_id=detections.class_ids.astype(int),
            )
            out = self._t.update(dets)
            if out.tracker_id is None or len(out) == 0:
                return Detections.empty()
            # trackers 2.6.0 marks a not-yet-confirmed track with tracker_id -1 (seen
            # fewer than minimum_consecutive_frames times). That is a sentinel, not an
            # identity: passed through, every unconfirmed detection in the room would
            # pool into one phantom track downstream.
            keep = out.tracker_id >= 0
            if not bool(keep.any()):
                return Detections.empty()
            return Detections(
                xyxy=np.asarray(out.xyxy[keep], np.float32),
                scores=np.asarray(out.confidence[keep], np.float32),
                class_ids=np.asarray(out.class_id[keep], np.int32),
                track_ids=np.asarray(out.tracker_id[keep], np.int32),
            )

        return self._timed("track", max(len(detections), 1), _run)

    def reset(self) -> None:
        self._t = ByteTrackTracker(**self._kwargs)


class BoTSORT(TimedModel):
    """The v2 shipped tracker, adopted in the 2026-08-31 review
    (research/v2-training-stack.md section 7a). Three deliberate departures from the
    library defaults, each carrying a measurement:

    - `enable_cmc` is refused, not defaulted off: camera-motion compensation needs the
      frame (this Tracker protocol is detection-only), the target cameras are statically
      mounted, and CMC costs a measured 7.0 ms/frame for nothing.
    - `lost_track_buffer` is counted in 30-fps frames, so seconds equals value/30 whatever
      `frame_rate` says, because trackers 2.6.0 base.py normalizes by frame_rate/30. The
      library default of 30 is one second of identity memory; a waiting room where people
      sit still for minutes needs 1800 (60 s) to 9000 (300 s). Default here is 1800;
      the burn-in prices the re-attachment risk of going higher.
    - `velocity_damping` ports OC-SORT's stationary fix (process-noise velocity block
      x0.01), the entire mechanism behind its measured 3.5x lower drift on motionless
      people. The tracklet rebuilds Q from `_SIGMA_V` every frame, so the damping is
      stamped per-instance after each update (sigma scales by sqrt of the variance
      factor). 1.0 disables.

    `score_floor` drops detections below it before the tracker sees them, a second line
    of defence over the port's own 0.1 low-pass floor, and the seam that cuts the 300-box
    tail of the shipped detector, RT-DETRv4-S person v2 seed 0 (checkpoint 345c30a5, ONNX
    40262049, factory key `rtdetr_onnx`). Per-class dedup of that head's duplicate top-k
    boxes belongs in the detector backend, not here.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 1800,
        minimum_iou_threshold_first_assoc: float = 0.2,
        minimum_iou_threshold_second_assoc: float = 0.5,
        minimum_iou_threshold_unconfirmed_assoc: float = 0.3,
        minimum_consecutive_frames: int = 2,
        high_conf_det_threshold: float = 0.6,
        frame_rate: float = 15.0,
        enable_cmc: bool = False,
        score_floor: float = 0.1,
        velocity_damping: float = 0.01,
    ) -> None:
        if enable_cmc:
            raise ValueError(
                "enable_cmc is not supported: CMC needs the raw frame, the Vividly "
                "Tracker protocol is detection-only, the cameras are statically mounted, "
                "and CMC costs a measured 7.0 ms/frame "
                "(research/v2-training-stack.md section 7a)"
            )
        if not 0.0 <= score_floor < 1.0:
            raise ValueError(f"score_floor must be in [0, 1), got {score_floor}")
        if not 0.0 < velocity_damping <= 1.0:
            raise ValueError(f"velocity_damping must be in (0, 1], got {velocity_damping}")
        super().__init__(
            ModelProvenance(
                name="botsort-trackers-2.6.0",
                task="tracking",
                weights_uri="pkg:pypi/trackers@2.6.0",
                sha256=None,
                train_data=(),
                pretrain_data=(),
                code_license="Apache-2.0",
                shippable="yes",
                input_size=(0, 0),
                runtime="cpu-numpy",
                notes=(
                    "cmc off (static cameras); lost_track_buffer in 30-fps frames "
                    "(seconds = value/30); OC-SORT velocity damping via per-tracklet "
                    "sigma stamp; score floor ahead of the tracker"
                ),
            )
        )
        # trackers 2.6.0's own parameter names, passed through untranslated (the BoT-SORT
        # port splits the association gate into three IoU thresholds; there is no single
        # minimum_iou_threshold here).
        self._kwargs = dict(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_iou_threshold_first_assoc=minimum_iou_threshold_first_assoc,
            minimum_iou_threshold_second_assoc=minimum_iou_threshold_second_assoc,
            minimum_iou_threshold_unconfirmed_assoc=minimum_iou_threshold_unconfirmed_assoc,
            minimum_consecutive_frames=minimum_consecutive_frames,
            high_conf_det_threshold=high_conf_det_threshold,
            frame_rate=frame_rate,
            enable_cmc=False,
        )
        self._score_floor = float(score_floor)
        if velocity_damping < 1.0:
            base_sigma = float(BoTSORTTracklet._SIGMA_V)
            self._damped_sigma_v: float | None = base_sigma * float(velocity_damping) ** 0.5
        else:
            self._damped_sigma_v = None
        self._t = BoTSORTTracker(**self._kwargs)

    def _stamp_velocity_damping(self) -> None:
        """Q is rebuilt from `self._SIGMA_V` on every tracklet step, so an instance
        attribute set once keeps every subsequent rebuild damped, including tracklets
        created inside the update call that just returned."""
        if self._damped_sigma_v is None:
            return
        for tracklet in self._t.tracks:
            if self._damped_sigma_v != tracklet._SIGMA_V:
                tracklet._SIGMA_V = self._damped_sigma_v

    def update(self, detections: Detections, frame_idx: int) -> Detections:
        def _run() -> Detections:
            dets = detections
            if len(dets) > 0 and self._score_floor > 0.0:
                keep = dets.scores >= self._score_floor
                if not bool(keep.any()):
                    dets = Detections.empty()
                elif not bool(keep.all()):
                    dets = Detections(
                        xyxy=dets.xyxy[keep],
                        scores=dets.scores[keep],
                        class_ids=dets.class_ids[keep],
                    )
            if len(dets) == 0:
                # Still step the tracker so lost-track ages advance on empty frames.
                self._t.update(sv.Detections.empty())
                self._stamp_velocity_damping()
                return Detections.empty()
            out = self._t.update(
                sv.Detections(
                    xyxy=dets.xyxy.astype(np.float32),
                    confidence=dets.scores.astype(np.float32),
                    class_id=dets.class_ids.astype(int),
                )
            )
            self._stamp_velocity_damping()
            if out.tracker_id is None or len(out) == 0:
                return Detections.empty()
            # Same sentinel rule as ByteTrack above: -1 marks a not-yet-confirmed track.
            keep = out.tracker_id >= 0
            if not bool(keep.any()):
                return Detections.empty()
            return Detections(
                xyxy=np.asarray(out.xyxy[keep], np.float32),
                scores=np.asarray(out.confidence[keep], np.float32),
                class_ids=np.asarray(out.class_id[keep], np.int32),
                track_ids=np.asarray(out.tracker_id[keep], np.int32),
            )

        return self._timed("track", max(len(detections), 1), _run)

    def reset(self) -> None:
        self._t = BoTSORTTracker(**self._kwargs)
