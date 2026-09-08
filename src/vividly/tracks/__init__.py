"""Track state layer for the replay pipeline (POLICY.md, section 1; recorded 2026-08-07).

Between the tracker and the rules. Pure stdlib: no numpy, no cv2, no model runtime, so it
imports cleanly on a Linux CI runner that has none of them installed. Thresholds live in
TrackConfig and remain PROVISIONAL until pilot shadow-mode data exists
(docs/CALIBRATION.md). The fine-tuned lane weights they also once waited on now exist and
are measured: final_models/eyes/README.md and final_models/joints/README.md.
"""

from __future__ import annotations

from vividly.tracks.state import (
    LEFT_HIP,
    LEFT_SHOULDER,
    MIN_TORSO_PX,
    NUM_KEYPOINTS,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    TORSO_KEYPOINTS,
    BaselineStats,
    Keypoints,
    TrackBook,
    TrackConfig,
    TrackIdentity,
    TrackObservation,
    TrackState,
    displacement_variance,
    posture_angle_deg,
)

__all__ = [
    "LEFT_HIP",
    "LEFT_SHOULDER",
    "MIN_TORSO_PX",
    "NUM_KEYPOINTS",
    "RIGHT_HIP",
    "RIGHT_SHOULDER",
    "TORSO_KEYPOINTS",
    "BaselineStats",
    "Keypoints",
    "TrackBook",
    "TrackConfig",
    "TrackIdentity",
    "TrackObservation",
    "TrackState",
    "displacement_variance",
    "posture_angle_deg",
]
