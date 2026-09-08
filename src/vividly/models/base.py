"""Model-layer contracts for Vividly.

Every perception backend in this repository implements one of the four Protocols below.
The following rules hold for every backend, with no exceptions:

  * numpy in, numpy out. Images are HWC, uint8, BGR (OpenCV order). No torch tensors,
    no PIL, no framework objects cross this boundary.
  * Coordinates are absolute pixels in the coordinate frame of the image that was
    passed in. No normalisation, no letterbox padding leaking out.
  * Keypoints are COCO-17, in COCO order, always. A backend with a different skeleton
    converts before returning.
  * Appearance embeddings are L2-normalised float32 rows, (N, D), index-aligned with the
    crops that produced them, so similarity is a dot product and no caller has to know
    which distance a particular backend was trained with.
  * Batch calls return a list of the same length as the input, index-aligned.
  * Timing is not optional. `TimedModel.__call__` records a Timing for every call;
    subclasses implement `_forward` and never override `__call__`.
  * Provenance is not optional. Every backend carries a ModelProvenance describing
    what data its weights were trained on, so an evaluation row can never quote a number
    without saying which checkpoint produced it. One average precision value can belong to
    two different checkpoints, which is why POLICY.md, rule 10 requires the hash.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

COCO17_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
NUM_KEYPOINTS = 17

# The task vocabulary is closed: it grows by editing this line under review, never by a
# config string. There is no face task and there never will be one, under the ban in
# POLICY.md, rule 2; a face model cannot be described by this registry, which is the
# structural half of the deny-list enforcement.
# "appearance_embedding" was added for the session-scoped re-identification work that
# POLICY.md, section 4 authorises. It describes a colour and texture vector over a
# whole person crop and nothing else: no face branch, no face landmarks, no gait or body
# shape, all of which POLICY.md, section 4 excludes and none of which is expressible here.
Task = Literal["person_detection", "body_pose", "tracking", "appearance_embedding"]

# "counsel" means the provenance question is open (AIC-pretrained backbone, Objects365
# detector data). Only "yes" may be quoted as a shippable number. See POLICY.md rules 8
# and 10.
Shippability = Literal["yes", "no", "counsel"]


@dataclass(frozen=True)
class ModelProvenance:
    """Everything needed to reproduce and to defend a number."""

    name: str
    task: Task
    weights_uri: str  # URL or local path of the exact artifact
    sha256: str | None  # None only for not-yet-downloaded assets
    train_data: tuple[str, ...]  # e.g. ("COCO",) or ("body7",)
    pretrain_data: tuple[str, ...]  # e.g. ("AIC", "COCO") for pt-aic-coco
    code_license: str  # Apache-2.0 / MIT / AGPL-3.0
    shippable: Shippability
    input_size: tuple[int, int]  # (w, h) the backend feeds the network
    runtime: str  # "onnxruntime-cpu" | "onnxruntime-cuda" | "cpu-numpy"
    notes: str = ""

    def assert_shippable(self) -> None:
        if self.shippable != "yes":
            raise RuntimeError(
                f"{self.name}: shippable={self.shippable!r} "
                f"(train={self.train_data}, pretrain={self.pretrain_data}). "
                "This checkpoint may be used for eval reference numbers only."
            )

    def label(self) -> str:
        """Short string that goes in every results table cell group."""
        return f"{self.name}[train={'+'.join(self.train_data)};ship={self.shippable}]"


@dataclass(frozen=True)
class Timing:
    stage: str  # "detect" | "pose" | "track" | "embed" | "e2e"
    ms: float
    n_items: int  # images for detect, person crops for pose and embed, tracks for track
    backend: str

    @property
    def ms_per_item(self) -> float:
        return self.ms / self.n_items if self.n_items else float("nan")


@dataclass(frozen=True)
class Detections:
    """N person boxes in one image."""

    xyxy: np.ndarray  # (N, 4) float32, absolute pixels
    scores: np.ndarray  # (N,)   float32
    class_ids: np.ndarray  # (N,)   int32, always 0 for person here
    track_ids: np.ndarray | None = None  # (N,)   int32, set only by a Tracker

    def __post_init__(self) -> None:
        n = self.xyxy.shape[0]
        if self.xyxy.shape != (n, 4):
            raise ValueError(f"xyxy must be (N,4), got {self.xyxy.shape}")
        if self.scores.shape != (n,) or self.class_ids.shape != (n,):
            raise ValueError("scores/class_ids must be (N,)")
        if self.track_ids is not None and self.track_ids.shape != (n,):
            raise ValueError("track_ids must be (N,)")

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    @classmethod
    def empty(cls) -> Detections:
        return cls(
            xyxy=np.zeros((0, 4), np.float32),
            scores=np.zeros((0,), np.float32),
            class_ids=np.zeros((0,), np.int32),
        )

    def filter(self, min_score: float) -> Detections:
        keep = self.scores >= min_score
        return Detections(
            xyxy=self.xyxy[keep],
            scores=self.scores[keep],
            class_ids=self.class_ids[keep],
            track_ids=None if self.track_ids is None else self.track_ids[keep],
        )


@dataclass(frozen=True)
class Poses:
    """N person poses in one image, index-aligned with the boxes they came from."""

    keypoints: np.ndarray  # (N, 17, 2) float32, absolute pixels
    scores: np.ndarray  # (N, 17)    float32, per-keypoint conf
    boxes: np.ndarray  # (N, 4)     float32, source boxes (xyxy)

    def __post_init__(self) -> None:
        n = self.keypoints.shape[0]
        if self.keypoints.shape != (n, NUM_KEYPOINTS, 2):
            raise ValueError(f"keypoints must be (N,17,2), got {self.keypoints.shape}")
        if self.scores.shape != (n, NUM_KEYPOINTS):
            raise ValueError(f"scores must be (N,17), got {self.scores.shape}")
        if self.boxes.shape != (n, 4):
            raise ValueError(f"boxes must be (N,4), got {self.boxes.shape}")

    def __len__(self) -> int:
        return int(self.keypoints.shape[0])

    @classmethod
    def empty(cls) -> Poses:
        return cls(
            keypoints=np.zeros((0, NUM_KEYPOINTS, 2), np.float32),
            scores=np.zeros((0, NUM_KEYPOINTS), np.float32),
            boxes=np.zeros((0, 4), np.float32),
        )


class TimedModel:
    """Base class that makes latency measurement structural.

    Subclasses implement `_forward`; nobody overrides `__call__`. POLICY.md, rule 9 requires
    latency beside accuracy in every evaluation, and the only reliable way to get that is
    to make it impossible to run a model without producing a Timing.
    """

    provenance: ModelProvenance

    def __init__(self, provenance: ModelProvenance) -> None:
        self.provenance = provenance
        self._timings: list[Timing] = []

    # -- timing bookkeeping -------------------------------------------------
    @property
    def timings(self) -> list[Timing]:
        return list(self._timings)

    def reset_timings(self) -> None:
        self._timings.clear()

    def _record(self, stage: str, ms: float, n_items: int) -> None:
        self._timings.append(
            Timing(stage=stage, ms=ms, n_items=n_items, backend=self.provenance.runtime)
        )

    def _timed(self, stage: str, n_items: int, fn, *args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        self._record(stage, (time.perf_counter() - t0) * 1000.0, n_items)
        return out

    # -- warmup -------------------------------------------------------------
    def warmup(self, shape: tuple[int, int] = (720, 1280), n: int = 2) -> None:
        """Run `n` throwaway forwards so the first measured call is not the JIT/allocator
        cost. Timings produced during warmup are discarded."""
        blank = np.zeros((shape[0], shape[1], 3), np.uint8)
        for _ in range(n):
            self._warmup_once(blank)
        self.reset_timings()

    def _warmup_once(self, image: np.ndarray) -> None:  # pragma: no cover - trivial
        raise NotImplementedError


@runtime_checkable
class PersonDetector(Protocol):
    """Person box detector. Batch call; single-image call is a convenience wrapper."""

    provenance: ModelProvenance

    def detect(self, images: Sequence[np.ndarray]) -> list[Detections]: ...
    def detect_one(self, image: np.ndarray) -> Detections: ...
    def warmup(self, shape: tuple[int, int] = ..., n: int = ...) -> None: ...
    @property
    def timings(self) -> list[Timing]: ...
    def reset_timings(self) -> None: ...


@runtime_checkable
class PoseEstimator(Protocol):
    """Top-down pose. `boxes` is (N,4) xyxy in the frame of `image`.

    A bottom-up or one-stage model (RTMO) implements the same protocol and simply
    ignores `boxes` when it is empty; when boxes are supplied it must still return
    poses index-aligned to them, so the two families stay interchangeable.
    """

    provenance: ModelProvenance

    def estimate(self, image: np.ndarray, boxes: np.ndarray) -> Poses: ...
    def estimate_batch(
        self, images: Sequence[np.ndarray], boxes: Sequence[np.ndarray]
    ) -> list[Poses]: ...
    def warmup(self, shape: tuple[int, int] = ..., n: int = ...) -> None: ...
    @property
    def timings(self) -> list[Timing]: ...
    def reset_timings(self) -> None: ...


@runtime_checkable
class Tracker(Protocol):
    """Stateful, single-camera, online. `update` is called once per frame in order."""

    provenance: ModelProvenance

    def update(self, detections: Detections, frame_idx: int) -> Detections: ...
    def reset(self) -> None: ...
    @property
    def timings(self) -> list[Timing]: ...
    def reset_timings(self) -> None: ...


@runtime_checkable
class AppearanceEmbedder(Protocol):
    """One appearance vector per person crop, for the session-scoped re-identification
    scope that POLICY.md, section 4 defines.

    Stateless and order-free: `embed` is a pure function of the crops it is handed, so the
    same crop embeds to the same row whether it arrives in a batch of one or a batch of
    thirty. All identity state (galleries, thresholds, seat priors) lives above this layer.

    The contract:

      * `crops` are HxWx3 uint8 BGR person boxes at whatever size they were cut. Resizing,
        colour conversion and normalisation are the backend's job, because they are
        properties of the weights, not of the caller.
      * the return is (N, D) float32, L2-normalised row by row, index-aligned with `crops`.
        Cosine similarity is therefore `a @ b`, and no caller needs to know which distance
        a particular checkpoint was trained under.
      * an empty batch returns a (0, D) array, never a bare (0,) one, so a frame with no
        detections still multiplies cleanly against a gallery matrix.
      * `embedding_dim` is D, known before the first call, so a gallery can be allocated.

    Guardrails this protocol inherits and cannot be used to escape:

      * The input is a whole-person box, under the facial-recognition ban in POLICY.md
        rule 2. There is no face crop, no face branch, no landmark input, and `Task` has
        no member that could describe one.
      * `provenance.assert_shippable()` behaves exactly as it does for the other three
        protocols: anything that is not "yes" raises, and an embedding produced by such a
        checkpoint may appear in a benchmark row only. Under the weight-provenance rule in
        POLICY.md, rule 11, every LUPerson-derived weight (SOLIDER, TransReID-SSL and
        related) is "no" permanently; see docs/LICENCE_ALLOWLIST.md, re-ID section.
    """

    provenance: ModelProvenance

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray: ...
    @property
    def embedding_dim(self) -> int: ...
    def warmup(self, shape: tuple[int, int] = ..., n: int = ...) -> None: ...
    @property
    def timings(self) -> list[Timing]: ...
    def reset_timings(self) -> None: ...
