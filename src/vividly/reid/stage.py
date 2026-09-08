"""The optional re-ID stage: it enrols departed tracks, probes new ones, and asks.

This module is the wiring between three things that already exist and deliberately do not
know about each other: `vividly.tracks.TrackBook` (which owns the automatic association
window), `vividly.reid.gallery.SessionGallery` (which scores a probe against the people
seen this visit and binds nothing), and `vividly.events.EventStore` (which holds the
question and the human's answer at file version 3: run-scoped proposals plus the
confirmed-row unique indexes, both of which this module depends on). POLICY.md, section 4,
re-identification scope, sets the scope; the replay stack is what puts it in the tree at
all.

The two regimes, restated here because this file is the seam between them
=========================================================================

  1. Inside the tracker's own re-association window, nothing here runs and nothing here
     changes. `trackers.BoTSORTTracker` re-issues the same track id across an occlusion
     (`lost_track_buffer` counts 30-fps frames whatever `frame_rate` says, so the shipped
     1800 is 60 s), and inside `TrackConfig.reap_after_s` (PROVISIONAL 30 s) the TrackBook
     keeps the same TrackState alive, so a person who steps out and comes back keeps their
     id, their baseline, their staff tag and their mock chart binding. That is unchanged,
     it is automatic, and it is what a run with re-ID switched off still does.

  2. Beyond that window there is no automatic binding. A track that has been reaped is
     enrolled into the session gallery under its own track id. A track born afterwards is
     probed against it, and a candidate at or above tau becomes a pending row in
     `reassoc_proposals`. Nothing else happens. The new track keeps its own new id, its
     own empty chart binding and its own blank slate until a human presses Confirm, and
     the confirmation is what calls `TrackBook.merge_identity`. A rejection is written
     down too, and both answers are labels (`EventStore.labels_as_rows`).

A false merge joins two patients' identities, which in a chart-bound deployment is a
patient-safety event rather than an accuracy point, so the machine is not permitted to make
that call. That asymmetry is the whole design, and every question in this file about why
a step is not automatic resolves to it.

When the stage runs at all
==========================

Never, unless a pipeline config both enables it and names a source of embeddings. With no
`reid:` block, with `enabled: false`, or with `embeddings.source: none`, `build_reid_stage`
returns None and the pipeline is byte-for-byte the pipeline that shipped before this file
existed. `tests/test_reid_pipeline.py` pins that against a recorded alert stream, because
being optional is a claim about behaviour and not about a flag.

Where embeddings come from
==========================

Pluggable, per config, and there are exactly two sources.

  * `synthetic` draws from a `vividly.reid.synth` identity bank keyed by the fixture's
    own ground-truth identity, declared in the config as `identity_by_track`. Deterministic,
    zero footage, no weights, no cv2, no network: this is the flow that runs in CI and the
    one to demo on a laptop that has downloaded nothing. It proves the wiring and it proves
    nothing whatsoever about re-identification (read `vividly.reid.synth`'s docstring).
  * `embedder` cuts the crop at each tracked box out of the replayed frame and pushes it
    through an `AppearanceEmbedder` built by `vividly.models.factory` (today: the ONNX
    OSNet backend). The factory is the only place a backend is named, and it refuses
    LUPerson-derived lineages outright.

A track with no declared identity under the synthetic source contributes nothing and is
counted in `ReidStats.n_unmapped`. Guessing an identity for it would be fabricating the one
thing the fixture is supposed to supply.

Why the stage is not one of the six timed stages
================================================

`scripts/run_pipeline.py` times six stages whose names are fixed by the runner's stage
contract (source, detect, pose, track, rules, render) and shared with the dashboard's
`STAGES` tuple. Re-ID is none of them. Folding it into `rules` would make the rules row
describe work the rules did not do, and adding a seventh name would break a contract
another surface reads. It is measured on its own instead: wall milliseconds in
`ReidStats`, plus the gallery's own `propose` latency window with the gallery size beside
it, which is the only form in which a match cost can be read. Under POLICY.md, rule 9, none
of it is a platform number.

Thresholds. Every number in `ReidConfig` and `GalleryConfig` is PROVISIONAL and lives in
configs/pipeline/reid.yaml with the reason written next to it. Not one has been fitted: no
deployment-domain embedding-score distribution has been measured, and no pilot shadow data
exists to fit against. docs/CALIBRATION.md supplies the reasoning and scopes itself to
configs/pipeline/rules.yaml, so it contains no re-identification procedure;
`vividly.reid.calibrate` is the only tool that picks tau at a stated false-accept budget
today.

Session scope. `close()` clears the gallery, the per-track embedding buffers and the reaped
identity snapshots, and `scripts/run_pipeline.py` calls it on every exit path before it
prints anything. CQ-16 in docs/COUNSEL_QUESTIONS.md cites the gallery for that property;
this module is the thing that has to actually call it.

Imports. numpy at module level, which is this package's convention (`vividly.reid.gallery`,
`vividly.reid.metrics`, `vividly.models.base` all do it). cv2 and onnxruntime appear
nowhere, not even lazily: cropping is array slicing and the embedder owns its own runtime.
`vividly.models.factory` is imported inside the builder so that a synthetic run never pays
for the model package at all.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from vividly.events import ConfirmConflictError, DuplicateProposalError, EventStore
from vividly.reid.gallery import GalleryConfig, Proposal, SessionGallery
from vividly.reid.synth import make_identity_bank, sample_embeddings
from vividly.schemas.reassoc import REASSOC_DECISIONS, ReassocProposalRecord
from vividly.tracks import TrackBook, TrackIdentity, TrackObservation, TrackState

#: The embedding sources this stage understands. Closed, like every other registry in this
#: repository: an unknown name is an error, never a dynamic import of a module path.
EMBEDDING_SOURCES: tuple[str, ...] = ("none", "synthetic", "embedder")

#: Written into the run summary so a screenshot of a re-ID run always carries the caveat.
PROVISIONAL_NOTE = (
    "every re-ID threshold in this run is PROVISIONAL (docs/CALIBRATION.md) and no "
    "count here is a false-merge rate"
)


class ReidConfigError(ValueError):
    """The `reid:` block is malformed, or asks for a source that cannot be built."""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def _check_keys(data: Mapping[str, Any], known: Iterable[str], what: str) -> None:
    """Unknown keys raise. Same stance as TrackConfig and GalleryConfig, same reason: a
    typo in a threshold name that silently kept the default is the failure this project
    cannot afford."""
    allowed = set(known)
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ReidConfigError(f"unknown {what} keys: {unknown}; known keys are {sorted(allowed)}")


@dataclass(frozen=True)
class EmbeddingsSpec:
    """Where this run's appearance vectors come from, and nothing about what to do with them.

    `frozen` guards rebinding; the two mapping fields are copied in `__post_init__`, so a
    caller that keeps editing its own dict cannot change what this run actually used.
    """

    #: One of EMBEDDING_SOURCES. "none" is the default and switches the whole stage off.
    source: str = "none"

    # -- synthetic source ---------------------------------------------------
    #: Embedding width for the synthetic bank. Sizing, not a threshold; it must equal the
    #: gallery's `dim`, and the builder refuses a mismatch rather than reshaping anything.
    dim: int = 512

    # PROVISIONAL and invented (the re-identification scope, POLICY.md;
    # docs/CALIBRATION.md, and the vividly.reid.synth docstring): the cone half-angle that
    # sets how far apart two synthetic identities sit. This is a property of a generated
    # fixture, not of people. No value of it is evidence about anything, and a demo tuned so
    # the right answer falls out is a demo of wiring.
    separability: float = 0.8

    # PROVISIONAL and invented: per-sample perturbation magnitude around an identity's
    # centroid, dimension-independent by construction. Same status as `separability`.
    noise: float = 0.2

    #: Determinism handle. Same config in, same embeddings out, on any platform.
    seed: int = 20260807

    #: Fixture ground truth: tracker id to synthetic identity index. This is the fixture's
    #: own answer key and it exists only because a generated scene has one. Nothing derives
    #: it, nothing infers it, and a track missing from it produces no embedding at all.
    identity_by_track: Mapping[int, int] = field(default_factory=dict)

    # -- embedder source ----------------------------------------------------
    #: Handed straight to `vividly.models.factory.build_embedder`, which is the only place
    #: a backend is named. Empty unless `source` is "embedder".
    embedder: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in EMBEDDING_SOURCES:
            raise ReidConfigError(
                f"unknown embeddings source {self.source!r}; available: {list(EMBEDDING_SOURCES)}"
            )
        object.__setattr__(
            self, "identity_by_track", {int(k): int(v) for k, v in self.identity_by_track.items()}
        )
        object.__setattr__(self, "embedder", dict(self.embedder))
        if self.dim < 3:
            # synth.make_identity_bank refuses below 3: a 1-D orthogonal complement holds
            # two directions and collapses every identity onto a line.
            raise ReidConfigError(f"embeddings dim must be >= 3, got {self.dim}")
        if not 0.0 <= self.separability <= 1.0:
            raise ReidConfigError(f"separability must be in [0, 1], got {self.separability}")
        if self.noise < 0.0:
            raise ReidConfigError(f"noise must be >= 0, got {self.noise}")
        if any(v < 0 for v in self.identity_by_track.values()):
            raise ReidConfigError("identity_by_track values are 0-based bank rows and must be >= 0")
        if self.source == "embedder" and not self.embedder:
            raise ReidConfigError(
                "embeddings.source is 'embedder' but no `embedder:` spec was given; it is "
                "passed to vividly.models.factory.build_embedder and needs at least a "
                "`backend` key"
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> EmbeddingsSpec:
        _check_keys(data, (f.name for f in fields(cls)), "embeddings")
        kwargs: dict[str, Any] = {}
        if "source" in data:
            kwargs["source"] = str(data["source"])
        if "dim" in data:
            kwargs["dim"] = int(data["dim"])
        if "seed" in data:
            kwargs["seed"] = int(data["seed"])
        for name in ("separability", "noise"):
            if name in data:
                kwargs[name] = float(data[name])
        if "identity_by_track" in data:
            block = data["identity_by_track"]
            if not isinstance(block, Mapping):
                raise ReidConfigError(
                    f"identity_by_track must be a mapping of track id to identity, got "
                    f"{type(block).__name__}"
                )
            kwargs["identity_by_track"] = dict(block)
        if "embedder" in data:
            block = data["embedder"]
            if not isinstance(block, Mapping):
                raise ReidConfigError(
                    f"embedder must be a mapping naming a backend, got {type(block).__name__}"
                )
            kwargs["embedder"] = dict(block)
        return cls(**kwargs)


@dataclass(frozen=True)
class ReidConfig:
    """How the stage behaves. Every decision number below is PROVISIONAL.

    Read configs/pipeline/reid.yaml before changing one: it carries the same values with
    the reasoning next to each, and tests/test_reid_pipeline.py pins the two together so
    the file and this dataclass cannot drift apart.
    """

    #: Off unless a config says otherwise. Combined with `embeddings.source`, this is the
    #: switch that makes the whole stage optional.
    enabled: bool = False

    #: Gallery sizing plus tau. Owned by GalleryConfig, which documents each field.
    gallery: GalleryConfig = field(default_factory=GalleryConfig)

    #: Where the vectors come from.
    embeddings: EmbeddingsSpec = field(default_factory=EmbeddingsSpec)

    # PROVISIONAL (the re-identification scope, POLICY.md; docs/CALIBRATION.md): how many
    # candidates one probe may raise. This is a cap on how many questions one arriving
    # person can put in front of a nurse at once, not a statement about how many of them
    # are right. 3 is a guess at a short list a person can actually compare, chosen with no
    # data about how often a real gallery produces two plausible candidates.
    top_k: int = 3

    # PROVISIONAL (the re-identification scope, POLICY.md; docs/CALIBRATION.md): the total
    # number of questions one new track may ever generate, across the first probe and every
    # re-probe. Without a cap, a session in which many people leave would keep asking about
    # one newcomer as the gallery grows.
    max_proposals_per_track: int = 3

    # PROVISIONAL (the re-identification scope, POLICY.md; docs/CALIBRATION.md): the
    # confidence a similarity exactly at tau maps to. See `confidence_from_similarity`,
    # which explains why a mapping is needed at all and why this one is a presentation
    # transform rather than a probability.
    confidence_at_tau: float = 0.5

    # PROVISIONAL: minimum source-time gap between two embeddings kept for the same
    # live track. Consecutive frames of a seated person are near-duplicates, and enrolling
    # every one of them spends the identity's whole row budget on one moment of the visit.
    # 1 s is a guess with the right shape at a 15 fps working point, not a measured rate.
    enroll_every_s: float = 1.0

    #: How many embeddings may queue up for one LIVE track before the oldest is dropped.
    #: Sizing, not a decision threshold: it bounds memory for a track that never leaves.
    #: The gallery re-refines whatever it is handed down to `gallery.max_per_identity`.
    pending_rows_per_track: int = 32

    #: The gap beyond which a candidate may be proposed at all, in source seconds. None
    #: means "take it from the track layer", which is the right answer: this is not a new
    #: threshold, it IS `TrackConfig.reap_after_s`, and a second copy of it that could
    #: disagree with the first is exactly what `vividly.reid.gallery` refuses to hold.
    min_gap_s: float | None = None

    #: How many stored proposals are read back at construction to rebuild the "already
    #: asked" set. Sizing, not a decision threshold. It only bites when a session is
    #: resumed over a store holding more than this many proposals, which for a
    #: session-scoped gallery means a store being reused across runs of the same replay.
    resume_history_limit: int = 1000

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ReidConfigError(f"top_k must be >= 1, got {self.top_k}")
        if self.max_proposals_per_track < 1:
            raise ReidConfigError(
                f"max_proposals_per_track must be >= 1, got {self.max_proposals_per_track}"
            )
        if not 0.0 <= self.confidence_at_tau <= 1.0:
            raise ReidConfigError(
                f"confidence_at_tau must be in [0, 1], got {self.confidence_at_tau}"
            )
        if self.enroll_every_s < 0.0:
            raise ReidConfigError(f"enroll_every_s must be >= 0, got {self.enroll_every_s}")
        if self.pending_rows_per_track < 1:
            raise ReidConfigError(
                f"pending_rows_per_track must be >= 1, got {self.pending_rows_per_track}"
            )
        if self.min_gap_s is not None and self.min_gap_s < 0.0:
            raise ReidConfigError(f"min_gap_s must be >= 0 or None, got {self.min_gap_s}")
        if self.resume_history_limit < 1:
            raise ReidConfigError(
                f"resume_history_limit must be >= 1, got {self.resume_history_limit}"
            )
        if self.enabled and self.embeddings.source == "none":
            raise ReidConfigError(
                "reid.enabled is true but embeddings.source is 'none'. An enabled stage "
                "with nothing to embed would silently do nothing; say which source it uses "
                f"({[s for s in EMBEDDING_SOURCES if s != 'none']}) or leave it disabled."
            )
        if self.embeddings.source == "synthetic" and self.embeddings.dim != self.gallery.dim:
            raise ReidConfigError(
                f"synthetic embeddings are {self.embeddings.dim}-D but the gallery is "
                f"{self.gallery.dim}-D. Both come from one embedder in a real run and a "
                "gallery cannot mix widths; set them equal rather than reshaping."
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ReidConfig:
        _check_keys(data, (f.name for f in fields(cls)), "ReidConfig")
        kwargs: dict[str, Any] = {}
        if "enabled" in data:
            kwargs["enabled"] = bool(data["enabled"])
        for name in ("top_k", "max_proposals_per_track", "pending_rows_per_track"):
            if name in data:
                kwargs[name] = int(data[name])
        if "resume_history_limit" in data:
            kwargs["resume_history_limit"] = int(data["resume_history_limit"])
        for name in ("confidence_at_tau", "enroll_every_s"):
            if name in data:
                kwargs[name] = float(data[name])
        if "min_gap_s" in data:
            value = data["min_gap_s"]
            kwargs["min_gap_s"] = None if value is None else float(value)
        for name, builder in (
            ("gallery", GalleryConfig.from_mapping),
            ("embeddings", EmbeddingsSpec.from_mapping),
        ):
            if name in data:
                block = data[name]
                if not isinstance(block, Mapping):
                    raise ReidConfigError(f"`{name}` must be a mapping, got {type(block).__name__}")
                try:
                    kwargs[name] = builder(block)
                except ValueError as exc:  # GalleryConfig raises plain ValueError
                    raise ReidConfigError(f"`{name}`: {exc}") from exc
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# similarity -> the number a human is shown
# ---------------------------------------------------------------------------
def confidence_from_similarity(similarity: float, tau: float, confidence_at_tau: float) -> float:
    """Map a cosine similarity onto the [0, 1] score `ReassocProposalRecord` demands.

    A mapping is required, not stylistic: the record's contract says confidence is a score
    in [0, 1] and that a raw cosine is not one, because the number reaches a human who is
    about to merge two patient identities. The store refuses anything outside [0, 1].

    The map is anchored on tau and is affine above it:

        confidence = confidence_at_tau + (1 - confidence_at_tau) * (s - tau) / (1 - tau)

    so a candidate scoring exactly at tau reads `confidence_at_tau` (the weakest thing the
    system is willing to ask about) and a perfect match reads 1.0. Below tau nothing is
    proposed at all, and the clamp exists only so a caller scoring outside the accept range
    still gets a legal value rather than an exception at the store boundary.

    What this is not. It is not a calibrated probability and nothing here makes it one. It
    is a monotone rescale that puts the accept threshold at a readable place, and it is
    PROVISIONAL exactly like tau itself. When `vividly.reid.calibrate` has been run
    against real overhead data, the calibration that comes out of it is what belongs here;
    until then the honest reading of a 0.5 is the weakest match the pipeline proposes, not
    a 50 percent chance that the two tracks are the same person.
    """
    if not math.isfinite(similarity):
        raise ValueError(f"similarity must be finite, got {similarity!r}")
    if tau >= 1.0:
        # A tau of exactly 1 accepts only an identical vector; every accepted candidate is
        # a perfect match, so the affine map degenerates rather than dividing by zero.
        return 1.0
    scaled = confidence_at_tau + (1.0 - confidence_at_tau) * (similarity - tau) / (1.0 - tau)
    return min(1.0, max(0.0, scaled))


# ---------------------------------------------------------------------------
# embedding sources
# ---------------------------------------------------------------------------
class EmbeddingSource(Protocol):
    """One frame of tracked people in, one row per track out.

    Rows are `(dim,)` float32 and L2-normalised, which is the `AppearanceEmbedder`
    contract and what `SessionGallery` validates rather than repairs. A track the source
    cannot embed is simply absent from the result: an empty dict is a legal answer and
    means this frame contributed nothing.
    """

    @property
    def label(self) -> str:
        """Short name for the run summary, e.g. "synthetic" or "embedder:onnx"."""

    @property
    def dim(self) -> int: ...

    def embed(
        self, observations: Sequence[TrackObservation], image: Any
    ) -> dict[int, np.ndarray]: ...


def _stream_seed(seed: int, track_id: int, t: float) -> int:
    """A stable 64-bit seed from (seed, track id, source millisecond).

    blake2b rather than `hash()`, which is salted per process and would make a "same config
    in, same bytes out" claim false across runs. Same reasoning as `vividly.reid.synth._mix`;
    written out here rather than importing that module's private helper.
    """
    key = f"{int(seed)}|{int(track_id)}|{int(t * 1000)}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")


class SyntheticEmbeddingSource:
    """Deterministic embeddings for a generated fixture, keyed by its own ground truth.

    Zero footage: no video, no weights, no cv2, no network. Each observation draws one
    sample around its identity's centroid in a `vividly.reid.synth` bank, seeded by
    (seed, track id, source millisecond), so two frames of the same person differ the way
    two crops of the same person would and two runs of the same config do not differ at all.

    NOTHING computed through this source is evidence about re-identification. Read the
    module docstring of `vividly.reid.synth`: the bank geometry is invented, and a fixture
    on which the right answer falls out is a demonstration that the wiring carries an
    answer, not that the answer is right.
    """

    def __init__(self, spec: EmbeddingsSpec) -> None:
        self.spec = spec
        self._identity_by_track = dict(spec.identity_by_track)
        # The bank has to be at least as tall as the largest identity referenced, and at
        # least one row wide even for an empty map, because make_identity_bank refuses 0.
        n_identities = max(self._identity_by_track.values(), default=-1) + 1
        self._bank = make_identity_bank(
            max(n_identities, 1), spec.dim, spec.separability, spec.seed
        )
        self.n_unmapped = 0

    @property
    def label(self) -> str:
        return "synthetic"

    @property
    def dim(self) -> int:
        return int(self.spec.dim)

    def embed(
        self, observations: Sequence[TrackObservation], image: Any = None
    ) -> dict[int, np.ndarray]:
        """One row per observation whose track appears in `identity_by_track`.

        `image` is ignored and is present only so this source and the crop source are
        interchangeable. A track with no declared identity is counted and skipped.
        """
        out: dict[int, np.ndarray] = {}
        for obs in observations:
            identity = self._identity_by_track.get(obs.track_id)
            if identity is None:
                self.n_unmapped += 1
                continue
            rows = sample_embeddings(
                self._bank,
                identity,
                1,
                self.spec.noise,
                _stream_seed(self.spec.seed, obs.track_id, obs.t),
            )
            out[obs.track_id] = np.asarray(rows[0], dtype=np.float32)
        return out


def crop_from_frame(image: Any, obs: TrackObservation) -> np.ndarray | None:
    """The pixels inside one tracked box, or None when the box has no pixels in frame.

    `TrackObservation` carries a centre and a size; the box is centre minus half size,
    floored and clamped to the frame. A box entirely outside the image, or one that clamps
    to zero width or height, returns None rather than an empty array: an embedder handed a
    zero-pixel crop raises, and a person walking off the edge of the frame is normal,
    not an error.

    A view into the frame, not a copy. The embedder reads it inside the same call and the
    frame is not mutated between here and there, so copying every crop would be pure cost.
    """
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"frame must be HxWx3, got shape {array.shape}")
    height, width = int(array.shape[0]), int(array.shape[1])
    x1 = max(0, math.floor(obs.cx - obs.w / 2.0))
    y1 = max(0, math.floor(obs.cy - obs.h / 2.0))
    x2 = min(width, math.ceil(obs.cx + obs.w / 2.0))
    y2 = min(height, math.ceil(obs.cy + obs.h / 2.0))
    if x2 <= x1 or y2 <= y1:
        return None
    return array[y1:y2, x1:x2]


class CropEmbeddingSource:
    """Crops from the replayed frame through an `AppearanceEmbedder`.

    The embedder owns its own preprocessing (resize, channel order, normalisation) because
    that recipe is a property of the weights, and it owns its own runtime. This class does
    geometry and batching and nothing else, which is why it needs no cv2 and no
    onnxruntime and can be tested against a stub.

    A tick with no image (the zero-footage flow) or with no boxes in frame returns nothing.
    """

    def __init__(self, embedder: Any, label: str = "embedder") -> None:
        for method in ("embed", "embedding_dim"):
            if not hasattr(embedder, method):
                raise ReidConfigError(
                    f"{type(embedder).__name__} is not an AppearanceEmbedder: it has no "
                    f"{method!r}. Build one through vividly.models.factory.build_embedder."
                )
        self._embedder = embedder
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    @property
    def dim(self) -> int:
        return int(self._embedder.embedding_dim)

    def embed(self, observations: Sequence[TrackObservation], image: Any) -> dict[int, np.ndarray]:
        if image is None:
            return {}
        crops: list[np.ndarray] = []
        track_ids: list[int] = []
        for obs in observations:
            crop = crop_from_frame(image, obs)
            if crop is None:
                continue
            crops.append(crop)
            track_ids.append(obs.track_id)
        if not crops:
            return {}
        rows = np.asarray(self._embedder.embed(crops), dtype=np.float32)
        if rows.shape[0] != len(crops):
            raise ValueError(
                f"embedder returned {rows.shape[0]} rows for {len(crops)} crops; row i must "
                "be the embedding of crop i or the gallery is enrolling the wrong people"
            )
        return {track_id: rows[i] for i, track_id in enumerate(track_ids)}


# ---------------------------------------------------------------------------
# what a step and a decision produce
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StepResult:
    """What one tick of the stage did. Nothing in here has bound anything."""

    #: Track ids enrolled into the gallery on this tick, i.e. the people who just left.
    enrolled: tuple[int, ...] = ()
    #: Track ids probed against the gallery on this tick.
    probed: tuple[int, ...] = ()
    #: Rows written to `reassoc_proposals`, all of them pending, all of them questions.
    proposals: tuple[ReassocProposalRecord, ...] = ()


@dataclass(frozen=True)
class DecisionOutcome:
    """A human's answer, and whether it could still be applied to a live track.

    `record` is written first and always: it is the label, and a label must survive even
    when the person it is about has already left the room. `merged` says whether
    `TrackBook.merge_identity` also ran, which for a confirmation on a track that has since
    been reaped is False with the reason in `note`. A rejection never merges anything and
    reports `merged=False` with a note saying so, which is not a failure.
    """

    record: ReassocProposalRecord
    merged: bool
    note: str


@dataclass(frozen=True)
class ReidStats:
    """Counters for the run summary and the dashboard. Not one of them is a rate.

    In particular `n_proposals` is how often the system asked, and `n_confirmed` how often
    a human said yes. Neither is an accuracy number: the denominator that would make one is
    ground truth this pipeline does not have, which is the whole reason the questions are
    asked out loud and the answers are stored as labels.
    """

    source: str
    tau: float
    n_identities: int
    n_gallery_rows: int
    n_enrolled: int
    n_probes: int
    n_proposals: int
    n_confirmed: int
    n_rejected: int
    n_duplicate_proposals: int
    n_skipped_asked: int
    n_skipped_inside_window: int
    n_skipped_capped: int
    n_skipped_resolved: int
    n_unmapped: int
    embed_ms: float
    gallery_ms: float
    cleared: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, for the dashboard payload and the run artifacts."""
        return {
            "source": self.source,
            "tau": self.tau,
            "n_identities": self.n_identities,
            "n_gallery_rows": self.n_gallery_rows,
            "n_enrolled": self.n_enrolled,
            "n_probes": self.n_probes,
            "n_proposals": self.n_proposals,
            "n_confirmed": self.n_confirmed,
            "n_rejected": self.n_rejected,
            "n_duplicate_proposals": self.n_duplicate_proposals,
            "n_skipped_asked": self.n_skipped_asked,
            "n_skipped_inside_window": self.n_skipped_inside_window,
            "n_skipped_capped": self.n_skipped_capped,
            "n_skipped_resolved": self.n_skipped_resolved,
            "n_unmapped": self.n_unmapped,
            "embed_ms": self.embed_ms,
            "gallery_ms": self.gallery_ms,
            "cleared": self.cleared,
            "threshold_status": PROVISIONAL_NOTE,
        }


# ---------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------
class ReidStage:
    """Enrol the departed, probe the newly arrived, write questions, apply answers.

    Lifecycle, in the order it happens on one pipeline tick:

      1. `TrackBook.ingest` creates the tick's new tracks and reaps the absent ones. Each
         reaped TrackState arrives at `on_track_reaped`, which does nothing but move that
         track's queued embeddings and its identity snapshot onto a hand-off deque. Cheap
         and allocation-light, because it runs on the ingest path.
      2. `step` drains that deque and ENROLS, so by the time anything is probed the gallery
         already holds everyone who left on this tick.
      3. `step` then PROBES every live track that has not yet been probed against the
         gallery in its current state, and writes a pending proposal per surviving
         candidate. It binds nothing.
      4. `decide`, called later from the dashboard, writes the human's answer and, on a
         confirmation, calls `TrackBook.merge_identity`.

    RE-PROBING, and the gap it closes. A track is probed at birth, and again whenever the
    gallery has since gained an identity. That second case is not hypothetical: a departed
    person reaches the gallery only when the TrackBook reaps them, `reap_after_s` (30 s)
    after their last frame, by which time every track already in the room has been born and
    probed. A return inside the tracker's own window keeps its old id and never reaches this
    stage at all (`lost_track_buffer` 1800 frames, 60 s), so the tracks this mechanism is
    for are the ones that were already alive when the enrollment landed. Probing only at
    birth would make every such enrollment invisible. Re-probing is what makes the
    already-asked set load-bearing, and the set is what stops the same pair being put in
    front of a human twice.

    THREADS. The pipeline thread runs `step` and receives the reap hand-off; uvicorn's
    threadpool runs `decide` and `stats`. One RLock covers the mutable bookkeeping, in the
    `TrackBook` pattern and for the same reason. Lock order: this stage's lock is the
    outermost; while holding it the stage may call into the store, the gallery, or the
    book, and no path acquires this stage's lock while holding any of theirs. `decide`
    writes the label to the store before it touches the book, and the gallery never calls
    back into anything.
    """

    def __init__(
        self,
        config: ReidConfig,
        *,
        gallery: SessionGallery,
        store: EventStore,
        book: TrackBook,
        source: EmbeddingSource,
        min_gap_s: float,
        run_id: str | None = None,
    ) -> None:
        if min_gap_s < 0.0:
            raise ReidConfigError(f"min_gap_s must be >= 0, got {min_gap_s}")
        if source.dim != gallery.config.dim:
            raise ReidConfigError(
                f"the {source.label} embedding source is {source.dim}-D but the gallery is "
                f"{gallery.config.dim}-D; a gallery cannot mix widths"
            )
        self.config = config
        self.gallery = gallery
        self.store = store
        self.book = book
        self.source = source
        self.min_gap_s = float(min_gap_s)
        # The live run this stage belongs to. Every proposal it writes carries it, every
        # question it answers is scoped to it, and it seeds its resume/exclusivity state only
        # from rows tagged with it. None is the pre-run-scoping behaviour (unit tests and any
        # caller that never shares a store file): proposals carry NULL and the store's
        # per-run identity indexes never fire, so the stage's own in-memory exclusivity below
        # is the whole guard. See scripts/run_pipeline.py for how the digest is built.
        self._run_id = run_id

        self._lock = threading.RLock()
        # track_id -> the embeddings kept for it while it is still live. Bounded by
        # config.pending_rows_per_track; the gallery refines again on enrollment.
        self._pending: dict[int, deque[np.ndarray]] = {}
        # track_id -> source time of the last embedding kept, for enroll_every_s.
        self._last_kept_t: dict[int, float] = {}
        # Reaped tracks waiting to be enrolled, handed over by the book's reap watcher.
        self._to_enroll: deque[tuple[TrackIdentity, float, list[np.ndarray]]] = deque()
        # Identity snapshots of everyone this session has enrolled, so a confirmation can
        # carry a staff tag or a mock binding forward after the TrackState is gone.
        self._identities: dict[int, TrackIdentity] = {}
        # (new_track, candidate_track) -> "pending" | "confirmed" | "rejected". Seeded from
        # the store so a resumed run never re-asks a question somebody already answered.
        self._asked: dict[tuple[int, int], str] = {}
        # How many questions each new track has generated, against max_proposals_per_track.
        self._n_asked_for: dict[int, int] = {}
        # Bumps every time the gallery gains an identity; a track probed at an older epoch
        # is probed again. See the class docstring.
        self._epoch = 0
        self._probed_at: dict[int, int] = {}
        # Tracks a human has already given an identity to. They are never probed again:
        # the question has been answered.
        self._resolved: set[int] = set()
        # The two per-run identity properties, held in memory as defence in depth over the
        # store's partial-unique indexes (which cannot fire for an unscoped run_id=None run):
        # a live track is confirmed to at most one departed identity, and a departed identity
        # is claimed by at most one live track. new_track -> its confirmed candidate, and
        # candidate_track -> the live track that claimed it. Seeded from the store on resume.
        self._confirmed_new: dict[int, int] = {}
        self._confirmed_candidate: dict[int, int] = {}

        self._n_enrolled = 0
        self._n_probes = 0
        self._n_proposals = 0
        self._n_confirmed = 0
        self._n_rejected = 0
        self._n_duplicates = 0
        self._n_skipped_asked = 0
        self._n_skipped_inside_window = 0
        self._n_skipped_capped = 0
        self._n_skipped_resolved = 0
        self._embed_s = 0.0
        self._gallery_s = 0.0
        self._cleared = False

        self._resume_from_store()
        self.book.set_reap_watcher(self.on_track_reaped)

    @property
    def run_id(self) -> str | None:
        """The live run this stage belongs to, or None when the store is not run-scoped."""
        return self._run_id

    # -- resume -------------------------------------------------------------
    def _resume_from_store(self) -> None:
        """Rebuild the already-asked set from proposals THIS RUN already holds.

        A replay run over an existing alerts.sqlite is the normal case (the same clip, run
        twice, is how anyone works), and the rows in it may already carry a human's answer.
        Re-asking an answered question is worse than not asking: it puts a decision a person
        already made back in the queue as if it had never happened, and a rejected pair
        would come back every single run. Deterministic proposal ids make the duplicate
        insert fail safely as well, but only when the clock lands on the same millisecond,
        so the pair set is the real guard and the id collision is the backstop.

        Scoped to `self._run_id` on purpose. One store file can be reached by more than one
        run (its path derives from a run name, and two clips can collide on it), and a prior
        run's CONFIRMED row for the same track id would otherwise seed this run's `_resolved`
        and silence its probing of a completely different person. With `run_id=None` the
        scope is the whole file, the pre-run-scoping behaviour that unit tests rely on.
        """
        for record in self.store.proposal_history(
            run_id=self._run_id, limit=self.config.resume_history_limit
        ):
            pair = (int(record.new_track), int(record.candidate_track))
            self._asked[pair] = record.status
            self._n_asked_for[pair[0]] = self._n_asked_for.get(pair[0], 0) + 1
            if record.status == "confirmed":
                self._resolved.add(pair[0])
                self._confirmed_new[pair[0]] = pair[1]
                self._confirmed_candidate[pair[1]] = pair[0]

    # -- the reap hand-off --------------------------------------------------
    def on_track_reaped(self, state: TrackState) -> None:
        """Registered with `TrackBook.set_reap_watcher`. Runs on the ingest path.

        Deliberately trivial: it moves the track's queued embeddings and a snapshot of its
        human-supplied identity fields onto a deque and returns. The numpy work happens in
        `step`, so the cost of a departure does not land inside the timed rules stage and
        so this never holds anything while the book is mid-frame.

        A track with no queued embeddings is still recorded as departed (its identity
        snapshot is kept) but is not enrolled: an identity with no appearance cannot be a
        candidate for anything, and inventing a row for it would put a person in the
        gallery who was never seen by the embedder.

        Re-checks `_cleared` inside the lock. `close()` unregisters this watcher before it
        empties anything, but a `reap` already in flight can have captured this callable
        before that landed, so the same critical section that would append the appearance
        rows is the one that has to see the session has ended and drop them (CQ-16).
        """
        with self._lock:
            if self._cleared:
                return
            rows = self._pending.pop(state.track_id, None)
            self._last_kept_t.pop(state.track_id, None)
            self._probed_at.pop(state.track_id, None)
            self._to_enroll.append(
                (state.identity(), float(state.last_seen_t), [] if rows is None else list(rows))
            )

    # -- the tick -----------------------------------------------------------
    def step(
        self, observations: Sequence[TrackObservation], t: float, image: Any = None
    ) -> StepResult:
        """Enrol whoever just left, probe whoever is new, return what was asked.

        Call once per pipeline tick, AFTER `TrackBook.ingest` and `TrackBook.reap`, so that
        the tick's departures are already in the hand-off deque and the tick's arrivals are
        already in the book. Returns a `StepResult` whose `proposals` are all pending.

        Nothing here mutates a TrackState. The stage reads the book to skip tracks a human
        has resolved and writes to it only from `decide`.
        """
        if self._cleared:
            raise RuntimeError(
                "this ReidStage has been closed; its gallery was cleared at session end "
                "(CQ-16) and re-using it would enrol a second visit into a first one"
            )
        enrolled = self._drain_enrollments()
        if not observations:
            return StepResult(enrolled=tuple(enrolled))

        started = time.perf_counter()
        rows = self.source.embed(observations, image)
        self._embed_s += time.perf_counter() - started

        probed: list[int] = []
        proposals: list[ReassocProposalRecord] = []
        for obs in observations:
            row = rows.get(obs.track_id)
            if row is None:
                continue
            with self._lock:
                eligible = self._probed_at.get(obs.track_id, -1) < self._epoch
                resolved = obs.track_id in self._resolved
                due = eligible and not resolved
                if eligible:
                    # Mark this track seen at the current epoch whether it is probed or
                    # skipped-because-resolved, so a resolved track that stays in the room is
                    # counted once per new-evidence epoch rather than every single tick.
                    self._probed_at[obs.track_id] = self._epoch
                if eligible and resolved:
                    # A track a human has already given an identity to stops being asked
                    # about. Counted so a session that goes quiet says WHY it went quiet.
                    self._n_skipped_resolved += 1
            if due:
                probed.append(obs.track_id)
                proposals.extend(self._probe(obs.track_id, row, t))
            self._keep(obs.track_id, row, obs.t)
        return StepResult(
            enrolled=tuple(enrolled), probed=tuple(probed), proposals=tuple(proposals)
        )

    def _keep(self, track_id: int, row: np.ndarray, t: float) -> None:
        """Queue one embedding for a live track, no more often than enroll_every_s.

        Re-checks `_cleared` inside the lock. `step` read it once, unlocked, at the top; a
        `close()` that lands after that read and before this write would otherwise let a tick
        in flight repopulate `_pending` with an appearance vector AFTER the session-end
        deletion, and CQ-16's post-condition would be false. The check and the write share
        one critical section so they cannot straddle the deletion.
        """
        with self._lock:
            if self._cleared:
                return
            last = self._last_kept_t.get(track_id)
            if last is not None and t - last < self.config.enroll_every_s:
                return
            self._last_kept_t[track_id] = t
            queue = self._pending.get(track_id)
            if queue is None:
                queue = deque(maxlen=self.config.pending_rows_per_track)
                self._pending[track_id] = queue
            queue.append(np.asarray(row, dtype=np.float32))

    def _drain_enrollments(self) -> list[int]:
        """Enrol every track handed over since the last tick. Returns the enrolled ids.

        Runs entirely inside `self._lock`, and enrols into the gallery while holding it. That
        is what makes CQ-16 hold under contention: `close()` sets `_cleared` and empties the
        gallery under this same lock, so an enrollment and the session-end deletion can never
        interleave. `step` read `_cleared` once, unlocked, at the top; a tick in flight when
        `close()` lands either enrols before the deletion (and is deleted with everything
        else) or re-checks `_cleared` here and enrols nothing. It can never enrol AFTER the
        deletion. Holding the stage lock across `gallery.enroll` is allowed by the documented
        lock order (this lock is outermost and the gallery never calls back); the work is
        bounded by `pending_rows_per_track`.
        """
        enrolled: list[int] = []
        while True:
            with self._lock:
                if self._cleared or not self._to_enroll:
                    break
                identity, last_seen_t, rows = self._to_enroll.popleft()
                self._identities[identity.track_id] = identity
                if not rows or not math.isfinite(last_seen_t):
                    continue
                started = time.perf_counter()
                # Enrolled at the time the appearance was OBSERVED, not at the time the
                # reaper got round to it. Proposal.elapsed_s is then how long the person was
                # actually gone, which is the number a reviewer is being asked to reason
                # about.
                self.gallery.enroll(identity.track_id, np.stack(rows, axis=0), last_seen_t)
                self._gallery_s += time.perf_counter() - started
                self._n_enrolled += 1
                # A new identity in the gallery is new evidence for every live track that has
                # not been given an identity yet, so everyone becomes probe-eligible again.
                # See the class docstring on re-probing.
                self._epoch += 1
            enrolled.append(identity.track_id)
        return enrolled

    # -- the question -------------------------------------------------------
    def _probe(self, track_id: int, row: np.ndarray, t: float) -> list[ReassocProposalRecord]:
        """Score one probe and write a pending proposal per surviving candidate.

        Every filter below is a reason NOT to ask, and each one exists so a human's queue
        holds questions that are worth a person's attention:

          * a candidate inside the tracker's own window is not this mechanism's business
            (the automatic path either handled it or is still holding the track);
          * a pair already asked about is not asked about twice, whatever the answer was;
          * a track that has already generated `max_proposals_per_track` questions stops.
        """
        started = time.perf_counter()
        candidates = self.gallery.propose(row, top_k=self.config.top_k, t=t)
        n_identities = len(self.gallery.per_identity_counts())
        self._gallery_s += time.perf_counter() - started
        with self._lock:
            self._n_probes += 1

        written: list[ReassocProposalRecord] = []
        for candidate in candidates:
            record = self._maybe_ask(track_id, candidate, t, n_identities)
            if record is not None:
                written.append(record)
        return written

    def _maybe_ask(
        self, track_id: int, candidate: Proposal, t: float, n_identities: int
    ) -> ReassocProposalRecord | None:
        pair = (track_id, int(candidate.identity_key))
        with self._lock:
            if pair[0] == pair[1]:
                # A tracker that reused an id, not a person who returned. The store refuses
                # a self-pair and it is right to; there is no question to ask here.
                return None
            if candidate.elapsed_s < self.min_gap_s:
                self._n_skipped_inside_window += 1
                return None
            if pair in self._asked:
                self._n_skipped_asked += 1
                return None
            if self._n_asked_for.get(track_id, 0) >= self.config.max_proposals_per_track:
                self._n_skipped_capped += 1
                return None
            confidence = confidence_from_similarity(
                candidate.confidence, self.gallery.config.tau, self.config.confidence_at_tau
            )
            try:
                record = self.store.record_proposal(
                    new_track=pair[0],
                    candidate_track=pair[1],
                    confidence=confidence,
                    ts=t,
                    gallery_size=n_identities,
                    run_id=self._run_id,
                )
            except DuplicateProposalError:
                # The same pair at the same millisecond is the same question. Record it as
                # asked so the pair set agrees with the store from here on.
                self._asked[pair] = "pending"
                self._n_asked_for[track_id] = self._n_asked_for.get(track_id, 0) + 1
                self._n_duplicates += 1
                return None
            self._asked[pair] = "pending"
            self._n_asked_for[track_id] = self._n_asked_for.get(track_id, 0) + 1
            self._n_proposals += 1
            return record

    # -- the answer ---------------------------------------------------------
    def decide(
        self, proposal_id: str, decision: str, decided_ts: float, via: str
    ) -> DecisionOutcome:
        """Record a human's confirm or reject, then apply a confirmation to the live track.

        The label is always written; the merge is applied only when it is safe to. A merge
        joins two patient identities, so a confirm is refused (merged=False, label kept)
        rather than misapplied in three cases:

          * the new track already has a confirmed identity this run (one live track, one
            identity), or the candidate is already claimed by another live track this run
            (one departed identity, one live track). Either would break a property the store
            enforces per run in its own indexes; the stage enforces it too as defence in
            depth, because those indexes cannot fire for an unscoped run_id=None run. The
            losing confirm is recorded as a rejection, per the store's own contract;
          * the candidate was never enrolled in THIS session (an unrelated or foreign-run
            proposal). There is no identity of this run to carry forward, so the label is
            stored and nothing is merged. This is the removal of the old blank-identity
            fallback, which used to fabricate an empty identity and merge it;
          * the new track has already been reaped (it left before the answer arrived). The
            label is a true statement about two people worth keeping; there is simply nothing
            live left to carry it onto.

        A confirm the store refuses at commit time (a racing second confirm that won its
        partial-unique index) surfaces as `ConfirmConflictError`; it is caught and turned
        into the same recorded rejection, never a 500. The merge itself is
        `TrackBook.merge_identity`, which sets `reassociated_from` and carries a staff tag and
        a mock chart binding forward; it never renames a track and never touches an alert.
        """
        if decision not in REASSOC_DECISIONS:
            raise ValueError(
                f"unknown re-association decision {decision!r}; "
                f"expected one of {list(REASSOC_DECISIONS)}"
            )
        if decision == "rejected":
            return self._reject(proposal_id, decided_ts, via)

        # A confirm. Read the pending pair first so the stage can enforce the two identity
        # properties itself before it writes a confirmed row. The read is scoped to this run
        # by decide_proposal below; here it only tells the stage which pair it is about.
        existing = self.store.get_proposal(proposal_id)
        if existing is not None and existing.status == "pending":
            pair = (int(existing.new_track), int(existing.candidate_track))
            with self._lock:
                reason = self._confirm_conflict(pair)
            if reason is not None:
                return self._record_losing_confirm(proposal_id, pair, decided_ts, via, reason)

        # No stage-level conflict known: write the confirmed label. The store enforces the
        # same two properties per run in the database, so a racing confirm that got there
        # first raises ConfirmConflictError, which the stage turns into the same rejection.
        try:
            record = self.store.decide_proposal(
                proposal_id, "confirmed", decided_ts, via, run_id=self._run_id
            )
        except ConfirmConflictError:
            # A conflict means the row exists and was pending a moment ago, so `existing` is
            # populated; the (0, 0) fallback is unreachable and only satisfies the type checker.
            pair = (
                (int(existing.new_track), int(existing.candidate_track))
                if existing is not None
                else (0, 0)
            )
            return self._record_losing_confirm(
                proposal_id,
                pair,
                decided_ts,
                via,
                "another live track confirmed this identity first",
            )

        pair = (int(record.new_track), int(record.candidate_track))
        with self._lock:
            self._asked[pair] = record.status
            self._n_confirmed += 1
            self._resolved.add(pair[0])
            # Reflect the store's confirmed row in the in-memory exclusivity maps, whether or
            # not the forward merge below can be applied: the identity is confirmed in the
            # record either way, so a later confirm for the same track or candidate must be
            # refused either way.
            self._confirmed_new[pair[0]] = pair[1]
            self._confirmed_candidate[pair[1]] = pair[0]
            prior = self._identities.get(pair[1])
        if prior is None:
            # The candidate was never enrolled in this session. The label stands; there is
            # nothing of THIS run's to carry onto the live track, so nothing is merged.
            return DecisionOutcome(
                record=record,
                merged=False,
                note=(
                    f"confirmed and stored, but track {pair[1]} was not enrolled in this "
                    f"session, so there is no identity of this run to carry onto track {pair[0]}"
                ),
            )
        try:
            self.book.merge_identity(pair[0], prior)
        except KeyError:
            return DecisionOutcome(
                record=record,
                merged=False,
                note=(
                    f"confirmed and stored, but track {pair[0]} has already been reaped, so "
                    "there is nothing live to carry the identity onto"
                ),
            )
        return DecisionOutcome(
            record=record,
            merged=True,
            note=f"confirmed: track {pair[0]} continues track {pair[1]}",
        )

    def _reject(self, proposal_id: str, decided_ts: float, via: str) -> DecisionOutcome:
        """Write a rejection label. Scoped to this run; never merges anything."""
        record = self.store.decide_proposal(
            proposal_id, "rejected", decided_ts, via, run_id=self._run_id
        )
        pair = (int(record.new_track), int(record.candidate_track))
        with self._lock:
            self._asked[pair] = record.status
            self._n_rejected += 1
        return DecisionOutcome(
            record=record,
            merged=False,
            note=(
                f"rejected: tracks {pair[0]} and {pair[1]} are different people and "
                "this pair will not be proposed again this session"
            ),
        )

    def _confirm_conflict(self, pair: tuple[int, int]) -> str | None:
        """Why a confirm of `pair` would break a per-run identity property, or None.

        Called with `self._lock` held. The `!=` guards let a re-decide of the very same pair
        fall through to the store, which reports it as an already-decided proposal rather
        than as a conflict; only a DIFFERENT track or candidate is a real exclusivity break.
        """
        new_track, candidate_track = pair
        claimed_by = self._confirmed_new.get(new_track)
        if claimed_by is not None and claimed_by != candidate_track:
            return (
                f"track {new_track} already has a confirmed identity (candidate {claimed_by}); "
                "one live track holds one identity"
            )
        holder = self._confirmed_candidate.get(candidate_track)
        if holder is not None and holder != new_track:
            return (
                f"candidate {candidate_track} is already claimed by live track {holder}; "
                "one departed identity is claimed by one live track"
            )
        return None

    def _record_losing_confirm(
        self,
        proposal_id: str,
        pair: tuple[int, int],
        decided_ts: float,
        via: str,
        reason: str,
    ) -> DecisionOutcome:
        """A confirm that would break exclusivity: store it as a rejection, merge nothing.

        The losing confirm is a real answer a human gave and is not dropped; it is recorded
        as a rejection, which is what the store's ConfirmConflictError contract asks the
        stage to do. merged=False, and the pair is not put in front of anyone again.
        """
        record = self.store.decide_proposal(
            proposal_id, "rejected", decided_ts, via, run_id=self._run_id
        )
        with self._lock:
            self._asked[pair] = record.status
            self._n_rejected += 1
        return DecisionOutcome(
            record=record,
            merged=False,
            note=(
                f"not merged and recorded as a rejection: {reason}. The label is kept so the "
                "answer is not lost, but nothing is bound"
            ),
        )

    def pending(self, limit: int = 100) -> list[ReassocProposalRecord]:
        """The review queue for THIS run, oldest question first. A pass-through to the store.

        Scoped to `self._run_id`: a proposal another run raised is that run's question to
        answer, visible in the shared history but not in this run's queue. With `run_id=None`
        the queue spans the whole file, the pre-run-scoping behaviour.
        """
        return self.store.pending_proposals(run_id=self._run_id, limit=limit)

    # -- reporting ----------------------------------------------------------
    def stats(self) -> ReidStats:
        """A snapshot for the run summary and the dashboard. Safe from another thread.

        Take one BEFORE `close()` if the numbers are wanted: closing empties the gallery,
        which is the point of closing it.
        """
        with self._lock:
            counts = self.gallery.per_identity_counts()
            return ReidStats(
                source=self.source.label,
                tau=float(self.gallery.config.tau),
                n_identities=len(counts),
                n_gallery_rows=int(sum(counts.values())),
                n_enrolled=self._n_enrolled,
                n_probes=self._n_probes,
                n_proposals=self._n_proposals,
                n_confirmed=self._n_confirmed,
                n_rejected=self._n_rejected,
                n_duplicate_proposals=self._n_duplicates,
                n_skipped_asked=self._n_skipped_asked,
                n_skipped_inside_window=self._n_skipped_inside_window,
                n_skipped_capped=self._n_skipped_capped,
                n_skipped_resolved=self._n_skipped_resolved,
                n_unmapped=int(getattr(self.source, "n_unmapped", 0)),
                embed_ms=self._embed_s * 1000.0,
                gallery_ms=self._gallery_s * 1000.0,
                cleared=self._cleared,
            )

    # -- session end --------------------------------------------------------
    def close(self) -> None:
        """Empty the gallery and every in-memory trace of this session's appearances.

        This is the deletion CQ-16 names, performed rather than intended: the gallery's
        embeddings, the per-track queues that have not been enrolled yet, and the identity
        snapshots all go. What remains is the alert store, which holds no embedding, no
        crop and no appearance vector of any kind: a proposal row is two integer track ids,
        a score, a count and a human's answer.

        Idempotent, because session end is exactly the moment a shutdown path runs twice.
        The book's reap watcher is unregistered first, so a reap arriving from another
        thread during shutdown cannot queue an enrollment into a gallery that is being
        emptied.
        """
        if self._cleared:
            return
        self.book.set_reap_watcher(None)
        with self._lock:
            self._cleared = True
            self._to_enroll.clear()
            self._pending.clear()
            self._last_kept_t.clear()
            self._identities.clear()
            self._probed_at.clear()
        self.gallery.clear()

    def __enter__(self) -> ReidStage:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def build_embedding_source(
    spec: EmbeddingsSpec, repo_root: Path | None = None
) -> EmbeddingSource | None:
    """One of the two sources, or None when the config names no source at all.

    The `embedder` branch is the only place this module reaches into `vividly.models`, and
    it does so through the factory, which is a closed registry: an unknown backend key is
    an error rather than a dynamic import, and the re-ID backend refuses a LUPerson-derived
    lineage on top of that (POLICY.md, rule 11, docs/LICENCE_ALLOWLIST.md).
    """
    if spec.source == "none":
        return None
    if spec.source == "synthetic":
        return SyntheticEmbeddingSource(spec)
    if spec.source == "embedder":
        from vividly.models.factory import build_embedder

        kwargs = dict(spec.embedder)
        path = kwargs.get("model_path")
        if repo_root is not None and path is not None:
            candidate = Path(str(path)).expanduser()
            kwargs["model_path"] = str(
                candidate if candidate.is_absolute() else repo_root / candidate
            )
        embedder = build_embedder(kwargs)
        if embedder is None:  # pragma: no cover - EmbeddingsSpec rejects an empty spec
            raise ReidConfigError("embeddings.source is 'embedder' but the spec built nothing")
        return CropEmbeddingSource(embedder, label=f"embedder:{kwargs.get('backend', '?')}")
    raise ReidConfigError(  # pragma: no cover - EmbeddingsSpec validates the name first
        f"unknown embeddings source {spec.source!r}"
    )


def load_reid_config(path: str | Path) -> ReidConfig:
    """Load configs/pipeline/reid.yaml, or the `reid:` block of any pipeline config.

    Same shape and same reasoning as `vividly.rules.load_rule_config`: a file whose top
    level IS the settings mapping is accepted, and so is one that nests them under a
    `reid:` key, so a pipeline config can carry its settings inline or point at the shared
    file. Every value still goes through `ReidConfig` validation either way.

    yaml is imported here rather than at module level: the stage itself runs inside the
    live pipeline and has no business depending on a config parser.
    """
    import yaml

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ReidConfigError(f"{config_path} must contain a mapping, got {type(data).__name__}")
    block = data.get("reid", data)
    if not isinstance(block, Mapping):
        raise ReidConfigError(
            f"{config_path}: `reid` must be a mapping, got {type(block).__name__}"
        )
    return ReidConfig.from_mapping(block)


def build_reid_stage(
    config: ReidConfig,
    *,
    store: EventStore,
    book: TrackBook,
    reap_after_s: float,
    repo_root: Path | None = None,
    run_id: str | None = None,
) -> ReidStage | None:
    """A wired stage, or None when re-ID is off. None is the default and the quiet path.

    `reap_after_s` is the track layer's own window and becomes the stage's `min_gap_s`
    unless the config overrides it. It is passed in rather than read off the book so that
    the one number lives in `TrackConfig` and this module holds no second copy of it.

    `run_id` is the live run this stage belongs to, handed straight to :class:`ReidStage`.
    Any caller that can reach a store file more than one run can also reach should supply
    one: an unscoped `run_id=None` run writes NULL, and the store's per-run identity indexes
    cannot fire on NULL, which leaves the stage's own in-memory maps as the only guard. The
    shipped runner always supplies one; see `scripts/run_pipeline.py` for how it is derived.
    """
    if not config.enabled:
        return None
    source = build_embedding_source(config.embeddings, repo_root)
    if source is None:  # pragma: no cover - ReidConfig rejects enabled + source "none"
        return None
    gallery = SessionGallery(config.gallery)
    min_gap_s = float(reap_after_s if config.min_gap_s is None else config.min_gap_s)
    return ReidStage(
        config,
        gallery=gallery,
        store=store,
        book=book,
        source=source,
        min_gap_s=min_gap_s,
        run_id=run_id,
    )
