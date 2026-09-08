"""The session-scoped open-set gallery: the module that proposes a re-association and
never makes one.

Scope is POLICY.md, section 4, re-identification scope, sitting inside the replay pipeline.
It holds appearance embeddings for the people seen during one visit, and it answers exactly
one question: which enrolled identities does this probe look like, and how much.

The semantics this module exists to enforce
===========================================

There are two association regimes and this file is only ever the second one.

  1. Inside the tracker's own short re-association window, association is automatic and is
     unchanged by anything here. `trackers.BoTSORTTracker` re-associates across a brief
     occlusion, `vividly.tracks.TrackBook` keeps a track alive for `reap_after_s`, and a
     person who leaves and returns inside that window keeps their track id, their baseline,
     their staff tag and their mock chart binding because the same TrackState object is
     still there. No gallery is consulted, nothing is proposed, and nobody is asked.

  2. Beyond that window there is no automatic binding at all. `propose` returns candidates
     with a confidence; a human confirms or rejects one on the dashboard; the confirmation
     or the rejection is what becomes a label. That is why the method is called `propose`
     and why no method on this class binds, merges, renames or joins anything. There is no
     `bind`, no `accept`, no `auto_merge`, and adding one would reverse the whole design
     rather than extend it. A false merge joins two patients' identities, which
     in a chart-bound deployment is a patient-safety event rather than an accuracy point,
     so the system is not permitted to make that decision by itself.

The gallery deliberately does not know where the tracker's window ends. `Proposal.elapsed_s`
reports how long ago that identity was last enrolled and the caller compares it against its
own tracker configuration, because the window is a property of the tracker and the track
book, and duplicating it here would create a second copy that could disagree with the first.

Session scope, and why `clear` is a contract and not a convenience
==================================================================

Session-scoped is the product claim: identity lives for one visit, is never persisted
across visits, and is never a cross-site identifier. This class holds its embeddings in
memory only. It has no path to disk, no path to a database, no serialisation method, and no
constructor that loads anything, so there is nothing to forget to delete at session end
except the object itself, and `clear()` empties it outright and can be called again safely.
CQ-16 in docs/COUNSEL_QUESTIONS.md cites this file by name for exactly that property, so
`tests/test_reid_gallery.py` pins it. If a future change ever needs the gallery to survive a
process restart, that is a counsel question first and a code change second.

What may be in here. Appearance embeddings over whole-person crops, produced by
`vividly.models.reid_backend`, and nothing else. No face detection, no face embedding, no
face-localisation branch, ever, under POLICY.md, rule 2, and because the load-bearing element
of the Philips US 11,157,726 independent claims is a face-location output combined with
non-facial features. No gait, no pose-sequence identity signal, no body-shape signal: those
are re-ID v2, they are gated on a freedom-to-operate opinion and on biometric law, and the
scope in POLICY.md, section 4 keeps them out of the codebase. This module takes a float32
vector and a number; it cannot express any of the banned signals and must never be extended
so that it can.

Statistics-driven refinement, and why not FIFO
==============================================

Each identity keeps at most `max_per_identity` embeddings. When a new one arrives and the
set is full, the row evicted is the one farthest from the identity's own centroid, measured
as cosine to the L2-normalised mean of the candidate set (the kept rows plus the newcomer).
The rule is statistics-driven gallery refinement reduced to its usable core: score each
stored view against the identity's own distribution and drop the outlier, rather than
trusting every view equally.

FIFO would be simpler and is wrong here for two reasons.

  * The decision rule is a maximum over an identity's rows (`vividly.reid.metrics` documents
    why: a probe should match any enrolled view of that person). A maximum is fixed by
    whichever row is most extreme, so a single bad view (motion blur, half-occluded behind
    a passing person, a crop clipped at the frame edge) is precisely the row a different
    person's probe is most likely to hit. Bad rows do not average out under a maximum; they
    are the only rows that matter. Evicting the outlier attacks the false-merge rate
    directly, and the false-merge rate is the headline patient-safety measure.
  * FIFO evicts by arrival order, which throws away the clean early views of someone who
    has been sitting still for an hour and keeps whatever the most recent frames happened
    to look like. Recency is not evidence of quality, and in a waiting room the most recent
    frames are often the worst ones (the person is standing up and walking out of frame).

The eviction is greedy and repeated: with the newcomer appended, the farthest row is dropped
and the centroid is recomputed, until the set is back at capacity. That lets a wildly
off-distribution newcomer evict itself instead of displacing a good stored view, which is
the intended behaviour when the crop that produced it was unusable.

None of this is calibrated. `max_per_identity` and `tau` are PROVISIONAL placeholders like
every other number in this repository (docs/CALIBRATION.md); `tau` in particular should come
from `vividly.reid.calibrate`, which picks it at a stated false-accept budget and records
what that budget achieved.

Precision. Rows are stored as float32, which is the `AppearanceEmbedder` contract and half
the memory, and every similarity is accumulated in float64. `tau` is compared with `>=`
against those float64 scores, exactly as `vividly.reid.metrics.dir_far_curve` and
`false_merge_rate` compare, so a tau chosen by the calibration tool is the tau this gallery
applies. Cast-on-read costs nothing at session sizes (tens of identities, tens of rows each)
and removes a whole class of defects in which the threshold behaves differently in
production.

Threading. The pipeline thread enrols while uvicorn's threadpool serves the dashboard
reading `stats()` and `per_identity_counts()`, and a human confirming a proposal can call
`clear()` at session end from a third. Every public method serialises on one `threading.RLock`,
which is the `vividly.tracks.TrackBook` pattern and for the same reason: one uncontended
acquire per call is noise next to an embedder forward pass.

Imports. numpy at module level (this tree's convention, as in `vividly.models.base` and
`vividly.reid.metrics`) plus stdlib. Nothing else, on purpose: this object runs inside the
live pipeline, so it must import on a box with no inference stack, no evaluation tooling and
no web framework. `tests/test_reid_gallery.py` pins that with an AST check.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

#: Guards the division when a centroid is normalised. Same value and same reasoning as
#: `vividly.reid.metrics._L2_EPS` and `vividly.models.reid_backend.l2_normalise`: a zero
#: vector stays zero instead of becoming NaN, because a NaN score poisons every comparison
#: it touches while a zero one simply matches nothing. Duplicated rather than imported so
#: this module keeps depending on numpy and nothing else.
_L2_EPS = 1e-12


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GalleryConfig:
    """Gallery sizing, the accept threshold, and the refinement policy's one knob.

    EVERY numeric default below is PROVISIONAL. Not one has been fitted: no
    deployment-domain embedding-score distribution has been measured, and no pilot shadow
    data exists to fit against. They are placeholders chosen to be structurally sane, not
    operating points. docs/CALIBRATION.md supplies the reasoning and contains no
    re-identification procedure, and `vividly.reid.calibrate` is the tool that produces
    `tau` at a stated false-accept budget in the meantime. A pipeline config supplies these
    through `from_mapping`; nothing here is a number anyone may quote.
    """

    #: Embedding width. 512 matches the OSNet export in `vividly.models.reid_backend`.
    #: Sizing, not a threshold: it is whatever the embedder emits, and a mismatch is a
    #: wiring error that `enroll` and `propose` both refuse.
    dim: int = 512

    # PROVISIONAL (the re-identification scope, POLICY.md; docs/CALIBRATION.md): the accept
    # threshold. A candidate identity scoring below this is not returned at all, so a probe
    # with no candidate above tau is simply a new identity and no human is asked anything.
    # 0.75 is an invented placeholder with the right shape for cosine similarity between
    # L2-normalised appearance vectors: high enough that near-orthogonal strangers never
    # surface, low enough that the same person in the same clothes an hour later can. It is
    # not an operating point. Replace it with the output of `vividly.reid.calibrate`, which
    # chooses tau at a stated false-accept budget and records the achieved false-accept
    # rate, detection and identification rate, and false-merge rate next to it. Raising tau
    # cannot make the system bind anything by itself, because it never binds anything at
    # all; what tau controls is how often a human is asked, and how likely the top
    # suggestion is to be the wrong person.
    tau: float = 0.75

    # PROVISIONAL (the re-identification scope, POLICY.md; docs/CALIBRATION.md): how many
    # embeddings are kept per identity, and therefore how much of the visit's appearance
    # variation the max-over-rows decision rule can see. 16 is a guess balancing two known
    # effects with no measured magnitudes: more rows means a longer enrollment helps (which
    # is why the rule is a max), and more rows also means more chances for one bad view to
    # be the row a stranger matches. The keep-representative eviction in
    # `keep_representative` is what makes the second effect survivable; see the module
    # docstring for why it is not FIFO.
    max_per_identity: int = 16

    #: How many `propose` calls the rolling latency window retains. Sizing, not a decision
    #: threshold: it bounds memory for the dashboard's stats panel and nothing reads it as
    #: a number about people. 256 calls of history is a few seconds to a few minutes of a
    #: real room depending on how often returns happen.
    latency_window: int = 256

    # PROVISIONAL (validation tolerance, not a decision threshold): how far a row's L2 norm
    # may sit from 1.0 before `enroll`/`propose` refuse it. The AppearanceEmbedder contract
    # says rows arrive L2-normalised, and silently renormalising a caller's raw features
    # would hide the wiring bug that produced them, so this is a check and not a repair.
    # 1e-3 is loose enough for float32 accumulation from any sane embedder and tight enough
    # that unnormalised features (norms in the tens) are caught immediately.
    unit_norm_atol: float = 1e-3

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError(f"dim must be >= 1, got {self.dim}")
        if not -1.0 <= self.tau <= 1.0:
            raise ValueError(
                f"tau must be a cosine similarity in [-1, 1], got {self.tau}. A tau at or "
                "below 0 accepts near-orthogonal strangers as candidates and is almost "
                "certainly a units mistake (a distance, not a similarity)."
            )
        if self.max_per_identity < 1:
            raise ValueError(f"max_per_identity must be >= 1, got {self.max_per_identity}")
        if self.latency_window < 1:
            raise ValueError(f"latency_window must be >= 1, got {self.latency_window}")
        if not self.unit_norm_atol > 0.0:
            raise ValueError(f"unit_norm_atol must be > 0, got {self.unit_norm_atol}")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> GalleryConfig:
        """Build from a config block. Unknown keys raise rather than being ignored.

        Same stance as `vividly.tracks.TrackConfig.from_mapping` and
        `vividly.reid.calibrate._check_keys`: a typo in a threshold name that silently kept
        the default is the failure this project cannot afford.
        """
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"unknown GalleryConfig keys: {unknown}; known keys are {sorted(known)}"
            )
        kwargs: dict[str, Any] = {}
        for name in ("dim", "max_per_identity", "latency_window"):
            if name in data:
                kwargs[name] = int(data[name])
        for name in ("tau", "unit_norm_atol"):
            if name in data:
                kwargs[name] = float(data[name])
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# what comes back
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Proposal:
    """A SUGGESTION that a probe may be an already-seen person. It binds nothing.

    Producing one of these has no effect on any track, any chart binding or any alert. It
    is a row on a dashboard with a Confirm button and a Reject button next to it, and until
    a human presses one of them nothing has happened. Both presses are worth writing down:
    a rejection says two people who look alike are not the same person, which is the label
    a false-merge rate is measured against.
    """

    #: Whichever key the caller enrolled under. In the replay pipeline this is a track id
    #: from `vividly.tracks`; the gallery never invents, renames or reuses one.
    identity_key: int
    #: Maximum cosine between the probe and any embedding kept for this identity. Always
    #: >= the configured tau, because lower-scoring candidates are not returned at all.
    #: It is a similarity, not a probability, and it is not calibrated to one.
    confidence: float
    #: How many embeddings that identity currently has KEPT (post-refinement), which is the
    #: number of views the confidence was maximised over. A confidence backed by one view
    #: is weaker evidence than the same confidence backed by sixteen, and the dashboard
    #: should show it.
    n_support: int
    #: Seconds between that identity's most recent enrollment and the probe's own time.
    #: The caller compares this against its tracker's re-association window: inside it, the
    #: tracker already handled the association and no proposal was needed; beyond it, this
    #: is the number that tells a human how long the person was gone. Never clamped, so a
    #: negative value surfaces a caller whose clock ran backwards instead of hiding it.
    elapsed_s: float

    def to_dict(self) -> dict[str, float | int]:
        """JSON-serialisable form, for the dashboard payload."""
        return {
            "identity_key": self.identity_key,
            "confidence": self.confidence,
            "n_support": self.n_support,
            "elapsed_s": self.elapsed_s,
        }


@dataclass(frozen=True)
class ProposeSample:
    """One `propose` call's cost: wall seconds, and how big the gallery was at the time.

    The pair is the point. Match cost grows with gallery size, so a latency number without
    the size beside it cannot be read, and under POLICY.md, rule 9 it cannot be quoted at all
    as a platform figure, because it is a property of the machine that measured it.
    """

    elapsed_s: float
    gallery_size: int


@dataclass(frozen=True)
class ProposeStats:
    """A snapshot of the rolling `propose` latency window, for the dashboard.

    Deliberately as plain as `vividly.models.base.Timing`: counts, a last value, a mean and
    a max, with no percentile machinery. `n_calls` counts every call since construction or
    the last `clear`; every other field is computed over the RETAINED window only
    (`GalleryConfig.latency_window` samples), so on a long session the mean is a recent mean
    and says so here rather than in a comment somewhere downstream.
    """

    n_calls: int
    window: int
    last_s: float
    mean_s: float
    max_s: float
    last_gallery_size: int
    max_gallery_size: int

    @property
    def last_ms(self) -> float:
        return self.last_s * 1000.0

    @property
    def mean_ms(self) -> float:
        return self.mean_s * 1000.0

    def to_dict(self) -> dict[str, float | int]:
        """JSON-serialisable form, for the dashboard payload."""
        return {
            "n_calls": self.n_calls,
            "window": self.window,
            "last_s": self.last_s,
            "mean_s": self.mean_s,
            "max_s": self.max_s,
            "last_ms": self.last_ms,
            "mean_ms": self.mean_ms,
            "last_gallery_size": self.last_gallery_size,
            "max_gallery_size": self.max_gallery_size,
        }


# ---------------------------------------------------------------------------
# refinement
# ---------------------------------------------------------------------------
def keep_representative(rows: np.ndarray, max_keep: int) -> np.ndarray:
    """Trim `rows` to `max_keep` by repeatedly dropping the row farthest from the centroid.

    Distance is cosine to the L2-normalised mean of the rows still in play, recomputed after
    every eviction, so the set converges on the identity's own typical appearance instead of
    on an arithmetic accident of the first pass. Input order is otherwise preserved: the
    result is the surviving rows in the order they arrived.

    Ties (two rows equally far from the centroid, which happens with symmetric fixtures and
    essentially never with real embeddings) drop the EARLIER row, because `np.argmin` returns
    the first minimum and the rows are held in arrival order. Deterministic and written down;
    that is all a tie rule has to be.

    A degenerate set whose mean cancels to zero scores every row 0.0 and therefore drops the
    first one. That configuration cannot arise from an embedder (it needs exactly antipodal
    rows) and is not worth a special case beyond not dividing by zero.

    Returns `rows` itself when it already fits, so the common path allocates nothing.
    """
    if max_keep < 1:
        raise ValueError(f"max_keep must be >= 1, got {max_keep}")
    if rows.shape[0] <= max_keep:
        return rows
    keep = rows
    while keep.shape[0] > max_keep:
        wide = keep.astype(np.float64, copy=False)
        centroid = wide.mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), _L2_EPS)
        drop = int(np.argmin(wide @ centroid))
        keep = np.delete(keep, drop, axis=0)
    return keep


# ---------------------------------------------------------------------------
# the gallery
# ---------------------------------------------------------------------------
class SessionGallery:
    """Every identity seen this session, and the one question it answers: who does this look like.

    Construction takes a `GalleryConfig`, which carries the embedding width, the accept
    threshold tau and the refinement settings, because thresholds live in configs in this
    repository and never inline (POLICY.md, section 7, engineering conventions).

    Lifecycle: `enroll` for every track the pipeline is watching, `propose` for a track the
    tracker could not re-associate on its own, `clear` at session end. There is no fourth
    verb. In particular there is no verb that binds a proposal to an identity: that decision
    is a human's, and the record of it is a label written through the alert store, not a
    mutation of this object.
    """

    def __init__(self, config: GalleryConfig | None = None) -> None:
        self.config = GalleryConfig() if config is None else config
        self._lock = threading.RLock()
        # identity_key -> (k, dim) float32 rows, k <= config.max_per_identity, arrival order.
        self._rows: dict[int, np.ndarray] = {}
        # identity_key -> the newest `t` that identity was enrolled at.
        self._last_seen: dict[int, float] = {}
        self._samples: deque[ProposeSample] = deque(maxlen=self.config.latency_window)
        self._n_calls = 0
        # Newest `t` seen by any enrollment: the fallback clock for `propose`. nan until the
        # first enrollment, which is also the only state in which the gallery is empty and
        # every proposal list is empty anyway.
        self._now: float = math.nan

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        with self._lock:
            return (
                f"SessionGallery(identities={len(self._rows)}, rows={self.size()}, "
                f"tau={self.config.tau}, dim={self.config.dim})"
            )

    # -- ingest -------------------------------------------------------------
    def enroll(self, identity_key: int, embeddings: Any, t: float) -> None:
        """Add appearance rows for one identity, observed at source time `t`.

        `embeddings` is `(n, dim)` float32 and L2-normalised, which is the
        `vividly.models.AppearanceEmbedder` contract; a `(dim,)` vector is refused with a
        message rather than reshaped, because guessing whether one row or one dimension was
        meant is how a silent transpose bug starts. Rows are validated, not repaired: a
        non-finite value or a norm outside `unit_norm_atol` raises.

        `t` is source time in seconds (frame index / fps for a replay source), the same
        clock `vividly.tracks` uses, never wall clock. An identity's `last_seen` only ever
        moves forward, so an out-of-order batch cannot make someone look more recent than
        they are.

        The identity's kept set is refined immediately (see `keep_representative`), so the
        gallery's memory is bounded by `max_per_identity * n_identities` rows no matter how
        long a visit lasts.
        """
        key = _as_key(identity_key)
        when = _as_time(t)
        rows = self._as_rows(embeddings)
        with self._lock:
            current = self._rows.get(key)
            merged = rows if current is None else np.concatenate([current, rows], axis=0)
            self._rows[key] = keep_representative(merged, self.config.max_per_identity)
            previous = self._last_seen.get(key)
            self._last_seen[key] = when if previous is None else max(previous, when)
            if math.isnan(self._now) or when > self._now:
                self._now = when

    # -- the only decision this class makes, and it is not a binding --------
    def propose(self, embedding: Any, top_k: int = 3, *, t: float | None = None) -> list[Proposal]:
        """Candidate identities for one probe, best first. NOTHING is bound by calling this.

        `embedding` is a `(dim,)` float32 L2-normalised vector, validated like `enroll`'s
        rows. The score for an identity is the MAXIMUM cosine over that identity's kept
        embeddings, which is the same rule `vividly.reid.metrics` scores and the benchmark
        reports (a probe should match ANY enrolled view of that person, and a longer
        enrollment should only ever help).

        Candidates scoring below `config.tau` are NOT returned. An empty list therefore has
        a precise meaning and it is the useful one: no enrolled person is a plausible match,
        so this is a new identity and no human needs to be asked anything. An empty gallery
        gives the same answer for the same reason.

        `top_k` caps how many survivors come back, so a dashboard row cannot turn into
        twenty. Ordering is by descending confidence, and an exact tie resolves to the
        SMALLER `identity_key`, matching the tie rule in `vividly.reid.metrics.top1_match`
        so a proposal list and a benchmark row can never disagree about what the system
        would have suggested.

        `t` is the probe's own source time and defaults to the newest enrollment time the
        gallery has seen, which in a live pipeline is the current frame. It exists only to
        compute `Proposal.elapsed_s`; pass it explicitly whenever the caller has a real
        clock, because the fallback goes stale the moment the room empties out.

        Every call records its wall cost and the gallery size into the rolling window that
        `stats()` reads, including calls that return nothing: a rejection costs the same
        matrix work as an acceptance, and a latency panel that only sampled the hits would
        be measuring the wrong thing.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        probe = self._as_vector(embedding)
        started = time.perf_counter()
        with self._lock:
            now = self._now if t is None else _as_time(t)
            tau = self.config.tau
            found: list[Proposal] = []
            for key, rows in self._rows.items():
                confidence = float(np.max(rows.astype(np.float64, copy=False) @ probe))
                if confidence >= tau:
                    found.append(
                        Proposal(
                            identity_key=key,
                            confidence=confidence,
                            n_support=int(rows.shape[0]),
                            elapsed_s=float(now - self._last_seen[key]),
                        )
                    )
            found.sort(key=lambda p: (-p.confidence, p.identity_key))
            out = found[:top_k]
            self._n_calls += 1
            self._samples.append(
                ProposeSample(
                    elapsed_s=time.perf_counter() - started,
                    gallery_size=self._size_locked(),
                )
            )
            return out

    # -- read ---------------------------------------------------------------
    def size(self) -> int:
        """Total embeddings currently kept, across every identity."""
        with self._lock:
            return self._size_locked()

    def per_identity_counts(self) -> dict[int, int]:
        """Kept embeddings per identity, ascending by key. A copy; mutating it changes nothing.

        These are POST-refinement counts, so a value equal to `config.max_per_identity`
        means that identity is at capacity and has been evicting, not that it was seen
        exactly that many times.
        """
        with self._lock:
            return {key: int(rows.shape[0]) for key, rows in sorted(self._rows.items())}

    def stats(self) -> ProposeStats:
        """Snapshot the rolling `propose` latency window. Safe to call from another thread.

        Take one before `clear()` if the run summary needs it: `clear()` resets these
        counters along with everything else.
        """
        with self._lock:
            samples = list(self._samples)
            if not samples:
                return ProposeStats(
                    n_calls=self._n_calls,
                    window=0,
                    last_s=float("nan"),
                    mean_s=float("nan"),
                    max_s=float("nan"),
                    last_gallery_size=0,
                    max_gallery_size=0,
                )
            seconds = [s.elapsed_s for s in samples]
            sizes = [s.gallery_size for s in samples]
            return ProposeStats(
                n_calls=self._n_calls,
                window=len(samples),
                last_s=seconds[-1],
                mean_s=sum(seconds) / len(seconds),
                max_s=max(seconds),
                last_gallery_size=sizes[-1],
                max_gallery_size=max(sizes),
            )

    # -- session end --------------------------------------------------------
    def clear(self) -> None:
        """Forget the entire session: every embedding, every timestamp, every counter.

        This is the deletion CQ-16 names this file for, so it is total rather than
        selective, and it is idempotent because session end is exactly the moment a
        shutdown path is most likely to run twice. After it, the object is indistinguishable
        from a freshly constructed one carrying the same config.
        """
        with self._lock:
            self._rows.clear()
            self._last_seen.clear()
            self._samples.clear()
            self._n_calls = 0
            self._now = math.nan

    # -- internals ----------------------------------------------------------
    def _size_locked(self) -> int:
        return int(sum(rows.shape[0] for rows in self._rows.values()))

    def _as_rows(self, embeddings: Any) -> np.ndarray:
        arr = np.asarray(embeddings)
        if arr.ndim == 1:
            raise ValueError(
                f"embeddings must be a 2-D (n, {self.config.dim}) matrix, got a 1-D array of "
                f"length {arr.shape[0]}. One embedding is a (1, dim) matrix; reshape "
                "explicitly so a transposed batch cannot be read as a single row."
            )
        if arr.ndim != 2:
            raise ValueError(
                f"embeddings must be a 2-D (n, {self.config.dim}) matrix, got shape {arr.shape}"
            )
        if arr.shape[0] < 1:
            raise ValueError(
                "embeddings is empty (0 rows). Enrolling nothing is a caller bug, not an "
                "identity with no appearance; skip the call instead."
            )
        rows = arr.astype(np.float32, copy=True)
        self._check_dim(rows.shape[1])
        self._check_finite(rows)
        self._check_unit(np.linalg.norm(rows.astype(np.float64, copy=False), axis=1))
        return rows

    def _as_vector(self, embedding: Any) -> np.ndarray:
        arr = np.asarray(embedding, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(
                f"a probe is one ({self.config.dim},) embedding, got shape {arr.shape}. "
                "Score a batch by calling propose once per probe: each one is a separate "
                "question for a separate human."
            )
        self._check_dim(arr.shape[0])
        self._check_finite(arr)
        self._check_unit(np.array([float(np.linalg.norm(arr))]))
        return arr

    def _check_dim(self, dim: int) -> None:
        if dim != self.config.dim:
            raise ValueError(
                f"embedding width {dim} does not match the gallery's dim {self.config.dim}. "
                "Both must come from the same embedder; a gallery cannot mix checkpoints."
            )

    def _check_finite(self, arr: np.ndarray) -> None:
        if not np.isfinite(arr).all():
            raise ValueError(
                "embedding contains NaN or inf. A non-finite embedding does not fail loudly "
                "downstream: it silently loses (or wins) every comparison it takes part in."
            )

    def _check_unit(self, norms: np.ndarray) -> None:
        off = np.abs(norms - 1.0) > self.config.unit_norm_atol
        if bool(off.any()):
            worst = float(norms[np.argmax(np.abs(norms - 1.0))])
            raise ValueError(
                f"{int(off.sum())} of {norms.shape[0]} embedding row(s) are not L2-normalised "
                f"(worst norm {worst:.6g}, tolerance {self.config.unit_norm_atol:g}). The "
                "AppearanceEmbedder contract is unit rows so similarity is a dot product; "
                "normalise at the embedder (vividly.models.reid_backend.l2_normalise), not "
                "here, because a gallery that quietly repairs its input hides the bug that "
                "produced it."
            )


def _as_key(identity_key: Any) -> int:
    """Identity keys are integers and are compared for exact equality. Bools are not keys."""
    if isinstance(identity_key, bool):
        raise TypeError("identity_key must be an int, got a bool")
    try:
        return int(identity_key)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"identity_key must be an int, got {identity_key!r}") from exc


def _as_time(t: Any) -> float:
    """Source time in seconds. Finite, because every elapsed number is a difference of these."""
    value = float(t)
    if not math.isfinite(value):
        raise ValueError(
            f"t must be a finite source time in seconds, got {t!r}. Times are subtracted to "
            "produce Proposal.elapsed_s, and a nan there would silently disable every "
            "downstream comparison against the tracker's re-association window."
        )
    return value
