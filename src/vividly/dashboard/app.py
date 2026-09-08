"""Local review dashboard for replay mode.

One FastAPI app over one :class:`PipelineState`. The pipeline runs in its own thread and
publishes a frame at a time into the state; the socket handler reads snapshots out of it.
Nothing here decodes video, runs a model, or talks to a network beyond the loopback socket
uvicorn binds, which is what makes this module safe to import on a Linux CI runner that has
neither cv2 nor numpy installed.

What this is not. It is not patient-facing, it shows no identity, and its chart binding is
a labelled mock: :func:`mock_mrn` builds a string from the tracker id and nothing else, and
the page prints the word MOCK next to every one of them. There is no EHR call, no lookup,
no face detection, embedding or matching (POLICY.md, rule 2), and no vital-signs extraction
of any kind (POLICY.md, rule 3). Staff tagging is a manual button, not a classifier
(docs/PIPELINE.md section 6).

Import discipline. fastapi, uvicorn and websockets are pure-python and cross-platform, so
they are imported at module level. cv2 is darwin-marked in this repo's dev fork and numpy
is absent on the default Linux fork, so :func:`encode_jpeg_b64` imports cv2 inside the
function body and the numpy annotation is a string under TYPE_CHECKING. Nothing else in
this file touches either.

No external assets. ``GET /`` serves one self-contained HTML document: inline CSS, inline
JavaScript, and the frame as a ``data:`` URI. The FastAPI docs routes are disabled because
they load Swagger and ReDoc from a CDN, which would put a cloud dependency into an
appliance that POLICY.md, rule 7 requires to work with the hospital network unplugged.

JSON safety. Track clocks start as nan and a variance can be inf, and neither is valid
JSON: ``JSON.parse`` rejects the literals Python's json module emits for them. Every float
that leaves this module goes through :func:`finite_or_none` first, so a browser never has
to parse ``NaN``.

Re-association proposals, and the one semantic this file exists to hold up. Inside the
tracker's own short re-association window, association is automatic and is exactly what it
was before re-ID existed: the tracker joins the fragments and nothing reaches this module.
Beyond that window nothing is ever auto-bound. The pipeline writes a pending proposal with
a confidence, this page shows it as a card with a Confirm button and a Reject button, and
until a person presses one of them nothing has happened to anybody's identity. Both presses
are written through the versioned alert store as labels (``EventStore.decide_proposal``),
and a confirmation additionally reaches the pipeline's merge hook
(``ReidStage.decide`` -> ``TrackBook.merge_identity``) while a rejection excludes the pair
from ever being asked again this session. No route in this file binds an identity by
itself, and no number in this file gates a decision: the only thresholds in the mechanism
live in configs/pipeline/reid.yaml and every one of them is PROVISIONAL.

Session scope (CQ-16-session-scoped-retention in docs/COUNSEL_QUESTIONS.md). The gallery
those proposals come from is session-scoped: at session end its embeddings are deleted
(``ReidStage.close`` -> ``SessionGallery.clear``). The cards outlive it, because the
proposal rows live in sqlite and hold no embedding, no crop and no appearance vector of
any kind: two integer track ids, a score, a count and a human's answer. The page says so
in the panel header rather than leaving a reviewer to infer it.

Crops. In replay mode a card can show side-by-side representative crops of the two tracks,
served by :func:`create_app`'s crops route out of the run directory and nowhere else: a
name that resolves outside it is refused, whatever it is spelled with. In zero-footage mode
there are no pixels at all, so both sides render as an identity-coloured placeholder tile
carrying the track id. The colour is a display device for telling two cards apart, never an
identity claim.
"""

from __future__ import annotations

import asyncio
import base64
import html
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response

from vividly.events import (
    EventStore,
    EventStoreError,
    ProposalAlreadyDecidedError,
    StoreBusyError,
    UnknownAlertError,
    UnknownProposalError,
)
from vividly.schemas.events import FEEDBACK_LABELS
from vividly.schemas.reassoc import ReassocProposalRecord
from vividly.tracks import TrackBook

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import numpy as np

# Stage names the runner times and this page renders, in pipeline order. Declared here so
# the page can lay out a stable table even before the first frame has been processed.
STAGES: tuple[str, ...] = ("source", "detect", "pose", "track", "rules", "render")

MOCK_MRN_PREFIX = "MRN-"
MOCK_MRN_DIGITS = 4

# The literal the page prints beside every binding. Pinned as a constant because a test
# asserts it is present in the served HTML: an unlabelled fake MRN is the one thing in this
# layer that could be mistaken for a real integration.
MOCK_LABEL = "MOCK"

# ------------------------------------------------------------ proposal display
# Not one of the numbers below gates a decision. They are display caps and one file-size
# ceiling; the thresholds that decide whether a human is asked anything at all (tau,
# min_gap_s, top_k, max_proposals_per_track) live in configs/pipeline/reid.yaml and every
# one of them is PROVISIONAL until both of these exist: fine-tuned weights for the lane
# whose outputs feed the rule, and pilot shadow-mode data.
# See docs/CALIBRATION.md. Nothing here may grow into a threshold: the moment this module
# wants a number that changes what a reviewer is shown as acceptable, it belongs in that
# config file with the others.

#: Subdirectory of the run directory the crops route will serve, and the only one.
CROPS_DIRNAME = "reid_crops"

#: File name a representative crop for one track is looked up under, suffix appended from
#: CROP_MEDIA_TYPES in order. Zero padded so a listing sorts the way a human reads it.
CROP_STEM_TEMPLATE = "track_{track_id:06d}"

#: Image types the crops route will serve, mapped to the media type it answers with. A
#: closed set: this route hands bytes from a directory to a browser, and "whatever mimetypes
#: guesses" is not a policy.
CROP_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

#: Refusal ceiling for one crop, not a tuning knob. A person crop out of a camera frame is
#: tens of kilobytes; anything past this is a wrong file in the directory, and reading it
#: into memory to answer a review page is how a dashboard becomes a way to read a video off
#: the appliance.
MAX_CROP_BYTES = 4 * 1024 * 1024

#: Default provenance written into `decided_via` when the request body names none. A label
#: with no provenance cannot be audited or excluded from a training set later, so the store
#: requires one and this is the honest minimum: the surface, not a person.
DECIDE_VIA_DEFAULT = "dashboard"

#: Longest provenance string accepted. Free text goes into a database column that a later
#: export reads; a cap keeps an accidental paste out of it.
MAX_VIA_CHARS = 200

#: Seconds put in the `Retry-After` header of a 503 raised because another writer held the
#: store past its busy timeout. A hint to the client, not a threshold: nothing in the
#: mechanism reads it, no decision depends on it, and the reviewer's answer is unaffected
#: either way. One second because the contention it describes is one SQLite write, and a
#: number long enough to feel like an outage would be a lie about a lock.
BUSY_RETRY_AFTER_S = 1


def _busy(exc: StoreBusyError, what: str) -> HTTPException:
    """A locked store as a 503 the caller can act on, never a 500 and never a lost label.

    503 plus `Retry-After` is the precise statement: the store is temporarily unavailable,
    nothing was written, and the same request will work shortly. That matters more here than
    on a read, because the request being refused is a person's answer about two patients'
    identities. Refusing it leaves the proposal pending, so the card stays in the queue with
    its buttons live and the same click stores the same label a moment later; a 500 would
    tell a reviewer that something went wrong without telling them whether their answer had
    landed, and the honest reading of that is a label nobody can trust.
    """
    return HTTPException(
        status_code=503,
        detail=f"{what} could not be stored: the alert store is busy. Nothing was written; "
        f"try again. ({exc})",
        headers={"Retry-After": str(BUSY_RETRY_AFTER_S)},
    )


# Every POST here takes an optional JSON object body. Declared once, in Annotated form, so
# the FieldInfo is not a call in a default argument (ruff B008) and so all three routes
# agree on what "no body" means: an empty dict, validated inside the handler rather than by
# a pydantic model, because a 422 from a model is a worse error message for a two-key body
# than the explicit 400 these handlers raise.
JsonBody = Annotated[dict[str, Any], Body(default_factory=dict)]


def mock_mrn(track_id: int, digits: int = MOCK_MRN_DIGITS) -> str:
    """Deterministic mock chart id for a track: ``MRN-`` plus the zero-padded tracker id.

    There is no record system behind this string. It exists so the demo can show what a
    bound chart would look like in the alert feed, and the UI labels every one of them
    MOCK. Real chart binding is out of scope in replay mode (docs/PIPELINE.md section 7).
    """
    return f"{MOCK_MRN_PREFIX}{int(track_id):0{digits}d}"


def finite_or_none(value: float | None) -> float | None:
    """Return the value as a float, or None when it is nan, inf or missing.

    Applied to every float in a socket payload. A nan first_seen_t is the normal state of a
    book that has not seen an observation yet, and shipping it as a JSON literal breaks the
    page rather than the number.
    """
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _finite_map(values: Mapping[str, float | None] | None) -> dict[str, float | None]:
    return {} if values is None else {str(k): finite_or_none(v) for k, v in values.items()}


def _finite_tree(value: Any) -> Any:
    """finite_or_none applied through nested dicts and lists, everything else untouched.

    Alert records carry rule-authored ``details`` and ``baseline`` dicts whose floats never
    pass through the scrubbers above. No current rule can emit a non-finite value, but one
    Infinity in one alert would make JSON.parse reject the whole payload and blank the
    page, so the boundary scrubs rather than trusts.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return finite_or_none(value)
    if isinstance(value, dict):
        return {k: _finite_tree(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite_tree(v) for v in value]
    return value


def identity_color(track_id: int) -> str:
    """A stable CSS colour for one track id. A display device, never an identity claim.

    Hues walk the golden angle so consecutive ids land far apart on the wheel and two cards
    on screen at once are never two shades of the same blue. Deterministic, so the same
    track is the same colour on the candidate tile of one card and the new-track tile of
    another, which is the only thing this is for: telling two placeholder tiles apart in a
    run that has no pixels.

    It carries no information about the person, encodes nothing derived from appearance,
    and is computed from the tracker id alone.
    """
    hue = int(track_id) * 137 % 360
    return f"hsl({hue}, 52%, 42%)"


def crop_name(track_id: int, suffix: str = ".jpg") -> str:
    """File name a representative crop for `track_id` is looked up under in the run dir."""
    return CROP_STEM_TEMPLATE.format(track_id=int(track_id)) + suffix


def find_crop(crops_dir: Path | None, track_id: int) -> str | None:
    """Name of the representative crop on disk for one track, or None when there is none.

    None is the normal answer twice over: a zero-footage run has no pixels to crop, and a
    footage run has none either until something writes them, which nothing in the pipeline
    does today (docs/PIPELINE.md section 9). Both cases render as a placeholder tile rather
    than a broken image, so the card is legible either way.
    """
    if crops_dir is None:
        return None
    for suffix in CROP_MEDIA_TYPES:
        name = crop_name(track_id, suffix)
        if (crops_dir / name).is_file():
            return name
    return None


def is_servable_crop_name(name: str) -> bool:
    """True for a name this app is willing to look up at all, before any path work.

    Suffix allowlist plus a refusal of the two characters that turn a name into something
    other than a name. This is the cheap filter; :func:`resolve_crop_path` is the one that
    actually decides whether a path stays inside the run directory, and neither is trusted
    to be the only guard.
    """
    if not name or "\x00" in name or "\\" in name:
        return False
    return Path(name).suffix.lower() in CROP_MEDIA_TYPES


def resolve_crop_path(crops_dir: Path, name: str) -> Path | None:
    """Absolute path of `name` inside `crops_dir`, or None when it lands anywhere else.

    Both sides are fully resolved before they are compared, which is what makes this hold
    for the three ways a name escapes a directory: ``..`` segments, an absolute path, and a
    symlink inside the directory that points out of it. On macOS it is also what makes the
    comparison work at all, because /var is a symlink to /private/var and an unresolved
    prefix test would reject every legitimate path under a temporary directory.

    Returning None rather than raising keeps the decision testable without an HTTP client;
    the route turns it into a refusal.
    """
    base = crops_dir.resolve()
    try:
        target = (base / name).resolve()
    except (OSError, RuntimeError):  # pragma: no cover - symlink loop or a hostile name
        return None
    return target if target.is_relative_to(base) else None


def _proposal_view(
    record: ReassocProposalRecord,
    *,
    now: float,
    min_gap_s: float | None,
    live_tracks: frozenset[int],
    support: Mapping[int, int],
    crops_dir: Path | None,
) -> dict[str, Any]:
    """One proposal card's data. Read-only: building this decides nothing and binds nothing.

    Two different clocks appear here and they are deliberately not merged into one number.
    `away_at_least_s` is what is known about how long the candidate had been gone: every
    proposal clears the tracker's own window by construction, so the true gap is at least
    that, and the exact figure is not recoverable from a stored row today: `away_s` is the
    slot it would land in, and it is always None because no writer populates it.
    `asked_ago_s` is how long the question has been sitting in the queue on the source
    clock, which is a different fact about the review and not about the person.

    Nothing here is dressed up as a probability. `confidence` is passed through as the score
    it is, and the page renders it as a score.
    """
    ts = finite_or_none(record.ts)
    asked_ago_s = None
    if ts is not None and math.isfinite(now) and now >= ts:
        asked_ago_s = now - ts
    return {
        "proposal_id": record.proposal_id,
        "schema_version": int(record.schema_version),
        "ts": ts,
        "new_track": int(record.new_track),
        "candidate_track": int(record.candidate_track),
        "confidence": finite_or_none(record.confidence),
        "gallery_size": int(record.gallery_size),
        "status": record.status,
        "decided_ts": finite_or_none(record.decided_ts),
        "decided_via": record.decided_via,
        # What is known about the absence, and what is not. See the docstring.
        "away_s": None,
        "away_at_least_s": finite_or_none(min_gap_s),
        "asked_ago_s": asked_ago_s,
        # Evidence a reviewer should see next to the score: the same score against three
        # kept views and against sixteen is not the same evidence.
        "candidate_support": support.get(int(record.candidate_track)),
        # A confirmation on a track that has already left still stores the label, but there
        # is nothing live to carry the identity onto. The card says so before the click.
        "new_track_live": int(record.new_track) in live_tracks,
        "candidate_color": identity_color(record.candidate_track),
        "new_color": identity_color(record.new_track),
        "candidate_crop": find_crop(crops_dir, record.candidate_track),
        "new_crop": find_crop(crops_dir, record.new_track),
    }


def encode_jpeg_b64(image: np.ndarray, quality: int = 70) -> str | None:
    """JPEG-encode one BGR frame and base64 it for the ``data:`` URI on the page.

    Returns None when the encoder refuses the array, which the page renders as "no frame"
    rather than a broken image. cv2 is imported here, not at module level: this file is
    collected by a Linux CI runner that has no cv2 installed.
    """
    import cv2

    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")


@dataclass
class PipelineState:
    """Everything the dashboard reads, and the seam the pipeline writes through.

    The pipeline thread calls :meth:`publish` once per processed frame; the socket handler
    calls :meth:`snapshot` on the event loop thread. One lock covers the mutable fields of
    this dataclass.

    The book and the store are shared directly rather than copied, and each carries its
    own locking. EventStore opens SQLite with ``check_same_thread=False`` behind its own
    lock; TrackBook serializes ingest/reap/tracks/set_staff/bind_mrn on its own RLock
    (the staff-tag and chart-bind endpoints mutate it from uvicorn's threadpool), and
    ``TrackState.baseline()`` computes from a snapshot and writes once. What remains
    deliberately unlocked is cross-frame consistency: a snapshot taken mid-ingest can show
    one track a frame ahead of another. Display-only skew, accepted rather than paid for
    with a lock spanning whole pipeline frames.

    `mode` and `source_label` are display strings. They exist so a screenshot of the page
    always says which flow produced it: a synthetic run and a clip run look similar on
    screen and mean completely different things.

    `reid` is the optional `vividly.reid.stage.ReidStage`, or None when the run has no re-ID
    at all, which is the default and every config that shipped before it existed. It is
    typed `Any` and never imported: `vividly.reid.stage` imports numpy at module level, and
    this package is collected by a Linux CI runner that has none. The stage is reached only
    through the attributes used below, all of which are public on it.

    `run_dir` is the run's own output directory. The crops route serves files from
    `run_dir / CROPS_DIRNAME` and refuses everything else; None means this run serves no
    crops at all, which is the correct answer for a zero-footage run.
    """

    book: TrackBook
    store: EventStore
    run_name: str = "replay"
    mode: str = "replay"
    source_label: str = ""
    # The run's store-scoping identity, distinct from the display-only run_name: one store
    # file can be reached by more than one run, and this is what keeps run B's page from
    # listing (or, with no stage attached, answering) run A's pending questions. None keeps
    # the pre-run-scoping view, which only bare test harnesses should want; the pipeline
    # runner always supplies one (scripts/run_pipeline.py, derive_run_id).
    run_id: str | None = None
    stream_hz: float = 4.0
    alert_limit: int = 25
    # The optional re-ID stage. None disables the proposal panel rather than hiding it: the
    # page says re-ID is off in this run, because a silent empty queue and a queue that
    # cannot exist look identical and mean opposite things.
    #
    # `enabled` in the proposal payload reports whether a stage is attached to this
    # dashboard, not whether re-ID is configured for this run. The two can differ while a
    # caller has not wired the stage through, and the page reports the former because that
    # is what it knows.
    reid: Any = None
    run_dir: Path | None = None
    # The optional egress head (vividly.rules.egress.EgressMonitor), or None when the run
    # has no exit zones configured. It is what the departures panel lists and what its
    # "mark discharged" button reaches; None means the panel says the head is off rather
    # than showing an empty list that would read as "nobody left".
    egress: Any = None
    # Display caps, not thresholds. The pending queue is worked oldest first, so a cap here
    # hides the newest questions and never the oldest one.
    proposal_limit: int = 25
    decided_limit: int = 25

    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)
    _frame_b64: str | None = field(default=None, repr=False, compare=False)
    _frame_index: int = field(default=-1, repr=False, compare=False)
    _t_source: float = field(default=math.nan, repr=False, compare=False)
    _latency_ms: dict[str, float | None] = field(default_factory=dict, repr=False, compare=False)
    _latency_p50_ms: dict[str, float | None] = field(
        default_factory=dict, repr=False, compare=False
    )
    _latency_p95_ms: dict[str, float | None] = field(
        default_factory=dict, repr=False, compare=False
    )
    _frames_published: int = field(default=0, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.stream_hz > 0:
            raise ValueError(f"stream_hz must be > 0, got {self.stream_hz}")
        if self.alert_limit < 1:
            raise ValueError(f"alert_limit must be >= 1, got {self.alert_limit}")
        if self.proposal_limit < 1:
            raise ValueError(f"proposal_limit must be >= 1, got {self.proposal_limit}")
        if self.decided_limit < 1:
            raise ValueError(f"decided_limit must be >= 1, got {self.decided_limit}")
        if self.run_dir is not None:
            self.run_dir = Path(self.run_dir)

    @property
    def crops_dir(self) -> Path | None:
        """The one directory the crops route serves from, or None when the run has no dir."""
        return None if self.run_dir is None else self.run_dir / CROPS_DIRNAME

    @property
    def stream_interval_s(self) -> float:
        """Seconds between socket pushes. The page is a review surface, not a video wall."""
        return 1.0 / self.stream_hz

    @property
    def frames_published(self) -> int:
        with self._lock:
            return self._frames_published

    # -- pipeline side -----------------------------------------------------
    def publish(
        self,
        *,
        frame_b64: str | None = None,
        frame_index: int = -1,
        t_source: float = math.nan,
        latency_ms: Mapping[str, float | None] | None = None,
        latency_p50_ms: Mapping[str, float | None] | None = None,
        latency_p95_ms: Mapping[str, float | None] | None = None,
    ) -> None:
        """Hand the dashboard the frame just processed. Called once per pipeline tick.

        frame_b64 is None in zero-footage mode, because there is no image to show. A stage
        that did not run in this mode (detect, pose and track are all absent from a
        zero-footage run) reports None rather than a number, so the page prints a dash
        instead of implying a detector ran in 0.01 ms. The rest of the payload is identical
        in both modes, which is what keeps the two flows on one code path.
        """
        with self._lock:
            self._frame_b64 = frame_b64
            self._frame_index = int(frame_index)
            self._t_source = float(t_source)
            if latency_ms is not None:
                self._latency_ms = dict(latency_ms)
            if latency_p50_ms is not None:
                self._latency_p50_ms = dict(latency_p50_ms)
            if latency_p95_ms is not None:
                self._latency_p95_ms = dict(latency_p95_ms)
            self._frames_published += 1

    # -- dashboard side ----------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """The socket payload. The keys frame_b64, tracks, alerts, timers and latency_ms are
        fixed and callers may rely on them; the rest are display context.

        Timers are source-clock seconds since the track was first seen, not wall clock, so
        they mean the same thing under faster-than-realtime replay as they would live.
        """
        with self._lock:
            frame_b64 = self._frame_b64
            frame_index = self._frame_index
            t_source = self._t_source
            latency_ms = _finite_map(self._latency_ms)
            latency_p50_ms = _finite_map(self._latency_p50_ms)
            latency_p95_ms = _finite_map(self._latency_p95_ms)
            frames_published = self._frames_published

        now = self.book.now
        tracks = self.book.tracks()
        alerts = [
            _finite_tree(record.model_dump())
            for record in self.store.alerts(limit=self.alert_limit)
        ]
        return {
            "frame_b64": frame_b64,
            "tracks": [_track_view(tracks[tid], now) for tid in sorted(tracks)],
            "alerts": alerts,
            "timers": {str(tid): _elapsed_s(tracks[tid], now) for tid in sorted(tracks)},
            "latency_ms": latency_ms,
            "latency_p50_ms": latency_p50_ms,
            "latency_p95_ms": latency_p95_ms,
            "stages": list(STAGES),
            "frame_index": frame_index,
            "frames_published": frames_published,
            "t_source": finite_or_none(t_source),
            "now": finite_or_none(now),
            "run_name": self.run_name,
            "mode": self.mode,
            "source_label": self.source_label,
            "mock_label": MOCK_LABEL,
            # Pushed on the same socket as everything else, so a proposal written by the
            # pipeline thread is in front of a reviewer within one stream interval without
            # the page polling for it. A store that has been closed under the socket
            # degrades this one panel instead of killing the stream.
            "proposals": self._proposals_or_error(),
            # The egress head's departures still on the clock, or {enabled: false}.
            "departures": self.departures(),
        }

    # -- departures (the egress head) ---------------------------------------
    def departures(self) -> dict[str, Any]:
        """The departures panel payload: pending egress candidates, oldest first.

        `enabled` reports whether an EgressMonitor is attached to this dashboard. With
        none, the page says the head is off; it never shows an empty list for a head that
        does not exist, because an empty list reads as nobody having left, and that is a
        different claim.
        """
        if self.egress is None:
            return {"enabled": False, "pending": []}
        rows = self.egress.pending_view(self.book.now)
        return {
            "enabled": True,
            "rule_id": str(self.egress.rule_id),
            "zones": [zone.name for zone in self.egress.zones.zones],
            "absent_min_s": float(self.egress.cfg.egress_absent_min_s),
            "pending": [_finite_tree(row) for row in rows],
            "stats": _finite_tree(self.egress.stats()),
        }

    # -- re-association proposals ------------------------------------------
    def _proposals_or_error(self) -> dict[str, Any]:
        """`proposals()` for the socket, with a closed store reported rather than raised.

        The socket is a long-lived push loop. An EventStoreError out of here would end the
        stream and blank the whole page including the alert feed, which is a worse answer to
        "the store was closed" than one panel saying so.
        """
        try:
            return self.proposals()
        except EventStoreError as exc:
            return {"available": False, "enabled": self.reid is not None, "error": str(exc)}

    def proposals(self) -> dict[str, Any]:
        """The proposal panel's whole payload: the queue, recent answers, and the session.

        Two queries rather than one. The pending queue comes back oldest first because it is
        a queue to drain, and taking it from the history query instead would let a display
        cap hide the oldest unanswered question, which is the one whose track is most likely
        to have left the room. The decided list is a glance at recent answers and is allowed
        to under-fill when the queue is long; it is not a record of anything, the store is.

        Nothing in here is a rate. `n_proposals` is how often the system asked and
        `n_confirmed` how often a person said yes; the denominator that would turn either
        into an accuracy number is ground truth this pipeline does not have, which is the
        whole reason the questions are asked out loud.
        """
        stage = self.reid
        now = self.book.now
        min_gap_s = None if stage is None else float(stage.min_gap_s)
        support: dict[int, int] = {} if stage is None else dict(stage.gallery.per_identity_counts())
        live = frozenset(self.book.tracks())
        crops_dir = self.crops_dir

        def view(record: ReassocProposalRecord) -> dict[str, Any]:
            return _proposal_view(
                record,
                now=now,
                min_gap_s=min_gap_s,
                live_tracks=live,
                support=support,
                crops_dir=crops_dir,
            )

        # The queue is scoped to this run; the decided glance deliberately is not. A foreign
        # run's pending question must never appear as this run's work, but its answered rows
        # are history, and the store documents history as file-wide on purpose.
        pending = [
            view(r)
            for r in self.store.pending_proposals(run_id=self.run_id, limit=self.proposal_limit)
        ]
        history = self.store.proposal_history(limit=self.proposal_limit + self.decided_limit)
        decided = [view(r) for r in history if r.status != "pending"][: self.decided_limit]

        # One stats() call, not one per field: it takes the stage's lock and walks the
        # gallery, and two calls in one payload could also disagree with each other.
        reid_stats = None if stage is None else stage.stats()
        session: dict[str, Any] = {
            # None means "this run has no re-ID stage", which is a different statement from
            # "the gallery is empty" and the page prints a different sentence for each.
            "gallery_cleared": None if reid_stats is None else bool(reid_stats.cleared),
            "n_identities": len(support),
            "n_gallery_rows": sum(support.values()),
            "min_gap_s": finite_or_none(min_gap_s),
        }
        if stage is not None and reid_stats is not None:
            session["reid"] = _finite_tree(reid_stats.to_dict())
            session["gallery_latency"] = _finite_tree(stage.gallery.stats().to_dict())
        return {
            "available": True,
            "enabled": stage is not None,
            "pending": pending,
            "decided": decided,
            "n_pending_shown": len(pending),
            "n_decided_shown": len(decided),
            "limits": {"pending": self.proposal_limit, "decided": self.decided_limit},
            "crops_available": crops_dir is not None,
            "session": session,
        }


def _elapsed_s(track: Any, now: float) -> float | None:
    """Source-clock seconds between a track's first sighting and the book's clock."""
    if math.isnan(now) or math.isnan(track.first_seen_t):
        return None
    return finite_or_none(now - track.first_seen_t)


def _track_view(track: Any, now: float) -> dict[str, Any]:
    """One row of the people panel. Read-only; the dashboard never mutates a TrackState."""
    observation = track.last_observation
    baseline = track.baseline()
    return {
        "track_id": track.track_id,
        "is_staff": bool(track.is_staff),
        "is_discharged": bool(track.is_discharged),
        "mrn_binding": track.mrn_binding,
        "first_seen_t": finite_or_none(track.first_seen_t),
        "last_seen_t": finite_or_none(track.last_seen_t),
        "seconds_since_first_seen": _elapsed_s(track, now),
        "n_observations": int(track.n_observations),
        "cx": None if observation is None else finite_or_none(observation.cx),
        "cy": None if observation is None else finite_or_none(observation.cy),
        "w": None if observation is None else finite_or_none(observation.w),
        "h": None if observation is None else finite_or_none(observation.h),
        "baseline": None if baseline is None else _baseline_view(baseline),
    }


def _baseline_view(baseline: Any) -> dict[str, Any]:
    view = asdict(baseline)
    for key, value in list(view.items()):
        if isinstance(value, float):
            view[key] = finite_or_none(value)
    return view


async def _await_disconnect(sock: WebSocket) -> None:
    """Resolve as soon as the browser goes away.

    The stream is push-only, so without this the send loop would keep pushing into a closed
    socket until the next send raised. Watching for the disconnect frame in a sibling task
    lets the loop end on the tab closing rather than on an exception.
    """
    try:
        while True:
            message = await sock.receive()
            if message.get("type") == "websocket.disconnect":
                return
    except (WebSocketDisconnect, RuntimeError):
        return


def create_app(state: PipelineState) -> FastAPI:
    """Build the review app over one live pipeline state.

    docs_url and redoc_url are off on purpose: both pages fetch their JavaScript from a
    CDN, and this app has to work with the network cable out (POLICY.md, rule 7).
    """
    app = FastAPI(
        title="Vividly replay review",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
    )
    app.state.pipeline = state

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """The whole UI: one document, no external asset of any kind."""
        return PAGE_HTML

    @app.get("/state")
    def read_state() -> dict[str, Any]:
        """The socket payload over plain HTTP. Debugging aid and a no-websocket fallback."""
        return state.snapshot()

    @app.websocket("/ws")
    async def stream(sock: WebSocket) -> None:
        """Push a snapshot every 1/stream_hz seconds until the browser disconnects.

        The first frame is sent before the loop checks for a disconnect, so a client that
        connects and immediately closes still receives one well-formed payload.
        """
        await sock.accept()
        listener = asyncio.create_task(_await_disconnect(sock))
        try:
            while True:
                await sock.send_json(state.snapshot())
                if listener.done():
                    break
                await asyncio.sleep(state.stream_interval_s)
        except (WebSocketDisconnect, RuntimeError):
            # RuntimeError is what starlette raises for a send after the peer closed. Both
            # mean the same thing here: the reviewer closed the tab.
            pass
        finally:
            listener.cancel()

    @app.post("/alerts/{alert_id}/feedback")
    def set_feedback(alert_id: str, payload: JsonBody) -> dict[str, Any]:
        """Record a reviewer verdict. This route is the labelling interface, not a nicety.

        The two labels are the schema's own Literal values; anything else is a 400 rather
        than a string that quietly poisons every later agreement count.
        """
        label = payload.get("label")
        if label not in FEEDBACK_LABELS:
            raise HTTPException(
                status_code=400,
                detail=f"label must be one of {list(FEEDBACK_LABELS)}, got {label!r}",
            )
        try:
            state.store.set_feedback(alert_id, str(label))
        except UnknownAlertError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StoreBusyError as exc:
            # Ahead of the EventStoreError clause below on purpose: a lock is transient and
            # a retry fixes it, which is a different thing to tell a reviewer than "this
            # store is gone". This route writes, so the verdict is what was refused.
            raise _busy(exc, "the reviewer verdict") from exc
        except EventStoreError as exc:  # pragma: no cover - closed store, operator error
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"alert_id": alert_id, "label": label, "stored": True}

    @app.post("/tracks/{track_id}/staff")
    def set_staff(track_id: int, payload: JsonBody) -> dict[str, Any]:
        """Manual v0 staff tag. There is no staff classifier and there is no identity claim.

        No rule reads this flag since the no_staff_contact_v0 rule was withdrawn; it
        remains a reviewer affordance (the page distinguishes staff from patients) and it
        rides identity carry-forward in TrackBook.merge_identity.
        """
        if "staff" not in payload or not isinstance(payload["staff"], bool):
            raise HTTPException(status_code=400, detail='body must be {"staff": true|false}')
        staff = bool(payload["staff"])
        try:
            state.book.set_staff(track_id, staff)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"track_id": track_id, "staff": staff}

    @app.post("/tracks/{track_id}/discharged")
    def set_discharged(track_id: int, payload: JsonBody) -> dict[str, Any]:
        """Manual v0 discharged tag on a live track. A reviewer's statement, no integration.

        The egress head reads it at departure time: a discharged person walking out through
        an exit zone is not an event. For someone who has already left, the lever is
        POST /departures/{track_id}/discharged below.
        """
        if "discharged" not in payload or not isinstance(payload["discharged"], bool):
            raise HTTPException(status_code=400, detail='body must be {"discharged": true|false}')
        discharged = bool(payload["discharged"])
        try:
            state.book.set_discharged(track_id, discharged)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"track_id": track_id, "discharged": discharged}

    @app.get("/api/departures")
    def read_departures() -> dict[str, Any]:
        return state.departures()

    @app.post("/departures/{track_id}/discharged")
    def discharge_departed(track_id: int, payload: JsonBody) -> dict[str, Any]:
        """Retire a pending egress candidate: the reviewer says that person was discharged.

        409 when this run has no egress head (there is nothing to retire, and pretending
        otherwise would hide that the door is unwatched); 404 when no departure by that id
        is on the clock (it already fired, was already retired, or never existed). The body
        is ignored: there is nothing to type, only a person to vouch for.
        """
        if state.egress is None:
            raise HTTPException(status_code=409, detail="egress monitoring is off in this run")
        if not state.egress.mark_discharged(track_id):
            raise HTTPException(
                status_code=404, detail=f"track {track_id} is not a pending departure"
            )
        return {"track_id": track_id, "discharged": True, "departed": True}

    @app.post("/tracks/{track_id}/chart-bind")
    def chart_bind(track_id: int, payload: JsonBody) -> dict[str, Any]:
        """Bind the labelled MOCK chart id. No EHR call, no lookup, no patient data.

        The body is ignored: the id is derived from the tracker id so that a demo cannot
        accidentally type a real medical record number into this repository.
        """
        mrn = mock_mrn(track_id)
        try:
            state.book.bind_mrn(track_id, mrn)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"track_id": track_id, "mrn_binding": mrn, "mock": True, "label": MOCK_LABEL}

    # -- re-association proposals ------------------------------------------
    #
    # Three routes and one rule: beyond the tracker's own re-association window, this app
    # never binds an identity. GET asks what is waiting, the two POSTs record what a person
    # answered, and there is no fourth verb that a machine could use to answer for them.

    @app.get("/api/proposals")
    def read_proposals() -> dict[str, Any]:
        """The pending queue, recent answers, and the session's own retention state."""
        try:
            return state.proposals()
        except EventStoreError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def _decide(proposal_id: str, decision: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Shared body of the two decision routes. Never called with a decision from a body.

        The decision is the route, not a field: a single endpoint taking {"decision": ...}
        would make "confirm" and "reject" one typo apart on the wire, and the typo would
        merge two patients' identities. `via` is the only thing the body may set.

        `decided_ts` is wall clock, deliberately, and it is the one clock in this system that
        is not source time. It answers "when did a person answer", which is a fact about the
        review and not about the footage, and under faster-than-realtime replay the source
        clock would answer a different question entirely. `vividly.schemas.reassoc` records
        that the two clocks are exported side by side and never subtracted.
        """
        via = payload.get("via", DECIDE_VIA_DEFAULT)
        if not isinstance(via, str) or not via.strip():
            raise HTTPException(
                status_code=400,
                detail='"via" must be a non-empty string naming where the decision came from',
            )
        via = via.strip()
        if len(via) > MAX_VIA_CHARS:
            raise HTTPException(
                status_code=400, detail=f'"via" must be at most {MAX_VIA_CHARS} characters'
            )
        decided_ts = time.time()
        try:
            if state.reid is None:
                # No stage in this run: the label still lands, because a store that holds
                # the question can hold the answer, and there is simply nothing live to
                # merge onto. Said out loud in `note` rather than implied by merged=False.
                # Scoped to this run's id the same way the stage's decide path is: without
                # it, the no-stage fallback was the one door through which run B could
                # answer run A's question.
                record = state.store.decide_proposal(
                    proposal_id, decision, decided_ts, via, run_id=state.run_id
                )
                merged, note = False, "no re-ID stage in this run: the label is stored only"
            else:
                outcome = state.reid.decide(proposal_id, decision, decided_ts, via)
                record, merged, note = outcome.record, bool(outcome.merged), str(outcome.note)
        except UnknownProposalError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ProposalAlreadyDecidedError as exc:
            # 409, not 200 and not 400. Two reviewers on one proposal is the normal way this
            # happens, and the second one has to be told that the stored answer is not
            # theirs. Decisions are written once (vividly.schemas.reassoc).
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except StoreBusyError as exc:
            # A lock held past the busy timeout, on the one route in this app where losing a
            # write would lose a person's answer about two patients. 503 + Retry-After says
            # nothing was stored and the same click will work; the proposal is untouched and
            # still pending, so the card comes back with its buttons live on the next push.
            raise _busy(exc, "the re-association decision") from exc
        except EventStoreError as exc:  # pragma: no cover - closed store, operator error
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "proposal_id": record.proposal_id,
            "decision": decision,
            "status": record.status,
            "decided_ts": finite_or_none(record.decided_ts),
            "decided_via": record.decided_via,
            "merged": merged,
            "note": note,
            "stored": True,
        }

    @app.post("/api/proposals/{proposal_id}/confirm")
    def confirm_proposal(proposal_id: str, payload: JsonBody) -> dict[str, Any]:
        """A human says these two tracks are the same person. This writes the label.

        The label lands first and always. Whether the pipeline could also carry the identity
        forward onto a live track is reported in `merged` and `note`, and a confirmation on a
        track that has already left the room is a true statement worth storing even though
        there is nothing left to merge onto.
        """
        return _decide(proposal_id, "confirmed", payload)

    @app.post("/api/proposals/{proposal_id}/reject")
    def reject_proposal(proposal_id: str, payload: JsonBody) -> dict[str, Any]:
        """A human says these two tracks are different people. This writes the label too.

        A rejection is not a no-op and not a dismissal: it is the label a false-merge rate is
        measured against, and it excludes the pair from being proposed again this session.
        """
        return _decide(proposal_id, "rejected", payload)

    @app.get("/api/proposals/crops/{name:path}")
    def proposal_crop(name: str) -> Response:
        """One representative crop, read out of this run's crop directory and nowhere else.

        The path parameter takes the rest of the URL on purpose, so that a name carrying
        `..` reaches the containment check instead of failing to match a route. Every answer
        below is a refusal except the last one:

          * no run directory: this run serves no crops (a zero-footage run never can);
          * a name that is not an allowed image name: refused before any path work;
          * a name that resolves outside the crop directory: 403, whether it got there by
            `..`, by being absolute, or through a symlink inside the directory;
          * anything past MAX_CROP_BYTES: refused rather than read into memory.

        Nothing here consults the request for a base directory, so there is no header, query
        or body that can move it.
        """
        crops_dir = state.crops_dir
        if crops_dir is None:
            raise HTTPException(status_code=404, detail="this run serves no crops")
        if not is_servable_crop_name(name):
            raise HTTPException(
                status_code=404,
                detail=f"not a servable crop name; expected one of {sorted(CROP_MEDIA_TYPES)}",
            )
        target = resolve_crop_path(crops_dir, name)
        if target is None:
            raise HTTPException(
                status_code=403, detail="crop paths outside the run directory are not served"
            )
        if not target.is_file():
            raise HTTPException(status_code=404, detail="no such crop in this run")
        if target.stat().st_size > MAX_CROP_BYTES:
            raise HTTPException(
                status_code=413, detail=f"crop is larger than {MAX_CROP_BYTES} bytes"
            )
        return Response(
            content=target.read_bytes(),
            media_type=CROP_MEDIA_TYPES[target.suffix.lower()],
            # A crop is a review artifact of one run; a cached copy outliving the session
            # would be an appearance image sitting in a browser cache after the gallery that
            # produced it was deleted (CQ-16-session-scoped-retention in
            # docs/COUNSEL_QUESTIONS.md).
            headers={"Cache-Control": "no-store"},
        )

    return app


# --------------------------------------------------------------------- the page
# One document, inline everything. The only dynamic value is the socket payload, and every
# string from it reaches the DOM through textContent or createTextNode, never through
# innerHTML, so a rule id or a details value carrying an odd character cannot rewrite the
# page. The one server-side substitution (MOCK_LABEL) goes through html.escape for the same
# defence-in-depth reason, even though it is a module constant today.
_PAGE_TEMPLATE = """<!-- Vividly replay review. Structure only, replay mode. No external assets by design. -->
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vividly replay review</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #14161a; color: #e8eaed;
         font-family: ui-sans-serif, system-ui, sans-serif; font-size: 14px; }
  header { padding: 12px 18px; border-bottom: 1px solid #2a2e35; }
  h1 { font-size: 16px; margin: 0 0 4px 0; font-weight: 600; }
  .meta { color: #9aa0a6; font-size: 12px; }
  .banner { margin: 10px 18px 0 18px; padding: 8px 12px; border-radius: 6px;
            background: #3a2c12; color: #f7d08a; border: 1px solid #6b5220; font-size: 12px; }
  main { display: grid; grid-template-columns: minmax(320px, 3fr) minmax(280px, 2fr);
         gap: 14px; padding: 14px 18px; align-items: start; }
  section { background: #1b1e24; border: 1px solid #2a2e35; border-radius: 8px; padding: 12px; }
  h2 { font-size: 13px; margin: 0 0 10px 0; color: #9aa0a6; text-transform: uppercase;
       letter-spacing: 0.06em; font-weight: 600; }
  #frame { width: 100%; border-radius: 6px; display: none; background: #000; }
  #frame-empty { color: #9aa0a6; font-size: 12px; padding: 24px 0; text-align: center; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 5px 6px; border-bottom: 1px solid #262a31;
           vertical-align: middle; }
  th { color: #9aa0a6; font-weight: 500; font-size: 12px; }
  td.num { font-variant-numeric: tabular-nums; }
  button { font: inherit; font-size: 12px; padding: 3px 8px; border-radius: 5px;
           border: 1px solid #3a3f47; background: #23272e; color: #e8eaed; cursor: pointer; }
  button:hover { background: #2c313a; }
  button.on { background: #1f4d33; border-color: #2f7a51; color: #cdebd9; }
  button.agree { border-color: #2f7a51; }
  button.false-alarm { border-color: #7a3b2f; }
  .mock { font-size: 10px; letter-spacing: 0.08em; padding: 1px 5px; border-radius: 4px;
          background: #4a3410; color: #f7d08a; border: 1px solid #6b5220; margin-right: 6px; }
  .alert { border: 1px solid #2a2e35; border-radius: 6px; padding: 8px 10px; margin-bottom: 8px; }
  .alert .rule { font-weight: 600; }
  .alert .facts { color: #9aa0a6; font-size: 12px; margin: 3px 0 6px 0;
                  font-variant-numeric: tabular-nums; }
  .verdict { font-size: 12px; color: #8fd0a6; margin-left: 8px; }
  .empty { color: #6b7178; font-size: 12px; }

  /* Re-association proposals. Full width because two crops sit side by side. */
  #proposals-section { grid-column: 1 / -1; }
  .panel-note { color: #9aa0a6; font-size: 12px; margin: 0 0 10px 0; line-height: 1.5; }
  .panel-note strong { color: #e8eaed; font-weight: 600; }
  .proposal { border: 1px solid #2a2e35; border-radius: 6px; padding: 10px 12px;
              margin-bottom: 10px; }
  .proposal.decided { opacity: 0.72; }
  .pair { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin: 8px 0; }
  .side { display: flex; flex-direction: column; align-items: center; gap: 4px; width: 116px; }
  .side .cap { color: #9aa0a6; font-size: 11px; text-align: center; }
  .tile { width: 96px; height: 128px; border-radius: 6px; border: 1px solid #3a3f47;
          display: flex; align-items: center; justify-content: center; color: #0d0f12;
          font-weight: 700; font-size: 15px; font-variant-numeric: tabular-nums; }
  .tile img { width: 100%; height: 100%; object-fit: cover; border-radius: 5px; display: block; }
  .ask { color: #9aa0a6; font-size: 12px; max-width: 220px; }
  .score { font-weight: 600; font-variant-numeric: tabular-nums; }
  .pill { font-size: 10px; letter-spacing: 0.08em; padding: 1px 5px; border-radius: 4px;
          border: 1px solid #3a3f47; color: #9aa0a6; margin-left: 6px; }
  .pill.warn { background: #3a2c12; color: #f7d08a; border-color: #6b5220; }
  .pill.yes { background: #1f4d33; color: #cdebd9; border-color: #2f7a51; }
  .pill.no { background: #4a2620; color: #f0c4bb; border-color: #7a3b2f; }
  button.confirm { border-color: #2f7a51; }
  button.reject { border-color: #7a3b2f; }
  .pnote { color: #9aa0a6; font-size: 12px; margin-top: 6px; }
  footer { padding: 4px 18px 18px 18px; color: #6b7178; font-size: 11px; }
</style>

<header>
  <h1>Vividly replay review</h1>
  <div class="meta">
    run <span id="run-name">...</span>
    &middot; mode <span id="mode">...</span>
    &middot; source <span id="source-label">...</span>
    &middot; frame <span id="frame-index">...</span>
    &middot; source clock <span id="t-source">...</span>
    &middot; socket <span id="socket">connecting</span>
  </div>
</header>

<div class="banner">
  Replay mode only: every frame here comes from a licensed clip or a generated
  fixture. Every rule threshold is PROVISIONAL until pilot shadow-mode data exists, so no
  count on this page is a false-alarm rate. Chart binding is a __MOCK_LABEL__: the id is
  built from the tracker id and there is no record system behind it. Staff tagging is a
  manual tag, not a classifier.
</div>

<main>
  <section>
    <h2>Frame</h2>
    <img id="frame" alt="current replay frame">
    <div id="frame-empty">No frame in this run (zero-footage mode reads observations from a
      fixture, so there is no image to show).</div>
    <h2 style="margin-top:14px">Per-stage latency (ms)</h2>
    <table>
      <thead><tr><th>stage</th><th>last</th><th>p50</th><th>p95</th></tr></thead>
      <tbody id="latency"></tbody>
    </table>
  </section>

  <section>
    <h2>People in the room</h2>
    <table>
      <thead>
        <tr><th>track</th><th>waiting</th><th>staff tag</th><th>discharged</th><th>chart (__MOCK_LABEL__)</th></tr>
      </thead>
      <tbody id="tracks"></tbody>
    </table>
    <div id="tracks-empty" class="empty">No live tracks yet.</div>

    <h2 style="margin-top:16px">Alert feed</h2>
    <div id="alerts"></div>
    <div id="alerts-empty" class="empty">No alerts yet.</div>

    <h2 style="margin-top:16px">Departures on the egress clock</h2>
    <p class="panel-note">
      A person who left through a configured exit zone and has not been confirmed back.
      If they were discharged, say so here and nothing fires; otherwise the egress
      candidate fires when the clock runs out. Thresholds PROVISIONAL (docs/CALIBRATION.md).
    </p>
    <div id="departures"></div>
    <div id="departures-empty" class="empty">No departures on the clock.</div>
  </section>

  <section id="proposals-section">
    <h2>Re-association proposals (a human decides, this page never binds)</h2>
    <p class="panel-note">
      Inside the tracker's own short re-association window, association is
      <strong>automatic and unchanged</strong>. Beyond it nothing is ever bound automatically:
      each card below is a question the system declined to answer by itself. Confirm and
      Reject are both <strong>labels</strong> and both are written down; a rejection is the
      label a false-merge rate is measured against, not a dismissal. A decision is written
      once, so a card someone else has already answered comes back as a conflict rather than
      overwriting their answer.
    </p>
    <p class="panel-note">
      The gallery is <strong>session-scoped</strong> (CQ-16-session-scoped-retention in
      docs/COUNSEL_QUESTIONS.md): at session end its appearance
      embeddings are deleted. These cards keep being served afterwards because they come from
      the alert database, which holds
      no embedding, no crop and no appearance vector of any kind:
      two track ids, a score, a count and a person's answer.
      <span id="session-note" class="empty"></span>
    </p>
    <p class="panel-note">
      The score is PROVISIONAL and it is not a probability that two people are the same one.
      It is calibrated only in the sense of being mapped into 0 to 1, against thresholds that
      have never been fitted to this domain (configs/pipeline/reid.yaml, docs/CALIBRATION.md).
      Tile colours are a device for telling two cards apart and say nothing about anybody.
    </p>
    <div id="proposals"></div>
    <div id="proposals-empty" class="empty">No proposals awaiting a decision.</div>
    <h2 style="margin-top:16px">Recently decided</h2>
    <div id="proposals-decided"></div>
    <div id="proposals-decided-empty" class="empty">Nothing has been decided yet.</div>
    <div id="proposals-stats" class="panel-note" style="margin-top:10px"></div>
  </section>
</main>

<footer>
  Structure only. Not a pilot, not a deployment, nothing patient-facing.
  See docs/PIPELINE.md and docs/CALIBRATION.md.
</footer>

<script>
(function () {
  "use strict";

  var MOCK_LABEL_TEXT = "__MOCK_LABEL__";

  var byId = function (id) { return document.getElementById(id); };

  function setText(id, value) {
    var node = byId(id);
    if (node) { node.textContent = value; }
  }

  function fmtSeconds(value) {
    if (value === null || value === undefined) { return "-"; }
    var total = Math.max(0, Math.round(value));
    var mins = Math.floor(total / 60);
    var secs = total % 60;
    return mins + "m " + (secs < 10 ? "0" : "") + secs + "s";
  }

  function fmtMs(value) {
    if (value === null || value === undefined) { return "-"; }
    return value.toFixed(2);
  }

  function fmtClock(value) {
    if (value === null || value === undefined) { return "-"; }
    return value.toFixed(2) + " s";
  }

  function postJSON(url, body) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {})
    }).then(function (response) {
      if (!response.ok) {
        return response.text().then(function (text) {
          window.console.warn("POST " + url + " failed: " + response.status + " " + text);
        });
      }
      return null;
    });
  }

  function makeButton(label, className, onClick) {
    var button = document.createElement("button");
    button.textContent = label;
    if (className) { button.className = className; }
    button.addEventListener("click", onClick);
    return button;
  }

  function renderTracks(tracks, timers) {
    var body = byId("tracks");
    body.textContent = "";
    byId("tracks-empty").style.display = tracks.length ? "none" : "block";

    tracks.forEach(function (track) {
      var row = document.createElement("tr");

      var idCell = document.createElement("td");
      idCell.className = "num";
      idCell.textContent = String(track.track_id);
      row.appendChild(idCell);

      var timerCell = document.createElement("td");
      timerCell.className = "num";
      var seconds = timers[String(track.track_id)];
      timerCell.textContent = fmtSeconds(
        seconds === undefined ? track.seconds_since_first_seen : seconds
      );
      row.appendChild(timerCell);

      var staffCell = document.createElement("td");
      var staffButton = makeButton(
        track.is_staff ? "staff" : "not staff",
        track.is_staff ? "on" : "",
        function () {
          postJSON("/tracks/" + track.track_id + "/staff", { staff: !track.is_staff });
        }
      );
      staffCell.appendChild(staffButton);
      row.appendChild(staffCell);

      var dischargeCell = document.createElement("td");
      var dischargeButton = makeButton(
        track.is_discharged ? "discharged" : "not discharged",
        track.is_discharged ? "on" : "",
        function () {
          postJSON("/tracks/" + track.track_id + "/discharged",
                   { discharged: !track.is_discharged });
        }
      );
      dischargeCell.appendChild(dischargeButton);
      row.appendChild(dischargeCell);

      var chartCell = document.createElement("td");
      if (track.mrn_binding) {
        var badge = document.createElement("span");
        badge.className = "mock";
        badge.textContent = MOCK_LABEL_TEXT;
        chartCell.appendChild(badge);
        var bound = document.createElement("span");
        bound.textContent = track.mrn_binding;
        chartCell.appendChild(bound);
      } else {
        chartCell.appendChild(makeButton("bind " + MOCK_LABEL_TEXT + " chart", "", function () {
          postJSON("/tracks/" + track.track_id + "/chart-bind", {});
        }));
      }
      row.appendChild(chartCell);

      body.appendChild(row);
    });
  }

  function renderAlerts(alerts) {
    var wrap = byId("alerts");
    wrap.textContent = "";
    byId("alerts-empty").style.display = alerts.length ? "none" : "block";

    alerts.forEach(function (alert) {
      var card = document.createElement("div");
      card.className = "alert";

      var head = document.createElement("div");
      var rule = document.createElement("span");
      rule.className = "rule";
      rule.textContent = alert.rule_id + " " + alert.rule_version;
      head.appendChild(rule);
      var who = document.createElement("span");
      who.textContent = "  track " + alert.track_id;
      head.appendChild(who);
      if (alert.mrn_binding) {
        var badge = document.createElement("span");
        badge.className = "mock";
        badge.style.marginLeft = "8px";
        badge.textContent = MOCK_LABEL_TEXT + " " + alert.mrn_binding;
        head.appendChild(badge);
      }
      if (alert.feedback) {
        var verdict = document.createElement("span");
        verdict.className = "verdict";
        verdict.textContent = "reviewed: " + alert.feedback;
        head.appendChild(verdict);
      }
      card.appendChild(head);

      var facts = document.createElement("div");
      facts.className = "facts";
      facts.textContent =
        "onset " + fmtClock(alert.onset_ts) +
        "  detected " + fmtClock(alert.detected_ts) +
        "  id " + alert.alert_id;
      card.appendChild(facts);

      var buttons = document.createElement("div");
      buttons.appendChild(makeButton("Agree", "agree", function () {
        postJSON("/alerts/" + encodeURIComponent(alert.alert_id) + "/feedback",
                 { label: "agree" });
      }));
      var spacer = document.createTextNode(" ");
      buttons.appendChild(spacer);
      buttons.appendChild(makeButton("False alarm", "false-alarm", function () {
        postJSON("/alerts/" + encodeURIComponent(alert.alert_id) + "/feedback",
                 { label: "false_alarm" });
      }));
      card.appendChild(buttons);

      wrap.appendChild(card);
    });
  }

  function renderDepartures(block) {
    var wrap = byId("departures");
    var empty = byId("departures-empty");
    wrap.textContent = "";
    if (!block || !block.enabled) {
      empty.textContent = "Egress monitoring is off in this run (no exit zones configured).";
      empty.style.display = "block";
      return;
    }
    var rows = block.pending || [];
    empty.textContent = "No departures on the clock.";
    empty.style.display = rows.length ? "none" : "block";
    rows.forEach(function (d) {
      var line = document.createElement("div");
      var text = document.createElement("span");
      text.textContent = "track " + d.track_id + " left via " + d.zone + " at " +
        fmtClock(d.last_seen_t) + ", away " + fmtSeconds(d.absent_s) +
        ", fires at " + fmtClock(d.fires_at) +
        (d.mrn_binding ? " (" + MOCK_LABEL_TEXT + " " + d.mrn_binding + ")" : "");
      line.appendChild(text);
      line.appendChild(document.createTextNode(" "));
      line.appendChild(makeButton("mark discharged", "", function () {
        postJSON("/departures/" + d.track_id + "/discharged", {});
      }));
      wrap.appendChild(line);
    });
  }

  function renderLatency(stages, last, p50, p95) {
    var body = byId("latency");
    body.textContent = "";
    stages.forEach(function (stage) {
      var row = document.createElement("tr");
      var name = document.createElement("td");
      name.textContent = stage;
      row.appendChild(name);
      [last, p50, p95].forEach(function (table) {
        var cell = document.createElement("td");
        cell.className = "num";
        cell.textContent = fmtMs(table ? table[stage] : null);
        row.appendChild(cell);
      });
      body.appendChild(row);
    });
  }

  // ------------------------------------------------------ proposals
  // Nothing in this block binds anything. It draws a question and sends whichever answer a
  // person clicked. The two answers go to two different URLs on purpose: one endpoint
  // taking a decision field would put "confirm" and "reject" one typo apart on the wire.

  function fmtScore(value) {
    if (value === null || value === undefined) { return "-"; }
    return value.toFixed(2);
  }

  function postDecision(url) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    }).then(function (response) {
      return response.text().then(function (text) {
        var data = null;
        try { data = JSON.parse(text); } catch (err) { data = null; }
        return { ok: response.ok, status: response.status, data: data, text: text };
      });
    });
  }

  function identityTile(trackId, color, cropName) {
    var tile = document.createElement("div");
    tile.className = "tile";
    tile.style.backgroundColor = color;
    tile.textContent = "#" + trackId;
    if (cropName) {
      var image = document.createElement("img");
      image.alt = "representative crop of track " + trackId;
      // The crop route serves this run's crop directory and refuses everything else. On a
      // miss the tile falls back to its colour rather than showing a broken image.
      image.src = "/api/proposals/crops/" + encodeURIComponent(cropName);
      image.addEventListener("error", function () {
        tile.textContent = "#" + trackId;
      });
      tile.textContent = "";
      tile.appendChild(image);
    }
    return tile;
  }

  function side(trackId, color, cropName, caption) {
    var wrap = document.createElement("div");
    wrap.className = "side";
    wrap.appendChild(identityTile(trackId, color, cropName));
    var cap = document.createElement("div");
    cap.className = "cap";
    cap.textContent = caption;
    wrap.appendChild(cap);
    return wrap;
  }

  function proposalFacts(p) {
    var bits = ["score " + fmtScore(p.confidence) + " (PROVISIONAL)"];
    if (p.away_s !== null && p.away_s !== undefined) {
      bits.push("away " + fmtSeconds(p.away_s));
    } else if (p.away_at_least_s !== null && p.away_at_least_s !== undefined) {
      bits.push("away at least " + fmtSeconds(p.away_at_least_s));
    }
    bits.push("gallery " + p.gallery_size + " identities at the time");
    if (p.candidate_support !== null && p.candidate_support !== undefined) {
      bits.push(p.candidate_support + " kept view" + (p.candidate_support === 1 ? "" : "s") +
                " of the candidate");
    }
    if (p.asked_ago_s !== null && p.asked_ago_s !== undefined) {
      bits.push("asked " + fmtSeconds(p.asked_ago_s) + " ago on the source clock");
    }
    bits.push("id " + p.proposal_id);
    return bits.join("  .  ");
  }

  function proposalCard(p) {
    var card = document.createElement("div");
    card.className = p.status === "pending" ? "proposal" : "proposal decided";

    var head = document.createElement("div");
    var score = document.createElement("span");
    score.className = "score";
    score.textContent = "score " + fmtScore(p.confidence);
    head.appendChild(score);
    if (p.status !== "pending") {
      var verdict = document.createElement("span");
      verdict.className = p.status === "confirmed" ? "pill yes" : "pill no";
      verdict.textContent = p.status + (p.decided_via ? " by " + p.decided_via : "");
      head.appendChild(verdict);
    } else if (!p.new_track_live) {
      var gone = document.createElement("span");
      gone.className = "pill warn";
      gone.textContent = "track " + p.new_track + " has already left";
      head.appendChild(gone);
    }
    card.appendChild(head);

    var pair = document.createElement("div");
    pair.className = "pair";
    pair.appendChild(side(p.candidate_track, p.candidate_color, p.candidate_crop,
                          "candidate: track " + p.candidate_track + ", seen earlier"));
    var ask = document.createElement("div");
    ask.className = "ask";
    // A decided card is a record of an answer, not a question. Leaving the question on it
    // would invite a second click on something that is already written.
    ask.textContent = p.status === "pending"
      ? "Is the person on the right the person on the left? Nothing is bound until you answer."
      : "Answered: " + p.status + ". This is the stored label and it is written once.";
    pair.appendChild(ask);
    pair.appendChild(side(p.new_track, p.new_color, p.new_crop,
                          "new: track " + p.new_track + ", just appeared"));
    card.appendChild(pair);

    var facts = document.createElement("div");
    facts.className = "facts";
    facts.textContent = proposalFacts(p);
    card.appendChild(facts);

    var note = document.createElement("div");
    note.className = "pnote";

    if (p.status === "pending") {
      var buttons = document.createElement("div");
      var confirm = null;
      var reject = null;

      var answer = function (suffix) {
        return function () {
          // Disabled before the request goes out, not after it comes back. Two clicks on
          // one card is exactly how a person meets the write-once rule as a 409.
          confirm.disabled = true;
          reject.disabled = true;
          note.textContent = "sending" + suffix + " ...";
          postDecision("/api/proposals/" + encodeURIComponent(p.proposal_id) + suffix)
            .then(function (result) {
              if (result.ok && result.data) {
                note.textContent = result.data.note || ("stored: " + result.data.status);
              } else if (result.status === 409) {
                note.textContent = "another reviewer has already answered this card; your " +
                                   "click changed nothing. Decisions are written once.";
              } else {
                note.textContent = "not stored (" + result.status + "): " + result.text;
                confirm.disabled = false;
                reject.disabled = false;
              }
            });
        };
      };

      // The two answers go to two literal paths, spelled out here rather than assembled
      // from a variable, so that grepping this page for what it can do to a stored identity
      // finds both of them.
      confirm = makeButton("Confirm same person", "confirm", answer("/confirm"));
      reject = makeButton("Reject, different people", "reject", answer("/reject"));
      buttons.appendChild(confirm);
      buttons.appendChild(document.createTextNode(" "));
      buttons.appendChild(reject);
      card.appendChild(buttons);
    }

    card.appendChild(note);
    return card;
  }

  function renderProposalList(containerId, emptyId, items) {
    var wrap = byId(containerId);
    // Rebuilt from the payload on every push. A card whose buttons the reviewer just
    // disabled is replaced by one carrying the stored answer, which is the state the store
    // agrees with rather than the state this tab remembers.
    wrap.textContent = "";
    byId(emptyId).style.display = items.length ? "none" : "block";
    items.forEach(function (p) { wrap.appendChild(proposalCard(p)); });
  }

  function renderProposals(block) {
    var stats = byId("proposals-stats");
    if (!block || block.available === false) {
      renderProposalList("proposals", "proposals-empty", []);
      renderProposalList("proposals-decided", "proposals-decided-empty", []);
      setText("session-note", "");
      stats.textContent = block && block.error
        ? "proposal store unavailable: " + block.error
        : "";
      return;
    }
    if (!block.enabled) {
      renderProposalList("proposals", "proposals-empty", block.pending || []);
      renderProposalList("proposals-decided", "proposals-decided-empty", block.decided || []);
      setText("session-note", "No re-ID stage is attached to this dashboard, so nothing new " +
                              "can be proposed here and no gallery state is shown; any card " +
                              "above is history from the database.");
      stats.textContent = "";
      return;
    }

    renderProposalList("proposals", "proposals-empty", block.pending || []);
    renderProposalList("proposals-decided", "proposals-decided-empty", block.decided || []);

    var session = block.session || {};
    setText("session-note", session.gallery_cleared
      ? "This session has ended: the gallery has been CLEARED and its embeddings are gone. " +
        "The cards above are history from the database, and the labels persist."
      : "The gallery is live: " + session.n_identities + " identities, " +
        session.n_gallery_rows + " embeddings kept, deleted at session end.");

    var latency = session.gallery_latency || {};
    var reid = session.reid || {};
    var parts = [];
    if (latency.last_ms !== null && latency.last_ms !== undefined) {
      parts.push("match cost " + fmtMs(latency.last_ms) + " ms at gallery size " +
                 latency.last_gallery_size + " (mean " + fmtMs(latency.mean_ms) +
                 " ms over the last " + latency.window + " calls, max size " +
                 latency.max_gallery_size + ")");
    }
    if (reid.n_proposals !== null && reid.n_proposals !== undefined) {
      parts.push(reid.n_proposals + " asked, " + reid.n_confirmed + " confirmed, " +
                 reid.n_rejected + " rejected, none auto-bound");
    }
    parts.push("counts, not rates: no ground truth exists here to divide by");
    stats.textContent = parts.join("  .  ");
  }

  function render(payload) {
    setText("run-name", payload.run_name);
    setText("mode", payload.mode);
    setText("source-label", payload.source_label || "-");
    setText("frame-index", payload.frame_index < 0 ? "-" : String(payload.frame_index));
    setText("t-source", fmtClock(payload.t_source));

    var image = byId("frame");
    var empty = byId("frame-empty");
    if (payload.frame_b64) {
      image.src = "data:image/jpeg;base64," + payload.frame_b64;
      image.style.display = "block";
      empty.style.display = "none";
    } else {
      image.style.display = "none";
      empty.style.display = "block";
    }

    renderTracks(payload.tracks || [], payload.timers || {});
    renderAlerts(payload.alerts || []);
    renderDepartures(payload.departures);
    renderProposals(payload.proposals);
    renderLatency(payload.stages || [], payload.latency_ms,
                  payload.latency_p50_ms, payload.latency_p95_ms);
  }

  function connect() {
    var scheme = window.location.protocol === "https:" ? "wss://" : "ws://";
    var socket = new WebSocket(scheme + window.location.host + "/ws");
    socket.addEventListener("open", function () { setText("socket", "live"); });
    socket.addEventListener("message", function (event) {
      try {
        render(JSON.parse(event.data));
      } catch (err) {
        window.console.error("bad payload", err);
      }
    });
    socket.addEventListener("close", function () {
      setText("socket", "closed, retrying");
      window.setTimeout(connect, 1500);
    });
    socket.addEventListener("error", function () { setText("socket", "error"); });
  }

  connect();
}());
</script>
"""

PAGE_HTML = _PAGE_TEMPLATE.replace("__MOCK_LABEL__", html.escape(MOCK_LABEL))
