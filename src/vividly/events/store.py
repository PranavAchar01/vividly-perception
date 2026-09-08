"""Alert store: one SQLite file, stdlib sqlite3, no server and no network (POLICY.md, rule 7).

Why SQLite and not a service: the appliance has to keep working when the hospital network
does not, and an alert history that lives in a process is an alert history that a restart
erases. One file, copyable, greppable, with the schema version stamped inside it.

Concurrency shape, which the dashboard depends on:

* WAL journal mode, so the dashboard can read while the pipeline writes. Under the default
  rollback journal a writer blocks every reader, which shows up as a frozen UI.
* ``check_same_thread=False`` plus one lock. FastAPI runs sync endpoints on a threadpool,
  so the store is reached from more than one thread even in the single-process demo. The
  lock serialises use of the single connection; sqlite3 itself is compiled thread-safe.
* ``busy_timeout``, so a second writer waits instead of raising "database is locked".

The store speaks :class:`~vividly.schemas.events.AlertRecord` and, since file version 2,
:class:`~vividly.schemas.reassoc.ReassocProposalRecord`. Nested alert fields (baseline,
details) are stored as JSON text rather than exploded into columns: rules own those keys
and are expected to add to them, and a schema migration per new rule detail would be a tax
on exactly the work the replay pipeline exists to make routine.

The second table is a different kind of thing from the first and the difference is the
point. Alerts are what the system decided; proposals are what the system declined to
decide. Beyond the tracker's own short re-association window the pipeline never auto-binds
an identity: it writes a pending proposal with a confidence, a human confirms or rejects it
on the dashboard, and that answer is the label the flywheel consumes through
:meth:`EventStore.labels_as_rows`. Inside that window, association stays automatic and
nothing reaches this table at all. See `vividly.schemas.reassoc` for why the asymmetry is a
patient-safety decision rather than a UX one.

The schema itself, its version history, and the upgrade path live in
`vividly.events.migrations`. This module holds no DDL.

All SQL here is parameterised, with no interpolation of any kind.
"""

from __future__ import annotations

import contextlib
import json
import math
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any

from vividly.events.errors import (
    ConfirmConflictError,
    DuplicateAlertError,
    DuplicateProposalError,
    EventStoreError,
    ProposalAlreadyDecidedError,
    StoreBusyError,
    UnknownAlertError,
    UnknownProposalError,
)
from vividly.events.migrations import apply_migrations
from vividly.schemas.events import FEEDBACK_LABELS, AlertRecord
from vividly.schemas.reassoc import (
    REASSOC_DECISIONS,
    ReassocProposalRecord,
    make_proposal_id,
)

# ------------------------------------------------------------------------ SQL
_INSERT = """
INSERT INTO alerts (
    alert_id, schema_version, track_id, rule_id, rule_version,
    onset_ts, detected_ts, created_utc, baseline, details, feedback, mrn_binding
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_RECENT = """
SELECT alert_id, schema_version, track_id, rule_id, rule_version,
       onset_ts, detected_ts, created_utc, baseline, details, feedback, mrn_binding
FROM alerts
ORDER BY detected_ts DESC, rowid DESC
LIMIT ?
"""

_UPDATE_FEEDBACK = "UPDATE alerts SET feedback = ? WHERE alert_id = ?"

_INSERT_PROPOSAL = """
INSERT INTO reassoc_proposals (
    proposal_id, schema_version, ts, new_track, candidate_track,
    confidence, gallery_size, status, decided_ts, decided_via, run_id
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# The status guard is in the WHERE clause, not in a read-then-write in Python. Two
# dashboard tabs open on one proposal are two threads on one connection, and a check that
# is not part of the statement that acts on it is a race with a patient's identity on the
# other side of it. UPDATE ... WHERE status = 'pending' either wins or reports rowcount 0.
_DECIDE_PROPOSAL = """
UPDATE reassoc_proposals
SET status = ?, decided_ts = ?, decided_via = ?
WHERE proposal_id = ? AND status = 'pending'
"""

# Same statement, but scoped to one run: a caller that names its live run may decide only
# its own rows. run_id is bound, so a foreign run's proposal reports rowcount 0 and is
# refused rather than answered. Kept as a second literal so no run_id value is ever
# interpolated into SQL.
_DECIDE_PROPOSAL_SCOPED = """
UPDATE reassoc_proposals
SET status = ?, decided_ts = ?, decided_via = ?
WHERE proposal_id = ? AND status = 'pending' AND run_id = ?
"""

_SELECT_PENDING = """
SELECT proposal_id, schema_version, ts, new_track, candidate_track,
       confidence, gallery_size, status, decided_ts, decided_via, run_id
FROM reassoc_proposals
WHERE status = 'pending'
ORDER BY ts ASC, rowid ASC
LIMIT ?
"""

_SELECT_PENDING_SCOPED = """
SELECT proposal_id, schema_version, ts, new_track, candidate_track,
       confidence, gallery_size, status, decided_ts, decided_via, run_id
FROM reassoc_proposals
WHERE status = 'pending' AND run_id = ?
ORDER BY ts ASC, rowid ASC
LIMIT ?
"""

_SELECT_PROPOSAL_HISTORY = """
SELECT proposal_id, schema_version, ts, new_track, candidate_track,
       confidence, gallery_size, status, decided_ts, decided_via, run_id
FROM reassoc_proposals
ORDER BY ts DESC, rowid DESC
LIMIT ?
"""

_SELECT_PROPOSAL_HISTORY_SCOPED = """
SELECT proposal_id, schema_version, ts, new_track, candidate_track,
       confidence, gallery_size, status, decided_ts, decided_via, run_id
FROM reassoc_proposals
WHERE run_id = ?
ORDER BY ts DESC, rowid DESC
LIMIT ?
"""

_SELECT_ONE_PROPOSAL = """
SELECT proposal_id, schema_version, ts, new_track, candidate_track,
       confidence, gallery_size, status, decided_ts, decided_via, run_id
FROM reassoc_proposals
WHERE proposal_id = ?
"""

_SELECT_LABELS = """
SELECT proposal_id, schema_version, ts, new_track, candidate_track,
       confidence, gallery_size, status, decided_ts, decided_via
FROM reassoc_proposals
WHERE status <> 'pending'
ORDER BY ts ASC, rowid ASC
"""


# -------------------------------------------------------------------- helpers
def _json_default(obj: object) -> Any:
    """Serialise the scalar types rules leak into `details` without importing numpy.

    A rule that computes a slope with numpy hands back a numpy float, which json refuses.
    Duck-typing `.item()` unwraps every numpy scalar to its Python equivalent and keeps
    this module importable on a machine with no numpy at all (linux CI collects it).
    """
    item = getattr(obj, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"alert payload holds a non-JSON value of type {type(obj).__name__}")


def _dump(payload: dict | None) -> str | None:
    if payload is None:
        return None
    return json.dumps(payload, sort_keys=True, default=_json_default)


def _record_from_row(row: sqlite3.Row) -> AlertRecord:
    baseline = row["baseline"]
    return AlertRecord(
        schema_version=row["schema_version"],
        alert_id=row["alert_id"],
        track_id=row["track_id"],
        rule_id=row["rule_id"],
        rule_version=row["rule_version"],
        onset_ts=row["onset_ts"],
        detected_ts=row["detected_ts"],
        created_utc=row["created_utc"],
        baseline=None if baseline is None else json.loads(baseline),
        details=json.loads(row["details"]),
        feedback=row["feedback"],
        mrn_binding=row["mrn_binding"],
    )


def _proposal_from_row(row: sqlite3.Row) -> ReassocProposalRecord:
    return ReassocProposalRecord(
        schema_version=row["schema_version"],
        proposal_id=row["proposal_id"],
        ts=row["ts"],
        new_track=row["new_track"],
        candidate_track=row["candidate_track"],
        confidence=row["confidence"],
        gallery_size=row["gallery_size"],
        status=row["status"],
        decided_ts=row["decided_ts"],
        decided_via=row["decided_via"],
        run_id=row["run_id"],
    )


# SQLite primary result codes for the two "another connection holds it" outcomes. The driver
# reports these on `sqlite_errorcode` since Python 3.11 (Vividly pins 3.11); older codes and
# any extended bits are handled by masking to the low byte and by the message fallback.
_SQLITE_BUSY = 5
_SQLITE_LOCKED = 6


def _is_busy(exc: sqlite3.Error) -> bool:
    """True for SQLITE_BUSY / SQLITE_LOCKED, the transient lock cases a retry can resolve.

    Prefers the numeric result code, masking off the extended bits, and falls back to the
    message so the classification still holds if a driver ever stops setting the code.
    """
    code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(code, int):
        return (code & 0xFF) in (_SQLITE_BUSY, _SQLITE_LOCKED)
    text = str(exc).lower()
    return "database is locked" in text or "database is busy" in text


def _check_finite(name: str, value: float) -> float:
    """Reject NaN and the infinities at the boundary rather than storing them.

    SQLite stores a NaN as NULL in a REAL column, so a NaN confidence would come back as
    None from a column the reader is entitled to treat as NOT NULL, and the record would
    fail validation on read instead of on write. Fail where the caller can still see it.
    """
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


# ---------------------------------------------------------------------- store
class EventStore:
    """Append-and-annotate store for alerts and re-association proposals.

    Inserts are immutable. Exactly two fields ever move after a write, and both are a
    human's answer: an alert's ``feedback``, and a proposal's ``status`` (once, from
    pending). Nothing in this class rewrites what the pipeline recorded.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._closed = False
        self._version_on_open = -1  # replaced by _configure; -1 means "never got that far"
        try:
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        except sqlite3.Error as exc:
            self._closed = True
            raise EventStoreError(f"cannot open event store {self._path}: {exc}") from exc
        self._conn.row_factory = sqlite3.Row
        try:
            self._configure()
        except BaseException as exc:
            self._conn.close()
            self._closed = True
            # A corrupt or foreign file surfaces here. Callers handle EventStoreError; a
            # raw sqlite3 error would escape every except clause in the pipeline and the
            # dashboard, so it is translated. SchemaVersionError is already a store error, and
            # a file another process holds open past the timeout becomes StoreBusyError.
            if isinstance(exc, sqlite3.Error):
                msg = f"cannot open event store {self._path}: {exc}"
                if isinstance(exc, sqlite3.OperationalError) and _is_busy(exc):
                    raise StoreBusyError(msg) from exc
                raise EventStoreError(msg) from exc
            raise

    # -- lifecycle ---------------------------------------------------------
    def _configure(self) -> None:
        """Connection pragmas, then the schema. Order matters and is not adjustable.

        journal_mode and busy_timeout are connection pragmas that cannot run inside a
        transaction, and `apply_migrations` opens one, so they are set first. Everything
        about which tables exist and what version is stamped on the file belongs to
        `vividly.events.migrations`, including the refusals.
        """
        cur = self._conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")  # durable enough under WAL, far faster
        cur.execute("PRAGMA busy_timeout=5000")
        self._version_on_open = apply_migrations(self._conn, self._path)

    def _require_open(self) -> None:
        if self._closed:
            raise EventStoreError(f"event store {self._path} is closed")

    # -- the one translating choke point -----------------------------------
    #
    # Every statement this store runs after open, read or write, goes through `_execute`
    # (and every commit through `_commit`). That is what makes the promise in
    # `vividly.events.errors` hold rather than being aspirational: a raw sqlite3 error does
    # not reach a caller from a statement, because there is no `self._conn.execute` outside
    # these two methods, and an open-time failure is translated in the constructor. The one
    # uncovered call is `self._conn.close()` in `close`. All of these methods assume
    # `self._lock` is already held by the caller.
    def _translate(self, exc: sqlite3.Error) -> EventStoreError:
        """Map any sqlite3 error to the store's own hierarchy, busy first."""
        if isinstance(exc, sqlite3.OperationalError) and _is_busy(exc):
            return StoreBusyError(
                f"event store {self._path} is locked by another writer past the busy timeout: {exc}"
            )
        return EventStoreError(f"event store {self._path}: {exc}")

    def _safe_rollback(self) -> None:
        """Undo the failed statement so the connection is clean for the next caller.

        A rollback that itself fails must not mask the original error, so it is swallowed.
        """
        with contextlib.suppress(sqlite3.Error):
            self._conn.rollback()

    def _execute(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
        *,
        on_integrity: Callable[[sqlite3.IntegrityError], EventStoreError] | None = None,
    ) -> sqlite3.Cursor:
        """Run one statement, translating every sqlite3 error before it can escape.

        `on_integrity`, when given, maps a constraint violation to the caller's own
        ``Duplicate*`` / conflict error; that mapping is kept ahead of the generic one, so
        an expected uniqueness collision stays a precise, named signal. Every other
        sqlite3 error becomes :class:`StoreBusyError` (a lock past the timeout) or the base
        :class:`EventStoreError`. The failed statement is rolled back before the translated
        error is raised.
        """
        try:
            return self._conn.execute(sql, params)
        except sqlite3.IntegrityError as exc:
            self._safe_rollback()
            if on_integrity is not None:
                raise on_integrity(exc) from exc
            raise EventStoreError(f"event store {self._path}: constraint violation: {exc}") from exc
        except sqlite3.Error as exc:
            self._safe_rollback()
            raise self._translate(exc) from exc

    def _commit(self) -> None:
        """Commit, translating a failure the same way `_execute` does.

        A commit that cannot land (a lock taken during the flush) must surface as a store
        error, never as a success that silently wrote nothing.
        """
        try:
            self._conn.commit()
        except sqlite3.Error as exc:
            self._safe_rollback()
            raise self._translate(exc) from exc

    def close(self) -> None:
        """Idempotent. Safe to call from a shutdown hook that may run twice.

        This is the one sqlite3 call in the class outside `_execute` and `_commit`: a
        driver error raised by close is not translated and reaches the caller as itself.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.close()

    def __enter__(self) -> EventStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def version_on_open(self) -> int:
        """Schema version this file carried before it was opened. 0 means it was new.

        Exposed so a run can say out loud that it upgraded a hospital's alert database
        rather than doing it silently. Compare against
        :data:`vividly.events.migrations.USER_VERSION`; nothing in the store's behaviour
        depends on this value.
        """
        return self._version_on_open

    # -- writes ------------------------------------------------------------
    def insert_alert(self, a: AlertRecord) -> None:
        """Persist one alert. Raises DuplicateAlertError if its alert_id is already stored."""
        row = (
            a.alert_id,
            a.schema_version,
            a.track_id,
            a.rule_id,
            a.rule_version,
            a.onset_ts,
            a.detected_ts,
            a.created_utc,
            _dump(a.baseline),
            _dump(a.details) or "{}",
            a.feedback,
            a.mrn_binding,
        )

        def _on_integrity(exc: sqlite3.IntegrityError) -> EventStoreError:
            text = str(exc)
            if "UNIQUE" in text or "PRIMARY KEY" in text:
                return DuplicateAlertError(f"alert already stored: {a.alert_id!r}")
            return EventStoreError(f"alert {a.alert_id!r} rejected: {exc}")

        with self._lock:
            self._require_open()
            self._execute(_INSERT, row, on_integrity=_on_integrity)
            self._commit()

    def set_feedback(self, alert_id: str, label: str) -> None:
        """Record a reviewer's label on a stored alert.

        The label is checked against the schema's own Literal before it reaches SQL, so a
        typo from the dashboard is a ValueError at the boundary rather than a value that
        poisons every downstream count of agreement rate.
        """
        if label not in FEEDBACK_LABELS:
            raise ValueError(
                f"unknown feedback label {label!r}; expected one of {list(FEEDBACK_LABELS)}"
            )
        with self._lock:
            self._require_open()
            cur = self._execute(_UPDATE_FEEDBACK, (label, alert_id))
            if cur.rowcount == 0:
                self._safe_rollback()
                raise UnknownAlertError(f"no alert with id {alert_id!r}")
            self._commit()

    # -- reads -------------------------------------------------------------
    def alerts(self, limit: int = 100) -> list[AlertRecord]:
        """Most recently detected alerts first, capped at `limit`.

        Ties on detected_ts break by insertion order, newest first, so the order is stable
        across calls even when a batch of rules fires on one tick with identical clocks.
        """
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        with self._lock:
            self._require_open()
            rows = self._execute(_SELECT_RECENT, (limit,)).fetchall()
        return [_record_from_row(r) for r in rows]

    def get(self, alert_id: str) -> AlertRecord | None:
        """One alert by id, or None."""
        with self._lock:
            self._require_open()
            row = self._execute(
                "SELECT alert_id, schema_version, track_id, rule_id, rule_version, "
                "onset_ts, detected_ts, created_utc, baseline, details, feedback, mrn_binding "
                "FROM alerts WHERE alert_id = ?",
                (alert_id,),
            ).fetchone()
        return None if row is None else _record_from_row(row)

    def count(self) -> int:
        """Total stored alerts, counted in the file rather than tallied by a caller."""
        with self._lock:
            self._require_open()
            return int(self._execute("SELECT COUNT(*) FROM alerts").fetchone()[0])

    # -- re-association proposals -----------------------------------------
    #
    # Everything below exists because of one rule: beyond the tracker's own short
    # re-association window, the system never binds an identity by itself. It asks. A row
    # in this table is that question; a decision on it is a person's answer, and the answer
    # is the label. Inside the tracker's window nothing here is called at all.

    def record_proposal(
        self,
        *,
        new_track: int,
        candidate_track: int,
        confidence: float,
        ts: float,
        gallery_size: int,
        run_id: str | None = None,
    ) -> ReassocProposalRecord:
        """Propose that `new_track` is the person last seen as `candidate_track`.

        Returns the stored record, always with status ``pending``: this method cannot
        write a decided row, which is what makes "no automatic binding beyond the window"
        a property of the store rather than a convention callers are trusted to follow.

        Keyword-only on purpose. ``record_proposal(4, 9, ...)`` reads as if the argument
        order were obvious, and getting the pair backwards silently inverts the question a
        human is about to answer about two patients.

        ``run_id`` tags the proposal with the run that raised it. Pass it whenever one store
        file can be reached by more than one run, so that :meth:`pending_proposals` and
        :meth:`decide_proposal` can scope to a run and never let one run answer another's
        question. It does not enter ``proposal_id``: ids stay derived from the track pair and
        the timestamp, so a replay of one run is still idempotent against its own answers.

        Raises :class:`DuplicateProposalError` when the id is already stored, which is the
        expected signal on a replay of the same session: the existing row may already carry
        an answer, and re-asking a question a person has answered is a worse outcome than
        the exception.
        """
        # Coerce before anything else so the persisted key and the persisted row are built
        # from the same values. A numpy int from the gallery formats into an id perfectly
        # well today, and an id is the one thing that must not depend on that staying true.
        confidence = _check_finite("confidence", confidence)
        ts = _check_finite("ts", ts)
        new_track = int(new_track)
        candidate_track = int(candidate_track)
        gallery_size = int(gallery_size)

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"confidence must be a calibrated score in [0, 1], got {confidence!r}. "
                "A raw cosine similarity is not one: map it before proposing, because this "
                "number is shown to a human as the strength of an identity match."
            )
        if new_track == candidate_track:
            raise ValueError(
                f"a track cannot be re-associated with itself (track {new_track}); "
                "a self-pair proposal is a bug upstream, not a question for a reviewer"
            )
        if gallery_size < 0:
            raise ValueError(f"gallery_size must be >= 0, got {gallery_size}")

        record = ReassocProposalRecord(
            proposal_id=make_proposal_id(new_track, candidate_track, ts, run_id),
            ts=ts,
            new_track=new_track,
            candidate_track=candidate_track,
            confidence=confidence,
            gallery_size=gallery_size,
            status="pending",
            run_id=run_id,
        )
        row = (
            record.proposal_id,
            record.schema_version,
            record.ts,
            record.new_track,
            record.candidate_track,
            record.confidence,
            record.gallery_size,
            record.status,
            None,
            None,
            record.run_id,
        )

        def _on_integrity(exc: sqlite3.IntegrityError) -> EventStoreError:
            # The only uniqueness a pending insert can break is the proposal_id primary key;
            # the partial-unique identity indexes cover confirmed rows only, so they never
            # fire here. A duplicate id is the expected replay signal, not a rejection.
            text = str(exc)
            if "UNIQUE" in text or "PRIMARY KEY" in text:
                return DuplicateProposalError(
                    f"re-association proposal already stored: {record.proposal_id!r}"
                )
            return EventStoreError(f"proposal {record.proposal_id!r} rejected: {exc}")

        with self._lock:
            self._require_open()
            self._execute(_INSERT_PROPOSAL, row, on_integrity=_on_integrity)
            self._commit()
        return record

    def decide_proposal(
        self,
        proposal_id: str,
        decision: str,
        decided_ts: float,
        via: str,
        run_id: str | None = None,
    ) -> ReassocProposalRecord:
        """Record a human's confirm or reject on a pending proposal. This is the label.

        Exactly once per proposal. A second decision raises
        :class:`ProposalAlreadyDecidedError` and changes nothing, including when it agrees
        with the first: a stored identity label whose value is whichever click landed last
        is not a label anyone can audit, and the flywheel would be training on an edit
        history no one reviewed.

        ``via`` is required and must be non-empty. It is the provenance of the answer, and
        a label that cannot be attributed cannot be excluded from a training set later.

        ``run_id``, when given, is the caller's live run, and the decision is scoped to it:
        a proposal whose stored ``run_id`` is a different run is refused with
        :class:`UnknownProposalError`, because one run may never answer another's question,
        even though it can see that question in :meth:`proposal_history`. When ``run_id`` is
        ``None`` the decision is unscoped, the pre-run-scoping behaviour.

        A ``confirmed`` decision is subject to two per-run identity constraints the database
        enforces: a track is confirmed to at most one identity, and a departed identity is
        claimed by at most one live track. A confirm that would break either is refused
        atomically with :class:`ConfirmConflictError` and changes nothing, the way a
        double-decide is refused; the losing confirm belongs in the record as a rejection,
        which is the caller's to write.
        """
        if decision not in REASSOC_DECISIONS:
            raise ValueError(
                f"unknown re-association decision {decision!r}; "
                f"expected one of {list(REASSOC_DECISIONS)}"
            )
        decided_ts = _check_finite("decided_ts", decided_ts)
        via = via.strip()
        if not via:
            raise ValueError("via must name where the decision came from; it cannot be empty")

        if run_id is None:
            sql = _DECIDE_PROPOSAL
            params: tuple[Any, ...] = (decision, decided_ts, via, proposal_id)
        else:
            sql = _DECIDE_PROPOSAL_SCOPED
            params = (decision, decided_ts, via, proposal_id, run_id)

        def _on_integrity(exc: sqlite3.IntegrityError) -> EventStoreError:
            # A confirm broke a partial-unique identity index. rejected rows are outside the
            # partial predicate, so only a confirm reaches here.
            return ConfirmConflictError(
                f"confirming proposal {proposal_id!r} in run {run_id!r} would give a track "
                f"a second confirmed identity, or claim a departed identity twice: {exc}"
            )

        with self._lock:
            self._require_open()
            cur = self._execute(sql, params, on_integrity=_on_integrity)
            if cur.rowcount == 0:
                self._safe_rollback()
                existing = self._execute(_SELECT_ONE_PROPOSAL, (proposal_id,)).fetchone()
                if existing is None:
                    raise UnknownProposalError(
                        f"no re-association proposal with id {proposal_id!r}"
                    )
                if run_id is not None and existing["run_id"] != run_id:
                    raise UnknownProposalError(
                        f"proposal {proposal_id!r} belongs to run {existing['run_id']!r}, "
                        f"not the live run {run_id!r}; a run may only decide its own proposals"
                    )
                raise ProposalAlreadyDecidedError(
                    f"proposal {proposal_id!r} was already {existing['status']} "
                    f"via {existing['decided_via']!r}; decisions are written once"
                )
            self._commit()
            row = self._execute(_SELECT_ONE_PROPOSAL, (proposal_id,)).fetchone()
        return _proposal_from_row(row)

    def pending_proposals(
        self, run_id: str | None = None, limit: int = 100
    ) -> list[ReassocProposalRecord]:
        """Undecided proposals, oldest first. The review queue, in the order to work it.

        Oldest first, unlike :meth:`alerts`, because this list is a queue that has to be
        drained rather than a feed to glance at: newest-first plus a limit would hide the
        oldest unanswered question forever, and the oldest unanswered question is the one
        whose track is most likely to have left the room.

        ``run_id`` scopes the queue to one run: with it, only that run's pending proposals
        are returned, so a run never shows another run's unanswered question as its own to
        work. Without it the queue spans every run in the file, the pre-run-scoping view.
        """
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        with self._lock:
            self._require_open()
            if run_id is None:
                rows = self._execute(_SELECT_PENDING, (limit,)).fetchall()
            else:
                rows = self._execute(_SELECT_PENDING_SCOPED, (run_id, limit)).fetchall()
        return [_proposal_from_row(r) for r in rows]

    def proposal_history(
        self, run_id: str | None = None, limit: int = 100
    ) -> list[ReassocProposalRecord]:
        """Every proposal, decided or not, newest first. Matches :meth:`alerts` ordering.

        ``run_id`` scopes the history to one run. Without it the history spans every run in
        the file: a foreign run's rows stay visible here even though they can never be
        answered as this run's live questions, which is the point of history as against a
        queue.
        """
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        with self._lock:
            self._require_open()
            if run_id is None:
                rows = self._execute(_SELECT_PROPOSAL_HISTORY, (limit,)).fetchall()
            else:
                rows = self._execute(_SELECT_PROPOSAL_HISTORY_SCOPED, (run_id, limit)).fetchall()
        return [_proposal_from_row(r) for r in rows]

    def get_proposal(self, proposal_id: str) -> ReassocProposalRecord | None:
        """One proposal by id, or None."""
        with self._lock:
            self._require_open()
            row = self._execute(_SELECT_ONE_PROPOSAL, (proposal_id,)).fetchone()
        return None if row is None else _proposal_from_row(row)

    def labels_as_rows(self) -> list[dict[str, Any]]:
        """Decided proposals as flat JSON-ready rows: the flywheel's export of the labels.

        Only decided rows appear. A pending proposal is an unanswered question, not a
        negative, and counting it as one would teach the next threshold that every match a
        nurse has not got to yet was wrong.

        This is the one proposal reader that is not run-scoped: it selects every decided
        row in the file and the exported dict carries no ``run_id``, unlike
        ``pending_proposals``, ``proposal_history`` and ``decide_proposal``. That is
        deliberate and pinned by the tests: a label is a label whichever run raised the
        proposal, and a threshold sweep wants all of them.

        Each row pairs the score the model gave with the answer a person gave, which is the
        input a threshold sweep needs (docs/CALIBRATION.md): ``confidence`` beside
        ``label``, 1 for confirmed and 0 for rejected, with ``gallery_size`` because the
        same score against three identities and against forty is not the same evidence.

        ``ts`` and ``decided_ts`` are exported side by side and never subtracted. They may
        be different clocks (source time and whatever the deciding surface passed), and the
        store will not present a difference between them as a review latency.

        Ordered oldest first so two exports of the same store are byte-identical and a diff
        between two exports is the labels that were added. Unbounded by design: the gallery
        these proposals come from is session-scoped, so the row count is bounded by one
        session's returns, not by a retention period.
        """
        with self._lock:
            self._require_open()
            rows = self._execute(_SELECT_LABELS).fetchall()
        return [
            {
                "proposal_id": r["proposal_id"],
                "schema_version": int(r["schema_version"]),
                "ts": float(r["ts"]),
                "new_track": int(r["new_track"]),
                "candidate_track": int(r["candidate_track"]),
                "confidence": float(r["confidence"]),
                "gallery_size": int(r["gallery_size"]),
                "decision": r["status"],
                "label": 1 if r["status"] == "confirmed" else 0,
                # Nullable in SQL and so nullable here: decide_proposal always writes it,
                # but a row hand-edited outside this process must not crash an export.
                "decided_ts": None if r["decided_ts"] is None else float(r["decided_ts"]),
                "decided_via": r["decided_via"],
            }
            for r in rows
        ]
