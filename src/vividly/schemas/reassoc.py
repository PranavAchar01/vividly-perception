"""Versioned re-association proposal schema. Pydantic v2, wire format pinned by
``schema_version``, written and read by the SQLite alert store at file version 3.

What this record is for, and the semantics it exists to make unbreakable.

Inside the tracker's own short re-association window, association is automatic and stays
exactly as it stands: the tracker joins the fragments, nothing is written here, and no
human is asked anything. Beyond that window the system never auto-binds. It proposes a
re-association, with a confidence and the gallery size that confidence was computed
against, and a human confirms or rejects it on the dashboard. One row per proposal is the
whole mechanism: a row exists because the automatic path declined to act.

That asymmetry is a patient-safety decision, not a UX preference. A false merge joins two
patients' identities, which in a chart-bound deployment is a patient-safety event and the
headline KPI of every re-ID stage (POLICY.md, section 4, re-identification scope). Silence beyond
the tracker's window costs a re-ask; a wrong automatic merge costs the wrong chart.

The decisions are the labels. ``status`` moving from ``pending`` to ``confirmed`` or
``rejected`` is a human-supplied ground-truth answer to whether the two tracks are the
same person, carrying the score the model gave at the time, which is exactly the pair a
threshold calibration needs (docs/CALIBRATION.md; the store's ``labels_as_rows()`` is the
export).

Session scope. The gallery these proposals are drawn from is session-scoped: it is cleared
at session end and never persists across visits (CQ-16, src/vividly/reid/gallery.py). Rows
here hold no embedding, no crop, and no appearance vector of any kind. A proposal row is
two integer track ids, a score, a count, a run tag, and a human's answer. It carries
nothing biometric, nothing demographic, and no vital sign, as POLICY.md rules 2 and 3
require, and it is not an identifier that means anything outside the one session that
produced it. ``run_id`` is the opaque name of the run that raised the proposal; it is not
derived from and cannot be mapped back to a person, a visit, or a clip's content.

Versioning. ``ReassocProposalRecord.schema_version`` is 1 even though the store file it
lives in is at ``PRAGMA user_version = 3``. The two numbers count different things: the
record version counts the shape of one row, and the file version counts the set of tables
and columns. Two file versions moved this table without moving the record contract: version 2
added the table next to ``alerts``, and version 3 added the nullable ``run_id`` column that
scopes a proposal to one run. ``run_id`` is additive and nullable, so a row written before
version 3 reads back with ``run_id = None`` and is still a valid version-1 record; that is
why the added field does not bump ``schema_version``. A change that alters or drops an
existing field, by contrast, would mean a new Literal value plus a store migration, never
an edit to version 1 in place. ``AlertRecord`` is likewise still at version 1 in a version
3 file.
"""

from __future__ import annotations

import hashlib
from typing import Literal, get_args

from pydantic import BaseModel

REASSOC_SCHEMA_VERSION = 1
"""Version of this record. See the module docstring: not the store's file version."""

ReassocStatus = Literal["pending", "confirmed", "rejected"]
"""Lifecycle of one proposal. ``pending`` is the only state a machine may write."""

ReassocDecision = Literal["confirmed", "rejected"]
"""What a human may answer. Deliberately not a superset of :data:`ReassocStatus`.

A decision cannot be "pending": un-deciding a re-association would erase a label a person
gave, and the flywheel would then train on an edit history nobody reviewed. Withdrawing a
decision, if it is ever wanted, is a new row with its own provenance, not a mutation.
"""

REASSOC_STATUSES: tuple[str, ...] = get_args(ReassocStatus)
REASSOC_DECISIONS: tuple[str, ...] = get_args(ReassocDecision)


class ReassocProposalRecord(BaseModel):
    """One proposed re-association beyond the tracker's window, and its human answer.

    Field notes that are not obvious from the types:

    * ``proposal_id`` is derived, not random. See :func:`make_proposal_id`. Replaying the
      same session twice proposes the same pairs at the same source timestamps and so
      produces the same ids, which is what lets a re-run be idempotent against a store
      that already holds the human's answers instead of asking for them again.
    * ``ts`` is the source-clock second at which the proposal was raised, the same clock
      as ``AlertRecord.onset_ts``, so it stays meaningful under faster-than-realtime
      replay. ``decided_ts`` is whatever clock the deciding surface passes, and the two
      are exported side by side rather than subtracted: the store does not know that they
      are the same clock and will not pretend a difference is a review latency.
    * ``new_track`` is the track that just appeared and needs an identity;
      ``candidate_track`` is the earlier track the gallery matched it to. The direction
      matters when the label is read back: the question asked was "is this new track the
      same person as that older one", never the reverse.
    * ``confidence`` is a calibrated score in [0, 1], not a raw cosine similarity. It is
      shown to a human who is about to merge two patient identities, so a number that
      reads as a percentage must actually be one. Producers map their similarity into
      this range before proposing; the store rejects anything outside it.
    * ``gallery_size`` is how many identities the match was made against at that moment.
      Without it a stored label is uninterpretable later: a match against three people and
      a match against forty are not the same evidence at the same score.
    * ``decided_via`` is free-text provenance of the answer (which surface, which role).
      It is required at decision time because a label with no provenance cannot be audited
      and cannot be excluded from a training set later.
    * ``run_id`` scopes a proposal to the run that raised it. One store file can be shared
      by more than one run (its path derives from a run name, which two clips can collide
      on), and without a run tag one run's pending questions would show up in another run's
      review queue and could be answered there, binding an identity for the wrong clip. It
      is nullable: rows written before file version 3 carry ``None`` and belong to no run.
      It is an opaque run label, never a patient, visit, or content identifier.
    """

    schema_version: Literal[1] = 1
    proposal_id: str
    ts: float
    new_track: int
    candidate_track: int
    confidence: float
    gallery_size: int
    status: ReassocStatus = "pending"
    decided_ts: float | None = None
    decided_via: str | None = None
    run_id: str | None = None


def make_proposal_id(
    new_track: int, candidate_track: int, ts: float, run_id: str | None = None
) -> str:
    """Deterministic proposal id: ``reassoc-{new_track}-{candidate_track}-{int(ts*1000)}``,
    suffixed with a short digest of ``run_id`` when the caller supplies one.

    Same shape and same reasoning as :func:`vividly.schemas.events.make_alert_id`: no
    uuid, no wall clock, so the same session replayed twice yields the same keys and a
    second pass collides with the stored row instead of duplicating a question a human
    has already answered.

    Millisecond resolution truncates toward zero, so two proposals for the same ordered
    pair of tracks inside the same millisecond collide by design. Two proposals that close
    together for one pair are the same proposal.

    Why the run suffix exists. Track ids restart at 1 in every run and the store file is
    shared by every run that writes under one run name, so without it two different runs
    over two different clips mint the same id for two different questions about two
    different people. The primary key then makes the second run's question a duplicate of
    the first's: it is swallowed, it never enters this run's queue (that queue is scoped by
    run id), and the summary reports nothing awaiting a human while a real question has
    quietly gone missing. That failure is silent and it fails toward never asking, which is
    the one direction a propose-then-confirm design cannot tolerate. Scoping the key to the
    run keeps determinism exactly where it is wanted, within a run, and removes it exactly
    where it was never true, across runs.

    ``run_id=None`` keeps the unscoped legacy form, so rows written before run scoping
    existed still resolve and older stores keep working.
    """
    base = f"reassoc-{new_track}-{candidate_track}-{int(ts * 1000)}"
    if run_id is None:
        return base
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{digest}"
