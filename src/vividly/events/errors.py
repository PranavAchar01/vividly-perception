"""Every failure the alert store raises, in one module.

Why these are not in `store.py` any more: the schema migration path and the store both
need to raise them, and the store imports the migration path, so an exception hierarchy
living in either one of them makes the other one circular. A leaf module with no imports
of its own is the version of that dependency that cannot go wrong.

The contract these types encode is that a caller of `vividly.events` never has to catch
`sqlite3` directly. Every sqlite3 exception that escapes the driver is translated into one
of these before it leaves the store, so the pipeline and the dashboard can hold a single
`except EventStoreError` and know what it covers. That is not a promise this module can
keep on its own: it holds because the store routes every statement it issues after open
through one translating helper (`EventStore._execute` / `EventStore._commit`), and because
a failure at open time is translated in the constructor. The exception is
`EventStore.close`, which calls `sqlite3.Connection.close` unwrapped, so a driver error
raised while closing surfaces as itself. Two of the leaves below are refinements a
caller may special-case without giving up that guarantee, because both remain
`EventStoreError`: :class:`StoreBusyError` for a transient lock, and
:class:`ConfirmConflictError` for an identity-uniqueness violation.

Import them from `vividly.events`; this module is where they are defined, not where the
rest of the codebase should reach for them.
"""

from __future__ import annotations


class EventStoreError(RuntimeError):
    """Base for every store failure. Callers never see a raw sqlite3 exception."""


class DuplicateAlertError(EventStoreError):
    """An alert_id already present was inserted again.

    Not a crash case for the caller: alert ids are deterministic, so this is the expected
    signal that a firing has already been recorded (a re-run over the same clip, or a rule
    that fired twice inside one millisecond).
    """


class UnknownAlertError(EventStoreError):
    """Feedback was written against an alert_id this store has never seen."""


class SchemaVersionError(EventStoreError):
    """The database file on disk cannot be read at this build's schema version.

    Three ways to get here: a file stamped with a version from the future, a file whose
    tables carry this build's names but not its columns, and a migration that could not
    complete.
    All three are refusals at open time, which is the last moment an operator can still
    choose a different path.
    """


class DuplicateProposalError(EventStoreError):
    """A proposal_id already present was recorded again.

    Like DuplicateAlertError this is a normal outcome, not a bug: proposal ids are
    deterministic, so a replay of the same session re-proposes the same pairs. The right
    handling is to leave the stored row alone, because it may already carry a human's
    answer that a second insert would have thrown away.
    """


class UnknownProposalError(EventStoreError):
    """A decision was written against a proposal_id this store has never seen."""


class ProposalAlreadyDecidedError(EventStoreError):
    """A second decision arrived for a proposal a human has already answered.

    Refused rather than applied. A re-association decision is a label about a patient's
    identity: overwriting one silently would mean the stored answer is whichever click
    landed last, and the flywheel would train on an edit history no one reviewed.
    """


class StoreBusyError(EventStoreError):
    """A write could not proceed because another connection held the lock past the timeout.

    The SQLITE_BUSY / SQLITE_LOCKED case, surfaced as its own type so a caller can tell a
    transient contention apart from a corrupt file or a rejected value: a busy store is one
    a retry from the dashboard can still resolve, and losing a reviewer's
    label behind a raw ``OperationalError`` that no ``except`` clause named is the failure
    this type exists to prevent. It is an :class:`EventStoreError`, so a caller that only
    catches the base still catches it.
    """


class ConfirmConflictError(EventStoreError):
    """A confirm would give one track a second identity, or claim a departed one twice.

    The store enforces, per run and in the database itself, two properties of a confirmed
    re-association: a track is confirmed to at most one identity, and a departed identity is
    claimed by at most one live track (two partial-unique indexes over the confirmed rows).
    A second decision that would break either is refused atomically here, the way a
    double-decide on one proposal is refused, rather than written and reconciled later.

    The losing confirm is still a real answer a human gave, and it belongs in the record as
    a rejection rather than being dropped. Turning a refused confirm into that rejection is
    the re-identification stage's job on catching this; the store's job is only to make the
    conflicting confirm impossible to store.
    """
