"""Event persistence for the replay pipeline (POLICY.md, section 1): the SQLite alert store.

Depends on `vividly.schemas` and the standard library only. No perception imports, so the
dashboard can hold a store without pulling cv2 or a model backend into its process.

Two tables, two different jobs. `alerts` is what the rules decided. `reassoc_proposals` is
what the system declined to decide by itself: beyond the tracker's own short
re-association window an identity is never auto-bound, so the pipeline proposes and a human
confirms or rejects, and those answers are the labels (`EventStore.labels_as_rows`).

This is the import path for everything in the package. The schema, its version history, and
the upgrade path live in `vividly.events.migrations`; the exception hierarchy lives in
`vividly.events.errors`; both are re-exported here so no caller has to know that.
"""

from __future__ import annotations

from vividly.events.errors import (
    ConfirmConflictError,
    DuplicateAlertError,
    DuplicateProposalError,
    EventStoreError,
    ProposalAlreadyDecidedError,
    SchemaVersionError,
    StoreBusyError,
    UnknownAlertError,
    UnknownProposalError,
)
from vividly.events.migrations import USER_VERSION
from vividly.events.store import EventStore

__all__ = [
    "USER_VERSION",
    "ConfirmConflictError",
    "DuplicateAlertError",
    "DuplicateProposalError",
    "EventStore",
    "EventStoreError",
    "ProposalAlreadyDecidedError",
    "SchemaVersionError",
    "StoreBusyError",
    "UnknownAlertError",
    "UnknownProposalError",
]
