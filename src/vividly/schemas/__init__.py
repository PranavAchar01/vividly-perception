"""Versioned event and alert schemas.

Schema only. This package holds no pipeline logic and imports nothing from the rest of
Vividly, so both the event store and the dashboard can depend on it without dragging in
perception code. Import the models from here; `vividly.schemas.events` is where the alert
contract lives and `vividly.schemas.reassoc` is where the re-association proposal contract
lives. Each record carries its own `schema_version`, independent of the store file's
`PRAGMA user_version`.
"""

from __future__ import annotations

from vividly.schemas.events import (
    FEEDBACK_LABELS,
    SCHEMA_VERSION,
    AlertRecord,
    FeedbackLabel,
    make_alert_id,
    utc_now_iso,
)
from vividly.schemas.reassoc import (
    REASSOC_DECISIONS,
    REASSOC_SCHEMA_VERSION,
    REASSOC_STATUSES,
    ReassocDecision,
    ReassocProposalRecord,
    ReassocStatus,
    make_proposal_id,
)

__all__ = [
    "FEEDBACK_LABELS",
    "REASSOC_DECISIONS",
    "REASSOC_SCHEMA_VERSION",
    "REASSOC_STATUSES",
    "SCHEMA_VERSION",
    "AlertRecord",
    "FeedbackLabel",
    "ReassocDecision",
    "ReassocProposalRecord",
    "ReassocStatus",
    "make_alert_id",
    "make_proposal_id",
    "utc_now_iso",
]
