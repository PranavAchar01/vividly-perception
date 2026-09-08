"""Versioned alert schema. Pydantic v2; the wire format is pinned by schema_version.

This is the record the SQLite event store writes and the dashboard reads. Rows outlive the
process that wrote them, so version 1 is frozen: a field change means adding a new Literal
value plus a store migration, never editing v1 in place.

Nothing here touches identity. `mrn_binding` holds the dashboard's clearly labelled mock
binding (of the form MRN-0007), never a real medical record number, and there is no
biometric, demographic, or vital-sign field of any kind, as POLICY.md rules 2 and 3
require.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, get_args

from pydantic import BaseModel

SCHEMA_VERSION = 1

FeedbackLabel = Literal["agree", "false_alarm"]
"""Reviewer verdict on a shown alert.

Two values on purpose. The v0 question is whether the alert was worth surfacing, not what
happened to the patient: a clinical outcome taxonomy is a different artifact with
different governance, and adding one is a schema bump, not a string.
"""

FEEDBACK_LABELS: tuple[str, ...] = get_args(FeedbackLabel)


class AlertRecord(BaseModel):
    """One rule firing, as persisted.

    Field notes that are not obvious from the types:

    * ``alert_id`` is derived, not random. See :func:`make_alert_id`: replaying the same
      footage twice produces the same ids, so a re-run updates nothing and duplicates
      nothing (the store rejects the second insert instead of silently doubling counts).
    * ``onset_ts`` is when the condition began, ``detected_ts`` when the rule noticed it.
      Both are source-clock seconds from the frame source, not wall clock, so they stay
      meaningful under faster-than-realtime replay. ``created_utc`` is the wall-clock
      stamp of the write and is the only field that moves between two replays.
    * ``baseline`` is the per-track BaselineStats snapshot as a plain dict, or None when
      the track had not finished its baseline window. Kept denormalised on purpose: an
      alert has to stay readable after the TrackBook that produced it is gone.
    * ``details`` carries the rule's own evidence (the fall cascade's transition time and
      drop, the convulsion's reversal fraction and amplitude). Rule authors own its keys;
      the schema deliberately does not constrain them.
    """

    schema_version: Literal[1] = 1
    alert_id: str
    track_id: int
    rule_id: str
    rule_version: str
    onset_ts: float
    detected_ts: float
    created_utc: str
    baseline: dict | None
    details: dict
    feedback: FeedbackLabel | None = None
    mrn_binding: str | None = None


def make_alert_id(rule_id: str, track_id: int, onset_ts: float) -> str:
    """Deterministic alert id: ``{rule_id}-{track_id}-{int(onset_ts * 1000)}``.

    No uuid, no clock. The same firing on the same clip always yields the same key, which
    is what makes a replay run idempotent against an existing store and makes an alert
    citable in a report without carrying the database around.

    Millisecond resolution truncates toward zero, so two firings of one rule on one track
    inside the same millisecond collide by design. The engine's refire debounce is orders
    of magnitude larger than that, so a collision means a real duplicate.
    """
    return f"{rule_id}-{track_id}-{int(onset_ts * 1000)}"


def utc_now_iso() -> str:
    """Wall-clock stamp for ``created_utc``: RFC 3339 UTC, milliseconds, Z suffix.

    Timezone-aware and always UTC. The appliance may sit on a clock nobody has audited,
    so a local-time stamp would be unreadable off-site.
    """
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
