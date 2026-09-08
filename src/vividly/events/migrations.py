"""The store file's schema, its version history, and the one path that moves a file forward.

Why this is its own module. An alert database outlives the build that wrote it: it is one
file on an appliance that a hospital keeps, copies, and reopens after an update. That makes
"what shape is this file, and how does an older one become a newer one" a contract, not an
implementation detail of the store class. Keeping it here means the DDL for every version
is readable in one place next to the upgrade that produced it, and that the store never
holds two ideas about what a table looks like.

Rules this module keeps, all of them load-bearing:

* **Forward only, never down.** A file stamped with a version this build does not know is
  refused loudly at open. Opening it read-anyway would mean writing rows in the old shape
  into a newer file, which is how a database ends up half in each version with no record
  of when it happened.
* **One transaction per open.** Creating tables and stamping `PRAGMA user_version` happen
  inside a single explicit `BEGIN IMMEDIATE`. Python's sqlite3 does not open a transaction
  for DDL on its own, so without the explicit begin a crash between the CREATE and the
  stamp would leave a file with version 1 written on the outside and version 2 tables
  inside. `PRAGMA user_version` is journalled like any other page, so it rolls back with
  everything else.
* **Existing rows are never rewritten.** Every migration so far is additive: version 2
  adds a table beside `alerts`, and version 3 adds a nullable column to that table with
  ALTER TABLE ADD COLUMN, which is metadata-only in SQLite. Neither step touches a byte of
  `alerts`. That is a deliberate constraint on how the schema is allowed to grow. An
  alert that a nurse acted on is evidence, and evidence does not get rewritten by a
  software update.
* **A known table name must mean a known table.** `CREATE TABLE IF NOT EXISTS` is a silent
  no-op against a table that already exists under that name with different columns, so an
  unrelated database would otherwise open cleanly and fail on the first insert, mid-run.
  Columns are verified inside the transaction and a mismatch rolls the whole open back,
  leaving the foreign file exactly as it was found.

Two statements in this module are built by interpolation, both because SQLite does not
allow the value to be bound: `PRAGMA user_version = {int}` and `PRAGMA table_info({name})`.
Both interpolate module-level constants defined in this file and never anything that
reached the process from outside it. Every other statement here is a literal.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from vividly.events.errors import SchemaVersionError

USER_VERSION = 3
"""Stamped into `PRAGMA user_version`. The version of the file, not of a record.

Version history:

1. `alerts` plus its detected_ts index. AlertRecord v1, the replay pipeline (POLICY.md,
   section 1).
2. adds `reassoc_proposals`. ReassocProposalRecord v1: proposed re-associations beyond
   the tracker's automatic window and the human confirm/reject answers that are the
   labels (POLICY.md, section 4, re-identification scope).
3. adds a nullable `run_id` column to `reassoc_proposals`, and two partial-unique indexes
   over its confirmed rows: (run_id, new_track) and (run_id, candidate_track). The column
   scopes a proposal to the run that raised it, so one store file shared by two runs stops
   cross-contaminating their review queues; the indexes make "a track has at most one
   confirmed identity" and "a departed identity is claimed by at most one live track"
   properties the database enforces, per run, not conventions a caller is trusted to keep.
   ReassocProposalRecord stays at v1: `run_id` is additive and nullable, so a pre-v3 row
   reads back as a valid v1 record with `run_id = None`.

Record versions and this number count different things and are expected to disagree.
`AlertRecord.schema_version` is still 1 inside a version 3 file, because none of these
steps moved the alert row shape. Bump this when the set of tables or any column changes;
bump a record's own version when that record's fields change.
"""

FIRST_VERSION = 1
"""Oldest file version this build can still open and carry forward."""


# ------------------------------------------------------------------ v1 schema
# Frozen. This DDL is what shipped as version 1 and what is already on disk in every
# existing store file. It is reproduced, not regenerated: changing a character here does
# not change any existing file, it only changes what new files get, which is precisely the
# way two versions of one table name come to exist. A real change to `alerts` is a new
# version number and an ALTER in a _upgrade_N_to_N_plus_1 function.
CREATE_ALERTS = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id       TEXT    PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    track_id       INTEGER NOT NULL,
    rule_id        TEXT    NOT NULL,
    rule_version   TEXT    NOT NULL,
    onset_ts       REAL    NOT NULL,
    detected_ts    REAL    NOT NULL,
    created_utc    TEXT    NOT NULL,
    baseline       TEXT,
    details        TEXT    NOT NULL,
    feedback       TEXT,
    mrn_binding    TEXT
)
"""

CREATE_ALERTS_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_alerts_detected_ts ON alerts (detected_ts DESC)"
)

ALERT_COLUMNS = frozenset(
    {
        "alert_id",
        "schema_version",
        "track_id",
        "rule_id",
        "rule_version",
        "onset_ts",
        "detected_ts",
        "created_utc",
        "baseline",
        "details",
        "feedback",
        "mrn_binding",
    }
)


# ------------------------------------------------------------------ v2 schema
# `status` is constrained by the database itself and not only by pydantic. The dashboard
# and the pipeline are not the only things that will ever hold this file open: an operator
# with the sqlite3 CLI is, and a hand-edited status of 'confirmed ' or 'accepted' would
# turn a patient-identity label into a value no reader knows how to count. The CHECK is the
# one guarantee that survives a process this codebase does not control.
#
# decided_ts and decided_via are nullable because a pending proposal has no decision yet.
# They are written together, exactly once, by decide_proposal.
#
# Frozen at the v2 shape (no run_id). This DDL is what shipped as version 2 and is what a
# version 2 file on disk already holds; version 3 reaches its shape by ALTER, not by
# rewriting this. A v1 -> v3 open builds this first (v1 -> v2) and then adds run_id
# (v2 -> v3), so each historical step stays reproducible.
CREATE_PROPOSALS = """
CREATE TABLE IF NOT EXISTS reassoc_proposals (
    proposal_id     TEXT    PRIMARY KEY,
    schema_version  INTEGER NOT NULL,
    ts              REAL    NOT NULL,
    new_track       INTEGER NOT NULL,
    candidate_track INTEGER NOT NULL,
    confidence      REAL    NOT NULL,
    gallery_size    INTEGER NOT NULL,
    status          TEXT    NOT NULL CHECK (status IN ('pending', 'confirmed', 'rejected')),
    decided_ts      REAL,
    decided_via     TEXT
)
"""

CREATE_PROPOSALS_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_reassoc_status_ts ON reassoc_proposals (status, ts)"
)

PROPOSAL_COLUMNS_V2 = frozenset(
    {
        "proposal_id",
        "schema_version",
        "ts",
        "new_track",
        "candidate_track",
        "confidence",
        "gallery_size",
        "status",
        "decided_ts",
        "decided_via",
    }
)


# ------------------------------------------------------------------ v3 schema
# Version 3 is version 2 plus one nullable column and two partial-unique indexes.
#
# `run_id` scopes a proposal to the run that raised it. A fresh file gets it in the CREATE
# below; an existing v2 file gets it by ALTER TABLE ADD COLUMN, which SQLite records as a
# metadata-only change that rewrites no existing row (the additive-only rule). Rows written
# before v3 carry run_id = NULL: they predate run scoping and belong to no run.
#
# The two indexes are UNIQUE and PARTIAL (only over confirmed rows). They are the store-side
# statement of two identity properties, per run: a track is confirmed to at most one
# identity, and a departed identity is claimed by at most one live track. Because a NULL in
# a UNIQUE index makes its row distinct, legacy run_id = NULL rows never collide, so adding
# the indexes to an existing file cannot fail on old confirmed rows. `pending` and
# `rejected` rows are outside the partial predicate and are never constrained by identity.
CREATE_PROPOSALS_V3 = """
CREATE TABLE IF NOT EXISTS reassoc_proposals (
    proposal_id     TEXT    PRIMARY KEY,
    schema_version  INTEGER NOT NULL,
    ts              REAL    NOT NULL,
    new_track       INTEGER NOT NULL,
    candidate_track INTEGER NOT NULL,
    confidence      REAL    NOT NULL,
    gallery_size    INTEGER NOT NULL,
    status          TEXT    NOT NULL CHECK (status IN ('pending', 'confirmed', 'rejected')),
    decided_ts      REAL,
    decided_via     TEXT,
    run_id          TEXT
)
"""

ADD_RUN_ID = "ALTER TABLE reassoc_proposals ADD COLUMN run_id TEXT"

CREATE_CONFIRMED_NEW_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_reassoc_confirmed_new "
    "ON reassoc_proposals (run_id, new_track) WHERE status = 'confirmed'"
)

CREATE_CONFIRMED_CANDIDATE_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_reassoc_confirmed_candidate "
    "ON reassoc_proposals (run_id, candidate_track) WHERE status = 'confirmed'"
)

PROPOSAL_COLUMNS = PROPOSAL_COLUMNS_V2 | {"run_id"}


# -------------------------------------------------------------------- helpers
def _columns(cur: sqlite3.Cursor, table: str) -> set[str]:
    """The column names of `table` right now. Interpolates only the module-known name."""
    return {str(row["name"]) for row in cur.execute(f"PRAGMA table_info({table})")}


def _verify_table(cur: sqlite3.Cursor, path: Path, table: str, expected: frozenset[str]) -> None:
    """Refuse a file whose table of that name is foreign. Called inside the transaction."""
    missing = expected - _columns(cur, table)
    if missing:
        raise SchemaVersionError(
            f"{path} holds a table named {table!r} that is missing {sorted(missing)}; "
            f"it was not written by this store. Migrate it, do not open it."
        )


# Create, then verify, then index, per table. The order is not cosmetic: an index over a
# column a foreign table does not have raises a raw sqlite3.OperationalError, which reaches
# the operator as "no such column" instead of as the refusal that names the file and says
# what is missing. Verify between the two so the friendly error is the one that wins.
def _ensure_alerts(cur: sqlite3.Cursor, path: Path) -> None:
    """Version 1 objects: the alerts table and its detected_ts index."""
    cur.execute(CREATE_ALERTS)
    _verify_table(cur, path, "alerts", ALERT_COLUMNS)
    cur.execute(CREATE_ALERTS_INDEX)


def _ensure_proposals_v2(cur: sqlite3.Cursor, path: Path) -> None:
    """Version 2 objects: the reassoc_proposals table (no run_id) and its status index."""
    cur.execute(CREATE_PROPOSALS)
    _verify_table(cur, path, "reassoc_proposals", PROPOSAL_COLUMNS_V2)
    cur.execute(CREATE_PROPOSALS_INDEX)


def _ensure_proposals_v3(cur: sqlite3.Cursor, path: Path) -> None:
    """Version 3 objects: reassoc_proposals with run_id, the status index, and the two
    partial-unique identity indexes. Built directly for a fresh file, and run idempotently
    on reopen to verify columns and repair an index dropped by hand.
    """
    cur.execute(CREATE_PROPOSALS_V3)
    _verify_table(cur, path, "reassoc_proposals", PROPOSAL_COLUMNS)
    cur.execute(CREATE_PROPOSALS_INDEX)
    cur.execute(CREATE_CONFIRMED_NEW_INDEX)
    cur.execute(CREATE_CONFIRMED_CANDIDATE_INDEX)


def _upgrade_1_to_2(cur: sqlite3.Cursor, path: Path) -> None:
    """Add the proposals table beside the alerts table.

    Purely additive: not one byte of `alerts` is read, rewritten, or re-indexed. That is
    what lets a migrated file be compared row for row against its backup.
    """
    _ensure_proposals_v2(cur, path)


def _upgrade_2_to_3(cur: sqlite3.Cursor, path: Path) -> None:
    """Add run_id to reassoc_proposals, then the two partial-unique identity indexes.

    ALTER TABLE ADD COLUMN with a NULL default is metadata-only in SQLite: not one existing
    proposal row is rewritten, so the additive-only rule holds exactly as it did for the
    v1 -> v2 step. Existing rows carry run_id = NULL and so never collide in the new
    indexes. The column-presence guard keeps this idempotent if it is ever re-entered.
    """
    if "run_id" not in _columns(cur, "reassoc_proposals"):
        cur.execute(ADD_RUN_ID)
    _verify_table(cur, path, "reassoc_proposals", PROPOSAL_COLUMNS)
    cur.execute(CREATE_PROPOSALS_INDEX)
    cur.execute(CREATE_CONFIRMED_NEW_INDEX)
    cur.execute(CREATE_CONFIRMED_CANDIDATE_INDEX)


# ------------------------------------------------------------------- entry point
def apply_migrations(conn: sqlite3.Connection, path: Path) -> int:
    """Bring `conn` up to :data:`USER_VERSION`, or refuse the file. Returns what was found.

    The return value is the version the file carried on the way in: 0 for a database with
    nothing in it yet, otherwise the stamped version. A caller that wants to log that a file
    was upgraded compares it against USER_VERSION; nothing in the store's behaviour depends
    on it.

    Raises :class:`~vividly.events.errors.SchemaVersionError` for a file from the future
    and for a file whose tables carry these names with other columns. Both refusals leave the
    file byte-identical to how it was found.

    The caller is responsible for having set journal_mode and busy_timeout first. Those are
    connection pragmas that cannot run inside a transaction, and this function opens one.
    """
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    found = int(cur.execute("PRAGMA user_version").fetchone()[0])

    if found > USER_VERSION:
        raise SchemaVersionError(
            f"{path} was written with schema user_version={found}; this build reads "
            f"{USER_VERSION}. Upgrade the build, or migrate the file forward with one that "
            f"knows that version. This store never downgrades a database."
        )
    if found < 0 or (found != 0 and found < FIRST_VERSION):
        raise SchemaVersionError(
            f"{path} carries schema user_version={found}, which no build ever wrote."
        )

    # commit() with no transaction open is a no-op, and guarantees the explicit BEGIN below
    # cannot hit "cannot start a transaction within a transaction" on a reused connection.
    conn.commit()
    cur.execute("BEGIN IMMEDIATE")
    try:
        # Present at every version, and idempotent: this is also the check that the file's
        # `alerts` table is a known one before anything else in the open touches it.
        _ensure_alerts(cur, path)

        if found == 0:
            # A fresh file goes straight to the current version rather than being built as
            # v1 and then migrated: replaying history to create a new database only adds
            # code paths nothing else exercises.
            _ensure_proposals_v3(cur, path)
        elif found == 1:
            # An old file chains every step, so each historical upgrade stays exercised:
            # add the proposals table (v1 -> v2), then run_id and its indexes (v2 -> v3).
            _upgrade_1_to_2(cur, path)
            _upgrade_2_to_3(cur, path)
        elif found == 2:
            _upgrade_2_to_3(cur, path)
        else:
            # An already-current file (found == USER_VERSION) runs the same idempotent
            # creates, which verify its columns and repair an index dropped by hand.
            _ensure_proposals_v3(cur, path)

        if found != USER_VERSION:
            # PRAGMA values cannot be bound; USER_VERSION is a module int, never input.
            cur.execute(f"PRAGMA user_version = {int(USER_VERSION)}")
    except BaseException:
        conn.rollback()
        raise
    conn.commit()
    return found
