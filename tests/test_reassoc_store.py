"""Alert-store schema v3: the migrations, and the re-association proposal API.

What this file is defending, in one sentence: beyond the tracker's own short
re-association window nothing binds an identity without a person saying so, and the record
of a person saying so survives a software update that changes the database underneath it.

Deterministic, network-free, model-free. Every fixture is either a hand-built record or a
database this file writes with raw SQL, so it runs identically on a laptop and on linux CI
with no weights and no footage present.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from vividly.events import (
    USER_VERSION,
    ConfirmConflictError,
    DuplicateProposalError,
    EventStore,
    EventStoreError,
    ProposalAlreadyDecidedError,
    SchemaVersionError,
    StoreBusyError,
    UnknownProposalError,
)
from vividly.events import migrations as mig
from vividly.schemas import ReassocProposalRecord, make_proposal_id

# ---------------------------------------------------------------- v1 fixture
# A frozen copy of the version 1 schema, exactly as it shipped and exactly as it exists on
# disk in every store file written before this migration. It is duplicated here on purpose:
# building the fixture from the current code would make the migration test tautological,
# because a change to the shipped DDL would silently change the "old" database too.
V1_CREATE_ALERTS = """
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

V1_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_alerts_detected_ts ON alerts (detected_ts DESC)"

V1_INSERT = """
INSERT INTO alerts (
    alert_id, schema_version, track_id, rule_id, rule_version,
    onset_ts, detected_ts, created_utc, baseline, details, feedback, mrn_binding
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# Three rows that between them exercise every nullable column, a non-ascii payload, and a
# reviewer verdict that was already recorded before the upgrade.
V1_ROWS = [
    (
        "stillness_v0-7-120500",
        1,
        7,
        "stillness_v0",
        "0.1.0",
        120.5,
        2520.5,
        "2026-08-07T18:30:00.000Z",
        '{"complete": true, "movement_var": 12.5, "n_samples": 900}',
        '{"minutes_still": 41.5, "note": "seat by the window (caf\\u00e9)"}',
        "agree",
        "MRN-0007",
    ),
    (
        "drift_v0-12-60000",
        1,
        12,
        "drift_v0",
        "0.1.0",
        60.0,
        3660.0,
        "2026-08-07T18:31:00.000Z",
        None,
        '{"slope_px_per_min": -3.25}',
        None,
        None,
    ),
    (
        "variance_collapse_v0-3-900000",
        1,
        3,
        "variance_collapse_v0",
        "0.2.0",
        900.0,
        1800.0,
        "2026-08-07T18:32:00.000Z",
        '{"complete": false}',
        "{}",
        "false_alarm",
        None,
    ),
]


def _write_v1_db(path: Path, rows: list[tuple] | None = None) -> None:
    """Build a real version 1 store file with raw SQL. No current code is involved."""
    raw = sqlite3.connect(str(path))
    try:
        raw.execute("PRAGMA journal_mode=WAL")
        raw.execute(V1_CREATE_ALERTS)
        raw.execute(V1_CREATE_INDEX)
        raw.executemany(V1_INSERT, V1_ROWS if rows is None else rows)
        raw.execute("PRAGMA user_version = 1")
        raw.commit()
    finally:
        raw.close()


def _raw(path: Path, sql: str, args: tuple = ()) -> list[tuple]:
    """Read a database with a connection that knows nothing about this codebase."""
    raw = sqlite3.connect(str(path))
    try:
        return list(raw.execute(sql, args).fetchall())
    finally:
        raw.close()


def _alert_bytes(path: Path) -> list[tuple]:
    """Every alerts row including its rowid, in storage order. The byte-for-byte witness."""
    return _raw(path, "SELECT rowid, * FROM alerts ORDER BY rowid")


def _schema_text(path: Path) -> list[tuple]:
    """The stored DDL of every object, which is what an ALTER would change."""
    return _raw(path, "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY name")


def _user_version(path: Path) -> int:
    return int(_raw(path, "PRAGMA user_version")[0][0])


def _tables(path: Path) -> set[str]:
    return {r[0] for r in _raw(path, "SELECT name FROM sqlite_master WHERE type = 'table'")}


def _propose(store: EventStore, **over) -> ReassocProposalRecord:
    fields = {
        "new_track": 41,
        "candidate_track": 12,
        "confidence": 0.87,
        "ts": 1830.25,
        "gallery_size": 6,
    }
    fields.update(over)
    return store.record_proposal(**fields)


# --------------------------------------------------------------- fresh create
def _proposal_columns(path: Path) -> set[str]:
    return {r[1] for r in _raw(path, "PRAGMA table_info(reassoc_proposals)")}


def _index_names(path: Path) -> set[str]:
    return {r[0] for r in _raw(path, "SELECT name FROM sqlite_master WHERE type = 'index'")}


def test_a_fresh_store_is_created_at_v3_with_run_id_and_the_identity_indexes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "alerts.db"
    store = EventStore(path)
    assert store.version_on_open == 0  # nothing was there
    store.close()

    assert USER_VERSION == 3
    assert _user_version(path) == USER_VERSION
    assert {"alerts", "reassoc_proposals"} <= _tables(path)
    assert _raw(path, "PRAGMA journal_mode")[0][0].lower() == "wal"

    # v3 built directly, not v2-then-migrated: the run_id column and both partial-unique
    # identity indexes exist on a store that was never anything but current.
    assert "run_id" in _proposal_columns(path)
    assert {"idx_reassoc_confirmed_new", "idx_reassoc_confirmed_candidate"} <= _index_names(path)


def test_the_frozen_v1_ddl_is_still_the_shipped_alerts_ddl() -> None:
    """Drift guard on the fixture above.

    If this fails, either the alerts table changed without a migration to carry existing
    files across (which is the bug), or it changed with one, in which case this fixture is
    now a v1 fixture in name only and the migration test below is testing nothing.
    """
    assert V1_CREATE_ALERTS == mig.CREATE_ALERTS
    assert V1_CREATE_INDEX == mig.CREATE_ALERTS_INDEX


def test_reopening_a_current_file_changes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "alerts.db"
    first = EventStore(path)
    _propose(first)
    first.close()

    before_schema, before_version = _schema_text(path), _user_version(path)
    second = EventStore(path)
    assert second.version_on_open == USER_VERSION
    assert len(second.pending_proposals()) == 1
    second.close()

    assert _schema_text(path) == before_schema
    assert _user_version(path) == before_version


# ------------------------------------------------------------------ migration
def test_a_v1_file_migrates_to_current_preserving_every_alert_row(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    _write_v1_db(path)
    assert _user_version(path) == 1
    before = _alert_bytes(path)
    before_alerts_sql = [r for r in _schema_text(path) if r[2] == "alerts"]

    store = EventStore(path)
    assert store.version_on_open == 1
    store.close()

    assert _user_version(path) == USER_VERSION
    assert "reassoc_proposals" in _tables(path)
    # The v1 -> v2 -> v3 chain ran to completion: the proposals table exists with run_id
    # and both identity indexes, not just the v2 shape.
    assert "run_id" in _proposal_columns(path)
    assert {"idx_reassoc_confirmed_new", "idx_reassoc_confirmed_candidate"} <= _index_names(path)
    # Additive means additive: same rows, same rowids, same order, same DDL for alerts.
    assert _alert_bytes(path) == before
    assert [r for r in _schema_text(path) if r[2] == "alerts"] == before_alerts_sql
    assert _raw(path, "PRAGMA journal_mode")[0][0].lower() == "wal"


def test_alerts_written_before_the_upgrade_read_back_through_the_current_store(
    tmp_path: Path,
) -> None:
    """The rows survive as bytes above; here they survive as records, feedback included."""
    path = tmp_path / "legacy.db"
    _write_v1_db(path)

    with EventStore(path) as store:
        alerts = store.alerts()
        assert store.count() == 3
        by_id = {a.alert_id: a for a in alerts}
        old = by_id["stillness_v0-7-120500"]
        assert old.feedback == "agree"
        assert old.mrn_binding == "MRN-0007"
        assert old.schema_version == 1  # the record contract did not move with the file
        assert old.details["note"].endswith("(café)")
        assert by_id["drift_v0-12-60000"].baseline is None


def test_a_migrated_file_immediately_accepts_proposals(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    _write_v1_db(path)

    with EventStore(path) as store:
        proposal = _propose(store)
        assert store.pending_proposals() == [proposal]
        assert store.count() == 3  # and the alerts are still there


def test_a_v1_file_with_no_alerts_still_migrates(tmp_path: Path) -> None:
    path = tmp_path / "empty_legacy.db"
    _write_v1_db(path, rows=[])

    with EventStore(path) as store:
        assert store.version_on_open == 1
        assert store.count() == 0
    assert _user_version(path) == USER_VERSION


# ------------------------------------------------------------ v2 -> v3 migration
# A frozen copy of the version 2 proposals schema, exactly as it shipped: no run_id, no
# identity indexes. Duplicated here for the same reason as the v1 alerts fixture, so a
# change to the current DDL cannot quietly change the "old" database this migrates from.
V2_CREATE_PROPOSALS = """
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

V2_CREATE_PROPOSALS_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_reassoc_status_ts ON reassoc_proposals (status, ts)"
)

V2_INSERT_PROPOSAL = """
INSERT INTO reassoc_proposals (
    proposal_id, schema_version, ts, new_track, candidate_track,
    confidence, gallery_size, status, decided_ts, decided_via
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# One of each status, plus two confirmed rows that share a new_track. The pair proves the
# v2 -> v3 index build tolerates legacy confirmed rows: their run_id is NULL after the
# ALTER, and a NULL makes each row distinct in the partial-unique index, so the CREATE does
# not fail on data that predates run scoping.
V2_PROPOSAL_ROWS = [
    ("reassoc-1-2-1000", 1, 1.0, 1, 2, 0.90, 4, "pending", None, None),
    ("reassoc-5-10-2000", 1, 2.0, 5, 10, 0.80, 4, "confirmed", 3.0, "dashboard:nurse_a"),
    ("reassoc-5-20-3000", 1, 3.0, 5, 20, 0.70, 4, "confirmed", 4.0, "dashboard:nurse_b"),
    ("reassoc-9-8-4000", 1, 5.0, 9, 8, 0.60, 4, "rejected", 6.0, "dashboard:nurse_a"),
]


def _write_v2_db(path: Path) -> None:
    """Build a real version 2 store file with raw SQL: alerts (v1) plus a v2 proposals
    table, both stamped user_version = 2. No current code is involved."""
    raw = sqlite3.connect(str(path))
    try:
        raw.execute("PRAGMA journal_mode=WAL")
        raw.execute(V1_CREATE_ALERTS)
        raw.execute(V1_CREATE_INDEX)
        raw.executemany(V1_INSERT, V1_ROWS)
        raw.execute(V2_CREATE_PROPOSALS)
        raw.execute(V2_CREATE_PROPOSALS_INDEX)
        raw.executemany(V2_INSERT_PROPOSAL, V2_PROPOSAL_ROWS)
        raw.execute("PRAGMA user_version = 2")
        raw.commit()
    finally:
        raw.close()


def test_a_v2_file_migrates_to_v3_adding_run_id_and_indexes_preserving_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v2.db"
    _write_v2_db(path)
    assert _user_version(path) == 2
    before_alerts = _alert_bytes(path)

    store = EventStore(path)
    assert store.version_on_open == 2
    store.close()

    # The file moved to v3, gained the column and both identity indexes, and did not fail
    # the index build on the two legacy confirmed rows that share a new_track.
    assert _user_version(path) == USER_VERSION
    assert "run_id" in _proposal_columns(path)
    assert {"idx_reassoc_confirmed_new", "idx_reassoc_confirmed_candidate"} <= _index_names(path)

    # Every proposal row is preserved, and each legacy row now carries run_id = NULL.
    migrated = _raw(
        path,
        "SELECT proposal_id, status, decided_via, run_id FROM reassoc_proposals ORDER BY ts",
    )
    assert [(r[0], r[1], r[2]) for r in migrated] == [
        (row[0], row[7], row[9]) for row in V2_PROPOSAL_ROWS
    ]
    assert all(r[3] is None for r in migrated)  # run_id NULL on every pre-v3 row

    # Alerts are untouched, and the migrated file reads back through the current store.
    assert _alert_bytes(path) == before_alerts
    with EventStore(path) as reopened:
        assert reopened.count() == 3
        assert {p.proposal_id for p in reopened.proposal_history()} == {
            row[0] for row in V2_PROPOSAL_ROWS
        }
        assert [p.proposal_id for p in reopened.pending_proposals()] == ["reassoc-1-2-1000"]


# -------------------------------------------------------------------- run scope
def test_run_id_round_trips_on_a_recorded_proposal(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "alerts.db")
    proposal = _propose(store, run_id="run-a")
    assert proposal.run_id == "run-a"

    (read_back,) = store.pending_proposals()
    assert read_back == proposal
    assert read_back.run_id == "run-a"
    assert store.get_proposal(proposal.proposal_id).run_id == "run-a"
    store.close()


def test_run_id_scopes_the_pending_queue_and_the_decision(tmp_path: Path) -> None:
    """Two runs sharing one file: neither sees nor can answer the other's live questions,
    yet both are visible in the unscoped history."""
    path = tmp_path / "shared.db"
    store = EventStore(path)
    a = store.record_proposal(
        new_track=41, candidate_track=12, confidence=0.9, ts=10.0, gallery_size=4, run_id="run-a"
    )
    b = store.record_proposal(
        new_track=50, candidate_track=20, confidence=0.8, ts=20.0, gallery_size=4, run_id="run-b"
    )

    # The live queue is scoped: run-a sees only its own question, never run-b's.
    assert [p.proposal_id for p in store.pending_proposals(run_id="run-a")] == [a.proposal_id]
    assert [p.proposal_id for p in store.pending_proposals(run_id="run-b")] == [b.proposal_id]
    # Unscoped, the queue and the history span both runs.
    assert {p.proposal_id for p in store.pending_proposals()} == {a.proposal_id, b.proposal_id}
    assert {p.proposal_id for p in store.proposal_history()} == {a.proposal_id, b.proposal_id}
    assert [p.proposal_id for p in store.proposal_history(run_id="run-a")] == [a.proposal_id]

    # run-a may not answer run-b's proposal, even though it can see it in history.
    with pytest.raises(UnknownProposalError, match="belongs to run 'run-b'"):
        store.decide_proposal(b.proposal_id, "confirmed", 30.0, "dashboard:a", run_id="run-a")
    assert store.get_proposal(b.proposal_id).status == "pending"  # unchanged

    # The owning run decides its own, and b is still answerable by run-b afterward.
    store.decide_proposal(a.proposal_id, "confirmed", 31.0, "dashboard:a", run_id="run-a")
    decided_b = store.decide_proposal(
        b.proposal_id, "confirmed", 32.0, "dashboard:b", run_id="run-b"
    )
    assert store.get_proposal(a.proposal_id).status == "confirmed"
    assert decided_b.status == "confirmed"
    store.close()


# ------------------------------------------------------- identity uniqueness
def test_a_second_confirm_of_one_track_is_refused_by_the_new_track_index(
    tmp_path: Path,
) -> None:
    """A track gets at most one confirmed identity: the (run_id, new_track) partial index
    rejects the second confirm, and the loser can still be recorded as a rejection."""
    store = EventStore(tmp_path / "alerts.db")
    p1 = store.record_proposal(
        new_track=41, candidate_track=12, confidence=0.9, ts=10.0, gallery_size=4, run_id="run-a"
    )
    p2 = store.record_proposal(
        new_track=41, candidate_track=99, confidence=0.8, ts=20.0, gallery_size=4, run_id="run-a"
    )

    store.decide_proposal(p1.proposal_id, "confirmed", 30.0, "dashboard:a", run_id="run-a")

    with pytest.raises(ConfirmConflictError) as caught:
        store.decide_proposal(p2.proposal_id, "confirmed", 31.0, "dashboard:a", run_id="run-a")
    assert isinstance(caught.value, EventStoreError)
    assert isinstance(caught.value.__cause__, sqlite3.IntegrityError)

    # The refused confirm left p2 pending, so the stage's fallback (record it as a
    # rejection) still lands: the losing confirm is not lost, it becomes a rejected label.
    assert store.get_proposal(p2.proposal_id).status == "pending"
    losing = store.decide_proposal(p2.proposal_id, "rejected", 32.0, "dashboard:a", run_id="run-a")
    assert losing.status == "rejected"
    assert [r["label"] for r in store.labels_as_rows()] == [1, 0]
    store.close()


def test_a_second_confirm_of_one_identity_is_refused_by_the_candidate_track_index(
    tmp_path: Path,
) -> None:
    """A departed identity is claimed by at most one live track: the (run_id,
    candidate_track) partial index rejects the second confirm."""
    store = EventStore(tmp_path / "alerts.db")
    p1 = store.record_proposal(
        new_track=41, candidate_track=12, confidence=0.9, ts=10.0, gallery_size=4, run_id="run-a"
    )
    p3 = store.record_proposal(
        new_track=77, candidate_track=12, confidence=0.8, ts=20.0, gallery_size=4, run_id="run-a"
    )

    store.decide_proposal(p1.proposal_id, "confirmed", 30.0, "dashboard:a", run_id="run-a")
    with pytest.raises(ConfirmConflictError):
        store.decide_proposal(p3.proposal_id, "confirmed", 31.0, "dashboard:a", run_id="run-a")
    assert store.get_proposal(p3.proposal_id).status == "pending"
    store.close()


def test_the_identity_indexes_are_per_run_not_global(tmp_path: Path) -> None:
    """Two different runs may each confirm their own track 41 against their own identity 12;
    the uniqueness is scoped by run_id, so this is not a conflict.

    The timestamps differ only so the two proposals get distinct ids (ids are derived from
    the pair and ts, never the run); the point under test is the same track and candidate
    numbers confirmed in two runs without the identity indexes objecting.
    """
    store = EventStore(tmp_path / "alerts.db")
    a = store.record_proposal(
        new_track=41, candidate_track=12, confidence=0.9, ts=10.0, gallery_size=4, run_id="run-a"
    )
    b = store.record_proposal(
        new_track=41, candidate_track=12, confidence=0.9, ts=11.0, gallery_size=4, run_id="run-b"
    )
    assert a.proposal_id != b.proposal_id
    store.decide_proposal(a.proposal_id, "confirmed", 30.0, "dashboard:a", run_id="run-a")
    # Same track and candidate numbers, different run: no conflict.
    decided_b = store.decide_proposal(
        b.proposal_id, "confirmed", 31.0, "dashboard:b", run_id="run-b"
    )
    assert decided_b.status == "confirmed"
    store.close()


# --------------------------------------------------------------- busy timeout
def test_a_write_blocked_past_the_timeout_is_translated_to_store_busy_error(
    tmp_path: Path,
) -> None:
    """A second connection holding the write lock makes the store's write raise
    StoreBusyError, not a raw sqlite3.OperationalError. The busy timeout is set to zero so
    the translation, not a five-second wait, is what the test exercises."""
    path = tmp_path / "alerts.db"
    store = EventStore(path)
    store._conn.execute("PRAGMA busy_timeout = 0")  # fail fast instead of waiting

    blocker = sqlite3.connect(str(path))
    blocker.isolation_level = None
    blocker.execute("BEGIN IMMEDIATE")  # takes the single write lock and holds it
    try:
        with pytest.raises(StoreBusyError) as caught:
            _propose(store)
        assert isinstance(caught.value, EventStoreError)
        assert isinstance(caught.value.__cause__, sqlite3.OperationalError)
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()

    # The store is not wedged by the rejected write: once the lock frees, the write lands.
    proposal = _propose(store)
    assert store.pending_proposals() == [proposal]
    store.close()


# ------------------------------------------------------------------- refusals
def test_a_future_version_file_is_refused_and_never_downgraded(tmp_path: Path) -> None:
    path = tmp_path / "from_the_future.db"
    _write_v1_db(path)
    raw = sqlite3.connect(str(path))
    raw.execute(f"PRAGMA user_version = {USER_VERSION + 1}")
    raw.commit()
    raw.close()
    before = _alert_bytes(path)

    with pytest.raises(SchemaVersionError, match=f"user_version={USER_VERSION + 1}"):
        EventStore(path)

    assert _user_version(path) == USER_VERSION + 1  # not stamped back down
    assert "reassoc_proposals" not in _tables(path)  # and not half migrated
    assert _alert_bytes(path) == before


def test_a_version_from_far_in_the_future_is_refused_too(tmp_path: Path) -> None:
    path = tmp_path / "v99.db"
    raw = sqlite3.connect(str(path))
    raw.execute("PRAGMA user_version = 99")
    raw.commit()
    raw.close()

    with pytest.raises(SchemaVersionError, match="never downgrades"):
        EventStore(path)
    assert _tables(path) == set()


def test_a_version_no_build_ever_wrote_is_refused(tmp_path: Path) -> None:
    """PRAGMA user_version is a signed integer, so a negative one is a file someone edited."""
    path = tmp_path / "negative.db"
    raw = sqlite3.connect(str(path))
    raw.execute("PRAGMA user_version = -5")
    raw.commit()
    raw.close()

    with pytest.raises(SchemaVersionError, match="no build ever wrote"):
        EventStore(path)
    assert _user_version(path) == -5
    assert _tables(path) == set()


def test_a_foreign_table_under_our_proposal_name_rolls_the_whole_open_back(
    tmp_path: Path,
) -> None:
    """A squatted table name must leave the file exactly as it was found, still at v1."""
    path = tmp_path / "squatted.db"
    _write_v1_db(path)
    raw = sqlite3.connect(str(path))
    raw.execute("CREATE TABLE reassoc_proposals (proposal_id TEXT, something_else INTEGER)")
    raw.commit()
    raw.close()
    before = _alert_bytes(path)

    with pytest.raises(SchemaVersionError, match="candidate_track"):
        EventStore(path)

    assert _user_version(path) == 1  # the migration did not half happen
    assert _alert_bytes(path) == before
    assert _raw(
        path, "SELECT COUNT(*) FROM sqlite_master WHERE name = 'idx_reassoc_status_ts'"
    ) == [(0,)]


def test_the_database_itself_rejects_a_status_outside_the_three(tmp_path: Path) -> None:
    """The CHECK constraint is the guarantee that survives a process outside this codebase.

    An operator with the sqlite3 CLI is a real reader and writer of this file. A status of
    'accepted' hand-typed into it would turn an identity label into a value no counter
    knows how to read, so the constraint lives in the schema and not only in pydantic.
    """
    path = tmp_path / "alerts.db"
    EventStore(path).close()

    raw = sqlite3.connect(str(path))
    try:
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            raw.execute(
                "INSERT INTO reassoc_proposals (proposal_id, schema_version, ts, new_track, "
                "candidate_track, confidence, gallery_size, status) "
                "VALUES ('reassoc-1-2-0', 1, 0.0, 1, 2, 0.5, 3, 'accepted')"
            )
    finally:
        raw.close()


# ------------------------------------------------------------------ proposal ids
def test_proposal_ids_are_deterministic_and_match_the_contract_formula() -> None:
    assert make_proposal_id(41, 12, 1830.25) == "reassoc-41-12-1830250"
    assert make_proposal_id(41, 12, 1830.25) == make_proposal_id(41, 12, 1830.25)
    # Truncation toward zero, like alert ids: two proposals inside one ms are one proposal.
    assert make_proposal_id(41, 12, 1830.2509) == "reassoc-41-12-1830250"
    # The pair is ordered. Asking "is 41 the person who was 12" is not the other question.
    assert make_proposal_id(12, 41, 1830.25) != make_proposal_id(41, 12, 1830.25)
    # Determinism is scoped to a run: same run, same key; different run, different key.
    assert make_proposal_id(41, 12, 1830.25, "run-a") == make_proposal_id(41, 12, 1830.25, "run-a")
    assert make_proposal_id(41, 12, 1830.25, "run-a") != make_proposal_id(41, 12, 1830.25, "run-b")
    # An unscoped id keeps the legacy form, so rows written before scoping still resolve.
    assert make_proposal_id(41, 12, 1830.25, None) == "reassoc-41-12-1830250"
    assert make_proposal_id(41, 12, 1830.25, "run-a").startswith("reassoc-41-12-1830250-")


def test_a_foreign_runs_row_cannot_swallow_this_runs_question(tmp_path: Path) -> None:
    """Two runs, one store file, same track ids and same clock: two questions, not one.

    Track ids restart at 1 every run and the store is shared per run name, so an unscoped
    key made the second run's question a primary-key duplicate of the first's. The store
    swallowed it, this run's queue never showed it (that queue is run-scoped), and the
    summary said nothing was awaiting a human. It fails toward never asking, silently,
    which is the one direction propose-then-confirm cannot tolerate.
    """
    store = EventStore(tmp_path / "alerts.sqlite")
    try:
        first = store.record_proposal(
            new_track=3, candidate_track=1, confidence=0.9, ts=40.0, gallery_size=1, run_id="run-a"
        )
        second = store.record_proposal(
            new_track=3, candidate_track=1, confidence=0.9, ts=40.0, gallery_size=1, run_id="run-b"
        )

        assert first.proposal_id != second.proposal_id
        # Each run sees exactly its own question, and both are really awaiting an answer.
        assert [p.proposal_id for p in store.pending_proposals(run_id="run-a")] == [
            first.proposal_id
        ]
        assert [p.proposal_id for p in store.pending_proposals(run_id="run-b")] == [
            second.proposal_id
        ]
    finally:
        store.close()


def test_recording_the_same_proposal_twice_is_refused_not_duplicated(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "alerts.db")
    first = _propose(store)

    with pytest.raises(DuplicateProposalError) as caught:
        _propose(store)
    assert isinstance(caught.value, EventStoreError)
    assert isinstance(caught.value.__cause__, sqlite3.IntegrityError)

    assert store.pending_proposals() == [first]
    # The store is still usable after the rejected insert.
    assert _propose(store, new_track=42).proposal_id != first.proposal_id
    store.close()


def test_a_replay_never_re_asks_a_question_that_was_answered(tmp_path: Path) -> None:
    """The whole point of a derived id: the second pass collides with the human's answer."""
    path = tmp_path / "alerts.db"
    first = EventStore(path)
    proposal = _propose(first)
    first.decide_proposal(proposal.proposal_id, "rejected", 1900.0, "dashboard:charge_nurse")
    first.close()

    second = EventStore(path)
    with pytest.raises(DuplicateProposalError):
        _propose(second)
    stored = second.get_proposal(proposal.proposal_id)
    assert stored is not None
    assert stored.status == "rejected"
    assert second.pending_proposals() == []
    second.close()


# -------------------------------------------------------------- record and read
def test_a_recorded_proposal_is_pending_and_round_trips(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "alerts.db")
    proposal = _propose(store)

    assert proposal.status == "pending"
    assert proposal.decided_ts is None
    assert proposal.decided_via is None
    assert proposal.schema_version == 1
    assert proposal.proposal_id == make_proposal_id(41, 12, 1830.25)

    (read_back,) = store.pending_proposals()
    assert read_back == proposal
    assert read_back.confidence == pytest.approx(0.87)
    assert read_back.gallery_size == 6
    assert store.get_proposal(proposal.proposal_id) == proposal
    assert store.get_proposal("reassoc-404-404-0") is None
    store.close()


def test_record_proposal_refuses_inputs_that_cannot_mean_anything(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "alerts.db")

    with pytest.raises(ValueError, match="calibrated score"):
        _propose(store, confidence=1.4)
    with pytest.raises(ValueError, match="calibrated score"):
        _propose(store, confidence=-0.2)  # a raw cosine, which is not a confidence
    with pytest.raises(ValueError, match="finite"):
        _propose(store, confidence=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        _propose(store, ts=float("inf"))
    with pytest.raises(ValueError, match="itself"):
        _propose(store, new_track=5, candidate_track=5)
    with pytest.raises(ValueError, match="gallery_size"):
        _propose(store, gallery_size=-1)

    assert store.pending_proposals() == []
    store.close()


def test_the_endpoints_of_the_confidence_range_are_allowed(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "alerts.db")
    assert _propose(store, confidence=0.0, ts=1.0).confidence == 0.0
    assert _propose(store, confidence=1.0, ts=2.0).confidence == 1.0
    store.close()


def test_proposals_survive_close_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "alerts.db"
    writer = EventStore(path)
    proposal = _propose(writer)
    writer.close()

    reader = EventStore(path)
    assert reader.pending_proposals() == [proposal]
    reader.close()


# ------------------------------------------------------------------- decisions
def test_a_decision_writes_the_label_and_leaves_the_queue(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "alerts.db")
    proposal = _propose(store)

    decided = store.decide_proposal(
        proposal.proposal_id, "confirmed", 1902.5, "dashboard:charge_nurse"
    )
    assert decided.status == "confirmed"
    assert decided.decided_ts == pytest.approx(1902.5)
    assert decided.decided_via == "dashboard:charge_nurse"
    # Everything the machine wrote is untouched by the human's answer.
    assert (decided.proposal_id, decided.ts, decided.confidence, decided.gallery_size) == (
        proposal.proposal_id,
        proposal.ts,
        proposal.confidence,
        proposal.gallery_size,
    )

    assert store.pending_proposals() == []
    assert store.proposal_history() == [decided]
    store.close()


def test_a_second_decision_is_refused_and_changes_nothing(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "alerts.db")
    proposal = _propose(store)
    first = store.decide_proposal(proposal.proposal_id, "confirmed", 1902.5, "dashboard:nurse_a")

    with pytest.raises(ProposalAlreadyDecidedError, match="already confirmed"):
        store.decide_proposal(proposal.proposal_id, "rejected", 2000.0, "dashboard:nurse_b")
    # Agreeing with the stored answer is refused too: a label is written once, by one person.
    with pytest.raises(ProposalAlreadyDecidedError):
        store.decide_proposal(proposal.proposal_id, "confirmed", 2000.0, "dashboard:nurse_b")

    assert store.get_proposal(proposal.proposal_id) == first
    # The connection is not wedged by the two rejected writes.
    other = _propose(store, new_track=42)
    assert store.pending_proposals() == [other]
    store.close()


def test_a_decision_on_an_unknown_proposal_raises(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "alerts.db")
    with pytest.raises(UnknownProposalError):
        store.decide_proposal("reassoc-9-9-9", "confirmed", 1.0, "dashboard")
    store.close()


def test_a_decision_outside_the_two_words_is_rejected_before_it_reaches_sql(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "alerts.db")
    proposal = _propose(store)

    for bad in ("pending", "Confirmed", "accept", ""):
        with pytest.raises(ValueError, match="unknown re-association decision"):
            store.decide_proposal(proposal.proposal_id, bad, 1902.5, "dashboard")

    assert store.get_proposal(proposal.proposal_id) == proposal
    store.close()


def test_a_decision_with_no_provenance_is_rejected(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "alerts.db")
    proposal = _propose(store)

    with pytest.raises(ValueError, match="via"):
        store.decide_proposal(proposal.proposal_id, "confirmed", 1902.5, "   ")

    assert store.get_proposal(proposal.proposal_id) == proposal
    store.close()


# ----------------------------------------------------------------- queue order
def test_pending_is_a_fifo_queue_and_hides_decided_proposals(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "alerts.db")
    for i in range(4):
        _propose(store, new_track=100 + i, ts=float(i * 10))

    assert [p.new_track for p in store.pending_proposals()] == [100, 101, 102, 103]
    # A limit drops the newest questions, never the oldest: the oldest is the one whose
    # track is most likely to have walked out of the room already.
    assert [p.new_track for p in store.pending_proposals(limit=2)] == [100, 101]

    store.decide_proposal(make_proposal_id(100, 12, 0.0), "rejected", 50.0, "dashboard")
    assert [p.new_track for p in store.pending_proposals()] == [101, 102, 103]

    with pytest.raises(ValueError):
        store.pending_proposals(limit=0)
    store.close()


def test_history_is_newest_first_and_keeps_the_decided_ones(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "alerts.db")
    for i in range(3):
        _propose(store, new_track=100 + i, ts=float(i * 10))
    store.decide_proposal(make_proposal_id(100, 12, 0.0), "confirmed", 50.0, "dashboard")

    assert [p.new_track for p in store.proposal_history()] == [102, 101, 100]
    assert [p.status for p in store.proposal_history()] == ["pending", "pending", "confirmed"]
    assert [p.new_track for p in store.proposal_history(limit=1)] == [102]

    with pytest.raises(ValueError):
        store.proposal_history(limit=0)
    store.close()


# --------------------------------------------------------------- label export
def test_labels_as_rows_exports_decisions_only(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "alerts.db")
    yes = _propose(store, new_track=100, ts=10.0, confidence=0.91, gallery_size=6)
    no = _propose(store, new_track=101, ts=20.0, confidence=0.42, gallery_size=7)
    _propose(store, new_track=102, ts=30.0)  # still pending, still not a label

    store.decide_proposal(yes.proposal_id, "confirmed", 55.0, "dashboard:charge_nurse")
    store.decide_proposal(no.proposal_id, "rejected", 65.0, "dashboard:charge_nurse")

    rows = store.labels_as_rows()
    assert [r["proposal_id"] for r in rows] == [yes.proposal_id, no.proposal_id]  # oldest first
    assert [r["label"] for r in rows] == [1, 0]
    assert [r["decision"] for r in rows] == ["confirmed", "rejected"]
    assert rows[0] == {
        "proposal_id": yes.proposal_id,
        "schema_version": 1,
        "ts": 10.0,
        "new_track": 100,
        "candidate_track": 12,
        "confidence": pytest.approx(0.91),
        "gallery_size": 6,
        "decision": "confirmed",
        "label": 1,
        "decided_ts": 55.0,
        "decided_via": "dashboard:charge_nurse",
    }
    # Two exports of one store are identical, so a diff between exports is new labels only.
    assert store.labels_as_rows() == rows
    store.close()


def test_an_export_with_nothing_decided_is_empty_not_negative(tmp_path: Path) -> None:
    """A question nobody answered yet is not a rejection, and must not train as one."""
    store = EventStore(tmp_path / "alerts.db")
    _propose(store)
    assert store.labels_as_rows() == []
    store.close()


def test_labels_survive_the_migration_boundary_in_both_directions(tmp_path: Path) -> None:
    """A migrated file collects labels, and they are still there on the next open."""
    path = tmp_path / "legacy.db"
    _write_v1_db(path)

    first = EventStore(path)
    proposal = _propose(first)
    first.decide_proposal(proposal.proposal_id, "confirmed", 1902.5, "dashboard:charge_nurse")
    first.close()

    second = EventStore(path)
    assert second.version_on_open == USER_VERSION
    (row,) = second.labels_as_rows()
    assert row["label"] == 1
    assert row["decided_via"] == "dashboard:charge_nurse"
    assert second.count() == 3  # the pre-existing alerts are untouched by any of this
    second.close()


# ------------------------------------------------------------------------- WAL
def test_a_second_open_connection_reads_committed_proposals(tmp_path: Path) -> None:
    """Mirrors the alert-side WAL test: the dashboard reads while the pipeline writes."""
    path = tmp_path / "alerts.db"
    writer = EventStore(path)
    reader = EventStore(path)

    assert reader.pending_proposals() == []
    proposal = writer.record_proposal(
        new_track=41, candidate_track=12, confidence=0.9, ts=100.0, gallery_size=4
    )
    assert reader.pending_proposals() == [proposal]

    # The decision comes from the reader side, which is where a dashboard lives.
    reader.decide_proposal(proposal.proposal_id, "confirmed", 150.0, "dashboard")
    assert writer.pending_proposals() == []
    assert [r["label"] for r in writer.labels_as_rows()] == [1]

    reader.close()
    writer.close()


def test_a_reader_holding_a_snapshot_does_not_block_the_writer(tmp_path: Path) -> None:
    """WAL's actual promise. Under a rollback journal this write would wait, then fail."""
    path = tmp_path / "alerts.db"
    store = EventStore(path)
    _propose(store, new_track=100, ts=10.0)

    raw = sqlite3.connect(str(path))
    raw.isolation_level = None  # transactions are explicit below, nothing implicit
    try:
        raw.execute("BEGIN")
        assert raw.execute("SELECT COUNT(*) FROM reassoc_proposals").fetchone()[0] == 1

        _propose(store, new_track=101, ts=20.0)  # writes while that read is open

        # The reader keeps its snapshot until it ends its own transaction.
        assert raw.execute("SELECT COUNT(*) FROM reassoc_proposals").fetchone()[0] == 1
        raw.execute("ROLLBACK")
        assert raw.execute("SELECT COUNT(*) FROM reassoc_proposals").fetchone()[0] == 2
    finally:
        raw.close()
    store.close()


# ------------------------------------------------------------------- lifecycle
def test_the_proposal_api_is_a_clean_error_after_close(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "alerts.db")
    proposal = _propose(store)
    store.close()

    with pytest.raises(EventStoreError):
        _propose(store, new_track=42)
    with pytest.raises(EventStoreError):
        store.decide_proposal(proposal.proposal_id, "confirmed", 1.0, "dashboard")
    with pytest.raises(EventStoreError):
        store.pending_proposals()
    with pytest.raises(EventStoreError):
        store.proposal_history()
    with pytest.raises(EventStoreError):
        store.get_proposal(proposal.proposal_id)
    with pytest.raises(EventStoreError):
        store.labels_as_rows()


# ---------------------------------------------------------------------- schema
def test_the_record_version_is_pinned_to_one_inside_a_version_three_file() -> None:
    record = ReassocProposalRecord(
        proposal_id="reassoc-41-12-1830250",
        ts=1830.25,
        new_track=41,
        candidate_track=12,
        confidence=0.87,
        gallery_size=6,
    )
    assert record.schema_version == 1
    assert record.status == "pending"
    assert record.run_id is None  # the added field is nullable and defaults to no run
    # The file is at user_version 3 and this record is at 1, on purpose: the two numbers
    # count different things (a set of tables and columns, and one row shape). Pinning is
    # what makes a field change a new version plus a migration; run_id did not bump the
    # record version because it is additive and nullable, not a change to an existing field.
    assert USER_VERSION == 3
    with pytest.raises(ValidationError):
        ReassocProposalRecord.model_validate({**record.model_dump(), "schema_version": 2})


def test_the_record_rejects_a_status_outside_the_literal() -> None:
    fields = {
        "proposal_id": "reassoc-41-12-1830250",
        "ts": 1830.25,
        "new_track": 41,
        "candidate_track": 12,
        "confidence": 0.87,
        "gallery_size": 6,
    }
    with pytest.raises(ValidationError):
        ReassocProposalRecord(**fields, status="accepted")
    with pytest.raises(ValidationError):
        ReassocProposalRecord(**fields, status="Confirmed")
