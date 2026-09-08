"""Event store + alert schema tests.

Deterministic, network-free, model-free: every fixture is a hand-built AlertRecord, so
this file runs identically on a laptop and on linux CI with no weights present. Anything
that needs a real detector belongs to the acceptance flows in docs/PIPELINE.md, section 4,
not to a test.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from vividly.events import (
    USER_VERSION,
    DuplicateAlertError,
    EventStore,
    EventStoreError,
    SchemaVersionError,
    UnknownAlertError,
)
from vividly.schemas import AlertRecord, make_alert_id, utc_now_iso


# --------------------------------------------------------------------- helpers
def _alert(
    rule_id: str = "stillness_v0",
    track_id: int = 7,
    onset_ts: float = 120.5,
    **over,
) -> AlertRecord:
    """A fully populated record. Every field is set so round-trip tests prove fidelity."""
    fields = {
        "alert_id": make_alert_id(rule_id, track_id, onset_ts),
        "track_id": track_id,
        "rule_id": rule_id,
        "rule_version": "0.1.0",
        "onset_ts": onset_ts,
        "detected_ts": onset_ts + 2400.0,
        "created_utc": "2026-08-07T18:30:00.000Z",
        "baseline": {
            "movement_var": 12.5,
            "posture_deg_mean": 4.25,
            "posture_deg_std": 1.125,
            "cy_mean": 512.0,
            "n_samples": 900,
            "complete": True,
        },
        "details": {
            "minutes_still": 41.5,
            "max_posture_dev_deg": 0.75,
            "note": "seat by the window (café), unicode survives",
        },
        "feedback": None,
        "mrn_binding": "MRN-0007",
    }
    fields.update(over)
    return AlertRecord(**fields)


def _store(tmp_path: Path, name: str = "alerts.db") -> EventStore:
    return EventStore(tmp_path / name)


# ---------------------------------------------------------------------- schema
def test_alert_id_matches_the_contract_formula() -> None:
    assert make_alert_id("drift_v0", 12, 60.0) == "drift_v0-12-60000"
    assert make_alert_id("drift_v0", 12, 60.0) == make_alert_id("drift_v0", 12, 60.0)
    # Truncation toward zero, not rounding: two firings inside one ms share an id.
    assert make_alert_id("drift_v0", 12, 60.0009) == "drift_v0-12-60000"


def test_schema_version_is_pinned_to_one() -> None:
    assert _alert().schema_version == 1
    with pytest.raises(ValidationError):
        _alert(schema_version=2)


def test_model_rejects_a_feedback_label_outside_the_literal() -> None:
    with pytest.raises(ValidationError):
        _alert(feedback="maybe")


def test_utc_now_iso_is_utc_and_millisecond_resolution() -> None:
    stamp = utc_now_iso()
    assert stamp.endswith("Z")
    assert stamp[10] == "T"
    assert len(stamp.split(".")[-1]) == 4  # three digits plus the Z


# ------------------------------------------------------------------ round trip
def test_insert_read_round_trip_is_field_for_field_equal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    original = _alert()
    store.insert_alert(original)

    (read_back,) = store.alerts()
    assert read_back == original
    assert read_back.baseline == original.baseline
    assert read_back.details["minutes_still"] == pytest.approx(41.5)
    assert read_back.details["note"] == original.details["note"]
    store.close()


def test_null_baseline_round_trips_as_none_not_empty(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.insert_alert(_alert(baseline=None, details={}))
    (read_back,) = store.alerts()
    assert read_back.baseline is None
    assert read_back.details == {}
    store.close()


def test_numpy_style_scalars_in_details_serialise(tmp_path: Path) -> None:
    """Rules compute with numpy; `.item()` is duck-typed so no numpy import is needed."""

    class FakeNumpyFloat:
        def item(self) -> float:
            return 3.25

    store = _store(tmp_path)
    store.insert_alert(_alert(details={"slope": FakeNumpyFloat()}))
    (read_back,) = store.alerts()
    assert read_back.details == {"slope": 3.25}
    store.close()


def test_alerts_are_newest_first_and_respect_limit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for i in range(5):
        store.insert_alert(_alert(track_id=i, onset_ts=float(i * 10)))

    ids = [a.alert_id for a in store.alerts()]
    assert ids == [make_alert_id("stillness_v0", i, float(i * 10)) for i in (4, 3, 2, 1, 0)]
    assert len(store.alerts(limit=2)) == 2
    assert store.alerts(limit=2)[0].track_id == 4
    assert store.count() == 5

    with pytest.raises(ValueError):
        store.alerts(limit=0)
    store.close()


# -------------------------------------------------------------------- feedback
def test_feedback_persists_across_close_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "alerts.db"
    writer = EventStore(path)
    a = _alert()
    writer.insert_alert(a)
    assert writer.alerts()[0].feedback is None
    writer.set_feedback(a.alert_id, "false_alarm")
    writer.close()

    reader = EventStore(path)
    (read_back,) = reader.alerts()
    assert read_back.feedback == "false_alarm"
    assert read_back.alert_id == a.alert_id
    reader.close()


def test_invalid_feedback_label_is_rejected_and_changes_nothing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    a = _alert()
    store.insert_alert(a)

    with pytest.raises(ValueError, match="unknown feedback label"):
        store.set_feedback(a.alert_id, "AGREE")  # right word, wrong case
    with pytest.raises(ValueError):
        store.set_feedback(a.alert_id, "escalated")

    assert store.alerts()[0].feedback is None
    store.close()


def test_feedback_on_an_unknown_alert_raises_unknown_alert_error(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(UnknownAlertError):
        store.set_feedback("stillness_v0-99-1", "agree")
    store.close()


# ------------------------------------------------------------------- duplicates
def test_duplicate_alert_id_surfaces_as_duplicate_alert_error(tmp_path: Path) -> None:
    store = _store(tmp_path)
    a = _alert()
    store.insert_alert(a)

    with pytest.raises(DuplicateAlertError) as caught:
        store.insert_alert(a)

    assert isinstance(caught.value, EventStoreError)
    assert not isinstance(caught.value, sqlite3.IntegrityError)
    assert isinstance(caught.value.__cause__, sqlite3.IntegrityError)
    # The first row survives the rejected insert, and the store stays usable.
    assert store.count() == 1
    store.insert_alert(_alert(track_id=8))
    assert store.count() == 2
    store.close()


def test_a_rerun_of_the_same_clip_is_idempotent_by_construction(tmp_path: Path) -> None:
    """Same footage, same firings, same ids: the second pass adds nothing."""
    path = tmp_path / "alerts.db"
    firings = [("stillness_v0", 1, 100.0), ("drift_v0", 2, 250.0)]

    first = EventStore(path)
    for rule, track, onset in firings:
        first.insert_alert(_alert(rule_id=rule, track_id=track, onset_ts=onset))
    first.close()

    second = EventStore(path)
    for rule, track, onset in firings:
        with pytest.raises(DuplicateAlertError):
            second.insert_alert(_alert(rule_id=rule, track_id=track, onset_ts=onset))
    assert second.count() == 2
    second.close()


# ------------------------------------------------------------------------- WAL
def test_two_sequential_connections_see_the_same_data(tmp_path: Path) -> None:
    path = tmp_path / "alerts.db"
    a = _alert()

    with EventStore(path) as writer:
        writer.insert_alert(a)

    with EventStore(path) as annotator:
        annotator.set_feedback(a.alert_id, "agree")

    with EventStore(path) as reader:
        (read_back,) = reader.alerts()
        assert read_back.feedback == "agree"
        assert read_back == a.model_copy(update={"feedback": "agree"})


def test_a_second_open_connection_reads_committed_writes(tmp_path: Path) -> None:
    """WAL's point: the dashboard reads while the pipeline writes, no blocking."""
    path = tmp_path / "alerts.db"
    writer = EventStore(path)
    reader = EventStore(path)

    assert reader.alerts() == []
    writer.insert_alert(_alert(track_id=1, onset_ts=10.0))
    assert [r.track_id for r in reader.alerts()] == [1]

    writer.insert_alert(_alert(track_id=2, onset_ts=20.0))
    reader.set_feedback(make_alert_id("stillness_v0", 1, 10.0), "agree")
    assert [(r.track_id, r.feedback) for r in writer.alerts()] == [(2, None), (1, "agree")]

    reader.close()
    writer.close()


def test_file_carries_wal_mode_and_user_version(tmp_path: Path) -> None:
    path = tmp_path / "alerts.db"
    store = EventStore(path)
    store.insert_alert(_alert())
    store.close()

    raw = sqlite3.connect(str(path))
    try:
        assert raw.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert raw.execute("PRAGMA user_version").fetchone()[0] == USER_VERSION
    finally:
        raw.close()


def test_a_foreign_schema_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "future.db"
    raw = sqlite3.connect(str(path))
    raw.execute("PRAGMA user_version = 99")
    raw.commit()
    raw.close()

    with pytest.raises(SchemaVersionError, match="user_version=99"):
        EventStore(path)


def test_an_alien_alerts_table_is_refused_at_open(tmp_path: Path) -> None:
    """CREATE TABLE IF NOT EXISTS is a no-op against a foreign table of the same name.

    Without the column check the file opens clean and dies on the first insert, mid-run.
    """
    path = tmp_path / "alien.db"
    raw = sqlite3.connect(str(path))
    raw.execute("CREATE TABLE alerts (alert_id TEXT, something_else INTEGER)")
    raw.commit()
    raw.close()

    with pytest.raises(SchemaVersionError, match="detected_ts"):
        EventStore(path)


def test_a_corrupt_file_raises_event_store_error_not_a_raw_sqlite_error(
    tmp_path: Path,
) -> None:
    """The store's promise is that callers never catch sqlite3 directly."""
    path = tmp_path / "garbage.db"
    path.write_bytes(b"this is not a sqlite database" * 20)

    with pytest.raises(EventStoreError) as caught:
        EventStore(path)

    assert not isinstance(caught.value, sqlite3.Error)
    assert isinstance(caught.value.__cause__, sqlite3.DatabaseError)


def test_an_unopenable_path_raises_event_store_error(tmp_path: Path) -> None:
    """A directory where a file belongs: sqlite3.OperationalError, translated."""
    target = tmp_path / "not_a_file"
    target.mkdir()

    with pytest.raises(EventStoreError, match="cannot open event store"):
        EventStore(target)


# ----------------------------------------------------------------- lifecycle
def test_store_creates_missing_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "outputs" / "pipeline" / "run_a" / "alerts.db"
    store = EventStore(path)
    store.insert_alert(_alert())
    store.close()
    assert path.exists()


def test_use_after_close_is_a_clean_error(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.close()
    store.close()  # idempotent

    with pytest.raises(EventStoreError):
        store.alerts()
    with pytest.raises(EventStoreError):
        store.insert_alert(_alert())
    with pytest.raises(EventStoreError):
        store.set_feedback("whatever", "agree")


def test_get_returns_one_record_or_none(tmp_path: Path) -> None:
    store = _store(tmp_path)
    a = _alert()
    store.insert_alert(a)
    assert store.get(a.alert_id) == a
    assert store.get("stillness_v0-404-0") is None
    store.close()
