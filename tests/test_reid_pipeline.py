"""The re-ID stage inside the replay pipeline: it enrols, it probes, and it asks.

What these tests are actually pinning, and it is one sentence: beyond the track layer's
reap window the system never binds an identity by itself. Everything else in this file is
scaffolding around that claim. `test_a_return_beyond_the_window_is_proposed_not_bound` is
the load-bearing one; `test_reid_off_leaves_the_alert_stream_byte_identical` is the one
that keeps the whole feature optional; `test_the_gallery_is_empty_after_shutdown` is the
one CQ-16 in docs/COUNSEL_QUESTIONS.md points at.

Three rules hold throughout, inherited from tests/test_dashboard.py.

  * No models. Nothing here downloads a weight, opens an ONNX session or decodes a video.
    The embedder path is exercised against a stub embedder so the crop geometry can be
    proved on a machine with no onnxruntime and no cv2.
  * No network, no server.
  * Deterministic. The committed fixture is generated in closed form with no RNG and no
    clock, and `test_committed_reid_fixture_is_deterministic` regenerates it and compares
    bytes, so it cannot drift under anyone's feet.

Nothing in this file is evidence about re-identification. The embeddings come from the
invented cone geometry in `vividly.reid.synth`, keyed by an answer key written by hand into
configs/pipeline/synthetic_reid_demo.yaml. A green run proves the wiring carries an answer.
Whether the answer would be right on real overhead footage is docs/CALIBRATION.md's problem
and it has not been touched.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from vividly.dashboard import mock_mrn
from vividly.events import EventStore, UnknownProposalError
from vividly.reid.gallery import GalleryConfig, SessionGallery
from vividly.reid.stage import (
    EMBEDDING_SOURCES,
    CropEmbeddingSource,
    EmbeddingsSpec,
    ReidConfig,
    ReidConfigError,
    SyntheticEmbeddingSource,
    build_reid_stage,
    confidence_from_similarity,
    crop_from_frame,
    load_reid_config,
)
from vividly.sources import SyntheticDetectionSource, write_observations_jsonl
from vividly.tracks import TrackBook, TrackConfig, TrackIdentity, TrackObservation

REPO = Path(__file__).resolve().parent.parent
RUN_PIPELINE = REPO / "scripts" / "run_pipeline.py"
REID_CONFIG = REPO / "configs" / "pipeline" / "reid.yaml"
REID_DEMO_CONFIG = REPO / "configs" / "pipeline" / "synthetic_reid_demo.yaml"
SYNTHETIC_CONFIG = REPO / "configs" / "pipeline" / "synthetic_demo.yaml"
SCENE_PATH = REPO / "tests" / "fixtures" / "pipeline_reid" / "leave_and_return.jsonl"


# ------------------------------------------------------------- the scene fixture
# 120 simulated seconds, 2 observation batches per simulated second, four abstract tracks.
# 2 Hz rather than 15 is a file-size decision and nothing else: every window downstream is
# expressed in seconds, so nothing can tell the difference.
#
# POLICY.md, rule 4: this is not a simulated medical event and it is not training or evaluation
# data. It is four points moving on a plane, generated so a departure, a return and a
# stranger exist for the stage to reason about with no footage in existence.
#
#   track 1  identity 0   sits, then leaves for good at t = 20
#   track 2  identity 1   never leaves: never enrolled, so never a candidate for anything
#   track 3  identity 0   the same person returning at t = 40 under a new tracker id
#   track 4  identity 2   a stranger arriving at t = 60, who must produce no question
#
# Track 3 returns to a different seat from track 1's. Position is not a signal anywhere in
# this stage, and putting the return in the same chair would leave that unprovable.
SCENE_HZ = 2.0
SCENE_SECONDS = 120.0
SCENE_DECIMALS = 3

LEAVER_TRACK = 1
STAYER_TRACK = 2
RETURN_TRACK = 3
STRANGER_TRACK = 4
LEAVER_LAST_T = 20.0
RETURN_FIRST_T = 40.0

#: track id -> (first t, last t), inclusive at both ends.
SCENE_SPANS: dict[int, tuple[float, float]] = {
    LEAVER_TRACK: (0.0, LEAVER_LAST_T),
    STAYER_TRACK: (0.0, SCENE_SECONDS),
    RETURN_TRACK: (RETURN_FIRST_T, SCENE_SECONDS),
    STRANGER_TRACK: (60.0, SCENE_SECONDS),
}
SCENE_SEATS: dict[int, tuple[float, float]] = {
    LEAVER_TRACK: (300.0, 300.0),
    STAYER_TRACK: (900.0, 300.0),
    RETURN_TRACK: (300.0, 900.0),
    STRANGER_TRACK: (900.0, 900.0),
}
BOX_W = 60.0
BOX_H = 150.0
SWAY_PX = 6.0
SWAY_PERIOD_S = 20.0


def build_leave_and_return_scene() -> list[list[TrackObservation]]:
    """The committed fixture, in closed form. No RNG, no clock, no dataset.

    One batch per sampled instant, every track that is present at that instant, sitting on
    its own seat with a deterministic sway so the boxes are not frozen. Keypoints are null
    throughout: this fixture is about identity, and a pose column nothing reads would
    quadruple the committed file for no information.
    """
    n_batches = round(SCENE_SECONDS * SCENE_HZ) + 1
    batches: list[list[TrackObservation]] = []
    for index in range(n_batches):
        t = round(index / SCENE_HZ, SCENE_DECIMALS)
        batch: list[TrackObservation] = []
        for track_id in sorted(SCENE_SPANS):
            first_t, last_t = SCENE_SPANS[track_id]
            if not first_t <= t <= last_t:
                continue
            seat_x, seat_y = SCENE_SEATS[track_id]
            angle = 2.0 * math.pi * (t - first_t) / SWAY_PERIOD_S
            batch.append(
                TrackObservation(
                    track_id=track_id,
                    t=t,
                    cx=round(seat_x + SWAY_PX * math.sin(angle), SCENE_DECIMALS),
                    cy=round(seat_y + SWAY_PX * math.cos(angle), SCENE_DECIMALS),
                    w=BOX_W,
                    h=BOX_H,
                    keypoints=None,
                )
            )
        batches.append(batch)
    return batches


def test_committed_reid_fixture_is_deterministic(tmp_path: Path) -> None:
    """Regenerate it and compare bytes. A fixture that can drift proves nothing."""
    regenerated = write_observations_jsonl(
        tmp_path / "leave_and_return.jsonl", build_leave_and_return_scene()
    )
    assert regenerated.read_bytes() == SCENE_PATH.read_bytes()


def test_the_fixture_contains_a_departure_and_a_return_under_a_new_id() -> None:
    """The scenario itself, asserted from the committed file rather than the generator."""
    batches = list(SyntheticDetectionSource(SCENE_PATH).observations())
    seen: dict[int, list[float]] = {}
    for batch in batches:
        for obs in batch:
            seen.setdefault(obs.track_id, []).append(obs.t)
    assert sorted(seen) == [LEAVER_TRACK, STAYER_TRACK, RETURN_TRACK, STRANGER_TRACK]
    assert max(seen[LEAVER_TRACK]) == pytest.approx(LEAVER_LAST_T)
    assert min(seen[RETURN_TRACK]) == pytest.approx(RETURN_FIRST_T)
    # The gap is what makes this a re-ID question rather than a tracker question: it is
    # wider than the demo config's reap_after_s, so the track layer has already given up.
    gap = min(seen[RETURN_TRACK]) - max(seen[LEAVER_TRACK])
    assert gap > demo_config()["tracks"]["reap_after_s"]


# ------------------------------------------------------------------- fixtures
def load_runner() -> Any:
    """Import scripts/run_pipeline.py as a module. It is a script, not a package.

    Same helper and same reasoning as tests/test_dashboard.py: the module goes into
    sys.modules before it is executed so a frozen dataclass can resolve its own module.
    """
    name = "vividly_run_pipeline_script_reid"
    spec = importlib.util.spec_from_file_location(name, RUN_PIPELINE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()


def demo_config() -> dict[str, Any]:
    """The shipped re-ID demo config, as a dict.

    Loading the real configs/pipeline/synthetic_reid_demo.yaml rather than writing a bespoke
    one is deliberate: the tau that produces the proposal in these tests is the tau that
    ships, and a bespoke config would let the two drift.
    """
    return yaml.safe_load(REID_DEMO_CONFIG.read_text(encoding="utf-8"))


def demo_pipeline(tmp_path: Path, **overrides: Any) -> Any:
    cfg = demo_config()
    cfg["run_name"] = "test_reid"
    cfg["output"] = {"dir": str(tmp_path / "outputs")}
    cfg.update(overrides)
    return RUNNER.Pipeline(cfg, REPO)


@pytest.fixture
def ran_demo(tmp_path: Path) -> Iterator[Any]:
    """A completed re-ID run, still open so its stage and store can be inspected."""
    pipeline = demo_pipeline(tmp_path)
    try:
        pipeline.run()
        yield pipeline
    finally:
        pipeline.close()


# ================================================================ The main claim
def test_a_return_beyond_the_window_is_proposed_not_bound(ran_demo: Any) -> None:
    """One return, one question, zero bindings. This is the whole feature.

    Track 1 leaves at t = 20 and the book reaps it 8 seconds later, which is the moment its
    appearance enters the gallery. Track 3 is the same person walking back in at t = 40
    under a new tracker id. The system notices, and the only thing it does about it is ask.
    """
    pending = ran_demo.reid.pending()
    assert len(pending) == 1, [p.model_dump() for p in pending]
    proposal = pending[0]
    assert proposal.new_track == RETURN_TRACK
    assert proposal.candidate_track == LEAVER_TRACK
    assert proposal.status == "pending"
    assert proposal.ts == pytest.approx(RETURN_FIRST_T)
    # One identity had been enrolled when the question was asked. Stored beside the score
    # because a match against one person and a match against forty are not the same
    # evidence at the same number.
    assert proposal.gallery_size == 1
    # Confidence is the tau-anchored rescale, so a candidate that only just cleared tau
    # would read 0.5. This one is a clean synthetic match and reads well above it.
    assert 0.5 < proposal.confidence <= 1.0

    # Nothing was bound. The returning person is still track 3, with a blank slate.
    tracks = ran_demo.book.tracks()
    assert sorted(tracks) == [STAYER_TRACK, RETURN_TRACK, STRANGER_TRACK]
    assert LEAVER_TRACK not in tracks, "the departed track must stay reaped, not resurrected"
    assert tracks[RETURN_TRACK].reassociated_from is None
    assert tracks[RETURN_TRACK].mrn_binding is None
    assert tracks[RETURN_TRACK].is_staff is False
    # And no label exists yet: an unanswered question is not a negative.
    assert ran_demo.store.labels_as_rows() == []


def test_the_stranger_and_the_stayer_produce_no_question(ran_demo: Any) -> None:
    """A gallery that matches nobody asks nobody. The empty list has a precise meaning.

    Track 4 arrives after track 1 was enrolled and is probed against it; track 2 is
    re-probed when the gallery gains its first identity. Neither is that person, so neither
    produces a row. A stage that asked about every arrival would be a stage nobody used.
    """
    stats = ran_demo.reid.stats()
    assert stats.n_proposals == 1
    # Every live track was probed at least once, so the silence is a decision and not a
    # skipped code path.
    assert stats.n_probes >= 4
    proposed_for = {p.new_track for p in ran_demo.reid.pending()}
    assert STRANGER_TRACK not in proposed_for
    assert STAYER_TRACK not in proposed_for


def test_confirming_merges_the_identity_forward(tmp_path: Path) -> None:
    """A human's yes is what moves a chart binding, and it moves it forward, not backward.

    The run is stopped while track 1 is still live so a reviewer can tag it the way the
    dashboard does (staff tag, mock chart bind), then resumed. When the confirmation lands,
    those two human-supplied fields travel to track 3 and the link is recorded. The track
    id itself does not change: renaming a live track to a dead one's id would orphan every
    alert already written against the current id.
    """
    pipeline = demo_pipeline(tmp_path)
    try:
        pipeline.run(max_frames=20)  # t = 9.5 s: track 1 is still in the room
        assert LEAVER_TRACK in pipeline.book.tracks()
        pipeline.book.bind_mrn(LEAVER_TRACK, mock_mrn(LEAVER_TRACK))
        pipeline.book.set_staff(LEAVER_TRACK, True)
        pipeline.run()  # continues the same source rather than restarting it

        proposal = pipeline.reid.pending()[0]
        outcome = pipeline.reid.decide(
            proposal.proposal_id, "confirmed", decided_ts=SCENE_SECONDS, via="pytest:nurse"
        )
        assert outcome.merged is True
        assert outcome.record.status == "confirmed"

        returned = pipeline.book.tracks()[RETURN_TRACK]
        assert returned.track_id == RETURN_TRACK, "a merge links, it never renames"
        assert returned.reassociated_from == LEAVER_TRACK
        assert returned.mrn_binding == mock_mrn(LEAVER_TRACK)
        assert returned.is_staff is True

        # The answer is a label, with the score the model gave beside it. That pairing is
        # what a threshold sweep consumes (docs/CALIBRATION.md).
        labels = pipeline.store.labels_as_rows()
        assert len(labels) == 1
        assert labels[0]["label"] == 1
        assert labels[0]["decision"] == "confirmed"
        assert labels[0]["decided_via"] == "pytest:nurse"
        assert labels[0]["confidence"] == pytest.approx(proposal.confidence)
    finally:
        pipeline.close()


def test_a_rejected_pair_is_never_proposed_again(tmp_path: Path) -> None:
    """Reject once, and the same two people are not put in front of anyone a second time.

    Replaying the same session over the same store is the normal way anyone works, and a
    rejected pair coming back every run would train reviewers to ignore the queue. The
    second run rebuilds its already-asked set from the store and skips the pair; the store
    still holds exactly one proposal, still rejected.
    """
    first = demo_pipeline(tmp_path)
    try:
        first.run()
        proposal = first.reid.pending()[0]
        outcome = first.reid.decide(
            proposal.proposal_id, "rejected", decided_ts=SCENE_SECONDS, via="pytest:nurse"
        )
        assert outcome.merged is False
        assert "will not be proposed again" in outcome.note
    finally:
        first.close()

    second = demo_pipeline(tmp_path)
    try:
        second.run()
        assert second.n_proposals == 0
        assert second.reid.pending() == []
        # Skipped on purpose, not missed: the pair was scored again and declined.
        assert second.reid.stats().n_skipped_asked >= 1
        history = second.store.proposal_history()
        assert len(history) == 1
        assert history[0].status == "rejected"
        assert second.book.tracks()[RETURN_TRACK].reassociated_from is None
    finally:
        second.close()


def test_a_confirmed_pair_is_not_re_proposed_either(tmp_path: Path) -> None:
    """Same guard, other answer. A confirmed track has an identity and stops being asked."""
    first = demo_pipeline(tmp_path)
    try:
        first.run()
        proposal = first.reid.pending()[0]
        first.reid.decide(proposal.proposal_id, "confirmed", SCENE_SECONDS, "pytest:nurse")
    finally:
        first.close()

    second = demo_pipeline(tmp_path)
    try:
        second.run()
        assert second.n_proposals == 0
        assert len(second.store.proposal_history()) == 1
    finally:
        second.close()


# ================================================================== Run scope
# One store file is reachable by more than one run: its path is
# <output.dir>/<run_name>/alerts.sqlite, and two different clips can be replayed under one
# name. Every row therefore carries the run that raised it, and the two properties below are
# what that column buys. Without it, run B answers run A's questions and run A's confirmed
# rows silence run B's probing of a completely different person.

SCOPED_REAP_S = 5.0


def scoped_config(dim: int = 16) -> ReidConfig:
    """A minimal enabled stage over two tracks of one synthetic identity."""
    return ReidConfig(
        enabled=True,
        gallery=GalleryConfig(dim=dim),
        embeddings=EmbeddingsSpec(source="synthetic", dim=dim, identity_by_track={1: 0, 2: 0}),
    )


def test_a_foreign_runs_pending_row_is_history_only_and_cannot_be_answered_here(
    tmp_path: Path,
) -> None:
    """Another run's unanswered question is visible in the file and untouchable from here.

    Visible, because the file is one file and the history spans it. Untouchable, because
    answering it would bind an identity for a clip this run never saw, on a card whose two
    track ids mean two different people here. The store refuses it as an unknown proposal,
    which is exactly what it is to this run, and the dashboard turns that into a 404.
    """
    store = EventStore(tmp_path / "alerts.sqlite")
    book = TrackBook(TrackConfig(reap_after_s=SCOPED_REAP_S, baseline_window_s=1.0, ring_size=100))
    foreign = store.record_proposal(
        new_track=2, candidate_track=1, confidence=0.9, ts=5.0, gallery_size=1, run_id="run-b"
    )
    stage = build_reid_stage(
        scoped_config(), store=store, book=book, reap_after_s=SCOPED_REAP_S, run_id="run-a"
    )
    try:
        assert stage is not None
        assert stage.run_id == "run-a"
        # Not in this run's queue, and not answerable by it.
        assert stage.pending() == []
        with pytest.raises(UnknownProposalError, match="belongs to run 'run-b'"):
            stage.decide(foreign.proposal_id, "confirmed", 10.0, "pytest:nurse")
        with pytest.raises(UnknownProposalError):
            stage.decide(foreign.proposal_id, "rejected", 10.0, "pytest:nurse")
        # Still pending, still that run's question, still in the shared history.
        assert store.get_proposal(foreign.proposal_id).status == "pending"
        assert [p.proposal_id for p in store.proposal_history()] == [foreign.proposal_id]
        assert store.labels_as_rows() == []
    finally:
        stage.close()
        store.close()


def test_a_foreign_runs_confirmed_row_does_not_silence_this_runs_probe(tmp_path: Path) -> None:
    """The same stored rows, two run identities, opposite and correct outcomes.

    A confirmed row says "track 2 has been given an identity, stop asking about it". That is
    only true inside the run that wrote it: another run's track 2 is a different person, in a
    different clip, and suppressing its probe would silently hide a real return. So the same
    file is opened twice, once as the run that owns the row and once as a stranger to it, and
    `n_skipped_resolved` has to report which of the two happened rather than leaving a quiet
    session unexplained.
    """

    def seeded(name: str, run_id: str) -> tuple[EventStore, TrackBook, Any]:
        room = tmp_path / name
        room.mkdir()
        store = EventStore(room / "alerts.sqlite")
        row = store.record_proposal(
            new_track=2, candidate_track=1, confidence=0.9, ts=5.0, gallery_size=1, run_id="run-b"
        )
        store.decide_proposal(row.proposal_id, "confirmed", 6.0, "dashboard:other", run_id="run-b")
        book = TrackBook(
            TrackConfig(reap_after_s=SCOPED_REAP_S, baseline_window_s=1.0, ring_size=100)
        )
        stage = build_reid_stage(
            scoped_config(), store=store, book=book, reap_after_s=SCOPED_REAP_S, run_id=run_id
        )
        assert stage is not None
        return store, book, stage

    def replay(book: TrackBook, stage: Any) -> Any:
        book.ingest([obs(1, 0.0)])
        stage.step([obs(1, 0.0)], 0.0)
        book.reap(30.0)  # track 1 leaves and is enrolled
        book.ingest([obs(2, 40.0)])
        return stage.step([obs(2, 40.0)], 40.0)

    # A stranger to the row: track 2 is probed and the question is asked.
    store, book, stage = seeded("stranger", "run-a")
    try:
        result = replay(book, stage)
        assert result.enrolled == (1,)
        assert result.probed == (2,)
        assert [(p.new_track, p.candidate_track) for p in result.proposals] == [(2, 1)]
        assert all(p.run_id == "run-a" for p in result.proposals)
        assert stage.stats().n_skipped_resolved == 0, "nothing here was resolved for THIS run"
        assert [p.proposal_id for p in stage.pending()] == [result.proposals[0].proposal_id]
        assert len(store.proposal_history()) == 2  # the foreign answer plus this run's question
    finally:
        stage.close()
        store.close()

    # The run that owns the row: track 2 already has its identity, so it is not asked about
    # again, and the counter says so out loud instead of the session just going quiet.
    store, book, stage = seeded("owner", "run-b")
    try:
        result = replay(book, stage)
        assert result.enrolled == (1,)
        assert result.probed == ()
        assert result.proposals == ()
        assert stage.stats().n_skipped_resolved == 1
        assert stage.pending() == []
        assert len(store.proposal_history()) == 1
    finally:
        stage.close()
        store.close()


def test_the_database_refuses_a_second_claim_the_stage_could_not_know_about(
    tmp_path: Path,
) -> None:
    """The store's own index is the backstop, and the losing confirm still becomes a label.

    Two stages on one run and one file, which is what two processes over one output
    directory look like. The second stage read the store before the first stage's
    confirmation existed, so its in-memory exclusivity map is empty and it has no reason to
    refuse. The partial-unique index on (run_id, candidate_track) refuses instead, the stage
    turns that into the same recorded rejection, and nothing is merged.
    """
    store = EventStore(tmp_path / "alerts.sqlite")
    book = TrackBook(TrackConfig(reap_after_s=SCOPED_REAP_S, baseline_window_s=1.0, ring_size=100))
    first = build_reid_stage(
        scoped_config(), store=store, book=book, reap_after_s=SCOPED_REAP_S, run_id="run-a"
    )
    assert first is not None
    try:
        book.ingest([obs(1, 0.0)])
        first.step([obs(1, 0.0)], 0.0)
        book.reap(30.0)
        book.ingest([obs(2, 40.0)])
        mine = first.step([obs(2, 40.0)], 40.0).proposals[0]
        theirs = store.record_proposal(
            new_track=7, candidate_track=1, confidence=0.9, ts=41.0, gallery_size=1, run_id="run-a"
        )

        # The second stage resumes now, while both rows are still pending.
        second = build_reid_stage(
            scoped_config(), store=store, book=book, reap_after_s=SCOPED_REAP_S, run_id="run-a"
        )
        assert second is not None

        assert first.decide(mine.proposal_id, "confirmed", 50.0, "pytest:nurse").merged is True

        outcome = second.decide(theirs.proposal_id, "confirmed", 51.0, "pytest:other")
        assert outcome.merged is False
        assert outcome.record.status == "rejected"
        assert "confirmed this identity first" in outcome.note
        assert store.get_proposal(theirs.proposal_id).status == "rejected"
        assert [row["label"] for row in store.labels_as_rows()] == [1, 0]
    finally:
        first.close()
        store.close()


# ============================================================ The off switch
def alert_stream(db_path: Path) -> list[dict[str, Any]]:
    """Stored alerts, minus the one field that cannot be identical across two runs.

    `created_utc` is wall clock by definition. Everything else about an alert is derived
    from the fixture, so two runs of the same config must agree on all of it.
    """
    store = EventStore(db_path)
    try:
        return [
            {k: v for k, v in record.model_dump().items() if k != "created_utc"}
            for record in store.alerts(limit=1000)
        ]
    finally:
        store.close()


def rules_demo_config(tmp_path: Path, **overrides: Any) -> Path:
    """The shipped synthetic rules demo, redirected to a temp directory.

    That config is used rather than the re-ID demo because it actually fires an alert: an
    "identical alert stream" test over two empty streams would pass no matter what broke.
    """
    cfg = yaml.safe_load(SYNTHETIC_CONFIG.read_text(encoding="utf-8"))
    cfg["run_name"] = "test_offswitch"
    cfg["output"] = {"dir": str(tmp_path / "outputs")}
    cfg.update(overrides)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


#: A re-ID block over the rules fixture's three tracks. Enabled, so it really runs.
RULES_SCENE_REID = {
    "enabled": True,
    "embeddings": {
        "source": "synthetic",
        "dim": 512,
        "identity_by_track": {1: 0, 2: 1, 99: 2},
    },
}


def test_reid_off_leaves_the_alert_stream_byte_identical(tmp_path: Path) -> None:
    """No `reid:` block, `enabled: false`, and re-ID switched on all produce one alert stream.

    The first pair is the off switch: a config that never mentions re-ID and one that
    mentions it and disables it must be indistinguishable downstream. The third is the
    stronger claim and the one worth having: turning the stage on must not perturb a single
    alert either, because it is an observer of the track layer and never a participant in it.
    """
    baseline_dir = tmp_path / "baseline"
    assert RUNNER.main(["--config", str(rules_demo_config(baseline_dir)), "--headless"]) == 0
    baseline = alert_stream(baseline_dir / "outputs" / "test_offswitch" / "alerts.sqlite")
    assert baseline, "the baseline config must fire something or this test proves nothing"

    disabled_dir = tmp_path / "disabled"
    config = rules_demo_config(disabled_dir, reid={"enabled": False})
    assert RUNNER.main(["--config", str(config), "--headless"]) == 0
    assert alert_stream(disabled_dir / "outputs" / "test_offswitch" / "alerts.sqlite") == baseline

    enabled_dir = tmp_path / "enabled"
    config = rules_demo_config(enabled_dir, reid=RULES_SCENE_REID)
    assert RUNNER.main(["--config", str(config), "--headless"]) == 0
    assert alert_stream(enabled_dir / "outputs" / "test_offswitch" / "alerts.sqlite") == baseline


def test_a_config_with_no_reid_block_builds_no_stage(tmp_path: Path) -> None:
    """The default is None, and None is not a disabled object: there is nothing there."""
    config = rules_demo_config(tmp_path)
    pipeline = RUNNER.Pipeline(yaml.safe_load(config.read_text(encoding="utf-8")), REPO)
    try:
        assert pipeline.reid is None
        assert pipeline.reid_cfg is None
        pipeline.run(max_frames=5)
        assert pipeline.n_proposals == 0
    finally:
        pipeline.close()
    assert "re-ID" not in pipeline.summary(), "a run with no re-ID must not report re-ID"


def test_a_disabled_reid_block_builds_no_stage_either(tmp_path: Path) -> None:
    config = rules_demo_config(tmp_path, reid={"enabled": False})
    pipeline = RUNNER.Pipeline(yaml.safe_load(config.read_text(encoding="utf-8")), REPO)
    try:
        assert pipeline.reid_cfg is not None and pipeline.reid_cfg.enabled is False
        assert pipeline.reid is None
    finally:
        pipeline.close()


def test_nothing_disappears_means_nothing_is_enrolled_or_asked(tmp_path: Path) -> None:
    """The rules fixture has no departures, so an enabled stage asks nothing at all."""
    config = rules_demo_config(tmp_path, reid=RULES_SCENE_REID)
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    pipeline = RUNNER.Pipeline(cfg, REPO)
    try:
        pipeline.run()
        stats = pipeline.reid.stats()
        assert stats.n_enrolled == 0
        assert stats.n_identities == 0
        assert stats.n_proposals == 0
        assert stats.n_probes >= 3, "every arriving track is still probed against an empty gallery"
    finally:
        pipeline.close()


# ==================================================== Session scope (CQ-16)
def test_the_gallery_is_empty_after_shutdown(tmp_path: Path) -> None:
    """Session-scoped retention, enforced rather than promised.

    CQ-16 in docs/COUNSEL_QUESTIONS.md cites src/vividly/reid/gallery.py for the property
    that identity lives for one visit and is deleted at the end of it. This is the test
    that the pipeline actually performs the deletion, on the ordinary exit path, with no
    caller having to remember to ask for it.
    """
    pipeline = demo_pipeline(tmp_path)
    pipeline.run()
    assert pipeline.reid.gallery.size() > 0, "there must be something to delete"

    pipeline.close()

    assert pipeline.reid.gallery.size() == 0
    assert pipeline.reid.gallery.per_identity_counts() == {}
    assert pipeline.reid.stats().cleared is True
    # And the object refuses to be reused: a second visit must not land in a first one.
    with pytest.raises(RuntimeError, match="cleared at session end"):
        pipeline.reid.step([], 0.0, None)


def test_the_gallery_is_empty_after_close_with_a_worker_still_ticking(tmp_path: Path) -> None:
    """CQ-16 under contention: the deletion wins even when a tick is in flight against it.

    The shipped served path does not race any more (`serve` stops and joins the worker
    before the caller closes), but a sequenced shutdown is a property of one call site and
    the deletion has to hold whatever the interleaving is. So this closes the pipeline while a
    worker thread is running it, twenty times, walking the close across different points of
    the run, and demands the same post-condition every time: no embedding survives, and the
    worker returns rather than dying on the store that vanished under it.

    Twenty trials rather than one because a single interleaving proves nothing about a race;
    the assertions at the bottom check that the stress actually stressed, i.e. that at least
    one trial closed on a live worker and at least one had a populated gallery to delete.
    """
    trials = 20
    closed_on_a_live_worker = 0
    had_something_to_delete = 0
    for trial in range(trials):
        pipeline = demo_pipeline(tmp_path / f"trial{trial}")
        errors: list[BaseException] = []

        def tick(target: Any = pipeline, sink: list[BaseException] = errors) -> None:
            try:
                target.run()
            except BaseException as exc:
                sink.append(exc)

        worker = threading.Thread(target=tick, name="stress-pipeline", daemon=True)
        worker.start()
        if trial % 2:
            # Odd trials wait for an observable state rather than a duration: close only
            # once the gallery actually holds something. A sleep alone is a bet on machine
            # speed, and on a loaded CI runner every close landed before the first
            # enrolment, so the deletion was never asked to delete anything and the stress
            # passed vacuously. Waiting on the state makes "there was something to delete"
            # true on any machine while the worker keeps ticking underneath it.
            deadline = time.monotonic() + 10.0
            while pipeline.reid.gallery.size() == 0 and worker.is_alive():
                if time.monotonic() > deadline:  # pragma: no cover, only on a stuck run
                    raise AssertionError(f"trial {trial}: nothing enrolled within 10 s")
                time.sleep(0.001)
        else:
            # Even trials walk the close across the early interleavings, including the ones
            # that land before any tick has run at all.
            time.sleep(trial * 0.001)
        closed_on_a_live_worker += int(worker.is_alive())
        pipeline.close()

        worker.join(timeout=30.0)
        assert not worker.is_alive(), "the worker must return once the pipeline is closed"
        assert errors == [], f"trial {trial} raised {errors}"
        assert pipeline.reid.gallery.size() == 0
        assert pipeline.reid.gallery.per_identity_counts() == {}
        assert pipeline.reid.stats().cleared is True
        # The snapshot close() takes before emptying the gallery, which is the only place a
        # test can see how much there was to delete.
        had_something_to_delete += int(pipeline._reid_stats.n_identities > 0)

    assert closed_on_a_live_worker > 0, "no trial actually raced a running worker"
    assert had_something_to_delete > 0, "no trial had an enrolled identity to delete"


def test_a_close_landing_mid_tick_cannot_be_out_enrolled_by_that_tick(tmp_path: Path) -> None:
    """The interleaving the stress loop can only hit by luck, arranged on purpose.

    A tick is parked exactly where it is about to run the re-ID stage, with a populated
    gallery behind it, and the session is ended while it sits there. When it is released it
    resumes into a stage that no longer exists: the deletion has already happened, and the
    tick has to be unable to put anything back. Two things are asserted about that. The
    gallery is still empty, which is CQ-16 holding against the worst-ordered tick rather than
    against a tidy one; and the worker returns instead of dying, because a shutdown that
    lands mid-tick is a shutdown and not a crash.

    The park is a wrapper around the private `_reid_step`, which is the only way to stop a
    tick at a chosen instant; nothing about the pipeline's behaviour is otherwise changed.
    """
    pipeline = demo_pipeline(tmp_path)
    parked = threading.Event()
    release = threading.Event()
    original = pipeline._reid_step

    def hooked(observations: Any, t: float, image: Any) -> None:
        if not parked.is_set() and pipeline.reid.gallery.size() > 0:
            parked.set()
            release.wait(timeout=30.0)
        original(observations, t, image)

    pipeline._reid_step = hooked
    errors: list[BaseException] = []

    def tick() -> None:
        try:
            pipeline.run()
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=tick, name="parked-pipeline", daemon=True)
    worker.start()
    try:
        assert parked.wait(timeout=30.0), "the tick never parked with a populated gallery"
        assert pipeline.reid.gallery.size() > 0, "there must be something to delete"
        pipeline.close()
    finally:
        release.set()
    worker.join(timeout=30.0)

    assert not worker.is_alive()
    assert errors == [], f"the worker died on shutdown: {errors}"
    assert pipeline.reid.gallery.size() == 0
    assert pipeline.reid.gallery.per_identity_counts() == {}


def test_close_is_idempotent_and_says_so_in_the_summary(tmp_path: Path) -> None:
    """Shutdown paths run twice. The second close must be a no-op, not an error."""
    pipeline = demo_pipeline(tmp_path)
    pipeline.run()
    pipeline.close()
    pipeline.close()

    summary = pipeline.summary()
    assert "CLEARED at session end" in summary
    assert "0 embeddings retained" in summary
    assert "CQ-16" in summary
    assert "NONE auto-bound" in summary
    assert "PROVISIONAL" in summary


def test_the_headless_run_records_reid_in_the_artifacts(tmp_path: Path) -> None:
    """The latency summary row carries the re-ID counters, and says they were deleted."""
    cfg = demo_config()
    cfg["run_name"] = "test_reid_cli"
    cfg["output"] = {"dir": str(tmp_path / "outputs")}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    assert RUNNER.main(["--config", str(path), "--headless"]) == 0
    out_dir = tmp_path / "outputs" / "test_reid_cli"
    rows = [
        json.loads(line)
        for line in (out_dir / "latency.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = rows[-1]
    assert summary["summary"] is True
    reid = summary["reid"]
    assert reid["source"] == "synthetic"
    assert reid["n_proposals"] == 1
    # The counts the session held, with the deletion flag set: reporting zero identities
    # here would read as "nobody was enrolled" rather than "everybody has been forgotten".
    assert reid["n_identities"] == 1
    assert reid["cleared"] is True
    assert "PROVISIONAL" in reid["threshold_status"]

    store = EventStore(out_dir / "alerts.sqlite")
    try:
        assert len(store.pending_proposals()) == 1
        assert store.labels_as_rows() == []
    finally:
        store.close()


def test_a_run_with_no_reid_writes_no_reid_key(tmp_path: Path) -> None:
    """Absent, not zeroed. A row of zeros reads as "it ran and found nothing"."""
    config = rules_demo_config(tmp_path)
    assert RUNNER.main(["--config", str(config), "--headless"]) == 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "outputs" / "test_offswitch" / "latency.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert "reid" not in rows[-1]


# =========================================== The track layer, left alone
def obs(track_id: int, t: float, cx: float = 0.0, cy: float = 0.0) -> TrackObservation:
    return TrackObservation(track_id=track_id, t=t, cx=cx, cy=cy, w=10.0, h=20.0)


def test_a_book_with_no_watcher_behaves_exactly_as_before() -> None:
    """The default TrackBook is the TrackBook that shipped: reap returns ids and tells nobody."""
    book = TrackBook(TrackConfig(reap_after_s=5.0, baseline_window_s=1.0, ring_size=100))
    book.ingest([obs(1, 0.0), obs(2, 0.0)])
    book.ingest([obs(2, 10.0)])
    assert sorted(book.tracks()) == [2]
    assert book.reap(30.0) == [2]
    assert book.tracks() == {}


def test_association_inside_the_reap_window_is_still_automatic() -> None:
    """The window this whole feature sits outside of, asserted from the inside.

    A person absent for less than reap_after_s keeps the same TrackState, and with it their
    baseline, their staff tag and their chart binding. No watcher fires, no gallery is
    consulted and no human is asked, exactly as before re-ID existed.
    """
    reaped: list[int] = []
    book = TrackBook(TrackConfig(reap_after_s=5.0, baseline_window_s=1.0, ring_size=100))
    book.set_reap_watcher(lambda state: reaped.append(state.track_id))

    book.ingest([obs(7, 0.0)])
    book.bind_mrn(7, "MOCK-7")
    book.set_staff(7, True)
    book.reap(4.0)  # inside the window: still there
    book.ingest([obs(7, 4.5)])  # and the same object catches the observation

    state = book.tracks()[7]
    assert state.mrn_binding == "MOCK-7"
    assert state.is_staff is True
    assert state.n_observations == 2
    assert state.reassociated_from is None
    assert reaped == [], "nothing left, so nothing may be announced"


def test_the_watcher_fires_once_per_reaped_track_with_its_identity() -> None:
    reaped: list[TrackIdentity] = []
    book = TrackBook(TrackConfig(reap_after_s=5.0, baseline_window_s=1.0, ring_size=100))
    book.set_reap_watcher(lambda state: reaped.append(state.identity()))
    book.ingest([obs(1, 0.0), obs(2, 0.0)])
    book.bind_mrn(1, "MOCK-1")
    book.reap(10.0)
    book.reap(20.0)  # already gone: announced once, not once per call

    assert [identity.track_id for identity in reaped] == [1, 2]
    assert reaped[0].mrn_binding == "MOCK-1"
    assert reaped[1].mrn_binding is None


def test_the_watcher_is_announced_with_the_book_lock_released() -> None:
    """A watcher may call back into the book. If the lock were held this would deadlock or
    lie; instead it observes a book that has already dropped the track."""
    seen: list[list[int]] = []
    book = TrackBook(TrackConfig(reap_after_s=5.0, baseline_window_s=1.0, ring_size=100))
    book.set_reap_watcher(lambda state: seen.append(sorted(book.tracks())))
    book.ingest([obs(1, 0.0), obs(2, 0.0)])
    book.ingest([obs(2, 10.0)])
    assert seen == [[2]]


def test_setting_a_watcher_to_none_stops_announcements() -> None:
    reaped: list[int] = []
    book = TrackBook(TrackConfig(reap_after_s=5.0, baseline_window_s=1.0, ring_size=100))
    book.set_reap_watcher(lambda state: reaped.append(state.track_id))
    book.set_reap_watcher(None)
    book.ingest([obs(1, 0.0)])
    book.reap(10.0)
    assert reaped == []


def test_a_watcher_must_be_callable() -> None:
    book = TrackBook(TrackConfig())
    with pytest.raises(TypeError, match="callable"):
        book.set_reap_watcher("not a function")  # type: ignore[arg-type]


def test_merge_identity_links_forward_and_never_renames() -> None:
    book = TrackBook(TrackConfig(reap_after_s=5.0, baseline_window_s=1.0, ring_size=100))
    book.ingest([obs(9, 0.0)])
    book.merge_identity(9, TrackIdentity(track_id=4, is_staff=True, mrn_binding="MOCK-4"))
    state = book.tracks()[9]
    assert state.track_id == 9
    assert state.reassociated_from == 4
    assert state.mrn_binding == "MOCK-4"
    assert state.is_staff is True


def test_merge_identity_never_overwrites_a_binding_a_human_made_on_the_live_track() -> None:
    """A binding on the live track is newer evidence than the reaped track's. It stands."""
    book = TrackBook(TrackConfig(reap_after_s=5.0, baseline_window_s=1.0, ring_size=100))
    book.ingest([obs(9, 0.0)])
    book.bind_mrn(9, "MOCK-9")
    book.merge_identity(9, TrackIdentity(track_id=4, mrn_binding="MOCK-4"))
    assert book.tracks()[9].mrn_binding == "MOCK-9"


def test_merge_identity_never_clears_a_staff_tag() -> None:
    """is_staff is OR-ed, never assigned: a match to an untagged track is not a retraction."""
    book = TrackBook(TrackConfig(reap_after_s=5.0, baseline_window_s=1.0, ring_size=100))
    book.ingest([obs(9, 0.0)])
    book.set_staff(9, True)
    book.merge_identity(9, TrackIdentity(track_id=4, is_staff=False))
    assert book.tracks()[9].is_staff is True


def test_merge_identity_refuses_a_self_merge_and_an_unknown_track() -> None:
    book = TrackBook(TrackConfig(reap_after_s=5.0, baseline_window_s=1.0, ring_size=100))
    book.ingest([obs(9, 0.0)])
    with pytest.raises(ValueError, match="itself"):
        book.merge_identity(9, TrackIdentity(track_id=9))
    with pytest.raises(KeyError):
        book.merge_identity(11, TrackIdentity(track_id=4))
    with pytest.raises(TypeError, match="TrackIdentity"):
        book.merge_identity(9, 4)  # type: ignore[arg-type]


def test_a_confirmation_on_a_track_that_already_left_is_still_a_label(tmp_path: Path) -> None:
    """The label is written before the merge is attempted, and it survives the failure."""
    pipeline = demo_pipeline(tmp_path)
    try:
        pipeline.run()
        proposal = pipeline.reid.pending()[0]
        # The room empties: the returning track is reaped long after the fixture ended.
        pipeline.book.reap(SCENE_SECONDS + 1000.0)
        assert RETURN_TRACK not in pipeline.book.tracks()

        outcome = pipeline.reid.decide(
            proposal.proposal_id, "confirmed", SCENE_SECONDS, "pytest:nurse"
        )
        assert outcome.merged is False
        assert "already been reaped" in outcome.note
        assert outcome.record.status == "confirmed"
        assert pipeline.store.labels_as_rows()[0]["label"] == 1
    finally:
        pipeline.close()


# ================================================ Confidence, crops, config
def test_confidence_is_anchored_on_tau_and_reaches_one_at_a_perfect_match() -> None:
    assert confidence_from_similarity(0.75, 0.75, 0.5) == pytest.approx(0.5)
    assert confidence_from_similarity(1.0, 0.75, 0.5) == pytest.approx(1.0)
    assert confidence_from_similarity(0.875, 0.75, 0.5) == pytest.approx(0.75)
    # Monotone, and clamped into the range the store will accept rather than raising there.
    assert confidence_from_similarity(0.5, 0.75, 0.5) == pytest.approx(0.0)
    assert confidence_from_similarity(-1.0, 0.75, 0.5) == 0.0
    # A different anchor moves only where tau lands, not where a perfect match lands.
    assert confidence_from_similarity(0.75, 0.75, 0.8) == pytest.approx(0.8)
    assert confidence_from_similarity(1.0, 0.75, 0.8) == pytest.approx(1.0)


def test_confidence_degenerates_rather_than_dividing_by_zero_at_tau_one() -> None:
    assert confidence_from_similarity(1.0, 1.0, 0.5) == 1.0


def test_confidence_refuses_a_non_finite_similarity() -> None:
    with pytest.raises(ValueError, match="finite"):
        confidence_from_similarity(float("nan"), 0.75, 0.5)


def test_crop_geometry_is_the_box_clamped_to_the_frame() -> None:
    frame = np.zeros((100, 200, 3), np.uint8)
    crop = crop_from_frame(
        frame, TrackObservation(track_id=1, t=0.0, cx=50.0, cy=50.0, w=20.0, h=40.0)
    )
    assert crop is not None and crop.shape[:2] == (40, 20)
    # Half out of frame on the left: clamped, not shifted.
    clipped = crop_from_frame(
        frame, TrackObservation(track_id=1, t=0.0, cx=5.0, cy=50.0, w=20.0, h=40.0)
    )
    assert clipped is not None and clipped.shape[1] == 15


def test_a_box_entirely_outside_the_frame_is_no_crop_at_all() -> None:
    """Someone walking off the edge of the camera frame is normal, not an error."""
    frame = np.zeros((100, 200, 3), np.uint8)
    outside = TrackObservation(track_id=1, t=0.0, cx=-500.0, cy=50.0, w=20.0, h=40.0)
    assert crop_from_frame(frame, outside) is None


class StubEmbedder:
    """An AppearanceEmbedder that encodes crop height, so rows are traceable to crops.

    Deliberately not a mock of onnxruntime: the crop source's whole job is geometry and
    batching, and that has to be provable on a machine with no inference stack at all.
    """

    embedding_dim = 4

    def __init__(self) -> None:
        self.calls: list[int] = []

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        self.calls.append(len(crops))
        rows = np.zeros((len(crops), self.embedding_dim), np.float32)
        for i in range(len(crops)):
            rows[i, i % self.embedding_dim] = 1.0
        return rows


def test_the_crop_source_pairs_every_row_with_the_track_it_came_from() -> None:
    frame = np.zeros((100, 200, 3), np.uint8)
    source = CropEmbeddingSource(StubEmbedder())
    rows = source.embed(
        [
            TrackObservation(track_id=5, t=0.0, cx=50.0, cy=50.0, w=20.0, h=40.0),
            TrackObservation(track_id=6, t=0.0, cx=150.0, cy=50.0, w=20.0, h=40.0),
        ],
        frame,
    )
    assert sorted(rows) == [5, 6]
    assert rows[5].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert rows[6].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert source.dim == 4


def test_the_crop_source_is_silent_with_no_image_and_with_no_boxes_in_frame() -> None:
    embedder = StubEmbedder()
    source = CropEmbeddingSource(embedder)
    off_frame = TrackObservation(track_id=5, t=0.0, cx=-500.0, cy=0.0, w=20.0, h=40.0)
    assert source.embed([off_frame], np.zeros((10, 10, 3), np.uint8)) == {}
    assert source.embed([off_frame], None) == {}
    assert embedder.calls == [], "no crops means the embedder is never called"


def test_the_crop_source_refuses_something_that_is_not_an_embedder() -> None:
    with pytest.raises(ReidConfigError, match="AppearanceEmbedder"):
        CropEmbeddingSource(object())


def test_the_synthetic_source_is_deterministic_and_identity_keyed() -> None:
    spec = EmbeddingsSpec(source="synthetic", dim=16, identity_by_track={1: 0, 2: 0, 3: 1})
    rows = SyntheticEmbeddingSource(spec).embed([obs(1, 4.0), obs(2, 4.0), obs(3, 4.0)], None)
    again = SyntheticEmbeddingSource(spec).embed([obs(1, 4.0)], None)
    assert np.array_equal(rows[1], again[1]), "same config in, same bytes out"
    # Two tracks of the same identity look alike; a third identity does not.
    same = float(rows[1] @ rows[2])
    other = float(rows[1] @ rows[3])
    assert same > other
    assert all(float(np.linalg.norm(row)) == pytest.approx(1.0, abs=1e-5) for row in rows.values())


def test_the_synthetic_source_counts_a_track_it_was_given_no_identity_for() -> None:
    """It does not invent one. The map is the fixture's answer key, not a suggestion."""
    source = SyntheticEmbeddingSource(
        EmbeddingsSpec(source="synthetic", dim=16, identity_by_track={1: 0})
    )
    assert sorted(source.embed([obs(1, 0.0), obs(2, 0.0)], None)) == [1]
    assert source.n_unmapped == 1


# ------------------------------------------------------------------- config
def test_shipped_reid_yaml_matches_the_dataclass_defaults() -> None:
    """The file and the code cannot drift: one of them would then be a lie about a run."""
    assert load_reid_config(REID_CONFIG) == ReidConfig()


def test_shipped_reid_yaml_is_off_and_says_its_numbers_are_provisional() -> None:
    text = REID_CONFIG.read_text(encoding="utf-8")
    assert "PROVISIONAL" in text and "CALIBRATION.md" in text
    assert load_reid_config(REID_CONFIG).enabled is False


def test_the_demo_config_points_at_the_committed_fixture_and_enables_reid() -> None:
    cfg = demo_config()
    assert (REPO / cfg["detections_from"]).resolve() == SCENE_PATH.resolve()
    assert cfg["headless"] is True
    assert "detector" not in cfg and "pose" not in cfg and "tracker" not in cfg
    reid = ReidConfig.from_mapping(cfg["reid"])
    assert reid.enabled is True
    assert reid.embeddings.source == "synthetic"
    # The map covers every track in the fixture; an unmapped one would silently contribute
    # nothing and the demo would quietly stop demonstrating anything.
    assert set(reid.embeddings.identity_by_track) == set(SCENE_SPANS)
    # tau, top_k and the confidence anchor are the shipped defaults, not demo-tuned.
    assert reid.gallery.tau == ReidConfig().gallery.tau
    assert reid.top_k == ReidConfig().top_k
    assert reid.confidence_at_tau == ReidConfig().confidence_at_tau


def test_an_enabled_stage_with_nothing_to_embed_is_refused() -> None:
    """Silently doing nothing is worse than being off, so it is an error instead."""
    with pytest.raises(ReidConfigError, match=r"embeddings\.source is 'none'"):
        ReidConfig(enabled=True)


def test_a_gallery_and_a_source_of_different_widths_are_refused() -> None:
    with pytest.raises(ReidConfigError, match="gallery"):
        ReidConfig(
            enabled=True,
            gallery=GalleryConfig(dim=128),
            embeddings=EmbeddingsSpec(source="synthetic", dim=64),
        )


def test_unknown_config_keys_raise_rather_than_being_ignored() -> None:
    with pytest.raises(ReidConfigError, match="unknown ReidConfig keys"):
        ReidConfig.from_mapping({"tau": 0.9})  # tau lives under `gallery`
    with pytest.raises(ReidConfigError, match="unknown embeddings keys"):
        EmbeddingsSpec.from_mapping({"sources": "synthetic"})


def test_an_unknown_embedding_source_is_refused() -> None:
    with pytest.raises(ReidConfigError, match="unknown embeddings source"):
        EmbeddingsSpec(source="magic")
    assert EMBEDDING_SOURCES == ("none", "synthetic", "embedder")


def test_the_runner_refuses_two_sources_of_truth_for_a_reid_threshold(tmp_path: Path) -> None:
    cfg = demo_config()
    cfg["reid_config"] = "configs/pipeline/reid.yaml"
    cfg["output"] = {"dir": str(tmp_path / "outputs")}
    with pytest.raises(RUNNER.ConfigError, match="both"):
        RUNNER.Pipeline(cfg, REPO)


def test_a_bad_reid_block_is_an_exit_three_config_error(tmp_path: Path) -> None:
    cfg = demo_config()
    cfg["run_name"] = "broken"
    cfg["output"] = {"dir": str(tmp_path / "outputs")}
    cfg["reid"] = {"enabled": True, "top_k": 0, "embeddings": {"source": "synthetic"}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    assert RUNNER.main(["--config", str(path)]) == 3


def test_a_reid_config_pointer_is_loaded_from_the_shared_file(tmp_path: Path) -> None:
    cfg = demo_config()
    cfg.pop("reid")
    cfg["reid_config"] = "configs/pipeline/reid.yaml"
    cfg["run_name"] = "pointer"
    cfg["output"] = {"dir": str(tmp_path / "outputs")}
    pipeline = RUNNER.Pipeline(cfg, REPO)
    try:
        assert pipeline.reid_cfg == ReidConfig()
        assert pipeline.reid is None, "the shared file ships disabled"
    finally:
        pipeline.close()


# ------------------------------------------------------- the window, directly
def test_a_candidate_inside_the_tracker_window_is_not_proposed(tmp_path: Path) -> None:
    """The gallery does not know where the window ends; the stage does, and it declines.

    In the pipeline this branch is structurally unreachable, because a track is only
    enrolled once it has been absent longer than reap_after_s. It is asserted here anyway:
    the contract in vividly.reid.gallery says the caller compares elapsed_s against its own
    tracker configuration, and a guarantee nobody checks is a guarantee that quietly stops
    being true.
    """
    store = EventStore(tmp_path / "alerts.sqlite")
    book = TrackBook(TrackConfig(reap_after_s=5.0, baseline_window_s=1.0, ring_size=100))
    config = ReidConfig(
        enabled=True,
        gallery=GalleryConfig(dim=16),
        embeddings=EmbeddingsSpec(source="synthetic", dim=16, identity_by_track={1: 0, 2: 0}),
        min_gap_s=1000.0,  # far wider than the gap the fixture below produces
    )
    stage = build_reid_stage(config, store=store, book=book, reap_after_s=5.0)
    try:
        book.ingest([obs(1, 0.0)])
        stage.step([obs(1, 0.0)], 0.0)
        book.reap(10.0)  # track 1 leaves and is enrolled
        book.ingest([obs(2, 20.0)])
        result = stage.step([obs(2, 20.0)], 20.0)
        assert result.enrolled == (1,)
        assert result.proposals == ()
        assert stage.stats().n_skipped_inside_window == 1
    finally:
        stage.close()
        store.close()


def test_a_track_born_before_an_enrollment_is_re_probed_after_it(tmp_path: Path) -> None:
    """The dead zone between the tracker's buffer and the book's reap window, closed.

    A person who returns 10 seconds after leaving gets a new tracker id while the old track
    is still sitting in the book waiting to be reaped. Probing only at birth would miss
    every one of those returns, which is the most common shape of the problem.
    """
    store = EventStore(tmp_path / "alerts.sqlite")
    book = TrackBook(TrackConfig(reap_after_s=30.0, baseline_window_s=1.0, ring_size=100))
    config = ReidConfig(
        enabled=True,
        gallery=GalleryConfig(dim=32),
        embeddings=EmbeddingsSpec(source="synthetic", dim=32, identity_by_track={1: 0, 2: 0, 3: 1}),
    )
    stage = build_reid_stage(config, store=store, book=book, reap_after_s=30.0)
    try:
        book.ingest([obs(1, 0.0)])
        stage.step([obs(1, 0.0)], 0.0)
        # Track 1 stops being seen at t = 0 but stays in the book until t > 30. Track 2 is
        # born at t = 10, well before that: an empty gallery, so nothing is asked.
        book.ingest([obs(2, 10.0)])
        assert stage.step([obs(2, 10.0)], 10.0).proposals == ()
        # Now the book gives up on track 1 and the gallery gains its first identity.
        book.reap(40.0)
        book.ingest([obs(2, 40.0)])
        result = stage.step([obs(2, 40.0)], 40.0)
        assert result.enrolled == (1,)
        assert [(p.new_track, p.candidate_track) for p in result.proposals] == [(2, 1)]

        # A probe costs nothing while the gallery stands still, so track 2 is not re-scored
        # frame after frame: it is probed again only when there is new evidence.
        book.ingest([obs(2, 41.0)])
        assert stage.step([obs(2, 41.0)], 41.0).probed == ()

        # New evidence arrives: a third person leaves and is enrolled. Track 2 is re-probed
        # and still matches track 1, and this is where the already-asked set earns its
        # keep, because that pair is a question somebody has already been given.
        book.ingest([obs(2, 42.0), obs(3, 42.0)])
        stage.step([obs(2, 42.0), obs(3, 42.0)], 42.0)
        book.ingest([obs(2, 60.0)])
        book.reap(80.0)
        book.ingest([obs(2, 80.0)])
        result = stage.step([obs(2, 80.0)], 80.0)
        assert result.enrolled == (3,)
        assert result.probed == (2,)
        assert result.proposals == ()
        assert stage.stats().n_skipped_asked == 1
    finally:
        stage.close()
        store.close()


def test_the_number_of_questions_one_track_can_raise_is_capped(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "alerts.sqlite")
    book = TrackBook(TrackConfig(reap_after_s=5.0, baseline_window_s=1.0, ring_size=100))
    # Three people leave, all of them the same synthetic identity, so the newcomer matches
    # every one of them. The cap is what stops three questions becoming a queue.
    config = ReidConfig(
        enabled=True,
        max_proposals_per_track=1,
        gallery=GalleryConfig(dim=32),
        embeddings=EmbeddingsSpec(
            source="synthetic", dim=32, identity_by_track={1: 0, 2: 0, 3: 0, 9: 0}
        ),
    )
    stage = build_reid_stage(config, store=store, book=book, reap_after_s=5.0)
    try:
        departures = [obs(1, 0.0), obs(2, 0.0), obs(3, 0.0)]
        book.ingest(departures)
        stage.step(departures, 0.0)
        book.reap(30.0)
        book.ingest([obs(9, 40.0)])
        result = stage.step([obs(9, 40.0)], 40.0)
        assert len(result.proposals) == 1
        assert stage.stats().n_skipped_capped >= 1
    finally:
        stage.close()
        store.close()


def test_the_stage_never_proposes_a_track_against_itself(tmp_path: Path) -> None:
    """A tracker that reuses an id is not a person who returned, and the store agrees."""
    store = EventStore(tmp_path / "alerts.sqlite")
    book = TrackBook(TrackConfig(reap_after_s=5.0, baseline_window_s=1.0, ring_size=100))
    config = ReidConfig(
        enabled=True,
        gallery=GalleryConfig(dim=32),
        embeddings=EmbeddingsSpec(source="synthetic", dim=32, identity_by_track={1: 0}),
    )
    stage = build_reid_stage(config, store=store, book=book, reap_after_s=5.0)
    try:
        book.ingest([obs(1, 0.0)])
        stage.step([obs(1, 0.0)], 0.0)
        book.reap(30.0)
        book.ingest([obs(1, 40.0)])  # the same id handed out again
        result = stage.step([obs(1, 40.0)], 40.0)
        assert result.proposals == ()
        assert store.proposal_history() == []
    finally:
        stage.close()
        store.close()


def test_the_stage_unregisters_itself_from_the_book_on_close(tmp_path: Path) -> None:
    """A reap arriving during shutdown must not queue an enrollment into an emptied gallery."""
    store = EventStore(tmp_path / "alerts.sqlite")
    book = TrackBook(TrackConfig(reap_after_s=5.0, baseline_window_s=1.0, ring_size=100))
    gallery = SessionGallery(GalleryConfig(dim=32))
    config = ReidConfig(
        enabled=True,
        gallery=GalleryConfig(dim=32),
        embeddings=EmbeddingsSpec(source="synthetic", dim=32, identity_by_track={1: 0}),
    )
    stage = build_reid_stage(config, store=store, book=book, reap_after_s=5.0)
    assert stage is not None and gallery.size() == 0
    try:
        book.ingest([obs(1, 0.0)])
        stage.step([obs(1, 0.0)], 0.0)
        stage.close()
        book.reap(100.0)  # must not raise and must not enrol
        assert stage.gallery.size() == 0
    finally:
        store.close()


def test_the_runner_hands_the_stage_and_run_dir_to_the_dashboard_state(tmp_path: Path) -> None:
    """PipelineState must carry the stage object and this run's directory.

    Without `reid=` the page lists the questions out of the store but Confirm takes the
    store-only branch and never reaches TrackBook.merge_identity; without `run_dir=` the
    crops route has no directory to serve. Both were missed once (the pipeline verification of
    2026-08-08); this pin keeps the loop closed.
    """
    pipeline = demo_pipeline(tmp_path)
    try:
        assert pipeline.state.reid is pipeline.reid
        assert pipeline.reid is not None, "demo config must enable the re-ID stage"
        assert pipeline.state.run_dir == pipeline.out_dir
    finally:
        pipeline.close()


# ==================================================== The runner's run identity
def test_the_shipped_runner_attaches_the_stage_with_a_real_run_id(tmp_path: Path) -> None:
    """The pin that exists because nothing pinned it: the runner must scope its own run.

    Run scoping is worth nothing if the one caller that shares store files across runs never
    supplies a run id, and `run_id=None` is not a mild default: the store's two identity
    indexes are over (run_id, ...), a NULL makes every row distinct in a unique index, and so
    an unscoped run is precisely the case the database cannot refuse a double confirm in.
    The stage's in-memory maps would be the only guard left. This asserts the shipped runner
    is never that case, and that the id reaches the rows rather than just the object.
    """
    pipeline = demo_pipeline(tmp_path)
    try:
        assert pipeline.reid is not None, "demo config must enable the re-ID stage"
        assert pipeline.run_id, "every run must have an identity"
        assert pipeline.reid.run_id == pipeline.run_id
        pipeline.run()

        (proposal,) = pipeline.reid.pending()
        assert proposal.run_id == pipeline.run_id
        # The queue the dashboard drains is this run's queue, and only this run's.
        assert [
            p.proposal_id for p in pipeline.store.pending_proposals(run_id=pipeline.run_id)
        ] == [proposal.proposal_id]
        assert pipeline.store.pending_proposals(run_id="some-other-run") == []
    finally:
        pipeline.close()


def test_the_run_id_is_a_digest_that_survives_a_re_run_and_separates_two_clips(
    tmp_path: Path,
) -> None:
    """Stable across re-runs of one config, different for anything that makes it a new run.

    Stability is the whole reason it is a digest rather than a uuid: a replay of the same
    clip has to be the same run, or it re-asks every question a person already answered. The
    other three cases are the collision the column exists for, all of which land in one store
    file when the run name is shared.
    """
    cfg = demo_config()
    cfg["run_name"] = "identity"
    cfg["output"] = {"dir": str(tmp_path / "outputs")}
    source = REPO / cfg["detections_from"]

    same = RUNNER.derive_run_id("identity", source, cfg)
    assert RUNNER.derive_run_id("identity", source, cfg) == same, "a re-run is the same run"
    assert same.startswith("identity-"), "readable prefix, digest for the identity"

    # A different clip under the same name shares the store file and must not share the run.
    # The path below is deliberately absent from the tree: only the path text enters the
    # digest, so the file never has to exist.
    other_clip = RUNNER.derive_run_id("identity", REPO / "tests/fixtures/nowhere.jsonl", cfg)
    assert other_clip != same

    # A changed threshold asks different questions, so it is a different run.
    tweaked = dict(cfg)
    tweaked["tracks"] = dict(cfg["tracks"], reap_after_s=float(cfg["tracks"]["reap_after_s"]) + 1)
    assert RUNNER.derive_run_id("identity", source, tweaked) != same

    # And a different name is a different run whatever else matches.
    assert RUNNER.derive_run_id("other", source, cfg) != same

    # Config key order is not part of the identity: a reordered YAML is the same run.
    reordered = dict(reversed(list(cfg.items())))
    assert RUNNER.derive_run_id("identity", source, reordered) == same


def test_two_pipelines_over_one_config_agree_on_their_run_id(tmp_path: Path) -> None:
    """The property a resumed run depends on, asserted through Pipeline rather than the
    helper: build it twice over one config and the second run owns the first run's rows."""
    first = demo_pipeline(tmp_path)
    try:
        first_id = first.run_id
        first.run()
    finally:
        first.close()

    second = demo_pipeline(tmp_path)
    try:
        assert second.run_id == first_id
        second.run()
        # Same run: the question from the first pass is this run's own pending row, so it is
        # Resumed and skipped. The distinction matters, because a run that failed to
        # recognise its own rows would score the pair again and be saved only by the
        # deterministic proposal id colliding, which shows up as a duplicate rather than as
        # a skip. Both leave the store with one row; only one of them is the mechanism
        # working.
        assert second.n_proposals == 0
        assert second.reid.stats().n_skipped_asked >= 1
        assert second.reid.stats().n_duplicate_proposals == 0
        assert len(second.store.proposal_history(run_id=second.run_id)) == 1
    finally:
        second.close()


# ============================================================ Served shutdown
def test_serve_stops_and_joins_the_worker_before_the_caller_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`serve` owns stopping the pipeline thread; `main`'s finally owns closing it.

    That order is the fix: closing empties the session gallery (CQ-16) and shuts the store,
    and a daemon worker still ticking through both of those is a deletion racing an
    enrollment. uvicorn is replaced by a stand-in that returns once the run has enrolled
    somebody, which is what a reviewer closing the page looks like from here; no socket is
    bound and no server is started.

    Two things are asserted on the way out of serve: the worker thread is gone (joined, not
    abandoned), and nothing has been closed or cleared yet, because that is the caller's job
    and doing it here would put it back inside the race.
    """
    pipeline = demo_pipeline(tmp_path)

    class FakeUvicorn:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def run(self, app: Any, **kwargs: Any) -> None:
            self.calls.append(kwargs)
            deadline = time.monotonic() + 30.0
            while pipeline.reid.gallery.size() == 0 and time.monotonic() < deadline:
                time.sleep(0.001)

    fake = FakeUvicorn()
    monkeypatch.setattr(RUNNER, "uvicorn", fake)
    try:
        RUNNER.serve(pipeline)

        assert fake.calls, "serve must actually start the server"
        assert pipeline.stop_requested is True
        assert [t for t in threading.enumerate() if t.name == "vividly-pipeline"] == []
        # Stopping is not closing: the gallery is still live and the store still open, so
        # the caller's close() is what performs the deletion, with no worker left to race.
        assert pipeline.reid.gallery.size() > 0
        assert pipeline.reid.stats().cleared is False
        assert pipeline.store.count() >= 0
    finally:
        pipeline.close()

    assert pipeline.reid.gallery.size() == 0
    assert "CLEARED at session end" in pipeline.summary()
