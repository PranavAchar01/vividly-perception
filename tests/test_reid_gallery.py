"""Tests for the session-scoped gallery and the tau calibration tooling.

Every expected number here is derived by hand in a comment. No random draws in the scoring
tests, no golden files, no recorded blobs: if behaviour changes, a test fails with the
arithmetic sitting next to it.

The scoring fixture, shared by most of the gallery tests and lifted deliberately from
tests/test_reid_metrics.py so the gallery and the metrics can be compared without a second
set of numbers to reason about. Every row is a unit vector whose entries are 0, 0.6, 0.8 or
1, so every cosine below is exact on paper and exact to well under pytest.approx's tolerance
in float32-stored, float64-accumulated arithmetic.

    identity 10:  g0 = (1.0, 0.0, 0.0)     probe p = (0.6, 0.8, 0.0)
                  g1 = (0.8, 0.6, 0.0)
    identity 20:  g2 = (0.0, 1.0, 0.0)     cosines: g0 0.60, g1 0.96, g2 0.80, g3 0.00
    identity 30:  g3 = (0.0, 0.0, 1.0)     per identity (MAX): 10 -> 0.96
                                                               20 -> 0.80
                                                               30 -> 0.00

So a tau of 0.5 leaves candidates {10: 0.96, 20: 0.80}, a tau of 0.9 leaves {10: 0.96}, and
a tau of 0.99 leaves nothing at all, which is the answer "this is a new person".

The eviction fixture, hand-computed once and reused by three tests. Four unit rows in 3-D:

    a = (1.0, 0.0, 0.0)   b = (0.8, 0.6, 0.0)   c = (0.6, 0.8, 0.0)   d = (0.0, 0.0, 1.0)

    mean(a, b, c, d) = (0.6, 0.35, 0.25), norm sqrt(0.545) = 0.738241,
    unit centroid     = (0.812743, 0.474100, 0.338643)

    cosine to that centroid:  a 0.81274   b 0.93465   c 0.86693   d 0.33864

d is the outlier by a wide margin, so with max_per_identity = 3 the kept set is {a, b, c}
no matter which of the four arrived last. FIFO would instead have dropped a, the oldest and
the most representative of the three that survive; that difference is the whole point of the
policy and `test_eviction_is_not_fifo` pins it.
"""

from __future__ import annotations

import ast
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import yaml

from vividly.reid import metrics as M
from vividly.reid.calibrate import (
    CALIBRATION_SCHEMA_VERSION,
    PROVISIONAL_NOTE,
    CalibrationConfigError,
    ScoreArrays,
    calibrate_arrays_report,
    calibrate_scenario,
    calibrate_tau,
    config_from_mapping,
    render_yaml,
    scenario_score_arrays,
    score_arrays_from_npz,
    summarise,
    write_calibration,
)
from vividly.reid.gallery import (
    GalleryConfig,
    Proposal,
    ProposeStats,
    SessionGallery,
    keep_representative,
)
from vividly.reid.synth import ScenarioConfig, build_scenario, sequence_name

REPO_ROOT = Path(__file__).resolve().parent.parent

DIM = 3
G0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
G1 = np.array([0.8, 0.6, 0.0], dtype=np.float32)
G2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
G3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
PROBE = np.array([0.6, 0.8, 0.0], dtype=np.float32)

A = np.array([1.0, 0.0, 0.0], dtype=np.float32)
B = np.array([0.8, 0.6, 0.0], dtype=np.float32)
C = np.array([0.6, 0.8, 0.0], dtype=np.float32)
D = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _config(**overrides) -> GalleryConfig:
    """The scoring fixture's config: 3-D, tau 0.5, plenty of room per identity."""
    base = {"dim": DIM, "tau": 0.5, "max_per_identity": 8}
    base.update(overrides)
    return GalleryConfig(**base)


def _populated(**overrides) -> SessionGallery:
    """The three-identity scoring fixture, enrolled at three different times."""
    gallery = SessionGallery(_config(**overrides))
    gallery.enroll(10, np.stack([G0, G1]), t=100.0)
    gallery.enroll(20, G2[None, :], t=200.0)
    gallery.enroll(30, G3[None, :], t=50.0)
    return gallery


# ---------------------------------------------------------------------------
# enrollment and proposal ordering
# ---------------------------------------------------------------------------
def test_proposals_are_ordered_by_confidence_and_carry_the_max_over_rows():
    # 10 -> max(0.60, 0.96) = 0.96, 20 -> 0.80, 30 -> 0.00 (dropped, below tau 0.5).
    proposals = _populated().propose(PROBE, top_k=3, t=300.0)
    assert [p.identity_key for p in proposals] == [10, 20]
    assert proposals[0].confidence == pytest.approx(0.96)
    assert proposals[1].confidence == pytest.approx(0.80)


def test_proposal_carries_support_count_and_time_since_that_identity_was_last_seen():
    # identity 10 kept two rows and was last enrolled at t=100; the probe is at t=300.
    proposals = _populated().propose(PROBE, top_k=3, t=300.0)
    by_key = {p.identity_key: p for p in proposals}
    assert by_key[10].n_support == 2
    assert by_key[20].n_support == 1
    assert by_key[10].elapsed_s == pytest.approx(200.0)
    assert by_key[20].elapsed_s == pytest.approx(100.0)


def test_propose_without_a_time_falls_back_to_the_newest_enrollment_clock():
    # The newest enrollment is identity 20 at t=200, so that is "now" when no t is given.
    proposals = _populated().propose(PROBE, top_k=3)
    by_key = {p.identity_key: p for p in proposals}
    assert by_key[20].elapsed_s == pytest.approx(0.0)
    assert by_key[10].elapsed_s == pytest.approx(100.0)


def test_a_proposal_is_json_serialisable_for_the_dashboard():
    payload = json.loads(json.dumps(_populated().propose(PROBE, t=300.0)[0].to_dict()))
    assert payload["identity_key"] == 10
    assert payload["confidence"] == pytest.approx(0.96)


def test_enrolling_the_same_identity_twice_accumulates_rows_and_moves_last_seen_forward():
    gallery = SessionGallery(_config())
    gallery.enroll(7, G0[None, :], t=10.0)
    gallery.enroll(7, G1[None, :], t=40.0)
    assert gallery.per_identity_counts() == {7: 2}
    assert gallery.propose(PROBE, t=100.0)[0].elapsed_s == pytest.approx(60.0)


def test_an_out_of_order_enrollment_cannot_pull_last_seen_backwards():
    gallery = SessionGallery(_config())
    gallery.enroll(7, G0[None, :], t=40.0)
    gallery.enroll(7, G1[None, :], t=10.0)  # late batch, older timestamp
    assert gallery.propose(PROBE, t=100.0)[0].elapsed_s == pytest.approx(60.0)


def test_enroll_copies_so_a_caller_reusing_its_buffer_cannot_rewrite_the_gallery():
    # The embedder path reuses output buffers; a gallery aliasing one would silently mutate
    # an enrolled identity mid-session.
    scratch = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    gallery = SessionGallery(_config())
    gallery.enroll(7, scratch, t=0.0)
    scratch[0] = [0.0, 1.0, 0.0]
    assert gallery.propose(G0, t=1.0)[0].confidence == pytest.approx(1.0)


def test_stored_rows_stay_float32():
    # The AppearanceEmbedder contract, and half the memory of the float64 the scores use.
    gallery = SessionGallery(_config())
    gallery.enroll(7, np.stack([G0, G1]).astype(np.float64), t=0.0)
    assert gallery._rows[7].dtype == np.float32


# ---------------------------------------------------------------------------
# tau gating: below tau is a new identity, not a weak proposal
# ---------------------------------------------------------------------------
def test_a_probe_below_tau_returns_no_proposals_at_all():
    # Top score is 0.96, so a tau of 0.99 accepts nothing: this probe is a new identity and
    # no human is asked anything.
    assert _populated(tau=0.99).propose(PROBE, top_k=3, t=300.0) == []


def test_tau_gates_each_candidate_independently():
    # tau 0.9 keeps identity 10 (0.96) and drops identity 20 (0.80).
    proposals = _populated(tau=0.9).propose(PROBE, top_k=3, t=300.0)
    assert [p.identity_key for p in proposals] == [10]


def test_acceptance_is_greater_or_equal_so_a_score_exactly_at_tau_survives():
    # metrics.dir_far_curve and false_merge_rate both compare >= tau; the gallery must agree
    # or a tau read off a calibration curve would mean something different in production.
    gallery = SessionGallery(_config(tau=1.0))
    gallery.enroll(1, G0[None, :], t=0.0)
    assert [p.identity_key for p in gallery.propose(G0, t=1.0)] == [1]


def test_an_empty_gallery_proposes_nothing():
    assert SessionGallery(_config()).propose(PROBE, t=0.0) == []


def test_ties_resolve_to_the_smaller_identity_key():
    # Two identities holding the same row score identically, bit for bit. metrics.top1_match
    # resolves an exact tie to the smaller label; so does this.
    gallery = SessionGallery(_config())
    gallery.enroll(40, G2[None, :], t=1.0)
    gallery.enroll(20, G2[None, :], t=1.0)
    assert [p.identity_key for p in gallery.propose(PROBE, top_k=2, t=2.0)] == [20, 40]


def test_top_k_truncates_the_survivors_not_the_ordering():
    proposals = _populated(tau=-1.0).propose(PROBE, top_k=2, t=300.0)
    assert [p.identity_key for p in proposals] == [10, 20]


def test_top_k_must_be_at_least_one():
    with pytest.raises(ValueError, match="top_k"):
        _populated().propose(PROBE, top_k=0)


# ---------------------------------------------------------------------------
# statistics-driven refinement
# ---------------------------------------------------------------------------
def test_eviction_keeps_the_representative_set():
    # See the module docstring: cosines to the unit centroid of {a, b, c, d} are
    # a 0.81274, b 0.93465, c 0.86693, d 0.33864, so d is evicted at capacity 3.
    gallery = SessionGallery(_config(max_per_identity=3))
    gallery.enroll(1, np.stack([A, B, C, D]), t=0.0)
    kept = gallery._rows[1]
    assert kept.shape == (3, DIM)
    assert np.array_equal(kept, np.stack([A, B, C]))


def test_an_outlier_newcomer_evicts_itself_rather_than_a_good_stored_view():
    gallery = SessionGallery(_config(max_per_identity=3))
    gallery.enroll(1, np.stack([A, B, C]), t=0.0)
    gallery.enroll(1, D[None, :], t=1.0)  # same four rows, different arrival order
    assert np.array_equal(gallery._rows[1], np.stack([A, B, C]))


def test_eviction_is_not_fifo():
    # a arrives first. FIFO would drop it when the fourth row lands. Keep-representative
    # drops d instead, because d is the row a stranger's probe would match.
    gallery = SessionGallery(_config(max_per_identity=3))
    for row, t in ((A, 0.0), (D, 1.0), (B, 2.0), (C, 3.0)):
        gallery.enroll(1, row[None, :], t=t)
    kept = gallery._rows[1]
    assert np.array_equal(kept, np.stack([A, B, C]))
    assert not np.array_equal(kept, np.stack([D, B, C]))  # what FIFO would have kept


def test_keep_representative_is_a_no_op_below_capacity_and_returns_the_same_array():
    rows = np.stack([A, B])
    assert keep_representative(rows, 8) is rows


def test_keep_representative_trims_to_exactly_max_keep():
    rows = np.stack([A, B, C, D])
    assert keep_representative(rows, 1).shape == (1, DIM)
    assert keep_representative(rows, 2).shape == (2, DIM)


def test_keep_representative_refuses_a_capacity_below_one():
    with pytest.raises(ValueError, match="max_keep"):
        keep_representative(np.stack([A, B]), 0)


def test_refinement_bounds_memory_no_matter_how_long_the_visit_lasts():
    gallery = SessionGallery(_config(max_per_identity=4))
    for i in range(50):
        gallery.enroll(1, np.stack([A, B]), t=float(i))
    assert gallery.per_identity_counts() == {1: 4}
    assert gallery.size() == 4


def test_n_support_reports_the_kept_count_not_the_number_ever_enrolled():
    gallery = SessionGallery(_config(max_per_identity=2))
    gallery.enroll(1, np.stack([A, B, C, D]), t=0.0)
    assert gallery.propose(PROBE, t=1.0)[0].n_support == 2


# ---------------------------------------------------------------------------
# session scope: CQ-16 cites src/vividly/reid/gallery.py for exactly this behaviour, and
# these are the tests behind it
# ---------------------------------------------------------------------------
def test_clear_empties_everything_and_is_idempotent():
    gallery = _populated()
    gallery.propose(PROBE, t=300.0)
    assert gallery.size() == 4

    gallery.clear()
    assert gallery.size() == 0
    assert gallery.per_identity_counts() == {}
    assert gallery.propose(PROBE, t=300.0) == []
    stats = gallery.stats()
    # The propose above is the only call since the clear.
    assert stats.n_calls == 1

    gallery.clear()
    gallery.clear()
    assert gallery.size() == 0
    assert gallery.per_identity_counts() == {}
    assert gallery.stats().n_calls == 0


def test_a_cleared_gallery_behaves_like_a_fresh_one():
    gallery = _populated()
    gallery.clear()
    gallery.enroll(10, G0[None, :], t=1000.0)
    assert gallery.per_identity_counts() == {10: 1}
    assert gallery.propose(G0, t=1000.0)[0].elapsed_s == pytest.approx(0.0)


def test_the_gallery_has_no_persistence_surface():
    # CQ-16's conservative posture is "no cross-session persistence path exists to widen".
    # A method that could write one would have to be added deliberately, and this is the
    # test that makes that addition visible in a diff.
    banned = {"save", "load", "dump", "to_json", "to_disk", "persist", "write", "flush"}
    assert banned.isdisjoint(dir(SessionGallery))


def test_the_gallery_never_binds_anything():
    # The hard semantic: beyond the tracker's re-association window the system proposes and
    # a human decides. There is no verb here that could do otherwise.
    banned = {"bind", "accept", "confirm", "merge", "auto_merge", "associate", "reassociate"}
    assert banned.isdisjoint(dir(SessionGallery))


# ---------------------------------------------------------------------------
# thread safety
# ---------------------------------------------------------------------------
def test_concurrent_enroll_and_propose_lose_nothing():
    # Each worker owns one identity, so the final per-identity counts are deterministic even
    # though the interleaving is not: 6 calls of 2 rows against a capacity of 5 must end at
    # exactly 5 for every identity, and no update may be lost to a read-modify-write race.
    n_workers, n_rounds, dim = 8, 6, 8
    gallery = SessionGallery(GalleryConfig(dim=dim, tau=0.0, max_per_identity=5))

    def rows_for(worker: int, round_index: int) -> np.ndarray:
        rng = np.random.default_rng(1000 * worker + round_index)
        raw = rng.normal(size=(2, dim))
        return (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)

    def work(worker: int) -> int:
        seen = 0
        for round_index in range(n_rounds):
            gallery.enroll(worker, rows_for(worker, round_index), t=float(round_index))
            seen += len(gallery.propose(rows_for(worker, round_index)[0], top_k=3, t=99.0))
        return seen

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(work, range(n_workers)))

    assert all(count >= 0 for count in results)
    assert gallery.per_identity_counts() == {worker: 5 for worker in range(n_workers)}
    assert gallery.size() == 5 * n_workers
    assert gallery.stats().n_calls == n_workers * n_rounds


def test_clear_races_with_readers_without_tearing():
    dim = 8
    gallery = SessionGallery(GalleryConfig(dim=dim, tau=0.0, max_per_identity=4))
    unit = np.zeros((1, dim), dtype=np.float32)
    unit[0, 0] = 1.0

    def churn(worker: int) -> None:
        for i in range(20):
            gallery.enroll(worker, unit, t=float(i))
            gallery.per_identity_counts()
            gallery.size()
            if i % 7 == 0:
                gallery.clear()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(churn, range(4)))
    # Whatever the interleaving, the invariant holds: no identity exceeds capacity.
    assert all(count <= 4 for count in gallery.per_identity_counts().values())


# ---------------------------------------------------------------------------
# latency stats
# ---------------------------------------------------------------------------
def test_latency_stats_populate_with_the_gallery_size_beside_the_time():
    gallery = _populated()
    for _ in range(3):
        gallery.propose(PROBE, t=300.0)
    stats = gallery.stats()
    assert isinstance(stats, ProposeStats)
    assert stats.n_calls == 3
    assert stats.window == 3
    assert stats.last_s >= 0.0
    assert stats.max_s >= stats.mean_s >= 0.0
    assert stats.last_gallery_size == gallery.size() == 4
    assert stats.max_gallery_size == 4
    assert stats.last_ms == pytest.approx(stats.last_s * 1000.0)
    assert json.loads(json.dumps(stats.to_dict()))["n_calls"] == 3


def test_rejected_probes_are_timed_too():
    gallery = _populated(tau=0.99)
    assert gallery.propose(PROBE, t=300.0) == []
    assert gallery.stats().n_calls == 1


def test_the_latency_window_rolls_while_the_call_count_does_not():
    gallery = _populated(latency_window=4)
    for _ in range(10):
        gallery.propose(PROBE, t=300.0)
    stats = gallery.stats()
    assert stats.n_calls == 10
    assert stats.window == 4


def test_stats_before_any_call_are_empty_rather_than_zero():
    stats = SessionGallery(_config()).stats()
    assert stats.n_calls == 0
    assert stats.window == 0
    assert np.isnan(stats.last_s)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def test_enroll_refuses_a_one_dimensional_array():
    with pytest.raises(ValueError, match="2-D"):
        SessionGallery(_config()).enroll(1, G0, t=0.0)


def test_enroll_refuses_an_empty_batch():
    with pytest.raises(ValueError, match="empty"):
        SessionGallery(_config()).enroll(1, np.zeros((0, DIM), np.float32), t=0.0)


def test_enroll_refuses_rows_that_are_not_l2_normalised():
    with pytest.raises(ValueError, match="L2-normalised"):
        SessionGallery(_config()).enroll(1, (G0 * 4.0)[None, :], t=0.0)


def test_enroll_refuses_a_non_finite_row():
    bad = np.array([[np.nan, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN or inf"):
        SessionGallery(_config()).enroll(1, bad, t=0.0)


def test_enroll_refuses_a_width_the_gallery_was_not_built_for():
    with pytest.raises(ValueError, match="does not match"):
        SessionGallery(_config()).enroll(1, np.eye(4, dtype=np.float32), t=0.0)


def test_enroll_refuses_a_non_finite_time():
    with pytest.raises(ValueError, match="finite source time"):
        SessionGallery(_config()).enroll(1, G0[None, :], t=float("nan"))


def test_enroll_refuses_a_boolean_identity_key():
    with pytest.raises(TypeError, match="bool"):
        SessionGallery(_config()).enroll(True, G0[None, :], t=0.0)


def test_propose_refuses_a_matrix():
    with pytest.raises(ValueError, match="one"):
        SessionGallery(_config()).propose(np.stack([G0, G1]))


def test_gallery_config_validates_its_own_numbers():
    with pytest.raises(ValueError, match="tau"):
        GalleryConfig(tau=1.5)
    with pytest.raises(ValueError, match="max_per_identity"):
        GalleryConfig(max_per_identity=0)
    with pytest.raises(ValueError, match="dim"):
        GalleryConfig(dim=0)
    with pytest.raises(ValueError, match="latency_window"):
        GalleryConfig(latency_window=0)
    with pytest.raises(ValueError, match="unit_norm_atol"):
        GalleryConfig(unit_norm_atol=0.0)


def test_gallery_config_from_mapping_refuses_unknown_keys():
    assert GalleryConfig.from_mapping({"dim": 8, "tau": 0.6}) == GalleryConfig(dim=8, tau=0.6)
    with pytest.raises(ValueError, match="unknown GalleryConfig keys"):
        GalleryConfig.from_mapping({"taus": 0.6})


def test_proposals_are_frozen():
    proposal = _populated().propose(PROBE, t=300.0)[0]
    with pytest.raises(AttributeError):
        proposal.confidence = 0.1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# import discipline: this object runs inside the live pipeline
# ---------------------------------------------------------------------------
def test_gallery_imports_numpy_and_stdlib_and_nothing_else():
    tree = ast.parse((REPO_ROOT / "src" / "vividly" / "reid" / "gallery.py").read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert imported <= {
        "__future__",
        "collections",
        "dataclasses",
        "math",
        "numpy",
        "threading",
        "time",
        "typing",
    }, f"unexpected import in gallery.py: {sorted(imported)}"


# ---------------------------------------------------------------------------
# calibration: the tool that replaces the placeholder tau
# ---------------------------------------------------------------------------
def _hand_built_arrays() -> ScoreArrays:
    """The tests/test_reid_metrics.py fixture, as a bundle.

    Return-probe top-1 scores {0.96, 0.80, 0.80} of which {0.96, 0.80} are correct, and
    new-identity top-1 scores {0.80, 0.60}. At a FAR budget of 0.5 the sweep's first
    eligible threshold is tau = 0.80 (one of two new probes accepted, FAR 0.5 exactly),
    where DIR = 2/3 and the false-merge rate is 1/3 (probe p2 binds to identity 10).
    """
    return ScoreArrays(
        gallery=np.array(
            [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        gallery_ids=np.array([10, 10, 20, 30], dtype=np.int64),
        probes=np.array([[0.6, 0.8, 0.0], [0.0, 0.8, 0.6], [0.8, 0.0, 0.6]], dtype=np.float64),
        probe_ids=np.array([10, 20, 30], dtype=np.int64),
        new_probes=np.array([[0.6, 0.0, 0.8], [0.6, -0.8, 0.0]], dtype=np.float64),
    )


def test_calibrate_tau_agrees_with_metrics_operating_point_on_a_hand_built_array():
    arrays = _hand_built_arrays()
    result = calibrate_tau(arrays, 0.5)

    curve = M.dir_far_curve(
        arrays.gallery, arrays.gallery_ids, arrays.probes, arrays.probe_ids, arrays.new_probes
    )
    expected = M.operating_point(curve, 0.5)
    assert result.tau == expected.tau
    assert result.dir == expected.dir
    assert result.far == expected.far
    assert result.far_target == expected.far_target

    # And the hand arithmetic the fixture docstring states.
    assert result.tau == pytest.approx(0.80)
    assert result.dir == pytest.approx(2.0 / 3.0)
    assert result.far == pytest.approx(0.5)
    assert result.false_merge_rate == pytest.approx(1.0 / 3.0)
    assert (result.n_probes, result.n_new_probes) == (3, 2)


def test_the_reported_false_merge_rate_is_the_one_that_tau_produces():
    arrays = _hand_built_arrays()
    result = calibrate_tau(arrays, 0.5)
    assert result.false_merge_rate == pytest.approx(
        M.false_merge_rate(
            arrays.gallery, arrays.gallery_ids, arrays.probes, arrays.probe_ids, result.tau
        )
    )


def test_a_tighter_budget_buys_a_higher_tau_and_a_lower_dir():
    arrays = _hand_built_arrays()
    loose = calibrate_tau(arrays, 0.5)
    tight = calibrate_tau(arrays, 0.0)
    assert tight.tau > loose.tau
    assert tight.dir <= loose.dir
    assert tight.far == 0.0


def _tie_at_one_arrays() -> ScoreArrays:
    """A fixture where a new probe ties the top score at exactly cosine 1.0.

    gallery is one row, the return probe is that same row (score 1.0, correct), and the new
    probe is that row too (score 1.0). The only threshold with FAR 0 is the curve's appended
    sentinel np.nextafter(1.0, inf), which is > 1.0. Selecting it is the bug under test.
    """
    return ScoreArrays(
        gallery=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        gallery_ids=np.array([10], dtype=np.int64),
        probes=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        probe_ids=np.array([10], dtype=np.int64),
        new_probes=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
    )


def test_an_unreachable_far_budget_clips_tau_into_the_gallerys_own_bounds():
    # Without the clip the operating point is np.nextafter(1.0, inf) > 1.0 and
    # GalleryConfig refuses it. The clip pins tau to 1.0, flags the budget unreachable, and
    # re-reads DIR/FAR at tau = 1.0 so the evidence stays consistent with the number.
    result = calibrate_tau(_tie_at_one_arrays(), 0.0)
    assert result.tau == 1.0
    assert result.far_target_unreachable is True
    GalleryConfig(dim=3, tau=float(result.tau))  # the whole point: it must be usable
    # At tau = 1.0 the stranger scoring exactly 1.0 is still accepted, so the achieved FAR is
    # the real 1.0, never the sentinel's misleading 0.0.
    assert result.far == pytest.approx(1.0)
    assert result.dir == pytest.approx(1.0)


def test_a_generated_unreachable_tau_validates_against_gallery_config_before_writing(tmp_path):
    # The emitted tau must always be one GalleryConfig can hold, even in the unreachable case.
    report = calibrate_arrays_report(_tie_at_one_arrays(), 0.0)
    document = yaml.safe_load(render_yaml(report))
    tau = document["reid"]["gallery"]["tau"]
    assert tau <= 1.0
    GalleryConfig(dim=3, tau=float(tau))
    assert document["calibration"]["recommended"]["far_target_unreachable"] is True


def test_calibrate_tau_refuses_a_far_target_outside_the_unit_interval():
    with pytest.raises(CalibrationConfigError, match="far_target"):
        calibrate_tau(_hand_built_arrays(), 1.5)


def test_calibrate_tau_refuses_a_source_it_does_not_understand():
    with pytest.raises(CalibrationConfigError, match="ScoreArrays"):
        calibrate_tau("a scenario, honestly", 0.5)  # type: ignore[arg-type]


def test_score_arrays_round_trip_through_npz(tmp_path):
    arrays = _hand_built_arrays()
    path = tmp_path / "scores.npz"
    np.savez(
        path,
        gallery=arrays.gallery,
        gallery_ids=arrays.gallery_ids,
        probes=arrays.probes,
        probe_ids=arrays.probe_ids,
        new_probes=arrays.new_probes,
    )
    loaded = score_arrays_from_npz(path)
    assert loaded.n_probes == 3
    assert loaded.n_new_probes == 2
    assert loaded.n_gallery_rows == 4
    assert calibrate_tau(loaded, 0.5).tau == calibrate_tau(arrays, 0.5).tau


def test_score_arrays_npz_missing_an_array_raises_rather_than_defaulting(tmp_path):
    path = tmp_path / "partial.npz"
    np.savez(path, gallery=np.eye(3), gallery_ids=np.array([1, 2, 3]))
    with pytest.raises(CalibrationConfigError, match="missing array"):
        score_arrays_from_npz(path)


def test_score_arrays_npz_must_exist(tmp_path):
    with pytest.raises(CalibrationConfigError, match="not found"):
        score_arrays_from_npz(tmp_path / "nope.npz")


def test_a_bare_npy_is_a_config_error_not_a_type_error(tmp_path):
    # A .npy loads as a single anonymous ndarray, which used to raise a raw TypeError on
    # the `with np.load(...)` context manager and escape the exit-3 contract. It is now the
    # malformed-score-file config error the CLI documents.
    npy = tmp_path / "one.npy"
    np.save(npy, np.eye(3))
    with pytest.raises(CalibrationConfigError, match="got a single"):
        score_arrays_from_npz(npy)


def test_a_truncated_npz_is_a_config_error_not_a_bad_zip_file(tmp_path):
    # A truncated archive used to raise zipfile.BadZipFile, which the CLI does not catch,
    # so it exited 1 with a stack trace instead of the documented exit 3.
    npz = tmp_path / "trunc.npz"
    np.savez(
        npz,
        gallery=np.eye(3),
        gallery_ids=np.array([1, 2, 3]),
        probes=np.eye(3),
        probe_ids=np.array([1, 2, 3]),
        new_probes=np.eye(3),
    )
    raw = npz.read_bytes()
    npz.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(CalibrationConfigError, match="score archive"):
        score_arrays_from_npz(npz)


# ---------------------------------------------------------------------------
# calibration over the synthetic scenario
# ---------------------------------------------------------------------------
def _small_scenario_config(**overrides) -> ScenarioConfig:
    base = {
        "n_identities": 4,
        "n_distractor_identities": 1,
        "n_new_identities": 4,
        "dim": 8,
        "gap_tiers_s": (30.0, 300.0),
        "enrollment_s": 10.0,
        "return_s": 5.0,
        "distractor_s": 5.0,
        "sample_hz": 1.0,
        "fps": 1.0,
        "seed": 7,
    }
    base.update(overrides)
    return ScenarioConfig(**base)


def test_calibrate_scenario_reports_every_tier_and_a_pooled_recommendation():
    cfg = config_from_mapping(
        {"scenario": dict(_small_scenario_config().__dict__), "far_target": 0.25}
    )
    report = calibrate_scenario(cfg)
    assert [tier.label for tier in report.tiers] == [sequence_name(30.0), sequence_name(300.0)]
    assert report.recommended.label == "pooled"
    assert report.recommended.n_probes == sum(tier.n_probes for tier in report.tiers)
    assert report.recommended.far <= 0.25
    assert report.tau == report.recommended.tau
    assert report.scenario_digest is not None


def test_a_single_tier_scenario_pools_to_exactly_that_tier():
    # The pooled sweep is benchmark.pooled_dir_far_curve over decisions; the per-tier sweep
    # is metrics.dir_far_curve over matrices. On one tier they are the same curve, which is
    # what keeps the recommended tau on the same curve the benchmark reports.
    cfg = config_from_mapping(
        {
            "scenario": dict(_small_scenario_config(gap_tiers_s=(30.0,)).__dict__),
            "far_target": 0.25,
        }
    )
    report = calibrate_scenario(cfg)
    assert len(report.tiers) == 1
    assert report.recommended.tau == report.tiers[0].tau
    assert report.recommended.dir == pytest.approx(report.tiers[0].dir)
    assert report.recommended.false_merge_rate == pytest.approx(report.tiers[0].false_merge_rate)


def test_calibrate_tau_on_a_single_tier_scenario_needs_no_gap_argument():
    scenario = build_scenario(_small_scenario_config(gap_tiers_s=(30.0,)))
    assert calibrate_tau(scenario, 0.25).gap_s == pytest.approx(30.0)


def test_calibrate_tau_on_a_multi_tier_scenario_demands_a_tier():
    scenario = build_scenario(_small_scenario_config())
    with pytest.raises(CalibrationConfigError, match="gap tiers"):
        calibrate_tau(scenario, 0.25)
    assert calibrate_tau(scenario, 0.25, gap_s=300.0).label == sequence_name(300.0)


def test_scenario_score_arrays_defaults_to_single_frame_probes_the_gallery_scores():
    # SessionGallery.propose scores one per-frame (dim,) row, so calibration must too.
    # The default `single` mode makes one probe per return-window embedding row; `pooled`
    # makes one tracklet-mean probe, the benchmark construction the gallery never runs.
    scenario = build_scenario(_small_scenario_config(gap_tiers_s=(30.0,)))
    single = scenario_score_arrays(scenario, 30.0)
    pooled = scenario_score_arrays(scenario, 30.0, probe="pooled")
    # 4 enrolled identities, return_s=5 at sample_hz=1 -> 5 rows each: 4 pooled, 20 single.
    assert pooled.n_probes == len(scenario.returns_for_gap(30.0))
    assert single.n_probes == sum(t.n_samples for t in scenario.returns_for_gap(30.0))
    assert single.n_probes > pooled.n_probes
    assert single.n_new_probes == sum(t.n_samples for t in scenario.new_probes_for_gap(30.0))


def test_scenario_score_arrays_shares_the_benchmark_gallery_in_both_modes():
    # The gallery is the benchmark's construction (every enrollment row plus gap distractors)
    # regardless of probe granularity, so distractors change the gallery but never the probes.
    scenario = build_scenario(_small_scenario_config())
    for mode in ("single", "pooled"):
        with_d = scenario_score_arrays(scenario, 30.0, probe=mode)
        without_d = scenario_score_arrays(scenario, 30.0, include_distractors=False, probe=mode)
        assert with_d.n_gallery_rows > without_d.n_gallery_rows
        assert with_d.n_probes == without_d.n_probes


def test_scenario_score_arrays_rejects_an_unknown_probe_mode():
    scenario = build_scenario(_small_scenario_config(gap_tiers_s=(30.0,)))
    with pytest.raises(CalibrationConfigError, match="probe"):
        scenario_score_arrays(scenario, 30.0, probe="mean")  # type: ignore[arg-type]


def test_config_from_mapping_refuses_unknown_keys():
    with pytest.raises(CalibrationConfigError, match="unknown key"):
        config_from_mapping({"far_targets": [0.01]})


def test_config_from_mapping_refuses_a_far_target_outside_the_unit_interval():
    with pytest.raises(CalibrationConfigError, match="far_target"):
        config_from_mapping({"far_target": 2.0})


# ---------------------------------------------------------------------------
# the generated artifact
# ---------------------------------------------------------------------------
def test_the_generated_yaml_carries_tau_the_evidence_and_the_provisional_flag(tmp_path):
    report = calibrate_arrays_report(_hand_built_arrays(), 0.5)
    text = render_yaml(report)
    assert text.startswith("# GENERATED by vividly.reid.calibrate")
    assert "PROVISIONAL" in text

    document = yaml.safe_load(text)
    assert document["schema_version"] == CALIBRATION_SCHEMA_VERSION
    # tau is nested under reid.gallery, the GalleryConfig shape load_reid_config accepts.
    assert document["reid"]["gallery"]["tau"] == pytest.approx(0.80)
    assert document["calibration"]["provisional"] is True
    assert document["calibration"]["note"] == PROVISIONAL_NOTE
    assert document["calibration"]["recommended"]["far_target"] == 0.5
    assert document["calibration"]["recommended"]["false_merge_rate"] == pytest.approx(1 / 3)

    written = write_calibration(report, tmp_path / "nested" / "tau.yaml")
    assert yaml.safe_load(written.read_text())["reid"]["gallery"]["tau"] == pytest.approx(0.80)


def test_the_generated_reid_block_loads_through_the_stage_loader(tmp_path):
    # The tool's own output must survive its only consumer. The reid block was emitted as
    # reid.tau, but tau is a GalleryConfig field, so load_reid_config raised
    # "unknown ReidConfig keys ['tau']" on the file the tool just wrote. Nested under
    # reid.gallery.tau it round-trips.
    from vividly.reid.stage import load_reid_config

    report = calibrate_arrays_report(_hand_built_arrays(), 0.5)
    path = write_calibration(report, tmp_path / "tau.yaml")
    loaded = load_reid_config(path)
    assert loaded.gallery.tau == pytest.approx(report.tau)


def test_a_scenario_calibration_also_loads_through_the_stage_loader(tmp_path):
    cfg = config_from_mapping({"scenario": dict(_small_scenario_config().__dict__)})
    path = write_calibration(calibrate_scenario(cfg), tmp_path / "tau.yaml")
    from vividly.reid.stage import load_reid_config

    assert load_reid_config(path).gallery.tau == pytest.approx(
        yaml.safe_load(path.read_text())["reid"]["gallery"]["tau"]
    )


def test_the_arrays_report_marks_distractor_inclusion_not_applicable():
    # On a captured .npz the gallery composition is fixed by the capture, so the flag does
    # not apply. It used to be hardcoded True (the dataclass default); it is now rendered null.
    document = calibrate_arrays_report(_hand_built_arrays(), 0.5).to_dict()
    assert document["include_distractors_in_gallery"] is None
    assert document["probe"] == "captured"
    rendered = yaml.safe_load(render_yaml(calibrate_arrays_report(_hand_built_arrays(), 0.5)))
    assert rendered["calibration"]["include_distractors_in_gallery"] is None


@pytest.mark.parametrize("include", [True, False])
def test_a_scenario_report_forwards_the_real_distractor_flag(include):
    # The synthetic path must forward the config's actual value, not the default.
    cfg = config_from_mapping(
        {
            "scenario": dict(_small_scenario_config(gap_tiers_s=(30.0,)).__dict__),
            "include_distractors_in_gallery": include,
        }
    )
    document = calibrate_scenario(cfg).to_dict()
    assert document["include_distractors_in_gallery"] is include


def test_the_generated_yaml_is_deterministic():
    # No timestamp, no hostname: a diff means the numbers moved, not that the clock did.
    cfg = config_from_mapping({"scenario": dict(_small_scenario_config().__dict__)})
    assert render_yaml(calibrate_scenario(cfg)) == render_yaml(calibrate_scenario(cfg))


def test_the_generated_tau_is_not_rounded_away_from_the_far_it_bought():
    # A rounded threshold does not reproduce the FAR reported beside it.
    arrays = _hand_built_arrays()
    report = calibrate_arrays_report(arrays, 0.0)
    tau = yaml.safe_load(render_yaml(report))["reid"]["gallery"]["tau"]
    assert tau == report.tau
    assert M.false_merge_rate(
        arrays.gallery, arrays.gallery_ids, arrays.probes, arrays.probe_ids, tau
    ) == pytest.approx(report.recommended.false_merge_rate)


def test_summarise_leads_with_the_provisional_banner_and_the_false_merge_column():
    text = summarise(calibrate_arrays_report(_hand_built_arrays(), 0.5))
    assert text.startswith("PROVISIONAL.")
    header = next(line for line in text.splitlines() if "tau" in line and "DIR" in line)
    assert header.index("false-merge") < header.index("DIR")


def test_proposal_is_a_plain_record_the_dashboard_can_hold():
    proposal = Proposal(identity_key=3, confidence=0.9, n_support=2, elapsed_s=42.0)
    assert proposal.to_dict() == {
        "identity_key": 3,
        "confidence": 0.9,
        "n_support": 2,
        "elapsed_s": 42.0,
    }
