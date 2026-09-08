"""Hand-computable tests for the open-set re-ID metrics.

Every expected number below is derived by hand in a comment, from one tiny fixture: a
4-row, 3-identity gallery in 3 dimensions, 3 return probes and 2 new-identity probes. No
random draws, no recorded blobs, no golden files. If a metric changes behaviour, a test
here fails with arithmetic next to it rather than a diff against a number nobody can check.

The fixture. Every row is a unit vector whose entries are 0, +/-1, +/-0.6 or +/-0.8, so
every cosine is a sum of products of those and every score in this file is exact to well
under `pytest.approx`'s tolerance. The norms are exactly 1.0 in IEEE double, which is what lets
scores that are equal on paper be equal bit-for-bit here (the DIR/FAR sweep depends on
that: it de-duplicates observed scores).

    gallery                     identity      probes                    true identity
    g0 = (1.0, 0.0, 0.0)           10         p0 = (0.6, 0.8, 0.0)          10
    g1 = (0.8, 0.6, 0.0)           10         p1 = (0.0, 0.8, 0.6)          20
    g2 = (0.0, 1.0, 0.0)           20         p2 = (0.8, 0.0, 0.6)          30
    g3 = (0.0, 0.0, 1.0)           30
                                              new-identity probes (no gallery mate)
                                              n0 = (0.6, 0.0, 0.8)
                                              n1 = (0.6, -0.8, 0.0)

    cosines, probe by gallery row              per-identity scores (max over that
                g0     g1     g2     g3        identity's rows)
    p0        0.60   0.96   0.80   0.00            10: 0.96   20: 0.80   30: 0.00
    p1        0.00   0.48   0.80   0.60            10: 0.48   20: 0.80   30: 0.60
    p2        0.80   0.64   0.00   0.60            10: 0.80   20: 0.00   30: 0.60
    n0        0.60   0.48   0.00   0.80            10: 0.60   20: 0.00   30: 0.80
    n1        0.60   0.00  -0.80   0.00            10: 0.60   20: -0.80  30: 0.00

    top-1:  p0 -> 10 @ 0.96 (correct)   p1 -> 20 @ 0.80 (correct)
            p2 -> 10 @ 0.80 (wrong, true 30: this is the false merge)
            n0 -> 30 @ 0.80              n1 -> 10 @ 0.60

So Rank-1 = 2/3, and the whole curve follows from five numbers: correct-probe scores
{0.96, 0.80}, all-probe top-1 scores {0.96, 0.80, 0.80}, new-probe top-1 scores
{0.80, 0.60}.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

from vividly.reid.metrics import (
    DirFarCurve,
    OperatingPoint,
    cosine_similarity,
    dir_at_far,
    dir_far_curve,
    false_merge_rate,
    identity_scores,
    match_latency,
    operating_point,
    rank1,
    top1_match,
)

REPO = Path(__file__).resolve().parents[1]
METRICS_SOURCE = REPO / "src" / "vividly" / "reid" / "metrics.py"

GALLERY = np.array(
    [
        [1.0, 0.0, 0.0],  # g0, identity 10
        [0.8, 0.6, 0.0],  # g1, identity 10
        [0.0, 1.0, 0.0],  # g2, identity 20
        [0.0, 0.0, 1.0],  # g3, identity 30
    ]
)
GALLERY_IDS = np.array([10, 10, 20, 30])

PROBES = np.array(
    [
        [0.6, 0.8, 0.0],  # p0, identity 10
        [0.0, 0.8, 0.6],  # p1, identity 20
        [0.8, 0.0, 0.6],  # p2, identity 30
    ]
)
PROBE_IDS = np.array([10, 20, 30])

NEW_PROBES = np.array(
    [
        [0.6, 0.0, 0.8],  # n0, tops out at 0.80 against identity 30
        [0.6, -0.8, 0.0],  # n1, tops out at 0.60 against identity 10
    ]
)

#: A gallery holding one person only. Used for the single-identity edge cases.
SOLO_GALLERY = np.array([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]])
SOLO_IDS = np.array([7, 7])


def curve() -> DirFarCurve:
    return dir_far_curve(GALLERY, GALLERY_IDS, PROBES, PROBE_IDS, NEW_PROBES)


# ---------------------------------------------------------------------------
# cosine similarity
# ---------------------------------------------------------------------------
def test_cosine_similarity_matches_the_hand_computed_matrix():
    # p0 . g1 = 0.6*0.8 + 0.8*0.6 = 0.48 + 0.48 = 0.96, and so on for all twelve entries.
    expected = [
        [0.60, 0.96, 0.80, 0.00],
        [0.00, 0.48, 0.80, 0.60],
        [0.80, 0.64, 0.00, 0.60],
    ]
    got = cosine_similarity(PROBES, GALLERY)
    assert got.shape == (3, 4)
    assert got == pytest.approx(np.array(expected))


def test_cosine_similarity_is_scale_invariant():
    """Raw (unnormalised) features must give the same cosines as unit rows."""
    scaled_probes = PROBES * np.array([[3.0], [0.5], [17.0]])
    scaled_gallery = GALLERY * 4.0
    assert cosine_similarity(scaled_probes, scaled_gallery) == pytest.approx(
        cosine_similarity(PROBES, GALLERY)
    )


def test_cosine_similarity_of_a_zero_row_is_zero_not_nan():
    # The eps-guarded division keeps a dead embedding at 0 (matches nothing) instead of
    # turning it into NaN (poisons every comparison it touches).
    zero = np.zeros((1, 3))
    got = cosine_similarity(zero, GALLERY)
    assert got == pytest.approx(np.zeros((1, 4)))


def test_cosine_similarity_dimension_mismatch_raises():
    with pytest.raises(ValueError, match=r"dim 3 but .* dim 5"):
        cosine_similarity(PROBES, np.ones((2, 5)))


# ---------------------------------------------------------------------------
# per-identity aggregation
# ---------------------------------------------------------------------------
def test_identity_scores_aggregate_by_max_over_that_identitys_rows():
    identities, scores = identity_scores(GALLERY, GALLERY_IDS, PROBES)
    assert identities.tolist() == [10, 20, 30]
    # identity 10 has two rows: p0 takes max(0.60, 0.96) = 0.96, p1 max(0.00, 0.48) = 0.48,
    # p2 max(0.80, 0.64) = 0.80.
    assert scores == pytest.approx(
        np.array(
            [
                [0.96, 0.80, 0.00],
                [0.48, 0.80, 0.60],
                [0.80, 0.00, 0.60],
            ]
        )
    )


def test_identity_scores_do_not_depend_on_gallery_row_order():
    """Rows are grouped by label, not by position: an unsorted gallery scores identically."""
    order = [3, 1, 2, 0]  # g3(30), g1(10), g2(20), g0(10)
    identities, scores = identity_scores(GALLERY[order], GALLERY_IDS[order], PROBES)
    assert identities.tolist() == [10, 20, 30]
    _, reference = identity_scores(GALLERY, GALLERY_IDS, PROBES)
    assert scores == pytest.approx(reference)


def test_max_aggregation_is_load_bearing_mean_would_produce_a_false_merge():
    """The documented reason for max, as arithmetic.

    p0 is a near-exact match to g1, one of identity 10's two enrolled views:
        max  over identity 10 = max(0.60, 0.96)   = 0.96  -> beats identity 20's 0.80
        mean over identity 10 = (0.60 + 0.96) / 2 = 0.78  -> loses to identity 20's 0.80
    Mean punishes an identity for its weaker views and punishes it more the longer the
    enrollment, which would turn a correct re-association into a wrong-chart merge.
    """
    identities, scores = identity_scores(GALLERY, GALLERY_IDS, PROBES)
    sims = cosine_similarity(PROBES, GALLERY)
    mean_over_identity_10 = sims[0, :2].mean()
    assert mean_over_identity_10 == pytest.approx(0.78)
    assert mean_over_identity_10 < scores[0, 1]  # identity 20 at 0.80 would win on mean
    assert identities[scores[0].argmax()] == 10  # max gets it right


def test_top1_match_returns_the_decision_the_system_would_make():
    predicted, scores = top1_match(GALLERY, GALLERY_IDS, PROBES)
    assert predicted.tolist() == [10, 20, 10]  # p2 is predicted 10 but is really 30
    assert scores == pytest.approx(np.array([0.96, 0.80, 0.80]))


def test_top1_ties_break_to_the_smaller_identity_label():
    """Two identities holding the identical row: the tie resolves the same way every run."""
    gallery = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ids = np.array([7, 3])  # deliberately not ascending in row order
    predicted, scores = top1_match(gallery, ids, np.array([[1.0, 0.0, 0.0]]))
    assert predicted.tolist() == [3]
    assert scores == pytest.approx(np.array([1.0]))


# ---------------------------------------------------------------------------
# Rank-1 (closed set)
# ---------------------------------------------------------------------------
def test_rank1_is_two_of_three_on_the_fixture():
    # p0 -> 10 correct, p1 -> 20 correct, p2 -> 10 but is 30. 2/3.
    assert rank1(GALLERY, GALLERY_IDS, PROBES, PROBE_IDS) == pytest.approx(2.0 / 3.0)


def test_rank1_is_one_when_every_probe_is_its_own_gallery_row():
    assert rank1(GALLERY, GALLERY_IDS, GALLERY, GALLERY_IDS) == pytest.approx(1.0)


def test_rank1_is_one_for_a_single_identity_gallery():
    """Closed set with one enrolled person: there is nothing else to pick, so Rank-1 is 1.

    This is the number that shows why Rank-1 alone can never clear a re-ID stage. The
    embedding is doing no work here at all.
    """
    probes = np.array([[0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])  # the second is nothing like g0/g1
    assert rank1(SOLO_GALLERY, SOLO_IDS, probes, np.array([7, 7])) == pytest.approx(1.0)


def test_rank1_empty_probe_set_raises():
    with pytest.raises(ValueError, match="probes is empty"):
        rank1(GALLERY, GALLERY_IDS, np.zeros((0, 3)), np.zeros(0, dtype=int))


def test_rank1_rejects_a_probe_identity_that_was_never_enrolled():
    with pytest.raises(ValueError, match=r"probe identities \[99\] do not appear"):
        rank1(GALLERY, GALLERY_IDS, PROBES, np.array([10, 20, 99]))


def test_rank1_rejects_a_label_array_of_the_wrong_length():
    with pytest.raises(ValueError, match="4 labels for 3 rows"):
        rank1(GALLERY, GALLERY_IDS, PROBES, np.array([10, 20, 30, 10]))


def test_gallery_labels_are_per_row_not_per_identity():
    with pytest.raises(ValueError, match="3 labels for 4 rows"):
        rank1(GALLERY, np.array([10, 20, 30]), PROBES, PROBE_IDS)


# ---------------------------------------------------------------------------
# the DIR/FAR curve
# ---------------------------------------------------------------------------
def test_dir_far_curve_matches_the_hand_computed_step_function():
    """Four rows: the three observed scores plus the appended above-everything threshold.

    correct-probe scores {p0 0.96, p1 0.80}; new-probe scores {n0 0.80, n1 0.60}.

        tau     accepted correct   DIR      accepted new   FAR
        0.60    p0, p1             2/3      n0, n1         2/2 = 1.0
        0.80    p0, p1             2/3      n0             1/2 = 0.5
        0.96    p0                 1/3      none           0.0
        0.96+   none               0.0      none           0.0
    """
    got = curve()
    assert got.taus.tolist() == pytest.approx([0.60, 0.80, 0.96, np.nextafter(0.96, np.inf)])
    assert got.dirs.tolist() == pytest.approx([2 / 3, 2 / 3, 1 / 3, 0.0])
    assert got.fars.tolist() == pytest.approx([1.0, 0.5, 0.0, 0.0])
    assert got.n_probes == 3
    assert got.n_new_probes == 2


def test_curve_thresholds_are_the_observed_scores_plus_one_above_the_top():
    got = curve()
    observed = [0.60, 0.80, 0.96]
    assert sorted(got.taus[:-1].tolist()) == pytest.approx(observed)
    assert got.taus[-1] > max(observed)
    assert got.taus[-1] == np.nextafter(got.taus[-2], np.inf)


def test_curve_is_monotone_in_tau():
    got = curve()
    assert np.all(np.diff(got.taus) > 0)  # strictly increasing thresholds
    assert np.all(np.diff(got.dirs) <= 0)  # raising tau can only reject
    assert np.all(np.diff(got.fars) <= 0)
    assert np.all((got.dirs >= 0) & (got.dirs <= 1))
    assert np.all((got.fars >= 0) & (got.fars <= 1))


def test_curve_bottom_row_reproduces_closed_set_rank1():
    """At a threshold below every return-probe score nothing is rejected, so DIR is Rank-1."""
    got = curve()
    assert got.taus[0] <= 0.80  # below p1's score, the lowest return-probe score
    assert got.dirs[0] == pytest.approx(rank1(GALLERY, GALLERY_IDS, PROBES, PROBE_IDS))


def test_curve_top_row_is_the_reject_everything_point():
    # tau above every observed score: DIR 0, FAR 0, and (checked below) false merges 0.
    got = curve()
    assert got.dirs[-1] == pytest.approx(0.0)
    assert got.fars[-1] == pytest.approx(0.0)


def test_curve_acceptance_is_at_least_tau_not_strictly_greater():
    """p1 scores exactly 0.80, and at tau = 0.80 it is still accepted (DIR stays 2/3).

    If the comparison were `>` the row at 0.80 would read DIR 1/3, FAR 0.0, and every
    false-merge number quoted at a threshold taken from this curve would be wrong by one
    probe in the other direction.
    """
    got = curve()
    row = int(np.argmin(np.abs(got.taus - 0.80)))
    assert got.taus[row] == pytest.approx(0.80)
    assert got.dirs[row] == pytest.approx(2 / 3)
    assert got.fars[row] == pytest.approx(0.5)


def test_curve_single_identity_gallery():
    """One enrolled person, one returning probe, two new people.

    p = g0 exactly, so DIR is 1.0 until tau passes 1.0. Both new probes top out against
    the only identity there is: n0 at 0.60 (max(0.60, 0.48)) and n1 at 1.0 (it is g0's
    direction). FAR therefore stays 0.5 until tau passes 1.0.
    """
    probes = np.array([[1.0, 0.0, 0.0]])
    new_probes = np.array([[0.6, 0.0, 0.8], [1.0, 0.0, 0.0]])
    got = dir_far_curve(SOLO_GALLERY, SOLO_IDS, probes, np.array([7]), new_probes)
    assert got.taus.tolist() == pytest.approx([0.60, 1.0, np.nextafter(1.0, np.inf)])
    assert got.dirs.tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert got.fars.tolist() == pytest.approx([1.0, 0.5, 0.0])


def test_curve_with_no_correct_probe_has_dir_zero_everywhere():
    """p2 alone: its top-1 is identity 10 and it is really 30, so DIR is 0 at every tau.

    FAR still moves (n0 at 0.80, n1 at 0.60), which is the shape of the worst case worth
    naming: a threshold that admits new people and identifies nobody.
    """
    got = dir_far_curve(GALLERY, GALLERY_IDS, PROBES[2:], PROBE_IDS[2:], NEW_PROBES)
    assert got.dirs.tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert got.fars.tolist() == pytest.approx([1.0, 0.5, 0.0])
    assert dir_at_far(got, 0.5) == pytest.approx((0.0, 0.80))


def test_several_probes_may_share_one_identity():
    """A return tracklet contributes many probe embeddings, so repeated labels are normal.

    Probes p0, p0 and p2 with labels 10, 10, 30: two correct at 0.96 and one wrong at 0.80,
    so Rank-1 is 2/3 and DIR holds at 2/3 until tau passes 0.96.
    """
    probes = PROBES[[0, 0, 2]]
    probe_ids = np.array([10, 10, 30])
    assert rank1(GALLERY, GALLERY_IDS, probes, probe_ids) == pytest.approx(2 / 3)
    got = dir_far_curve(GALLERY, GALLERY_IDS, probes, probe_ids, NEW_PROBES)
    assert got.taus.tolist() == pytest.approx([0.60, 0.80, 0.96, np.nextafter(0.96, np.inf)])
    assert got.dirs.tolist() == pytest.approx([2 / 3, 2 / 3, 2 / 3, 0.0])
    assert got.fars.tolist() == pytest.approx([1.0, 0.5, 0.0, 0.0])
    assert false_merge_rate(GALLERY, GALLERY_IDS, probes, probe_ids, 0.80) == pytest.approx(1 / 3)


def test_curve_empty_probe_set_raises():
    with pytest.raises(ValueError, match="probes is empty"):
        dir_far_curve(GALLERY, GALLERY_IDS, np.zeros((0, 3)), np.zeros(0, dtype=int), NEW_PROBES)


def test_curve_empty_new_probe_set_raises():
    with pytest.raises(ValueError, match="new_probes is empty"):
        dir_far_curve(GALLERY, GALLERY_IDS, PROBES, PROBE_IDS, np.zeros((0, 3)))


def test_curve_rejects_unenrolled_probe_identities():
    with pytest.raises(ValueError, match="probe identities"):
        dir_far_curve(GALLERY, GALLERY_IDS, PROBES, np.array([10, 20, 41]), NEW_PROBES)


def test_curve_to_dict_is_json_serialisable():
    payload = json.loads(json.dumps(curve().to_dict()))
    assert payload["dirs"] == pytest.approx([2 / 3, 2 / 3, 1 / 3, 0.0])
    assert payload["n_probes"] == 3


def test_curve_rejects_misaligned_arrays():
    with pytest.raises(ValueError, match="curve arrays must align"):
        DirFarCurve(
            taus=np.array([0.1, 0.2]),
            dirs=np.array([1.0]),
            fars=np.array([1.0]),
            n_probes=1,
            n_new_probes=1,
        )


# ---------------------------------------------------------------------------
# operating points
# ---------------------------------------------------------------------------
def test_dir_at_far_zero_takes_the_threshold_that_admits_no_new_identity():
    # FAR reaches 0 first at tau = 0.96, where only p0 survives: DIR 1/3.
    got_dir, got_tau = dir_at_far(curve(), 0.0)
    assert got_dir == pytest.approx(1 / 3)
    assert got_tau == pytest.approx(0.96)


def test_dir_at_far_half_takes_the_0_80_row():
    got_dir, got_tau = dir_at_far(curve(), 0.5)
    assert got_dir == pytest.approx(2 / 3)
    assert got_tau == pytest.approx(0.80)


def test_dir_at_far_picks_the_smallest_qualifying_tau_not_the_largest():
    """Regression pin for the DIR@FAR definition.

    FAR falls as tau rises, so every threshold above the qualifying one also meets the
    budget; the largest of them is the appended reject-everything row, where DIR is 0 by
    construction. The operating point is the smallest qualifying tau, which is the one
    with the most DIR left in it.
    """
    got = curve()
    got_dir, got_tau = dir_at_far(got, 1.0)
    assert got_tau == pytest.approx(0.60)  # not got.taus[-1]
    assert got_dir == pytest.approx(2 / 3)  # not 0.0


def test_dir_at_far_gives_no_slack_on_the_budget():
    """A target just under an achievable FAR must step up to the next threshold.

    FAR = 0.5 is available at tau 0.80, but a 0.49 budget does not buy it: the answer is
    the tau 0.96 row at FAR 0. A patient-safety ceiling is not negotiable by an epsilon.
    """
    got_dir, got_tau = dir_at_far(curve(), 0.49)
    assert got_tau == pytest.approx(0.96)
    assert got_dir == pytest.approx(1 / 3)


@pytest.mark.parametrize("bad", [-0.01, 1.5, float("nan")])
def test_dir_at_far_rejects_targets_outside_the_unit_interval(bad):
    with pytest.raises(ValueError, match="far target must be in"):
        dir_at_far(curve(), bad)


def test_operating_point_reports_the_achieved_far_next_to_the_target():
    point = operating_point(curve(), 0.9)
    # The 0.9 budget is met first at tau 0.80, where the achieved far is 0.5, not 0.9.
    assert point == OperatingPoint(tau=0.80, dir=2 / 3, far=0.5, far_target=0.9)
    assert point.far <= point.far_target


def test_operating_point_agrees_with_dir_at_far():
    for target in (0.0, 0.25, 0.5, 0.75, 1.0):
        point = operating_point(curve(), target)
        assert (point.dir, point.tau) == pytest.approx(dir_at_far(curve(), target))


def test_operating_point_to_dict_is_json_serialisable():
    payload = json.loads(json.dumps(operating_point(curve(), 0.5).to_dict()))
    assert payload == {
        "tau": pytest.approx(0.80),
        "dir": pytest.approx(2 / 3),
        "far": 0.5,
        "far_target": 0.5,
    }


# ---------------------------------------------------------------------------
# the false-merge rate (the patient-safety KPI)
# ---------------------------------------------------------------------------
def test_false_merge_rate_counts_p2_and_only_p2():
    # At tau 0.75: p0 -> 10 correct, p1 -> 20 correct, p2 -> 10 at 0.80 >= 0.75 and is
    # really 30. One wrong chart out of three return probes.
    got = false_merge_rate(GALLERY, GALLERY_IDS, PROBES, PROBE_IDS, tau=0.75)
    assert got == pytest.approx(1 / 3)


def test_false_merge_rate_uses_at_least_tau_at_the_boundary():
    # p2's wrong match scores exactly 0.80.
    assert false_merge_rate(GALLERY, GALLERY_IDS, PROBES, PROBE_IDS, 0.80) == pytest.approx(1 / 3)
    just_above = float(np.nextafter(0.80, np.inf))
    assert false_merge_rate(GALLERY, GALLERY_IDS, PROBES, PROBE_IDS, just_above) == 0.0


def test_false_merge_rate_is_zero_when_tau_is_above_every_score():
    # Reject everything and nothing can be merged. This is the safe, useless corner.
    assert false_merge_rate(GALLERY, GALLERY_IDS, PROBES, PROBE_IDS, 1.1) == 0.0


def test_false_merge_rate_below_every_score_is_one_minus_rank1():
    """Accept everything and every closed-set mistake becomes a merged chart."""
    got = false_merge_rate(GALLERY, GALLERY_IDS, PROBES, PROBE_IDS, -1.1)
    assert got == pytest.approx(1 - rank1(GALLERY, GALLERY_IDS, PROBES, PROBE_IDS))
    assert got == pytest.approx(1 / 3)


def test_a_wrong_identity_above_tau_is_not_a_merge_when_top1_is_correct():
    """p0 alone: identity 20 scores 0.80, above tau, but identity 10 wins at 0.96.

    The system binds to its top-1, so the runner-up never touches a chart and this is not
    a false merge. The KPI counts events that could actually happen.
    """
    got = false_merge_rate(GALLERY, GALLERY_IDS, PROBES[:1], PROBE_IDS[:1], tau=0.7)
    assert got == 0.0


def test_false_merge_rate_is_zero_for_a_single_identity_gallery():
    """With one enrolled person there is no wrong identity to merge into, at any tau."""
    probes = np.array([[0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
    for tau in (-1.0, 0.0, 0.5, 1.0):
        assert false_merge_rate(SOLO_GALLERY, SOLO_IDS, probes, np.array([7, 7]), tau) == 0.0


def test_false_merge_rate_at_a_curve_operating_point_is_consistent_with_that_curve():
    """The KPI quoted at an operating point must be the rate that point produces.

    At tau 0.80 the curve says DIR 2/3 and FAR 0.5; the false-merge rate there is 1/3, and
    2/3 + 1/3 = 1 only because no probe is rejected at that threshold. DIR and the
    false-merge rate are not complements in general.
    """
    point = operating_point(curve(), 0.5)
    got = false_merge_rate(GALLERY, GALLERY_IDS, PROBES, PROBE_IDS, point.tau)
    assert got == pytest.approx(1 / 3)
    assert point.dir + got == pytest.approx(1.0)


def test_false_merge_rate_empty_probe_set_raises():
    with pytest.raises(ValueError, match="probes is empty"):
        false_merge_rate(GALLERY, GALLERY_IDS, np.zeros((0, 3)), np.zeros(0, dtype=int), 0.5)


def test_false_merge_rate_rejects_unenrolled_probe_identities():
    with pytest.raises(ValueError, match="dir_far_curve"):
        false_merge_rate(GALLERY, GALLERY_IDS, PROBES, np.array([10, 20, 77]), 0.5)


# ---------------------------------------------------------------------------
# shared validation
# ---------------------------------------------------------------------------
def test_a_single_embedding_must_be_passed_as_a_matrix():
    with pytest.raises(ValueError, match="must be a 2-D"):
        rank1(GALLERY, GALLERY_IDS, np.array([0.6, 0.8, 0.0]), np.array([10]))


def test_non_finite_embeddings_raise():
    poisoned = PROBES.copy()
    poisoned[1, 1] = np.nan
    with pytest.raises(ValueError, match="contains NaN or inf"):
        rank1(GALLERY, GALLERY_IDS, poisoned, PROBE_IDS)


def test_float_identity_labels_raise():
    with pytest.raises(ValueError, match="integer dtype"):
        rank1(GALLERY, np.array([10.0, 10.0, 20.0, 30.0]), PROBES, PROBE_IDS)


def test_probe_and_gallery_dimensions_must_match():
    with pytest.raises(ValueError, match="probes has dim 4 but gallery has dim 3"):
        rank1(GALLERY, GALLERY_IDS, np.ones((2, 4)), np.array([10, 20]))


# ---------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------
def test_match_latency_returns_a_positive_per_probe_time():
    seconds = match_latency(GALLERY, PROBES, repeats=2)
    assert seconds > 0.0
    assert math.isfinite(seconds)
    assert seconds < 1.0  # a 3x4 matmul on any machine that can run this suite


def test_match_latency_accepts_float32_and_integer_input():
    assert match_latency(GALLERY.astype(np.float32), PROBES.astype(np.float32), 1) > 0.0
    assert match_latency(np.eye(3, dtype=np.int64), np.eye(3, dtype=np.int64), 1) > 0.0


def test_match_latency_requires_at_least_one_repeat():
    with pytest.raises(ValueError, match="repeats must be >= 1"):
        match_latency(GALLERY, PROBES, repeats=0)


def test_match_latency_empty_inputs_raise():
    with pytest.raises(ValueError, match="probes is empty"):
        match_latency(GALLERY, np.zeros((0, 3)))
    with pytest.raises(ValueError, match="gallery is empty"):
        match_latency(np.zeros((0, 3)), PROBES)


# ---------------------------------------------------------------------------
# module hygiene
# ---------------------------------------------------------------------------
def test_metrics_module_imports_nothing_but_numpy_and_the_standard_library():
    """Pure numpy and no sklearn is a contract here, not a preference.

    A Linux environment without the linux-harness group has no cv2, no onnxruntime and no
    torch, and this module has to score a benchmark there. sklearn is in neither the default
    development fork nor the cuda training fork, and nothing under src/vividly imports it.
    """
    tree = ast.parse(METRICS_SOURCE.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert roots == {"__future__", "time", "dataclasses", "typing", "numpy"}
