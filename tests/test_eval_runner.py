"""Eval-runner gate tests: the executable half of the measurement rule that every eval
reports latency alongside accuracy.

Model-free and offline by construction: every input here is a hand-built `Timing` or a
plain dict, so the file runs identically on the Mac dev env and on linux CI, where the
inference stack is not installed.

The two load-bearing cases are the ones that used to pass the gate. A condition where no
image yields a detection leaves the pose stage with zero timings; `latency_stats` then
emits NaN percentiles, and a NaN row both sailed through `assert_row_complete` and turned
`results.json` into a file that is not strict JSON. The other is a row with a blank
provenance label, which is an accuracy number nobody can attribute to a checkpoint.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from vividly.eval.runner import LATENCY_STAGES, assert_row_complete, latency_stats
from vividly.models.base import Timing

REQUIRED_CELLS = (
    "detect_p50_ms",
    "detect_p95_ms",
    "pose_p50_ms",
    "pose_p95_ms",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "host_cpu",
    "det_provenance",
    "pose_provenance",
)


# --------------------------------------------------------------------- helpers
def _row(**over) -> dict:
    """A row that a green run would produce. Tests damage one cell at a time."""
    row = {
        "condition": "clean",
        "kind": "passthrough",
        "n_images": 64,
        "det_AP": 0.512,
        "kpt_AP": 0.478,
        "detect_p50_ms": 18.5,
        "detect_p95_ms": 24.0,
        "pose_p50_ms": 31.25,
        "pose_p95_ms": 44.75,
        "e2e_p50_ms": 52.0,
        "e2e_p95_ms": 71.5,
        "host_cpu": "Apple M3 Pro",
        "det_provenance": "rtmdet-nano-person-onnx[train=COCO;ship=counsel]",
        "pose_provenance": "rtmpose-m-body7-onnx[train=body7;ship=no]",
    }
    row.update(over)
    return row


def _timing(stage: str, ms: float, n_items: int = 1) -> Timing:
    return Timing(stage=stage, ms=ms, n_items=n_items, backend="test")


# --------------------------------------------------- assert_row_complete: pass
def test_a_fully_populated_row_passes() -> None:
    assert_row_complete(_row())  # the assertion is that this does not raise


def test_extra_keys_are_never_a_reason_to_fail() -> None:
    # The gate is a floor, not a schema: a backend may add columns.
    assert_row_complete(_row(det_AP75=0.44, notes="ablation run"))


# ------------------------------------------------- assert_row_complete: missing
@pytest.mark.parametrize("cell", REQUIRED_CELLS)
def test_every_required_cell_is_actually_required(cell: str) -> None:
    with pytest.raises(AssertionError) as exc:
        assert_row_complete(_row(**{cell: None}))
    assert cell in str(exc.value)


@pytest.mark.parametrize("cell", REQUIRED_CELLS)
def test_a_dropped_key_is_the_same_failure_as_an_explicit_none(cell: str) -> None:
    row = _row()
    del row[cell]
    with pytest.raises(AssertionError) as exc:
        assert_row_complete(row)
    assert cell in str(exc.value)


def test_a_blank_provenance_label_is_missing_not_present() -> None:
    with pytest.raises(AssertionError) as exc:
        assert_row_complete(_row(det_provenance=""))
    assert "det_provenance" in str(exc.value)


def test_the_failure_names_the_condition_and_every_bad_cell_at_once() -> None:
    with pytest.raises(AssertionError) as exc:
        assert_row_complete(_row(condition="jpeg_q30", pose_p50_ms=None, host_cpu=""))
    msg = str(exc.value)
    assert "jpeg_q30" in msg
    assert "pose_p50_ms" in msg
    assert "host_cpu" in msg


# ---------------------------------------------- assert_row_complete: non-finite
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("cell", ["detect_p50_ms", "pose_p50_ms", "pose_p95_ms", "e2e_p95_ms"])
def test_a_non_finite_latency_is_rejected(cell: str, bad: float) -> None:
    with pytest.raises(AssertionError) as exc:
        assert_row_complete(_row(**{cell: bad}))
    assert cell in str(exc.value)


def test_a_numpy_nan_is_rejected_too() -> None:
    # latency_stats builds every cell with float(), but a sink or a merge step can hand
    # back np.float64; the gate must not care which float type it is looking at.
    with pytest.raises(AssertionError):
        assert_row_complete(_row(pose_p50_ms=float(np.float64("nan"))))


def test_zero_latency_is_finite_and_therefore_allowed() -> None:
    # 0.0 is falsy. A membership or truthiness test would reject it; a finiteness test
    # must not, because a sub-microsecond stage on a fast host legitimately rounds to 0.
    assert_row_complete(_row(detect_p50_ms=0.0, e2e_p50_ms=0.0))


def test_why_the_gate_rejects_nan_results_json_stops_being_json() -> None:
    row = _row(pose_p50_ms=float("nan"))
    # json.dumps emits a bare NaN token, which no strict JSON reader accepts.
    assert "NaN" in json.dumps(row)
    with pytest.raises(ValueError):
        json.dumps(row, allow_nan=False)


# ----------------------------------------------------------------- latency_stats
def test_percentiles_match_numpy_and_are_plain_floats() -> None:
    ms = [3.0, 5.0, 11.0, 17.0, 23.0, 29.0, 31.0]
    out = latency_stats([_timing("detect", v) for v in ms])
    assert out["detect_p50_ms"] == pytest.approx(float(np.percentile(ms, 50)))
    assert out["detect_p95_ms"] == pytest.approx(float(np.percentile(ms, 95)))
    assert out["detect_mean_ms"] == pytest.approx(float(np.mean(ms)))
    assert type(out["detect_p50_ms"]) is float


def test_stages_do_not_bleed_into_each_other() -> None:
    out = latency_stats([_timing("detect", 10.0), _timing("detect", 20.0), _timing("pose", 100.0)])
    assert out["detect_n_calls"] == 2
    assert out["pose_n_calls"] == 1
    assert out["pose_p50_ms"] == pytest.approx(100.0)


def test_ms_per_item_divides_by_items_not_by_calls() -> None:
    # One pose call over four crops is not four times the per-crop cost.
    out = latency_stats([_timing("pose", 40.0, n_items=4)])
    assert out["pose_ms_per_item"] == pytest.approx(10.0)


def test_a_stage_that_never_ran_states_n_calls_zero_explicitly() -> None:
    out = latency_stats([_timing("detect", 12.0), _timing("e2e", 15.0)])
    assert out["pose_n_calls"] == 0
    assert math.isnan(out["pose_p50_ms"])
    assert math.isnan(out["pose_p95_ms"])
    # No mean and no per-item cost are invented for a stage with no samples.
    assert "pose_mean_ms" not in out
    assert "pose_ms_per_item" not in out


def test_no_timings_at_all_still_reports_every_stage() -> None:
    out = latency_stats([])
    for stage in LATENCY_STAGES:
        assert out[f"{stage}_n_calls"] == 0
        assert math.isnan(out[f"{stage}_p50_ms"])


def test_the_empty_pose_stage_is_exactly_what_the_gate_catches() -> None:
    """The rtmlib pose backend returns Poses.empty() without recording a Timing when the
    detector found nothing, so a whole condition can end with zero pose samples. That
    row must never reach results.json."""
    row = _row(**latency_stats([_timing("detect", 12.0), _timing("e2e", 15.0)]))
    assert row["pose_n_calls"] == 0
    with pytest.raises(AssertionError) as exc:
        assert_row_complete(row)
    assert "pose_p50_ms" in str(exc.value)
    assert "pose_p95_ms" in str(exc.value)


def test_a_populated_matrix_of_stages_passes_the_gate() -> None:
    timings = [_timing("detect", 12.0), _timing("pose", 30.0, 3), _timing("e2e", 45.0)]
    row = _row(**latency_stats(timings))
    assert_row_complete(row)
    assert json.dumps(row, allow_nan=False)
