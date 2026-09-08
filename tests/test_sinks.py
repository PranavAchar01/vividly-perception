"""Metric-sink tests: POLICY.md, rule 7, no component assumes cloud connectivity.

This file is the executable form of that rule for the eval half, at the sink layer: with the
network dead and no W&B configuration present, the local JSONL sink still writes a row that
passes the runner's own gate, and `build_sink` never even imports wandb unless a config asks
for it.

Offline and model-free: no weights, no sockets, no wandb account.
"""

from __future__ import annotations

import builtins
import json
import socket
import sys
from pathlib import Path

import numpy as np
import pytest

from vividly.eval.runner import assert_row_complete
from vividly.eval.sinks import JsonlSink, MultiSink, build_sink


# --------------------------------------------------------------------- helpers
def _row() -> dict:
    """A row shaped like the runner's, with the numpy scalars a real run produces."""
    return {
        "condition": "clean",
        "kind": "passthrough",
        "n_images": np.int64(64),
        "det_AP": np.float32(0.512),
        "kpt_AP": np.float64(0.478),
        "detect_p50_ms": 18.5,
        "detect_p95_ms": 24.0,
        "pose_p50_ms": 31.25,
        "pose_p95_ms": 44.75,
        "e2e_p50_ms": 52.0,
        "e2e_p95_ms": 71.5,
        "host_cpu": "Apple M3 Pro",
        "det_provenance": "rtmdet-nano-person-onnx[train=COCO;ship=counsel]",
        "pose_provenance": "rtmpose-m-body7-onnx[train=body7;ship=no]",
        "gt_file": Path("data/eval_only/coco/annotations/person_keypoints_val2017.json"),
    }


def _cfg(**wandb) -> dict:
    return {"run_name": "unit", "logging": {"wandb": {"enabled": False, **wandb}}}


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _kill_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every outbound path raises. Nothing here should notice."""

    def _blocked(*args, **kwargs):
        raise OSError("network is blocked for this test")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)


# ----------------------------------------------------------------- JsonlSink
def test_the_jsonl_file_is_one_tagged_object_per_call(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    sink = JsonlSink(path)
    sink.log_config({"run_name": "unit"})
    sink.log_row(_row())
    sink.log_summary({"rows": 1})
    sink.close()

    assert [line["record"] for line in _lines(path)] == ["config", "row", "summary"]


def test_a_row_cannot_clobber_its_own_line_tag(tmp_path: Path) -> None:
    """An eval row carries `kind` (the condition kind), so the line tag lives under a key
    no payload owns. Otherwise every row line is tagged "passthrough" and nothing
    downstream can tell a row from a summary."""
    path = tmp_path / "metrics.jsonl"
    sink = JsonlSink(path)
    sink.log_row(_row())
    sink.close()

    (line,) = _lines(path)
    assert line["record"] == "row"
    assert line["kind"] == "passthrough"  # the row's own field survives untouched


def test_numpy_scalars_and_paths_survive_serialisation(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    sink = JsonlSink(path)
    sink.log_row({**_row(), "curve": np.array([0.1, 0.2], dtype=np.float32)})
    sink.close()

    (line,) = _lines(path)
    assert line["n_images"] == 64
    assert isinstance(line["det_AP"], float)
    assert line["det_AP"] == pytest.approx(0.512, abs=1e-6)
    assert line["gt_file"].endswith("person_keypoints_val2017.json")
    assert line["curve"] == pytest.approx([0.1, 0.2], abs=1e-6)


def test_the_sink_creates_its_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "unit_20260807" / "metrics.jsonl"
    JsonlSink(path).close()
    assert path.exists()


def test_rows_are_flushed_as_they_are_written_not_at_close(tmp_path: Path) -> None:
    # A run that dies on condition 4 must leave conditions 1-3 on disk.
    path = tmp_path / "metrics.jsonl"
    sink = JsonlSink(path)
    sink.log_row(_row())
    assert len(_lines(path)) == 1
    sink.close()


# ------------------------------------------------------------- offline by default
def test_a_row_survives_with_the_network_dead_and_no_wandb_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The offline claim, end to end at the sink layer: no WANDB_MODE, no WANDB_ENTITY,
    no reachable socket, and the run still records a row that passes assert_row_complete."""
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    _kill_the_network(monkeypatch)

    sink = build_sink(_cfg(), tmp_path)
    sink.log_config({"run_name": "unit"})
    sink.log_row(_row())
    sink.log_summary({"rows": 1})
    sink.close()

    (row,) = [line for line in _lines(tmp_path / "metrics.jsonl") if line["record"] == "row"]
    assert_row_complete(row)  # the round-tripped row is still a complete row


def test_build_sink_never_imports_wandb_when_it_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """wandb is a declared dependency, so `import wandb` would succeed, which is exactly
    why this is asserted rather than assumed. The eval runner is the pipeline's eval mode; a
    cloud client imported there is a cloud client on the appliance."""
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    real_import = builtins.__import__

    def _guarded(name, *args, **kwargs):
        if name == "wandb" or name.startswith("wandb."):
            raise AssertionError("build_sink imported wandb with wandb.enabled false")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded)
    sink = build_sink(_cfg(), tmp_path)
    sink.log_row(_row())
    sink.close()

    assert "wandb" not in sys.modules


def test_disabled_wandb_leaves_exactly_one_sink(tmp_path: Path) -> None:
    sink = build_sink(_cfg(), tmp_path)
    assert isinstance(sink, MultiSink)
    assert [type(s).__name__ for s in sink._s] == ["JsonlSink"]
    sink.close()


def test_a_missing_logging_block_still_produces_a_jsonl_sink(tmp_path: Path) -> None:
    sink = build_sink({"run_name": "unit"}, tmp_path)
    sink.log_row(_row())
    sink.close()
    assert len(_lines(tmp_path / "metrics.jsonl")) == 1


# -------------------------------------------- logging failure never kills a run
def test_a_broken_wandb_sink_is_reported_and_the_run_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An evaluation on a rented pod must not die because a metrics server was unreachable."""
    import vividly.eval.sinks as sinks_mod

    def _explode(**kwargs):
        raise RuntimeError("wandb.init: connection refused")

    monkeypatch.setattr(sinks_mod, "WandbSink", _explode)
    sink = build_sink({"logging": {"wandb": {"enabled": True}}}, tmp_path)
    sink.log_row(_row())
    sink.close()

    assert "W&B sink disabled" in capsys.readouterr().out
    assert len(_lines(tmp_path / "metrics.jsonl")) == 1


# ------------------------------------------------------------------- MultiSink
def test_multisink_fans_out_every_call(tmp_path: Path) -> None:
    a, b = JsonlSink(tmp_path / "a.jsonl"), JsonlSink(tmp_path / "b.jsonl")
    multi = MultiSink(a, b)
    multi.log_config({"run_name": "unit"})
    multi.log_row(_row())
    multi.log_summary({"rows": 1})
    multi.close()

    tags = [line["record"] for line in _lines(tmp_path / "a.jsonl")]
    assert tags == ["config", "row", "summary"]
    assert _lines(tmp_path / "a.jsonl") == _lines(tmp_path / "b.jsonl")
