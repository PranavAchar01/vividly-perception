"""Factory tests for the engineering convention in POLICY.md that model backends are
swappable behind configuration.

The claim being tested is that a shipped eval YAML alone selects a backend, and that an
unknown key is a loud error rather than a dynamic import of an arbitrary module path
(which would be a hole straight through the AGPL and face-biometric deny-lists).

No weights are fetched and no host is contacted. The routing tests swap the registry
tables for recorders, so nothing is constructed; the one test that touches a real backend
only binds the YAML spec against its constructor signature, and skips where that backend
does not belong (rtmlib is darwin-marked in the dev fork).
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from vividly.models import factory
from vividly.models.factory import build_detector, build_pose, build_tracker

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CFG = REPO_ROOT / "configs" / "eval" / "smoke_mac.yaml"


# --------------------------------------------------------------------- helpers
def _smoke() -> dict:
    return yaml.safe_load(SMOKE_CFG.read_text())


class _Recorder:
    """Stands in for a backend constructor. Records kwargs; builds nothing."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return f"built:{kw.get('name', '?')}"


# ------------------------------------------------------- the registry is closed
@pytest.mark.parametrize(
    ("build", "available"),
    [
        (build_detector, ["rtdetr_onnx", "rtmlib"]),
        (build_pose, ["rtmlib", "rtmlib_flip"]),
    ],
)
def test_an_unknown_backend_raises_keyerror_listing_what_exists(build, available) -> None:
    with pytest.raises(KeyError) as exc:
        build({"backend": "yolov8", "onnx": "anything.onnx"})
    msg = str(exc.value)
    assert "yolov8" in msg
    assert str(available) in msg, f"the error must list {available}; got {msg}"
    assert "factory.py" in msg, "the error should say where a backend is registered"


def test_an_unknown_tracker_raises_keyerror_listing_the_registered_backends() -> None:
    with pytest.raises(KeyError) as exc:
        build_tracker({"backend": "ocsort"})
    assert "['botsort', 'bytetrack']" in str(exc.value)


def test_a_spec_with_no_backend_key_is_an_error_not_a_default() -> None:
    with pytest.raises(KeyError):
        build_detector({"onnx": "anything.onnx"})


def test_no_tracker_is_a_valid_configuration() -> None:
    assert build_tracker(None) is None
    assert build_tracker({}) is None


def test_the_tables_hold_exactly_the_registered_backends() -> None:
    # The fourth table, factory.EMBEDDERS, is pinned in tests/test_reid_backend.py.
    # rtdetr_onnx is the v2 eyes lane (RT-DETRv4 export): two graph inputs, PIL-space
    # preprocessing, its own G1 parity gate. A separate key, never a flag on rtmlib.
    assert sorted(factory.DETECTORS) == ["rtdetr_onnx", "rtmlib"]
    # rtmlib_flip is the same ONNX with mmpose-equivalent flip TTA (2 forwards/crop). It
    # is a separate key, not a flag, so a config cannot change the pose stage's cost class
    # without changing the backend name a results row records.
    assert sorted(factory.POSE) == ["rtmlib", "rtmlib_flip"]
    assert sorted(factory.TRACKERS) == ["botsort", "bytetrack"]


def test_importing_the_factory_imports_no_backend() -> None:
    """Every table entry is a lazy thunk. If that regressed, importing the factory would
    drag in rtmlib, which the dev fork installs on darwin only, on every box that merely
    reads a config. A subprocess, because sys.modules in this one is shared with whatever
    ran before it."""
    code = (
        "import sys; import vividly.models.factory as f; "
        "assert sorted(f.DETECTORS) == ['rtdetr_onnx', 'rtmlib']; "
        "print([m for m in ('rtmlib', 'torch', 'onnxruntime') "
        "if m in sys.modules])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    assert proc.stdout.strip() == "[]"


# ------------------------------------------------ the shipped YAML routes correctly
def test_the_smoke_config_detector_routes_to_the_rtmlib_entry(monkeypatch) -> None:
    spec = _smoke()["detector"]
    assert spec["backend"] == "rtmlib"
    rec = _Recorder()
    monkeypatch.setitem(factory.DETECTORS, "rtmlib", rec)
    monkeypatch.setitem(factory.DETECTORS, "rtdetr_onnx", _Recorder())

    out = build_detector(spec)

    assert out == "built:rtmdet-nano-person-onnx"
    (kw,) = rec.calls
    assert "backend" not in kw, "the selector key must not reach the constructor"
    assert kw["input_size"] == (320, 320), "a YAML list must arrive as a tuple"
    assert kw["device"] == "cpu", "CoreML crashes both detector graphs; cpu is the pin"
    assert kw["onnx"] == spec["onnx"]
    assert factory.DETECTORS["rtdetr_onnx"].calls == [], "the other backend was never touched"


def test_the_smoke_config_pose_routes_to_the_rtmlib_entry(monkeypatch) -> None:
    spec = _smoke()["pose"]
    assert spec["backend"] == "rtmlib"
    rec = _Recorder()
    monkeypatch.setitem(factory.POSE, "rtmlib", rec)

    build_pose(spec)

    (kw,) = rec.calls
    assert kw["input_size"] == (192, 256)
    assert kw["name"] == "rtmpose-m-body7-onnx"


def test_swapping_the_backend_key_swaps_the_model_and_nothing_else(monkeypatch) -> None:
    """The convention restated as a test: one word in the YAML changes the backend."""
    spec = _smoke()["detector"]
    rtm, rtdetr = _Recorder(), _Recorder()
    monkeypatch.setitem(factory.DETECTORS, "rtmlib", rtm)
    monkeypatch.setitem(factory.DETECTORS, "rtdetr_onnx", rtdetr)

    build_detector({**spec, "backend": "rtdetr_onnx"})

    assert rtm.calls == []
    assert len(rtdetr.calls) == 1
    assert rtdetr.calls[0]["name"] == spec["name"]


@pytest.mark.parametrize("key", ["input_size", "det_input_size", "pose_input_size"])
def test_every_size_key_is_normalised_from_list_to_tuple(key: str, monkeypatch) -> None:
    rec = _Recorder()
    monkeypatch.setitem(factory.DETECTORS, "rtmlib", rec)
    build_detector({"backend": "rtmlib", key: [640, 480]})
    assert rec.calls[0][key] == (640, 480)


# ------------------------- constructible on the platform that owns the backend
def test_the_smoke_spec_binds_to_the_real_rtmlib_constructor() -> None:
    """Routing is not enough: the YAML's keys must be the constructor's parameters.

    Signature binding proves that without loading a single weight; importing the module
    is the only cost, and rtmlib is darwin-marked in the dev fork, so this skips wherever it
    is not installed.
    """
    pytest.importorskip("rtmlib", reason="rtmlib is darwin-marked in the dev fork")
    from vividly.models.rtmlib_backend import RtmlibDetector, RtmlibPose

    cfg = _smoke()
    for spec, cls in ((cfg["detector"], RtmlibDetector), (cfg["pose"], RtmlibPose)):
        kwargs = {k: v for k, v in spec.items() if k != "backend"}
        kwargs["input_size"] = tuple(kwargs["input_size"])
        inspect.signature(cls.__init__).bind(None, **kwargs)
