"""The `rtdetr_onnx` detector backend: registration, refusals, and the colour contract.

Nothing here loads the 40 MB eyes export of record. The registry tests swap the table for a
recorder, the refusal tests all raise before the onnxruntime import (which is why the
constructor validates in that order), and the preprocessing tests are pure-function checks
on `preprocess_bgr`, so this file stays green on a box with no inference stack and no
weights on disk. The one test that does touch the real export skips itself when the file
is absent.

The numerical claim, that this backend reproduces `scripts/eval/eval_detector_v2_onnx.py`
box for box, is not made here. It is the G1 gate, `scripts/eval/backend_parity_rtdetr.py`,
which needs the weights and 50 dev-val frames.
What this file pins is the one bug that gate cannot see coming back: the BGR->RGB
conversion, which is invisible in every shape assertion and costs accuracy silently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vividly.models import factory
from vividly.models.checkpoints import is_lfs_pointer
from vividly.models.factory import build_detector
from vividly.models.rtdetr_backend import INPUT_SIZE, orig_target_sizes, preprocess_bgr

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORD_ONNX = REPO_ROOT / "final_models" / "eyes" / "model.onnx"
SAMPLE_JPG = REPO_ROOT / "tests" / "fixtures" / "sample.jpg"

KEY = "rtdetr_onnx"


def _spec(**overrides) -> dict:
    spec = {
        "backend": KEY,
        "onnx": str(RECORD_ONNX),
        "input_size": [640, 640],
        "device": "cpu",
        "score_thr": 0.3,
    }
    spec.update(overrides)
    return spec


# ------------------------------------------------------------------ registration
def test_the_registry_holds_the_rtdetr_key() -> None:
    assert KEY in factory.DETECTORS
    assert sorted(factory.DETECTORS) == ["rtdetr_onnx", "rtmlib"]


def test_an_unknown_detector_error_now_advertises_the_rtdetr_key() -> None:
    """The KeyError is the discovery mechanism for a closed registry; a new backend that
    does not appear in it is a backend nobody can find from a config error."""
    with pytest.raises(KeyError) as exc:
        build_detector({"backend": "rtdetrv5", "onnx": "anything.onnx"})
    assert KEY in str(exc.value)


def test_importing_the_backend_module_costs_numpy_and_nothing_else() -> None:
    """cv2, Pillow and onnxruntime are imported inside functions, the way `reid_backend`
    does it. That is what lets the refusal tests above run on a box with no inference
    stack, and what keeps a missing weights file a FileNotFoundError there rather than a
    ModuleNotFoundError. A subprocess, because sys.modules here is shared with whatever
    ran before."""
    import subprocess
    import sys

    code = (
        "import sys; import vividly.models.rtdetr_backend as b; "
        "assert b.INPUT_SIZE == (640, 640); "
        "print([m for m in ('cv2', 'PIL', 'onnxruntime', 'torch') if m in sys.modules])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    assert proc.stdout.strip() == "[]"


def test_the_key_routes_to_the_rtdetr_constructor_and_nothing_else(monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setitem(factory.DETECTORS, KEY, lambda **kw: calls.append(kw) or "built")
    other: list[dict] = []
    monkeypatch.setitem(factory.DETECTORS, "rtmlib", lambda **kw: other.append(kw))

    assert build_detector(_spec()) == "built"

    (kw,) = calls
    assert "backend" not in kw, "the selector key must not reach the constructor"
    assert kw["input_size"] == (640, 640), "a YAML list must arrive as a tuple"
    assert kw["device"] == "cpu"
    assert kw["score_thr"] == 0.3
    assert other == [], "the other backend was never touched"


# ------------------------------------------------------- refusals, no weights needed
def test_a_missing_onnx_is_a_filenotfounderror() -> None:
    """Ordered before the onnxruntime import on purpose: on a CI box with no inference
    stack this must still be the honest error, not ModuleNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc:
        build_detector(_spec(onnx="outputs/eyes_b1/does_not_exist.onnx"))
    assert "does_not_exist.onnx" in str(exc.value)


def test_coreml_is_refused_by_name_before_anything_is_opened() -> None:
    with pytest.raises(ValueError) as exc:
        build_detector(_spec(device="mps", onnx="outputs/eyes_b1/does_not_exist.onnx"))
    assert "mps" in str(exc.value)
    assert "CoreML" in str(exc.value), "the refusal must say why, not just that"


def test_a_non_640_input_size_is_refused_rather_than_silently_resized() -> None:
    """The export has static 640x640 spatial dims. Accepting another size here would fail
    deep inside .run() with a rank message that names no config line."""
    with pytest.raises(ValueError) as exc:
        build_detector(_spec(input_size=[512, 512], onnx="outputs/eyes_b1/does_not_exist.onnx"))
    assert "512" in str(exc.value)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_a_score_threshold_outside_zero_to_one_is_refused(bad: float) -> None:
    with pytest.raises(ValueError) as exc:
        build_detector(_spec(score_thr=bad, onnx="outputs/eyes_b1/does_not_exist.onnx"))
    assert "score_thr" in str(exc.value)


# --------------------------------------------------- preprocessing is a pure function
@pytest.mark.parametrize(
    ("image", "why"),
    [
        ("not an array", "a path or PIL image must not be mistaken for a frame"),
        (np.zeros((8, 8), np.uint8), "a greyscale frame has no channel to convert"),
        (np.zeros((8, 8, 3), np.float32), "a float frame would be rescaled twice"),
    ],
)
def test_preprocess_refuses_anything_that_is_not_an_hwc_uint8_frame(image, why: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        preprocess_bgr(image)


def test_preprocess_produces_the_graph_tensor() -> None:
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    blob = preprocess_bgr(np.full((1080, 1920, 3), 128, np.uint8))
    assert blob.shape == (1, 3, INPUT_SIZE[1], INPUT_SIZE[0])
    assert blob.dtype == np.float32
    assert blob.min() >= 0.0 and blob.max() <= 1.0, "the recipe is /255 with no mean/std"
    assert blob.max() == pytest.approx(128 / 255.0)


def test_preprocess_converts_bgr_to_rgb() -> None:
    """The contract that a shape assertion cannot see.

    `base.py` hands this backend OpenCV BGR; the trainer's val pipeline and the I2 grader
    both feed RGB. A frame that is pure blue in BGR must come out with all its energy in
    the last channel of the tensor. Feeding this graph BGR does not raise; it just
    quietly costs accuracy, which is why this is a test and not a comment.
    """
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    blue_bgr = np.zeros((64, 64, 3), np.uint8)
    blue_bgr[..., 0] = 255  # channel 0 is B in OpenCV order

    blob = preprocess_bgr(blue_bgr)[0]

    assert blob[0].max() == pytest.approx(0.0), "R must be empty"
    assert blob[1].max() == pytest.approx(0.0), "G must be empty"
    assert blob[2].min() == pytest.approx(1.0), "B must be the full channel"


def test_orig_target_sizes_is_width_then_height() -> None:
    """[h, w] here would transpose every box on a non-square frame and still look sane.
    The trainer stores orig_size as [w, h] and the deployed postprocessor scales by it."""
    sizes = orig_target_sizes(np.zeros((1080, 1920, 3), np.uint8))
    assert sizes.shape == (1, 2)
    assert sizes.dtype == np.int64
    assert sizes.tolist() == [[1920, 1080]]


def test_preprocess_matches_the_grader_byte_for_byte() -> None:
    """The G1 claim in miniature, on the repo's own fixture and with no weights.

    Path A is this backend's (cv2 decode -> BGR -> RGB -> PIL resize); path B is
    `scripts/eval/eval_detector_v2_onnx.py`'s own `preprocess`, loaded by path because
    scripts/ is not a package. Bitwise equality is the bar: `cv2.resize` in place of PIL's
    antialiased bilinear moves these values by ~0.4 and detection scores by ~0.15.
    """
    import importlib.util
    import sys

    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    grader_path = REPO_ROOT / "scripts" / "eval" / "eval_detector_v2_onnx.py"
    spec = importlib.util.spec_from_file_location("eval_detector_v2_onnx", grader_path)
    assert spec is not None and spec.loader is not None
    grader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = grader
    spec.loader.exec_module(grader)

    mine = preprocess_bgr(cv2.imread(str(SAMPLE_JPG), cv2.IMREAD_COLOR))
    theirs = grader.preprocess(SAMPLE_JPG)

    assert np.array_equal(mine, theirs)


# ------------------------------------------------- the real export, when it is on disk
@pytest.mark.skipif(
    not RECORD_ONNX.is_file() or is_lfs_pointer(RECORD_ONNX),
    reason="final_models/eyes/model.onnx is not on disk (git lfs pull fetches it)",
)
def test_the_export_of_record_builds_and_returns_the_detections_contract() -> None:
    pytest.importorskip("onnxruntime")
    cv2 = pytest.importorskip("cv2")
    from vividly.models.base import Detections, PersonDetector

    detector = build_detector(_spec())

    assert isinstance(detector, PersonDetector)
    assert detector.provenance.shippable == "counsel", (
        "shippability is a question for counsel, never resolved in code"
    )
    dets = detector.detect_one(cv2.imread(str(SAMPLE_JPG), cv2.IMREAD_COLOR))
    assert isinstance(dets, Detections)
    assert dets.xyxy.dtype == np.float32
    assert dets.scores.dtype == np.float32
    assert dets.class_ids.dtype == np.int32
    assert set(np.unique(dets.class_ids)) <= {0}, "person_v2 is single-class"
    assert dets.scores.min(initial=1.0) >= 0.3, "score_thr must be applied by the backend"
    assert len(detector.timings) == 1, "every call bills exactly one Timing"
