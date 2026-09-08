"""LiveRTSPSource tests: no network, no camera.

The live source cannot be exercised end to end in CI (there is no camera), so these
pin the parts that must hold with the network absent: the protocol shape, the
forward-only contract, credential hygiene in repr, argument validation, and the
module-level import discipline that the whole sources package owes linux CI.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from vividly.sources import LiveRTSPSource
from vividly.sources.base import FrameSource

RTSP_PY = Path(__file__).resolve().parents[1] / "src" / "vividly" / "sources" / "rtsp.py"


def test_live_rtsp_source_is_a_frame_source() -> None:
    assert isinstance(LiveRTSPSource("rtsp://cam.invalid/stream"), FrameSource)


def test_live_rtsp_source_is_forward_only() -> None:
    src = LiveRTSPSource("rtsp://cam.invalid/stream")
    with pytest.raises(NotImplementedError, match="cannot seek"):
        src.seek(0)


def test_repr_never_leaks_the_credentialed_url() -> None:
    secret = "rtsp://admin:hunter2@10.0.0.9:554/h264Preview_01_main"
    src = LiveRTSPSource(secret)
    text = repr(src)
    assert "hunter2" not in text and "admin" not in text
    assert "scheme=rtsp" in text


@pytest.mark.parametrize("bad", ["", None, 123])
def test_rejects_a_missing_or_nonstring_url(bad) -> None:
    with pytest.raises(ValueError, match="URL"):
        LiveRTSPSource(bad)  # type: ignore[arg-type]


def test_rejects_a_nonpositive_fps_fallback() -> None:
    with pytest.raises(ValueError, match="fps_fallback"):
        LiveRTSPSource("rtsp://cam.invalid/s", fps_fallback=0.0)


def test_rtsp_module_has_no_top_level_numpy_or_cv2() -> None:
    """Same discipline the rest of sources/ owes: linux CI imports this with no cv2."""
    tree = ast.parse(RTSP_PY.read_text(encoding="utf-8"))
    for node in tree.body:  # module level only
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        assert "cv2" not in names and "numpy" not in names, f"top-level heavy import: {names}"
