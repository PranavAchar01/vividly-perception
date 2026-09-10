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


# ------------------------------------------------------------- wall clock (P4)
class _FakeCap:
    """A capture whose position counter advances one frame per read, with a controllable
    process clock beside it. No cv2, no network."""

    def __init__(self, fps: float = 15.0) -> None:
        self.fps = fps
        self.reads = 0
        self.released = False

    def get(self, prop: int) -> float:
        if prop == 0:  # CAP_PROP_POS_MSEC
            return self.reads * 1000.0 / self.fps
        if prop == 5:  # CAP_PROP_FPS
            return self.fps
        if prop == 3:
            return 640.0
        if prop == 4:
            return 360.0
        return 0.0

    def read(self) -> tuple[bool, object]:
        self.reads += 1
        return True, object()

    def release(self) -> None:
        self.released = True

    def isOpened(self) -> bool:
        return True


def _parse(iso: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def test_every_frame_carries_a_wall_clock_stamp_and_the_epoch_is_recorded(monkeypatch) -> None:
    clock = {"now": 1_800_000_000.0}
    cap = _FakeCap()
    src = LiveRTSPSource("rtsp://cam.invalid/s", clock=lambda: clock["now"])
    monkeypatch.setattr(src, "_connect", lambda: cap)
    info = src.open()
    assert info.n_frames is None and src.source_epoch_utc is not None
    assert abs(_parse(src.source_epoch_utc) - clock["now"]) < 0.01
    frames = src.frames()
    for i in range(3):
        clock["now"] += 1.0 / 15.0
        frame = next(frames)
        assert frame.index == i and frame.wall_utc is not None
        assert abs(_parse(frame.wall_utc) - clock["now"]) < 0.1


def test_a_stall_re_bases_the_epoch_so_the_next_stamp_is_within_a_second(monkeypatch) -> None:
    """A 400 s stall: the position counter kept counting frames, the wall clock did not
    wait for it. The next frame's stamp must follow the real clock, not the counter."""
    clock = {"now": 1_800_000_000.0}
    cap = _FakeCap()
    src = LiveRTSPSource("rtsp://cam.invalid/s", clock=lambda: clock["now"])
    monkeypatch.setattr(src, "_connect", lambda: cap)
    src.open()
    frames = src.frames()
    for _ in range(10):
        clock["now"] += 1.0 / 15.0
        next(frames)
    rebases_before = src.n_rebases
    clock["now"] += 400.0  # the stall: no frames arrive for 400 s
    frame = next(frames)  # the counter says 11 frames, the clock says 400 s later
    assert frame.wall_utc is not None
    assert abs(_parse(frame.wall_utc) - clock["now"]) < 1.0
    assert src.n_rebases == rebases_before + 1
    assert abs(_parse(src.source_epoch_utc or "") - (clock["now"] - 11 / 15.0)) < 0.01
    clock["now"] += 1.0 / 15.0
    frame = next(frames)
    assert abs(_parse(frame.wall_utc or "") - clock["now"]) < 0.1
    assert src.n_rebases == rebases_before + 1  # steady again: no further re-base


def test_a_reconnect_re_bases_the_epoch(monkeypatch) -> None:
    clock = {"now": 1_800_000_000.0}
    caps = [_FakeCap(), _FakeCap()]

    class _Dropping(_FakeCap):
        def read(self):  # type: ignore[override]
            return False, None

    dropping = _Dropping()
    order = [caps[0], caps[1]]
    src = LiveRTSPSource(
        "rtsp://cam.invalid/s", clock=lambda: clock["now"], sleep=lambda s: None, max_reconnects=2
    )
    monkeypatch.setattr(src, "_connect", lambda: order.pop(0))
    src.open()
    src._cap = dropping  # the live capture dies
    clock["now"] += 120.0
    frames = src.frames()
    frame = next(frames)
    assert src._cap is caps[1] and src.n_rebases == 2
    assert abs(_parse(frame.wall_utc or "") - clock["now"]) < 0.1
