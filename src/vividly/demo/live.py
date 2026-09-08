"""Live camera lane: browser frames in, annotated stream out, nothing persisted.

The project never records its own footage, and that rule shapes this module: a live
session is watch-only. Camera frames exist in memory for exactly one pipeline tick; no
MP4 is written, no thumbnail is grabbed, and the job directory holds only numeric
artefacts (alerts.sqlite, latency.jsonl, job.json). This lane is separate from the RTSP
lane: `LiveRTSPSource` in src/vividly/sources/rtsp.py reads a camera stream and is driven
by scripts/live_camera.py, while this is a demo-platform seam driven by the browser's own
camera permission.

The browser captures getUserMedia frames, JPEG-encodes them and pushes each over
``/ws/jobs/<id>/frames``; LiveSource turns that push stream into the pull-based
FrameSource the pipeline already speaks. t_source is wall time since the first frame, so
rule windows run on real seconds no matter how fast the machine processes: a 30 s stillness
requirement and a 12 s convulsion window in the clinical set (down_still_min_s and
convulse_window_s in configs/pipeline/rules.yaml), 5 s and 8 s in the demo profile
(vividly.demo.jobs RULES_PROFILES).

Import discipline: cv2/numpy only inside functions; this module imports heavy-free.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

#: Nominal capture rate the browser aims for. t_source is wall clock, so this number
#: only sizes ring buffers (tracks.fps) and labels the session; it is never a clock.
LIVE_FPS = 10.0

#: The camera permission prompt sits between session start and the first frame.
FIRST_FRAME_TIMEOUT_S = 60.0

#: No frame for this long means the page is gone; the session ends itself.
IDLE_TIMEOUT_S = 10.0

#: A session that nobody stops still ends (runaway-disk/CPU guard; max_s can lower it).
LIVE_MAX_S_DEFAULT = 300.0

MAX_FRAME_BYTES = 4 << 20  # one JPEG frame; anything larger is dropped, not buffered


class LiveStopped(RuntimeError):
    """The session was stopped before the camera delivered its first frame."""


class LiveSource:
    """Push-based FrameSource: a websocket puts JPEG bytes, the pipeline pulls frames.

    The mailbox holds exactly one frame and the newest wins: live mode wants the freshest
    picture, never a backlog, so a slow processing loop drops camera frames instead of
    drifting seconds behind the room.
    """

    def __init__(
        self,
        *,
        first_frame_timeout_s: float = FIRST_FRAME_TIMEOUT_S,
        idle_timeout_s: float = IDLE_TIMEOUT_S,
        max_s: float | None = None,
    ) -> None:
        self.first_frame_timeout_s = first_frame_timeout_s
        self.idle_timeout_s = idle_timeout_s
        self.max_s = max_s
        self.n_received = 0
        self.n_dropped = 0
        self._box: queue.Queue[bytes] = queue.Queue(maxsize=1)
        self._stopped = threading.Event()
        self._info: Any = None
        self._first_image: Any = None

    # -- websocket side ------------------------------------------------------
    def push(self, data: bytes) -> bool:
        """Offer one JPEG frame. Returns False once the session no longer wants frames."""
        if self._stopped.is_set():
            return False
        if not data or len(data) > MAX_FRAME_BYTES:
            self.n_dropped += 1
            return True
        self.n_received += 1
        try:
            self._box.put_nowait(data)
        except queue.Full:
            try:
                self._box.get_nowait()
                self.n_dropped += 1
            except queue.Empty:
                pass
            try:
                self._box.put_nowait(data)
            except queue.Full:  # only under put races; the frame is simply lost
                self.n_dropped += 1
        return True

    def stop(self) -> None:
        """End the session: unblocks any wait, refuses further pushes. Idempotent."""
        self._stopped.set()

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    @property
    def info(self) -> Any:
        return self._info

    # -- pipeline side (FrameSource protocol) --------------------------------
    def open(self) -> Any:
        """Block until the first decodable frame arrives; its size defines the session.

        The pipeline warms its models on (height, width), so open() cannot return until
        a real frame exists. Raises LiveStopped when stop() lands first, RuntimeError on
        timeout (camera permission denied, or the page never sent anything).
        """
        from vividly.sources.base import SourceInfo

        if self._info is not None:
            return self._info
        deadline = time.monotonic() + self.first_frame_timeout_s
        image = None
        while image is None:
            remaining = deadline - time.monotonic()
            if self._stopped.is_set():
                raise LiveStopped("live session stopped before the first camera frame")
            if remaining <= 0:
                raise RuntimeError(
                    "no camera frames arrived (camera permission denied, or the page closed)"
                )
            data = self._next_jpeg(remaining)
            if data is None:
                continue  # the while head re-checks stop and deadline
            image = _decode_jpeg(data)  # None on corrupt bytes: keep waiting
        self._first_image = image
        h, w = image.shape[:2]
        self._info = SourceInfo(fps=LIVE_FPS, width=int(w), height=int(h), n_frames=None)
        return self._info

    def frames(self) -> Any:
        """Yield frames until stop(), idle timeout, or the max_s cap."""
        self.open()
        image = self._first_image
        self._first_image = None
        t0 = time.monotonic()
        index = 0
        while True:
            if image is not None:
                from vividly.sources.base import Frame

                t = 0.0 if index == 0 else time.monotonic() - t0
                if self.max_s is not None and t > self.max_s:
                    return
                yield Frame(index=index, t_source=t, image=image)
                index += 1
            data = self._next_jpeg(self.idle_timeout_s)
            if data is None:
                return
            image = _decode_jpeg(data)  # None skips the corrupt frame and waits again

    def seek(self, index: int) -> None:
        raise RuntimeError("a live camera source cannot seek")

    def close(self) -> None:
        self.stop()

    # -- internals -----------------------------------------------------------
    def _next_jpeg(self, timeout_s: float) -> bytes | None:
        """Next frame, waiting in short slices so stop() is honoured within ~0.25 s."""
        deadline = time.monotonic() + timeout_s
        while not self._stopped.is_set():
            slice_s = min(0.25, deadline - time.monotonic())
            if slice_s <= 0:
                return None
            try:
                return self._box.get(timeout=slice_s)
            except queue.Empty:
                continue
        return None


def _decode_jpeg(data: bytes) -> Any:
    """JPEG bytes -> HxWx3 uint8 BGR, or None when the bytes are not an image."""
    import cv2  # heavy: function-local by package rule
    import numpy as np

    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
