"""LiveRTSPSource: a live network camera (RTSP/RTMP) as a FrameSource.

This is the working body the base.RTSPSource stub always pointed at. It is kept in a
separate module on purpose: base.RTSPSource stays a dead stub (its test pins that it
raises), and nothing here is reachable from a default pipeline config. The stream URL is
supplied explicitly, on the command line or through the environment, as docs/LIVE_CAMERA.md
shows, which is why this class takes a fully-formed URL and never assembles one from parts.

Standing rule: Vividly records nothing of its own. A live source consumes a hospital's
own camera, or a bring-up rig's, over the network, frame by frame, and writes no file.
The URL carries credentials, so it is read from the runner's --url argument or from the
environment and never committed. The one downstream consumer that keeps anything is the
Clip Library profile (POLICY.md owner decision, 2026-09-09): event Clips only, on premises,
cut by a separate process from its own memory ring; nothing from a bring-up rig enters any
corpus and the rig subject is the founder alone.

Import discipline: cv2 is darwin-marked in this repo's dev fork, so it is imported inside
functions, never at module level. Linux CI imports this file with no cv2 installed.

Two honest differences from ReplaySource, both intrinsic to a live stream:
  * Unbounded. SourceInfo.n_frames is None and iteration never ends on its own; the caller
    stops it (a frame budget, a duration, a signal).
  * No seeking. seek() raises, because a live camera cannot be scrubbed. t_source is still
    index/fps, so the rules layer sees the same clip-time contract, using fps_fallback when
    the stream does not report a usable frame rate (RTSP commonly reports 0).

Wall clock (Clip Library design P4, 2026-09-09). Every Frame carries `wall_utc`: the
stream position (CAP_PROP_POS_MSEC) added to the epoch taken when the capture opened,
cross-checked against the process clock on every read and re-based when the two disagree
by more than `rebase_tolerance_s` (a stall, a reconnect, a camera whose position counter
jumped). `source_epoch_utc` is the epoch currently in force. The camera reports no usable
frame rate and reconnect gaps of minutes have been observed, so an offset computed from
frame counts would drift without bound; the wall stamp is what a spool record's window is
measured from, and the Clip Library cuts its ring by that clock alone.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from typing import Any

from vividly.sources.base import Frame, FrameSource, SourceInfo

DEFAULT_FPS_FALLBACK = 15.0
_DEFAULT_FFMPEG_OPTS = "rtsp_transport;tcp|stimeout;5000000"
#: cv2.CAP_PROP_* values, stated here so `open` and `frames` never import cv2 (a test
#: drives them with a fake capture on a runner that has none).
_CAP_PROP_POS_MSEC = 0
_CAP_PROP_FRAME_WIDTH = 3
_CAP_PROP_FRAME_HEIGHT = 4
_CAP_PROP_FPS = 5
DEFAULT_REBASE_TOLERANCE_S = 1.0


def _utc_iso(epoch_s: float) -> str:
    from datetime import UTC, datetime

    return (
        datetime.fromtimestamp(epoch_s, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


class LiveRTSPSource(FrameSource):
    """A live RTSP/RTMP stream played as a FrameSource.

    url is fully formed, credentials included, and is never logged by this class. On a
    transient read failure the source reconnects up to `max_reconnects` times (with a
    short backoff) before giving up, because a real camera hiccups and a monitoring feed
    that dies on the first dropped packet is useless.
    """

    def __init__(
        self,
        url: str,
        *,
        fps_fallback: float = DEFAULT_FPS_FALLBACK,
        max_reconnects: int = 5,
        reconnect_backoff_s: float = 1.0,
        ffmpeg_opts: str = _DEFAULT_FFMPEG_OPTS,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        rebase_tolerance_s: float = DEFAULT_REBASE_TOLERANCE_S,
    ) -> None:
        if not url or not isinstance(url, str):
            raise ValueError("LiveRTSPSource needs a non-empty stream URL")
        if fps_fallback <= 0.0:
            raise ValueError(f"fps_fallback must be > 0, got {fps_fallback}")
        self.url = url
        self.fps_fallback = fps_fallback
        self.max_reconnects = max_reconnects
        self.reconnect_backoff_s = reconnect_backoff_s
        self.ffmpeg_opts = ffmpeg_opts
        self._sleep = sleep
        self._clock = clock
        self.rebase_tolerance_s = rebase_tolerance_s
        self._cap: Any | None = None
        self._info: SourceInfo | None = None
        self._next_index = 0
        self._epoch_s: float | None = None
        self.source_epoch_utc: str | None = None
        self.n_rebases = 0

    def __repr__(self) -> str:  # pragma: no cover - never leaks the credentialed URL
        scheme = self.url.split("://", 1)[0] if "://" in self.url else "?"
        return f"LiveRTSPSource(scheme={scheme}, fps_fallback={self.fps_fallback})"

    # -- lifecycle ----------------------------------------------------------
    def _connect(self) -> Any:
        import cv2

        # FFMPEG options must be set before the capture is constructed. setdefault, so an
        # OPENCV_FFMPEG_CAPTURE_OPTIONS already in the environment takes precedence over
        # the ffmpeg_opts argument, and the first value set here holds for the rest of the
        # process: a second source constructed with different options inherits the first's.
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", self.ffmpeg_opts)
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("could not open live stream (check URL, network, credentials)")
        return cap

    def open(self) -> SourceInfo:
        if self._cap is not None and self._info is not None:
            return self._info
        import math

        cap = self._connect()
        raw_fps = float(cap.get(_CAP_PROP_FPS))
        fps = raw_fps if (not math.isnan(raw_fps) and raw_fps > 0.0) else self.fps_fallback
        self._cap = cap
        self._info = SourceInfo(
            fps=fps,
            width=round(float(cap.get(_CAP_PROP_FRAME_WIDTH))),
            height=round(float(cap.get(_CAP_PROP_FRAME_HEIGHT))),
            n_frames=None,  # live: unbounded
        )
        self._next_index = 0
        self._rebase(cap)
        return self._info

    # -- wall clock ---------------------------------------------------------
    @staticmethod
    def _position_s(cap: Any) -> float:
        try:
            pos = float(cap.get(_CAP_PROP_POS_MSEC))
        except (TypeError, ValueError):
            return 0.0
        return pos / 1000.0 if pos == pos and pos > 0.0 else 0.0

    def _rebase(self, cap: Any) -> None:
        """Epoch = now - stream position, taken at open and on every reconnect or stall."""
        self._epoch_s = self._clock() - self._position_s(cap)
        self.source_epoch_utc = _utc_iso(self._epoch_s)
        self.n_rebases += 1

    def _wall_now(self, cap: Any) -> float:
        now = self._clock()
        if self._epoch_s is None:
            self._rebase(cap)
        assert self._epoch_s is not None
        wall = self._epoch_s + self._position_s(cap)
        if abs(wall - now) > self.rebase_tolerance_s:
            self._rebase(cap)
            wall = now
        return wall

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._info = None

    def seek(self, index: int) -> None:
        raise NotImplementedError("a live camera cannot seek; LiveRTSPSource is forward-only")

    # -- iteration ----------------------------------------------------------
    def frames(self) -> Iterator[Frame]:
        if self._cap is None or self._info is None:
            self.open()
        assert self._info is not None
        fps = self._info.fps
        reconnects = 0
        while True:
            ok, image = self._cap.read()  # type: ignore[union-attr]
            if not ok or image is None:
                if reconnects >= self.max_reconnects:
                    return
                reconnects += 1
                self._sleep(self.reconnect_backoff_s)
                if self._cap is not None:
                    self._cap.release()
                self._cap = self._connect()
                self._rebase(self._cap)
                continue
            reconnects = 0
            index = self._next_index
            self._next_index += 1
            yield Frame(
                index=index,
                t_source=index / fps,
                image=image,
                wall_utc=_utc_iso(self._wall_now(self._cap)),
            )
