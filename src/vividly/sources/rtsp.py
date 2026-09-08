"""LiveRTSPSource: a live network camera (RTSP/RTMP) as a FrameSource.

This is the working body the base.RTSPSource stub always pointed at. It is kept in a
separate module on purpose: base.RTSPSource stays a dead stub (its test pins that it
raises), and nothing here is reachable from a default pipeline config. The stream URL is
supplied explicitly, on the command line or through the environment, as docs/LIVE_CAMERA.md
shows, which is why this class takes a fully-formed URL and never assembles one from parts.

Standing rule: Vividly records nothing of its own. A live source consumes a hospital's
own camera, or a bring-up rig's, over the network, frame by frame, and writes no file.
The URL carries credentials, so it is read from the runner's --url argument or from the
environment and never committed.

Import discipline: cv2 is darwin-marked in this repo's dev fork, so it is imported inside
functions, never at module level. Linux CI imports this file with no cv2 installed.

Two honest differences from ReplaySource, both intrinsic to a live stream:
  * Unbounded. SourceInfo.n_frames is None and iteration never ends on its own; the caller
    stops it (a frame budget, a duration, a signal).
  * No seeking. seek() raises, because a live camera cannot be scrubbed. t_source is still
    index/fps, so the rules layer sees the same clip-time contract, using fps_fallback when
    the stream does not report a usable frame rate (RTSP commonly reports 0).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from typing import Any

from vividly.sources.base import Frame, FrameSource, SourceInfo

DEFAULT_FPS_FALLBACK = 15.0
_DEFAULT_FFMPEG_OPTS = "rtsp_transport;tcp|stimeout;5000000"


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
        self._cap: Any | None = None
        self._info: SourceInfo | None = None
        self._next_index = 0

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
        import cv2

        raw_fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = raw_fps if (not math.isnan(raw_fps) and raw_fps > 0.0) else self.fps_fallback
        self._cap = cap
        self._info = SourceInfo(
            fps=fps,
            width=round(float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
            height=round(float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            n_frames=None,  # live: unbounded
        )
        self._next_index = 0
        return self._info

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
                continue
            reconnects = 0
            index = self._next_index
            self._next_index += 1
            yield Frame(index=index, t_source=index / fps, image=image)
