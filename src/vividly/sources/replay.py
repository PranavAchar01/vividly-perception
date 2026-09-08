"""ReplaySource: a video file played back as a FrameSource, replay only.

Two properties this module owes the rest of the pipeline:

  * frame-exact seeking. seek(N) followed by the next read returns source frame N, so
    a dashboard scrub and a rule onset timestamp refer to the same picture. cv2's
    CAP_PROP_POS_FRAMES does this on every container in use here; when a backend refuses
    the set, the source falls back to rewind-and-decode-forward rather than silently
    drifting.
  * clip time, not wall time. t_source is index / fps exactly, whatever the pacing is.

Pacing is a playback convenience for the demo, never an input to the rules. rate=None
runs as fast as the decoder allows; rate=1.0 is wall clock; rate=2.0 is twice speed.
The sleep computation is the pure function pacing_delay(), and the clock and sleep
functions are injectable so the loop is testable without a wall-clock test.

Import discipline: cv2 is darwin-marked in this repo's dev fork, so it is imported
inside functions, never at module level. linux CI imports this file with no cv2.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from vividly.sources.base import Frame, FrameSource, SourceInfo, resolve_path


def pacing_delay(
    rate: float | None,
    fps: float,
    frames_since_epoch: int,
    elapsed_s: float,
) -> float:
    """Seconds to sleep before emitting a frame, given how far behind the clock it is.

    frames_since_epoch counts from the frame that started the current pacing epoch
    (iteration start, or the most recent seek). elapsed_s is the clock time since that
    same epoch. Returns 0.0 when playback is already late: it never tries to catch up
    by dropping frames, it just stops waiting.
    """
    if rate is None or rate <= 0.0 or fps <= 0.0:
        return 0.0
    target_s = frames_since_epoch / (fps * rate)
    return max(0.0, target_s - elapsed_s)


class ReplaySource(FrameSource):
    """Play a video file back as a FrameSource.

    clock and sleep are keyword-only injection seams for tests. The positional
    signature is exactly (path, rate), and callers depend on that order.
    """

    def __init__(
        self,
        path: str | Path,
        rate: float | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate is not None and rate <= 0.0:
            raise ValueError(
                f"rate must be positive or None (None means as fast as possible), got {rate!r}"
            )
        self.path = resolve_path(path)
        self.rate = rate
        self._clock = clock
        self._sleep = sleep
        self._cap: Any | None = None
        self._info: SourceInfo | None = None
        self._next_index: int = 0
        self._epoch_t: float | None = None
        self._epoch_index: int = 0

    # -- lifecycle ----------------------------------------------------------
    def open(self) -> SourceInfo:
        """Open the clip and report what it is. Idempotent while open."""
        if self._cap is not None and self._info is not None:
            return self._info

        # Existence first, cv2 second: a missing clip must be FileNotFoundError on every
        # platform, including the Linux CI fork where cv2 is absent and the import would
        # otherwise mask it as ModuleNotFoundError.
        if not self.path.exists():
            raise FileNotFoundError(f"replay clip not found: {self.path}")

        import cv2

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"OpenCV could not open replay clip: {self.path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if math.isnan(fps) or fps <= 0.0:
            cap.release()
            raise ValueError(
                f"replay clip reports fps={fps!r}: t_source is index / fps, so a clip "
                f"without a frame rate cannot be replayed ({self.path})"
            )
        raw_n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_frames = round(raw_n) if not math.isnan(raw_n) and raw_n > 0 else None

        self._cap = cap
        self._info = SourceInfo(
            fps=fps,
            width=round(float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
            height=round(float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            n_frames=n_frames,
        )
        self._next_index = 0
        self._reset_pacing_epoch()
        return self._info

    def close(self) -> None:
        """Release the capture. Safe to call twice, and safe to call before open()."""
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._info = None
        self._reset_pacing_epoch()

    def __enter__(self) -> ReplaySource:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- reading ------------------------------------------------------------
    @property
    def info(self) -> SourceInfo | None:
        """SourceInfo once open, None otherwise."""
        return self._info

    @property
    def next_index(self) -> int:
        """Source frame number the next read will return."""
        return self._next_index

    def frames(self) -> Iterator[Frame]:
        """Yield every frame from the current read head to the end of the clip.

        Opens the clip if it is not open yet. A seek() performed by the consumer
        between yields is honoured on the following iteration, including its pacing
        epoch reset, so a dashboard scrub does not make playback sprint to catch up.
        """
        info = self.open()
        while True:
            cap = self._cap
            if cap is None:  # closed by the consumer while iterating
                return
            ok, image = cap.read()
            if not ok:
                return
            index = self._next_index
            self._next_index = index + 1
            if self._epoch_t is None:
                self._epoch_t = self._clock()
                self._epoch_index = index
            delay = pacing_delay(
                self.rate, info.fps, index - self._epoch_index, self._clock() - self._epoch_t
            )
            if delay > 0.0:
                self._sleep(delay)
            yield Frame(index=index, t_source=index / info.fps, image=image)

    def seek(self, index: int) -> None:
        """Move the read head so the next read returns source frame `index`, exactly."""
        if index < 0:
            raise ValueError(f"seek index must be >= 0, got {index}")

        import cv2

        self.open()
        cap = self._cap
        assert cap is not None  # open() either set it or raised
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        reported = float(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if math.isnan(reported) or round(reported) != index:
            self._seek_by_decode(cap, index)
        self._next_index = index
        self._reset_pacing_epoch()

    # -- internals ----------------------------------------------------------
    def _seek_by_decode(self, cap: Any, index: int) -> None:
        """Fallback for backends whose POS_FRAMES set is advisory: rewind, then grab.

        grab() decodes without converting to an array, so this costs a decode pass and
        no allocation. If the clip is shorter than `index` the loop stops early and the
        next read simply reports end of file.
        """
        import cv2

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
        for _ in range(index):
            if not cap.grab():
                return

    def _reset_pacing_epoch(self) -> None:
        self._epoch_t = None
        self._epoch_index = self._next_index
