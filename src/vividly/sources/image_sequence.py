"""ImageSequenceSource: a directory of numbered still frames played back as a FrameSource.

Why it exists. CAUCAFall is the one registered event corpus that carries its imagery as a
directory of numbered stills rather than as a video file alone, and a directory of stills
records no frame rate. Every CAUCAFall sequence directory holds per-frame stills beside
the source video, and the rate that clocks them is the container-measured 20.00 fps written
down in configs/datasets/caucafall.yaml, not the rate the source paper declares. The rate
therefore comes from the caller, which reads it from the registry entry, and t_source is
index divided by fps exactly as in ReplaySource: clip time, never wall time, so the rules
layer cannot tell a paced replay from an unpaced one, or a frame directory from a clip.
That is why fps is a required argument with no default: a guessed rate would quietly put
every rule window on the wrong clock, and no number here is guessed where it can be
measured.

Ordering is by natural sort of the file name (numeric runs compare as numbers), so
`frame_2.jpg` precedes `frame_10.jpg`, and zero-padded names sort the same way they would
lexically. The order is fixed at open() and the file list is exposed, so a caller can check
it against an annotation document before trusting the clock.

Import discipline: cv2 is darwin-marked in this repo's dev fork, so it is imported inside
functions, never at module level. linux CI imports this file with no cv2.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from vividly.sources.base import Frame, FrameSource, SourceInfo, resolve_path
from vividly.sources.replay import pacing_delay

_NUMERIC_RUN = re.compile(r"(\d+)")


def natural_key(name: str) -> tuple[Any, ...]:
    """Sort key that compares digit runs numerically and everything else as text."""
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in _NUMERIC_RUN.split(name)
        if part != ""
    )


def list_frames(directory: str | Path, pattern: str = "*.jpg") -> tuple[Path, ...]:
    """Every file in `directory` matching `pattern`, in playback order. Never recursive.

    Raises FileNotFoundError when the directory does not exist and ValueError when nothing
    matches: an empty sequence is a configuration mistake, not a zero-frame clip.
    """
    root = resolve_path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"frame directory not found: {root}")
    if not pattern or "/" in pattern or "\\" in pattern or ".." in pattern:
        raise ValueError(f"pattern must be a plain file glob such as '*.jpg', got {pattern!r}")
    files = sorted(
        (p for p in root.glob(pattern) if p.is_file()), key=lambda p: natural_key(p.name)
    )
    if not files:
        raise ValueError(f"no frames match {pattern!r} in {root}")
    return tuple(files)


class ImageSequenceSource(FrameSource):
    """Play a directory of still frames back as a FrameSource at a declared rate.

    `fps` is the sequence's own frame rate and has no default on purpose (see the module
    docstring). `pattern` selects the frame files, because a frame directory often holds
    more than the frames: CAUCAFall, for instance, stores a YOLO label `.txt` beside every
    still and needs `*.png`. `rate` paces playback exactly as ReplaySource does and is never
    an input to the rules.
    """

    def __init__(
        self,
        directory: str | Path,
        fps: float,
        *,
        pattern: str = "*.jpg",
        rate: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError(
                f"fps must be a finite positive frame rate, got {fps!r}: no frame directory "
                "records its own rate, so the caller must supply the dataset's"
            )
        if rate is not None and rate <= 0.0:
            raise ValueError(
                f"rate must be positive or None (None means as fast as possible), got {rate!r}"
            )
        self.directory = resolve_path(directory)
        self.fps = fps
        self.pattern = pattern
        self.rate = rate
        self._clock = clock
        self._sleep = sleep
        self._files: tuple[Path, ...] = ()
        self._info: SourceInfo | None = None
        self._next_index: int = 0
        self._epoch_t: float | None = None
        self._epoch_index: int = 0

    # -- lifecycle ----------------------------------------------------------
    def open(self) -> SourceInfo:
        """List the frames, decode the first for its size, report. Idempotent while open."""
        if self._info is not None:
            return self._info
        files = list_frames(self.directory, self.pattern)
        first = _read_image(files[0])
        self._files = files
        self._info = SourceInfo(
            fps=self.fps,
            width=int(first.shape[1]),
            height=int(first.shape[0]),
            n_frames=len(files),
        )
        self._next_index = 0
        self._reset_pacing_epoch()
        return self._info

    def close(self) -> None:
        """Forget the listing. Safe to call twice, and safe to call before open()."""
        self._files = ()
        self._info = None
        self._next_index = 0
        self._reset_pacing_epoch()

    def __enter__(self) -> ImageSequenceSource:
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
    def files(self) -> tuple[Path, ...]:
        """The frames in playback order, empty until open()."""
        return self._files

    @property
    def next_index(self) -> int:
        """Source frame number the next read will return."""
        return self._next_index

    def frames(self) -> Iterator[Frame]:
        """Yield every frame from the read head to the end of the listing.

        Opens the directory if it is not open yet. A seek() between yields is honoured
        on the following iteration, with its pacing epoch reset. A frame that fails to
        decode raises: a hole in a numbered sequence is a broken input, and skipping it
        would shift every later timestamp.
        """
        info = self.open()
        while True:
            files = self._files
            if not files:  # closed by the consumer while iterating
                return
            index = self._next_index
            if index >= len(files):
                return
            image = _read_image(files[index])
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
        """Move the read head so the next read returns source frame `index`, exactly.

        Seeking past the end is allowed and makes the next read report end of sequence,
        which is what ReplaySource does on a clip as well.
        """
        if index < 0:
            raise ValueError(f"seek index must be >= 0, got {index}")
        self.open()
        self._next_index = index
        self._reset_pacing_epoch()

    # -- internals ----------------------------------------------------------
    def _reset_pacing_epoch(self) -> None:
        self._epoch_t = None
        self._epoch_index = self._next_index


def _read_image(path: Path) -> Any:
    """Decode one frame as HxWx3 uint8 BGR, raising on a missing or undecodable file."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not decode frame: {path}")
    return image


def frame_count(directory: str | Path, pattern: str = "*.jpg") -> int:
    """How many frames list_frames would play. Decodes nothing."""
    return len(list_frames(directory, pattern))


__all__: Sequence[str] = (
    "ImageSequenceSource",
    "frame_count",
    "list_frames",
    "natural_key",
)
