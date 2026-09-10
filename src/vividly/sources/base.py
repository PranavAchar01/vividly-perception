"""Frame source contracts for the replay pipeline.

Replay is the only product path, and real-time capture from a site's own cameras is
bring-up work rather than a deployment (POLICY.md, section 1). The standing rule is that
Vividly records no footage of its own and never will, so every frame this layer sees
comes out of a file that somebody else licensed. The replay-mode structure is the only
shape that rule allows, which is why the seam is written down here at all.

RTSPSource is a named, deliberately dead stub. It exists so the live-camera seam is
visible in code review, and it raises on every method so nobody can point this at a
patient today. It stays dead: the working live-camera body is LiveRTSPSource in
src/vividly/sources/rtsp.py, which the bring-up runner drives.

Import discipline: numpy and cv2 are darwin-marked in this repo's dev fork, so linux
CI imports this module with neither installed. Nothing here may import either at
module level. The numpy annotation on Frame.image is a string under TYPE_CHECKING.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import numpy as np

# The single message every RTSPSource method raises with. Tests pin it. No runner path
# constructs RTSPSource; the live-camera body is LiveRTSPSource in sources/rtsp.py.
RTSP_NOT_IMPLEMENTED = (
    "RTSPSource is a placeholder that raises; the live camera source is "
    "vividly.sources.rtsp.LiveRTSPSource"
)


@dataclass(frozen=True)
class SourceInfo:
    """What a source knows about itself once opened.

    n_frames is None when the container does not report a frame count (a stream, or a
    file with a broken index). Callers must not assume a length.
    """

    fps: float
    width: int
    height: int
    n_frames: int | None


@dataclass
class Frame:
    """One decoded frame.

    index is the source frame number, counted from 0 at the start of the file, not from
    the start of iteration. t_source is index / fps exactly: it is clip time, not wall
    time, so a paced replay and an unpaced replay produce identical timestamps and the
    rules layer cannot tell them apart.

    image is HxWx3 uint8 BGR, matching the model layer's numpy in / numpy out rule.

    wall_utc is the read wall time, RFC 3339 UTC with milliseconds, stamped by a live
    source (LiveRTSPSource, design P4) and None for every file source: a replay has no
    wall clock of its own, and inventing one would make a replayed rule window look live.
    """

    index: int
    t_source: float
    image: np.ndarray
    wall_utc: str | None = None


@runtime_checkable
class FrameSource(Protocol):
    """A pull-based frame source.

    Lifecycle: open() once, then frames() to iterate, seek(index) at any time to move
    the read head, close() when done. A source may be re-opened after close().
    """

    def open(self) -> SourceInfo: ...
    def frames(self) -> Iterator[Frame]: ...
    def seek(self, index: int) -> None: ...
    def close(self) -> None: ...


class RTSPSource:
    """Interface stub only. Every method raises NotImplementedError.

    Kept in the tree so the live-camera seam is reviewable and so no later change
    invents a second, differently shaped one. Nothing in this class touches a network.
    """

    def __init__(self, url: str) -> None:
        self.url = url

    def open(self) -> SourceInfo:
        raise NotImplementedError(RTSP_NOT_IMPLEMENTED)

    def frames(self) -> Iterator[Frame]:
        raise NotImplementedError(RTSP_NOT_IMPLEMENTED)

    def seek(self, index: int) -> None:
        raise NotImplementedError(RTSP_NOT_IMPLEMENTED)

    def close(self) -> None:
        raise NotImplementedError(RTSP_NOT_IMPLEMENTED)


def resolve_path(path: str | Path) -> Path:
    """Normalise a config-supplied path without touching the filesystem."""
    return Path(path).expanduser()
