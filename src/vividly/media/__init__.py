"""Media plumbing: the one ffmpeg surface.

`vividly.media.ffmpeg` locates the binary, probes a clip and runs an ffmpeg argv. Argv lists
only, never a shell string, so the exact command is recoverable from the returned record.

Only the standard library is imported here: the converters, the evaluation runner and the
demo platform all depend on this package, and none of them may be forced to install a model
stack to normalise a video.
"""

from vividly.media.ffmpeg import (
    BASE_FLAGS,
    ffmpeg_bin,
    ffmpeg_version,
    ffprobe_bin,
    probe_video,
    run_argv,
)

__all__ = [
    "BASE_FLAGS",
    "ffmpeg_bin",
    "ffmpeg_version",
    "ffprobe_bin",
    "probe_video",
    "run_argv",
]
