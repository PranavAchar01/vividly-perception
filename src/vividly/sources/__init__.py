"""Replay-mode frame and observation sources.

Replay is the default and the only product path; real-time capture is bring-up work, not a
deployment (POLICY.md, section 1). ReplaySource reads a licensed clip off disk;
ImageSequenceSource plays a directory of numbered frames (CAUCAFall ships per-frame stills
alongside its containers) at a rate the caller declares; SyntheticDetectionSource replays a
JSONL fixture so the rules, store and dashboard run with zero footage; RTSPSource is a
placeholder that raises on every call; LiveRTSPSource is the live camera source, reachable
only from an explicit URL passed on the command line or in the environment
(scripts/live_camera.py, docs/LIVE_CAMERA.md), never a default run.
"""

from vividly.sources.base import (
    RTSP_NOT_IMPLEMENTED,
    Frame,
    FrameSource,
    RTSPSource,
    SourceInfo,
    resolve_path,
)
from vividly.sources.image_sequence import ImageSequenceSource, frame_count, list_frames
from vividly.sources.replay import ReplaySource, pacing_delay
from vividly.sources.rtsp import LiveRTSPSource
from vividly.sources.synthetic import SyntheticDetectionSource, write_observations_jsonl

__all__ = [
    "RTSP_NOT_IMPLEMENTED",
    "Frame",
    "FrameSource",
    "ImageSequenceSource",
    "LiveRTSPSource",
    "RTSPSource",
    "ReplaySource",
    "SourceInfo",
    "SyntheticDetectionSource",
    "frame_count",
    "list_frames",
    "pacing_delay",
    "resolve_path",
    "write_observations_jsonl",
]
