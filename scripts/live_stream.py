"""Live MJPEG stream: RTMDet + RTMPose on a real RTSP feed, served as real-time video.

Decoupled 3-stage architecture so the feed is smooth and crisp enough to demo:

  Capture thread    decode the camera (4K H.265 via VideoToolbox) -> downscale to the
                    display size -> keep only the LATEST frame (drop stale; cap.read is
                    never on the inference path).
  Infer thread      run RTMDet + RTMPose on the latest frame as fast as the models allow
                    (~25-30 fps on Apple Silicon CPU) and publish the freshest overlay.
  Compositor thread at a fixed display FPS, draw the freshest overlay onto the freshest
                    frame, JPEG-encode once, and publish to all HTTP clients.

The video the browser sees updates at the display rate (smooth), while the AI overlay
refreshes at the inference rate — the standard real-time-CV decoupling. Records nothing
(D1). The stream URL carries credentials and is read from --url or REOLINK_RTSP_URL,
never committed.

    REOLINK_RTSP_URL='rtsp://admin:@172.16.25.10:554/h265Preview_01_main' \
        uv run python scripts/live_stream.py --host 0.0.0.0 --display-width 1280
"""

from __future__ import annotations

import argparse
import contextlib
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vividly.analysis import (  # noqa: E402
    FallAnalyzer,
    FallConfig,
    FallState,
    SeizureAnalyzer,
    SeizureConfig,
    SeizureState,
)
from vividly.models.factory import build_detector, build_pose  # noqa: E402
from vividly.models.tracker import ByteTrack  # noqa: E402
from vividly.sources import LiveRTSPSource  # noqa: E402


@dataclass
class TrackView:
    """One tracked person for the compositor to draw: identity, box, skeleton, and the
    per-person seizure and fall state."""

    tid: int
    box: np.ndarray
    kp: np.ndarray
    kp_sc: np.ndarray
    seizure: SeizureState
    fall: FallState


SKELETON: tuple[tuple[int, int], ...] = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (11, 12), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
)  # fmt: skip

# Brand palette (BGR).
_ACCENT = (242, 157, 169)  # lavender
_BOX = (130, 235, 150)  # mint (idle / monitoring)
_JOINT = (255, 120, 180)  # magenta
_BONE = (235, 170, 140)  # periwinkle
_INK = (24, 22, 18)
_AMBER = (60, 200, 250)  # analyzing / falling
_RED = (60, 60, 245)  # confirmed seizure / fall


def _bracket(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color, t: int = 2) -> None:
    """Corner-bracket box — reads more 'product' than a plain rectangle."""
    L = max(12, int(0.18 * min(x2 - x1, y2 - y1)))
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, t, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, t, cv2.LINE_AA)


def _severity(v: TrackView) -> int:
    """Higher = more urgent. Drives box color and which alert wins the banner."""
    if v.fall.state == "fall":
        return 5
    if v.seizure.state == "seizure":
        return 4
    if v.fall.state == "falling":
        return 3
    if v.seizure.state == "candidate":
        return 2
    return 0


def _track_style(v: TrackView) -> tuple[tuple[int, int, int], str]:
    """(box color, short tag) for a tracked person given their state."""
    s = _severity(v)
    if v.fall.state == "fall":
        return _RED, "FALL"
    if v.seizure.state == "seizure":
        return _RED, f"SEIZURE {v.seizure.dominant_hz:.0f}Hz"
    # Nothing amber. A building state used to tint the box and print "analyzing", which on
    # film reads as the system hedging, and it painted clutter yellow for a second at a
    # time. The box stays neutral until an event confirms, and then it is unmistakable.
    return _BOX, ""


def draw_skeleton(img: np.ndarray, kps: np.ndarray, sc: np.ndarray, kp_thr: float) -> None:
    for a, b in SKELETON:
        if sc[a] > kp_thr and sc[b] > kp_thr:
            cv2.line(
                img, tuple(kps[a].astype(int)), tuple(kps[b].astype(int)), _BONE, 2, cv2.LINE_AA
            )
    for j in range(len(kps)):
        if sc[j] > kp_thr:
            cv2.circle(img, tuple(kps[j].astype(int)), 3, _JOINT, -1, cv2.LINE_AA)


def draw_tracks(img: np.ndarray, tracks: list[TrackView], kp_thr: float) -> None:
    """Per-person box (colored by state), skeleton, and an id + state tag."""
    for v in tracks:
        x1, y1, x2, y2 = v.box.astype(int)
        color, tag = _track_style(v)
        _bracket(img, x1, y1, x2, y2, color, 2)
        draw_skeleton(img, v.kp, v.kp_sc, kp_thr)
        label = f"#{v.tid}" + (f"  {tag}" if tag else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.putText(
            img, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _INK, 1, cv2.LINE_AA
        )


def draw_hud(
    img: np.ndarray,
    persons: int,
    pose_fps: float,
    disp_fps: float,
    det_fps: float = 0.0,
    lat_ms: float = 0.0,
    mode: str = "demo",
) -> None:
    """Two overlays from one function.

    `demo` is what a camera pointed at this screen should see: the wordmark, how many people
    are in view, and the time. Frame rates and millisecond counters are engineering
    instruments, and on film they read as a system talking about itself rather than about
    the room. `debug` puts them back for the person tuning it.
    """
    w = img.shape[1]
    bar = img[0:34, 0:w]
    cv2.addWeighted(np.full_like(bar, _INK), 0.55, bar, 0.45, 0, bar)
    cv2.putText(
        img,
        "VIVIDLY  |  live perception",
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        _ACCENT,
        2,
        cv2.LINE_AA,
    )
    people = "no one in view" if persons == 0 else ("1 person" if persons == 1 else f"{persons} people")
    if mode == "debug":
        right = (
            f"{people}   pose {pose_fps:4.1f}   det {det_fps:4.1f}   "
            f"video {disp_fps:4.1f} fps   delay {lat_ms:3.0f} ms"
        )
    else:
        right = f"{people}     {time.strftime('%H:%M:%S')}"
    (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.putText(
        img,
        right,
        (w - tw - 12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )


def _fit_scale(text: str, target_w: int, base: float, thick: int) -> float:
    """Largest font scale (<= base) whose rendered text fits target_w."""
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, base, thick)
    return base if tw <= target_w else base * target_w / tw


def draw_banner(img: np.ndarray, tracks: list[TrackView]) -> None:
    """One aggregate top alert for the whole scene: whichever tracked person is in the most
    urgent state wins. Silent when everyone is idle; amber while an event builds; red alarm
    + border once a seizure or fall is CONFIRMED (for person #id)."""
    if not tracks:
        return
    worst = max(tracks, key=_severity)
    sev = _severity(worst)
    if sev < 4:  # idle, or an event still building: the screen stays quiet
        return
    w = img.shape[1]
    if sev >= 4:  # a CONFIRMED emergency
        if worst.fall.state == "fall":
            banner = f"FALL DETECTED   person #{worst.tid}   {worst.fall.held_s:0.0f}s"
        else:
            hz = worst.seizure.dominant_hz
            hz_txt = f"{hz:.1f} Hz jerking" if hz > 0 else "clonic jerking"
            banner = (
                f"SEIZURE DETECTED   person #{worst.tid}   {hz_txt}   {worst.seizure.held_s:0.0f}s"
            )
        cv2.rectangle(img, (0, 0), (w - 1, img.shape[0] - 1), _RED, 8)
        fs = _fit_scale(banner, w - 32, 0.8, 2)
        (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        cx = (w - tw) // 2
        cv2.rectangle(img, (cx - 16, 42), (cx + tw + 16, 42 + th + 22), _RED, -1)
        cv2.putText(
            img,
            banner,
            (cx, 42 + th + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            fs,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    # No banner for a building state: only a confirmed event says anything on screen.


class Rate:
    """Honest throughput: frames completed within a trailing window. Instantaneous 1/dt
    is far too noisy to put on screen (it swung to 128 fps on a 25 fps stream)."""

    def __init__(self, window: float = 1.5) -> None:
        self.window = window
        self._ts: list[float] = []

    def tick(self) -> float:
        now = time.time()
        self._ts.append(now)
        cut = now - self.window
        while self._ts and self._ts[0] < cut:
            self._ts.pop(0)
        span = now - self._ts[0]
        return (len(self._ts) - 1) / span if span > 0 else 0.0


class Shared:
    """Cross-thread state. The raw frame and the published JPEG each have a Condition; the
    overlay tuple is guarded by `_lock`. `infer_seq`/`seizure` are single-writer (infer
    thread) / single-reader (compositor) references, published atomically without a lock."""

    def __init__(self) -> None:
        # raw frame is guarded by its own condition so the compositor can block until a
        # genuinely new camera frame arrives (event-driven -> paces to the true stream
        # rate, one composite per frame, no duplicates, no busy-spin).
        self._raw_cond = threading.Condition()
        self.raw: np.ndarray | None = None
        self.raw_seq = 0
        self._lock = threading.Lock()
        self._tracks: list = []  # latest per-person TrackViews for the compositor
        self.infer_fps = 0.0
        # published jpeg for clients
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._pub_seq = 0

        self.infer_seq = 0  # frames the pose thread has completed

        # Wall time the newest raw frame came off the decoder. Every later stage carries it
        # so the HUD can state the age of what a viewer is looking at instead of implying
        # that a high frame rate means a low delay. The two are independent.
        self.raw_t = 0.0
        # Detector output, published by the detect thread and read by the pose thread.
        self._det_lock = threading.Lock()
        self._tracked = None
        self._tracked_seq = -1
        self._tracked_t = 0.0
        self.det_fps = 0.0
        self.pose_fps = 0.0
        self.lat_ms = 0.0        # capture -> composite, measured
        self.drop_fps = 0.0      # frames dropped by the drain, per second

    def put_raw(self, frame: np.ndarray, t_cap: float) -> None:
        with self._raw_cond:
            self.raw = frame
            self.raw_t = t_cap
            self.raw_seq += 1
            self._raw_cond.notify_all()

    def get_raw(self) -> tuple[np.ndarray | None, int, float]:
        with self._raw_cond:
            return (None if self.raw is None else self.raw, self.raw_seq, self.raw_t)

    def wait_raw(self, last_seq: int, timeout: float = 5.0) -> tuple[np.ndarray | None, int, float]:
        with self._raw_cond:
            self._raw_cond.wait_for(lambda: self.raw_seq != last_seq, timeout=timeout)
            return (None if self.raw is None else self.raw, self.raw_seq, self.raw_t)

    def put_tracked(self, tracked, seq: int, t_cap: float, det_fps: float) -> None:
        with self._det_lock:
            self._tracked = tracked
            self._tracked_seq = seq
            self._tracked_t = t_cap
            self.det_fps = det_fps

    def get_tracked(self):
        with self._det_lock:
            return self._tracked, self._tracked_seq, self._tracked_t

    def put_frame(self, tracks: list, infer_fps: float) -> None:
        with self._lock:
            self._tracks = tracks
            self.infer_fps = infer_fps

    def get_frame(self) -> tuple[list, float]:
        with self._lock:
            return self._tracks, self.infer_fps

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._pub_seq += 1
            self._cond.notify_all()

    def wait_next(self, last_seq: int, timeout: float = 5.0) -> tuple[bytes | None, int]:
        with self._cond:
            self._cond.wait_for(lambda: self._pub_seq != last_seq, timeout=timeout)
            return self._jpeg, self._pub_seq


#: FFmpeg options for a live camera, chosen for delay rather than smoothness. `nobuffer`
#: and `low_delay` stop the demuxer holding frames back, `reorder_queue_size 0` and
#: `max_delay 0` remove the RTP jitter buffer, and the small probe means the first frame is
#: decoded at once instead of after a second of stream analysis. `timeout` is the FFmpeg 9
#: spelling: `stimeout` was removed years ago and the value set here before was doing
#: nothing at all.
_LOW_LATENCY_OPTS = (
    "rtsp_transport;tcp"
    "|fflags;nobuffer"
    "|flags;low_delay"
    "|reorder_queue_size;0"
    "|max_delay;0"
    "|probesize;65536"
    "|analyzeduration;0"
    "|timeout;5000000"
)


class LowLatencyCapture:
    """A camera read that returns the newest decoded frame rather than the oldest queued one.

    Measured on this rig: a fresh connection consumes 13 more frames than the camera
    produced while it was connected, which is 528 ms of frames the camera had already sent
    and FFmpeg had queued. Reading at exactly the source rate keeps that queue full for the
    life of the session, so the viewer sits half a second behind the room forever and no
    amount of frame rate fixes it. Two things prevent that here: the reader never sleeps, so
    a queue drains at the difference between decode speed and source rate, and `drain`
    discards whole frames without converting them whenever a backlog is detected.
    """

    def __init__(self, url: str, *, fps_fallback: float = 25.0) -> None:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _LOW_LATENCY_OPTS
        self.url = url
        self.fps_fallback = fps_fallback
        self.cap: cv2.VideoCapture | None = None
        self.fps = fps_fallback
        self.width = 0
        self.height = 0
        self.dropped = 0

    def open(self) -> None:
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError("could not open live stream (check URL, network, credentials)")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        f = float(cap.get(cv2.CAP_PROP_FPS))
        self.fps = f if 1.0 < f < 121.0 else self.fps_fallback
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cap = cap
        # The camera replays from its last keyframe on connect. Throw that away before the
        # first frame reaches a viewer, or the session starts a keyframe interval behind.
        self.flush(0.6)

    def flush(self, seconds: float = 0.5) -> int:
        """Throw away everything the camera replayed on connect.

        Timing a grab cannot tell a queued frame from a fresh one here: decoding 4K HEVC
        costs about as long as a frame interval, so every grab looks like a queued one and a
        per-read drain silently discarded four fifths of the camera's output (measured: the
        viewer fell to 6 fps). A fixed flush at connect has no such ambiguity, and no drain
        is needed afterwards: the reader never sleeps, and decoding is faster than the camera
        produces, so a queue can only shrink.
        """
        cap = self.cap
        if cap is None:
            return 0
        end = time.perf_counter() + seconds
        n = 0
        while time.perf_counter() < end and cap.grab():
            n += 1
        self.dropped += n
        return n

    def read(self) -> tuple[np.ndarray | None, float, int]:
        """The newest frame and the wall time it came off the decoder."""
        cap = self.cap
        if cap is None:
            return None, 0.0, 0
        ok, img = cap.read()
        return (img if ok else None), time.time(), 0

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def capture_loop(url: str, shared: Shared, disp_w: int) -> None:
    backoff = 1.0
    drop_rate = Rate(window=3.0)
    while True:
        src = LowLatencyCapture(url)
        try:
            src.open()
            scale = disp_w / max(1, src.width)
            disp_h = int(round(src.height * scale / 2) * 2)
            print(
                f"capture: {src.width}x{src.height} @ {src.fps:.1f} -> {disp_w}x{disp_h}"
                f" (dropped {src.dropped} queued frames on connect)",
                flush=True,
            )
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            while True:
                img, t_cap, dropped = src.read()
                if img is None:
                    raise RuntimeError("stream ended")
                backoff = 1.0
                if abs(scale - 1.0) > 1e-3:
                    img = cv2.resize(img, (disp_w, disp_h), interpolation=interp)
                shared.put_raw(img, t_cap)
                for _ in range(dropped):
                    shared.drop_fps = drop_rate.tick()
                # Deliberately no pacing sleep. A live camera paces itself, and sleeping to
                # the nominal rate is what preserved the connect-time backlog for the whole
                # session. Consuming as fast as the decoder allows is what drains it.
        except Exception as exc:
            print(
                f"capture error ({type(exc).__name__}: {exc}); reconnect in {backoff:.0f}s",
                file=sys.stderr,
                flush=True,
            )
        finally:
            src.close()
        time.sleep(backoff)
        backoff = min(backoff * 2, 30.0)


_PRUNE_S = 5.0  # drop a track's analyzers this long after it was last seen
_BOX_KP_THR = 0.3  # keypoint confidence that counts towards a refreshed crop
_BODY_KP = 8  # keypoints averaged when asking "is this actually a person"
_CONF_ALPHA = 0.4  # weight of the newest frame in the running body-confidence score
_PERSON_HOLD = 3  # frames a box must look like a person before it is drawn (kills flicker)


def body_confidence(sc: np.ndarray) -> float:
    """How much like a body the pose model found this box to be.

    The detector of record was trained entirely on rendered warehouse scenes, and its own
    lane page says precision transfers worse than recall. On this camera it puts boxes on
    cartons and shelf clutter. A second opinion is cheap and already computed: a real person
    yields a set of confident, coherent joints, and a carton yields a scatter of weak ones,
    so the mean of the strongest few keypoint scores separates them without any new model.
    """
    if sc.size == 0:
        return 0.0
    return float(np.sort(sc)[-_BODY_KP:].mean())


def box_from_keypoints(
    kp: np.ndarray, sc: np.ndarray, fallback: np.ndarray, w: int, h: int
) -> np.ndarray:
    """A person's box for the next frame, taken from where their joints just were.

    The detector runs several times a second, the pose runs on every frame, and in between
    the two a walking person leaves the box they were found in. Cropping the next pose from
    a stale box costs keypoints exactly when someone is moving, which is when the analyzers
    need them. The joints found on this frame place the crop for the next one; the detector
    overwrites this whenever it reports.
    """
    m = sc > _BOX_KP_THR
    if int(m.sum()) < 4:
        return fallback
    xs, ys = kp[m, 0], kp[m, 1]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    mx, my = max(12.0, (x1 - x0) * 0.18), max(12.0, (y1 - y0) * 0.12)
    return np.array(
        [max(0.0, x0 - mx), max(0.0, y0 - my), min(float(w), x1 + mx), min(float(h), y1 + my)],
        dtype=np.float32,
    )


def detect_loop(shared: Shared, cfg_path: str, score: float, det_hz: float) -> None:
    """The detector, on its own thread.

    It costs about 120 ms a frame on this CPU. Run inline it set the pace for everything
    behind it, which pinned pose sampling near 8 Hz and left the seizure band under-sampled.
    Here it reads the newest frame whenever it is free, and the pose thread uses whatever it
    last reported, so a slow detector no longer slows the frame rate a person is measured at.
    """
    cfg = yaml.safe_load((REPO / cfg_path).read_text())
    print("building detector + tracker (ONNX/CPU)...", flush=True)
    det_m = build_detector(dict(cfg["detector"]))
    # frame_rate stays at the value the tracker was tuned with: its lost-track window is
    # counted in 30 fps frames whatever this says, and changing it would change which
    # tracks survive a gap, which is not a latency question.
    tracker = ByteTrack(frame_rate=15.0)
    rate = Rate()
    last_seq = -1
    frame_idx = 0
    min_period = 1.0 / det_hz if det_hz > 0 else 0.0
    while True:
        raw, last_seq, t_cap = shared.wait_raw(last_seq)
        if raw is None:
            continue
        t0 = time.perf_counter()
        d = det_m.detect_one(raw).filter(score)
        tracked = tracker.update(d, frame_idx)
        frame_idx += 1
        shared.put_tracked(tracked, last_seq, t_cap, rate.tick())
        # Leave the cores to the pose stage between refreshes. Detecting flat out would
        # starve the thread that has to keep up with the camera.
        spare = min_period - (time.perf_counter() - t0)
        if spare > 0:
            time.sleep(spare)


def infer_loop(
    shared: Shared,
    cfg_path: str,
    score: float,
    max_pose: int,
    seizure: bool,
    fall: bool,
    body_conf_min: float = 0.30,
    min_dim_frac: float = 0.06,
    seizure_hold_s: float = 5.0,
) -> None:
    cfg = yaml.safe_load((REPO / cfg_path).read_text())
    print("building pose (ONNX/CPU)...", flush=True)
    pose_m = build_pose(dict(cfg["pose"]))
    # Seconds a person must stay down-and-still before a fall CONFIRMS. Longer = fewer false
    # alarms (a trip-and-recover never sustains); tune down for a snappier live demo.
    fall_cfg = FallConfig(hold_s=float(os.environ.get("VIVIDLY_FALL_HOLD", "4.0")))
    rate = Rate()
    # tid -> [SeizureAnalyzer|None, FallAnalyzer|None, last_seen]. Each tracked person gets
    # their OWN analyzers so a seizure or fall is gated per-identity through a crowd.
    per_track: dict[int, list] = {}
    last_seq = -1
    last_det_seq = -1
    live_box: dict[int, np.ndarray] = {}
    conf: dict[int, float] = {}     # tid -> running body confidence
    good: dict[int, int] = {}       # tid -> consecutive frames that looked like a person
    while True:
        raw, last_seq, _t_cap = shared.wait_raw(last_seq)  # block until a NEW camera frame
        if raw is None:
            continue
        now = time.time()
        h_img, w_img = raw.shape[:2]
        tracked, det_seq, _det_t = shared.get_tracked()
        if tracked is None:
            continue
        if det_seq != last_det_seq:
            # A fresh detection is authoritative: it replaces every box carried forward.
            last_det_seq = det_seq
            live_box = {
                int(tracked.track_ids[i]): tracked.xyxy[i].astype(np.float32)
                for i in range(len(tracked))
            }

        # Pose only the confirmed tracks (bounded for latency: the largest/strongest first).
        all_tids = {int(t) for t in tracked.track_ids} if len(tracked) else set()

        # The pose cap is a latency bound, NOT a visibility gate. Prioritize posing tracks
        # that already have an active event (a latched fall / building seizure) so an
        # in-progress emergency is never starved by the cap; fill the rest by detection score.
        prio = []
        for i in range(len(tracked)):
            ent = per_track.get(int(tracked.track_ids[i]))
            active = 1 if (ent is not None and ent[3]) else 0
            prio.append((active, float(tracked.scores[i]), i))
        order = [i for *_, i in sorted(prio, reverse=True)]
        idxs = order[:max_pose] if max_pose else order
        views: list[TrackView] = []
        seen: set[int] = set()
        if idxs:
            boxes = np.stack(
                [live_box.get(int(tracked.track_ids[i]), tracked.xyxy[i].astype(np.float32)) for i in idxs]
            )
            poses = pose_m.estimate(raw, boxes)
            for k, i in enumerate(idxs):
                tid = int(tracked.track_ids[i])
                seen.add(tid)
                ent = per_track.get(tid)
                if ent is None:
                    ent = [
                        SeizureAnalyzer(SeizureConfig(min_hold_s=seizure_hold_s)) if seizure else None,
                        FallAnalyzer(fall_cfg) if fall else None,
                        now,
                        False,  # had an active (non-idle) event last frame
                    ]
                    per_track[tid] = ent
                kp, sc = poses.keypoints[k], poses.scores[k]
                box = boxes[k].astype(np.float32)
                live_box[tid] = box_from_keypoints(kp, sc, box, w_img, h_img)

                # Is this a person at all? Anything that fails is tracked but neither drawn
                # nor fed to an analyzer, so a carton cannot accumulate a seizure state.
                c = body_confidence(sc)
                conf[tid] = c if tid not in conf else (1 - _CONF_ALPHA) * conf[tid] + _CONF_ALPHA * c
                longest = max(float(box[2] - box[0]), float(box[3] - box[1]))
                big_enough = longest >= min_dim_frac * h_img
                if conf[tid] >= body_conf_min and big_enough:
                    good[tid] = good.get(tid, 0) + 1
                else:
                    good[tid] = 0
                if good[tid] < _PERSON_HOLD:
                    continue
                sz = ent[0].update(kp, sc, now) if ent[0] else SeizureState()
                fl = ent[1].update(kp, sc, box, now) if ent[1] else FallState()
                ent[2] = now
                ent[3] = sz.state != "idle" or fl.state != "idle"
                views.append(TrackView(tid, box, kp, sc, sz, fl))

        # A track PRESENT but not posed this frame (deprioritized by the cap) stays alive and
        # is NOT decayed — only tracks the tracker itself dropped are absent. on_absent HOLDS a
        # latched fall through a brief absence; a long-gone track is pruned.
        for tid in list(per_track):
            if tid in seen:
                continue
            ent = per_track[tid]
            if tid in all_tids:
                ent[2] = now  # present but pose-capped -> keep, do not decay or prune
                continue
            if ent[0]:
                ent[0].on_absent(now)
            if ent[1]:
                ent[1].on_absent(now)
            if now - ent[2] > _PRUNE_S:
                del per_track[tid]

        shared.pose_fps = rate.tick()
        shared.put_frame(views, shared.pose_fps)
        shared.infer_seq += 1


def composite_loop(
    shared: Shared, max_fps: float, kp_thr: float, quality: int, hud: str = "demo"
) -> None:
    """Draw and publish once per camera frame, and time the result.

    The old loop ran on a fixed 30 fps deadline against a 25 fps camera, so a fifth of the
    JPEGs a viewer received were re-encodings of a frame they already had, and a frame could
    sit up to a full period waiting for the next deadline. Composing on arrival removes both.
    The poll is 2 ms rather than a condition wait: a blocking wakeup measured about 40 ms
    under thread contention on this machine, which would have put back most of what the
    deadline cost.
    """
    enc = [cv2.IMWRITE_JPEG_QUALITY, quality]
    rate = Rate()
    last_seq = -1
    min_period = 1.0 / max_fps if max_fps and max_fps > 0 else 0.0
    while True:
        raw, seq, t_cap = shared.get_raw()
        if raw is None or seq == last_seq:
            time.sleep(0.002)
            continue
        last_seq = seq
        t0 = time.perf_counter()
        tracks, _ = shared.get_frame()
        canvas = raw.copy()
        if hud != "clean":
            draw_tracks(canvas, tracks, kp_thr)
            draw_banner(canvas, tracks)
        disp_fps = rate.tick()
        # Capture to composite, measured on every frame. This is the part of the delay this
        # process owns; the camera's own encode and the network sit in front of it.
        lat_ms = (time.time() - t_cap) * 1000.0 if t_cap else 0.0
        shared.lat_ms = lat_ms
        if hud != "clean":
            draw_hud(canvas, len(tracks), shared.pose_fps, disp_fps, shared.det_fps, lat_ms, hud)
        ok, buf = cv2.imencode(".jpg", canvas, enc)
        if ok:
            shared.publish(buf.tobytes())
        if min_period:
            spare = min_period - (time.perf_counter() - t0)
            if spare > 0:
                time.sleep(spare)


PAGE = b"""<!doctype html><meta charset=utf-8><title>Vividly - live</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>html,body{margin:0;height:100%;background:#0f1014;font-family:-apple-system,Arial}
.wrap{height:100%;display:flex;align-items:center;justify-content:center}
img{display:block;max-width:100vw;max-height:100vh;width:auto;height:auto}</style>
<div class=wrap><img src="/stream.mjpg"></div>
"""


def make_handler(shared: Shared):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a) -> None:
            return

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
                return
            if self.path.startswith("/stream.mjpg"):
                # Low-latency streaming: disable Nagle and write each frame as ONE buffer
                # so many viewers don't degrade each other on small-write syscalls.
                with contextlib.suppress(OSError):
                    self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last = -1
                try:
                    while True:
                        jpeg, last = shared.wait_next(last)
                        if jpeg is None:
                            continue
                        header = (
                            b"--frame\r\nContent-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                        )
                        self.wfile.write(header + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return
                return
            self.send_error(404)

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=os.environ.get("REOLINK_RTSP_URL", ""))
    ap.add_argument("--config", default="configs/eval/smoke_mac.yaml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--display-width", type=int, default=1280, help="downscale width for display")
    ap.add_argument(
        "--display-fps", type=float, default=30.0, help="upper FPS guard (event-driven)"
    )
    ap.add_argument(
        "--score",
        type=float,
        default=0.5,
        help="detector score cut. 0.5 is the eyes lane's own precision-leaning operating point",
    )
    ap.add_argument(
        "--body-conf",
        type=float,
        default=0.30,
        help="mean of the strongest keypoint scores a box needs before it counts as a person",
    )
    ap.add_argument(
        "--min-person",
        type=float,
        default=0.06,
        help="a person's longest side, as a fraction of frame height, below which a box is clutter",
    )
    ap.add_argument(
        "--seizure-hold",
        type=float,
        default=float(os.environ.get("VIVIDLY_SEIZURE_HOLD", "5.0")),
        help="seconds the candidate must persist before it confirms",
    )
    ap.add_argument(
        "--hud",
        choices=("demo", "debug", "clean"),
        default=os.environ.get("VIVIDLY_HUD", "demo"),
        help=(
            "clean: the camera picture and nothing else, for filming footage that will be "
            "edited and then fed back through the model. demo: wordmark, people in view, "
            "clock, and a red banner only when an event confirms. debug: adds frame rates "
            "and the measured delay"
        ),
    )
    ap.add_argument("--kp-thr", type=float, default=0.3)
    ap.add_argument("--max-pose", type=int, default=8, help="cap persons sent to pose (latency)")
    ap.add_argument(
        "--det-every",
        type=int,
        default=1,
        help="accepted and ignored: the detector now runs on its own thread, see --det-hz",
    )
    ap.add_argument(
        "--det-hz",
        type=float,
        default=float(os.environ.get("VIVIDLY_DET_HZ", "8")),
        help="how often the detector refreshes boxes; pose and the analyzers run on every frame",
    )
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument(
        "--seizure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run per-person seizure detection (default on)",
    )
    ap.add_argument(
        "--fall",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run per-person fall detection (default on)",
    )
    a = ap.parse_args()
    if not a.url:
        print("no stream URL: pass --url or set REOLINK_RTSP_URL", file=sys.stderr)
        return 2

    shared = Shared()
    for target, args in (
        (capture_loop, (a.url, shared, a.display_width)),
        (detect_loop, (shared, a.config, a.score, a.det_hz)),
        (
            infer_loop,
            (shared, a.config, a.score, a.max_pose, a.seizure, a.fall,
             a.body_conf, a.min_person, a.seizure_hold),
        ),
        (composite_loop, (shared, a.display_fps, a.kp_thr, a.quality, a.hud)),
    ):
        threading.Thread(target=target, args=args, daemon=True).start()

    if os.environ.get("VIVIDLY_TELEMETRY"):

        def _mon() -> None:
            p = (shared.raw_seq, shared.infer_seq, shared._pub_seq)
            while True:
                time.sleep(3.0)
                c, i, pub = shared.raw_seq, shared.infer_seq, shared._pub_seq
                print(
                    f"[rates] capture {(c - p[0]) / 3:.1f}  pose {(i - p[1]) / 3:.1f}  "
                    f"det {shared.det_fps:.1f}  composite {(pub - p[2]) / 3:.1f} fps  "
                    f"delay {shared.lat_ms:.0f} ms  dropped {shared.drop_fps:.1f}/s",
                    flush=True,
                )
                p = (c, i, pub)

        threading.Thread(target=_mon, daemon=True).start()

    server = ThreadingHTTPServer((a.host, a.port), make_handler(shared))
    print(f"live stream: http://{a.host}:{a.port}/  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
