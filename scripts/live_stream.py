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
    if v.fall.state == "falling":
        return _AMBER, "falling"
    if v.seizure.state == "candidate":
        return _AMBER, "analyzing"
    return (_BOX, "") if s == 0 else (_AMBER, "")


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


def draw_hud(img: np.ndarray, persons: int, infer_fps: float, disp_fps: float) -> None:
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
    right = f"{persons} person(s)   infer {infer_fps:4.1f} fps   video {disp_fps:4.1f} fps"
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
    if sev == 0:
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
    else:  # building: a candidate seizure or a fall in progress
        msg = (
            f"ANALYZING person #{worst.tid} — falling"
            if worst.fall.state == "falling"
            else f"ANALYZING person #{worst.tid} — {worst.seizure.n_limbs} limbs @ "
            f"{worst.seizure.dominant_hz:.1f} Hz"
        )
        fs = _fit_scale(msg, w - 24, 0.6, 2)
        cv2.putText(img, msg, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, fs, _AMBER, 2, cv2.LINE_AA)


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

        self.infer_seq = 0  # frames the infer thread has completed

    def put_raw(self, frame: np.ndarray) -> None:
        with self._raw_cond:
            self.raw = frame
            self.raw_seq += 1
            self._raw_cond.notify_all()

    def get_raw(self) -> tuple[np.ndarray | None, int]:
        with self._raw_cond:
            return (None if self.raw is None else self.raw, self.raw_seq)

    def wait_raw(self, last_seq: int, timeout: float = 5.0) -> tuple[np.ndarray | None, int]:
        with self._raw_cond:
            self._raw_cond.wait_for(lambda: self.raw_seq != last_seq, timeout=timeout)
            return (None if self.raw is None else self.raw, self.raw_seq)

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


def capture_loop(url: str, shared: Shared, disp_w: int) -> None:
    backoff = 1.0
    while True:
        src = LiveRTSPSource(url)
        try:
            info = src.open()
            scale = disp_w / max(1, info.width)
            disp_h = int(round(info.height * scale / 2) * 2)
            print(f"capture: {info.width}x{info.height} -> {disp_w}x{disp_h}", flush=True)
            # Never consume faster than the source frame rate. A live camera already paces
            # itself (so this never sleeps), but a file or a buffered backlog would otherwise
            # race ahead — which corrupts the wall-clock timing the seizure analyzer relies on.
            period = 1.0 / info.fps if info.fps and info.fps > 0 else 0.0
            deadline = time.time()
            for frame in src.frames():
                backoff = 1.0  # a real frame arrived; only now is the connection healthy
                img = frame.image
                if abs(scale - 1.0) > 1e-3:
                    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                    img = cv2.resize(img, (disp_w, disp_h), interpolation=interp)
                shared.put_raw(img)
                if period:
                    deadline += period
                    slack = deadline - time.time()
                    if slack > 0:
                        time.sleep(slack)
                    elif slack < -1.0:
                        deadline = time.time()  # fell far behind (real camera); resync
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


def infer_loop(
    shared: Shared,
    cfg_path: str,
    score: float,
    max_pose: int,
    seizure: bool,
    fall: bool,
    det_every: int = 1,
) -> None:
    cfg = yaml.safe_load((REPO / cfg_path).read_text())
    print("building RTMDet + RTMPose + ByteTrack (ONNX/CPU)...", flush=True)
    det_m = build_detector(dict(cfg["detector"]))
    pose_m = build_pose(dict(cfg["pose"]))
    tracker = ByteTrack(frame_rate=15.0)
    # Seconds a person must stay down-and-still before a fall CONFIRMS. Longer = fewer false
    # alarms (a trip-and-recover never sustains); tune down for a snappier live demo.
    fall_cfg = FallConfig(hold_s=float(os.environ.get("VIVIDLY_FALL_HOLD", "4.0")))
    rate = Rate()
    # tid -> [SeizureAnalyzer|None, FallAnalyzer|None, last_seen]. Each tracked person gets
    # their OWN analyzers so a seizure or fall is gated per-identity through a crowd.
    per_track: dict[int, list] = {}
    frame_idx = 0
    last_seq = -1
    tracked = None
    det_every = max(1, int(det_every))
    while True:
        raw, last_seq = shared.wait_raw(last_seq)  # block until a NEW camera frame
        if raw is None:
            continue
        now = time.time()
        # The eyes lane (RT-DETRv4-S) costs ~120 ms on this CPU, the pose ~30 ms per person.
        # Running the detector every iteration would hold pose sampling near 8 Hz, below what
        # the seizure analyzer's 3-8 Hz band needs. So the detector refreshes every det_every
        # iterations and the iterations between pose the last tracked boxes on the current
        # frame: a person moves little in the ~150 ms between refreshes, and the analyzers
        # see ~18 Hz keypoints instead of ~8.
        if tracked is None or frame_idx % det_every == 0:
            d = det_m.detect_one(raw).filter(score)
            tracked = tracker.update(d, frame_idx)  # Detections with stable track_ids
        frame_idx += 1

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
            boxes = tracked.xyxy[idxs]
            poses = pose_m.estimate(raw, boxes)
            for k, i in enumerate(idxs):
                tid = int(tracked.track_ids[i])
                seen.add(tid)
                ent = per_track.get(tid)
                if ent is None:
                    ent = [
                        SeizureAnalyzer() if seizure else None,
                        FallAnalyzer(fall_cfg) if fall else None,
                        now,
                        False,  # had an active (non-idle) event last frame
                    ]
                    per_track[tid] = ent
                kp, sc, box = (
                    poses.keypoints[k],
                    poses.scores[k],
                    tracked.xyxy[i].astype(np.float32),
                )
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

        shared.put_frame(views, rate.tick())
        shared.infer_seq += 1


def composite_loop(shared: Shared, max_fps: float, kp_thr: float, quality: int) -> None:
    enc = [cv2.IMWRITE_JPEG_QUALITY, quality]
    period = 1.0 / max_fps
    rate = Rate()
    deadline = time.time()
    while True:
        # Poll the latest frame (a blocking condition-wait here costs ~8 fps of wakeup
        # latency under 3-thread contention; polling + absolute-deadline pacing does not).
        raw, _ = shared.get_raw()
        if raw is None:
            time.sleep(0.01)
            continue
        tracks, infer_fps = shared.get_frame()
        persons = len(tracks)
        canvas = raw.copy()
        draw_tracks(canvas, tracks, kp_thr)
        draw_banner(canvas, tracks)
        disp_fps = rate.tick()
        draw_hud(canvas, persons, infer_fps, disp_fps)
        ok, buf = cv2.imencode(".jpg", canvas, enc)
        if ok:
            shared.publish(buf.tobytes())
        # Absolute-deadline pacing: steady cadence without drift, no per-iter sleep math.
        deadline += period
        sleep_for = deadline - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            deadline = time.time()  # fell behind; resync


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
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--kp-thr", type=float, default=0.3)
    ap.add_argument("--max-pose", type=int, default=8, help="cap persons sent to pose (latency)")
    ap.add_argument(
        "--det-every",
        type=int,
        default=int(os.environ.get("VIVIDLY_DET_EVERY", "1")),
        help="run the detector every N inference iterations; pose runs every iteration on the last tracked boxes",
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
        (infer_loop, (shared, a.config, a.score, a.max_pose, a.seizure, a.fall, a.det_every)),
        (composite_loop, (shared, a.display_fps, a.kp_thr, a.quality)),
    ):
        threading.Thread(target=target, args=args, daemon=True).start()

    if os.environ.get("VIVIDLY_TELEMETRY"):

        def _mon() -> None:
            p = (shared.raw_seq, shared.infer_seq, shared._pub_seq)
            while True:
                time.sleep(3.0)
                c, i, pub = shared.raw_seq, shared.infer_seq, shared._pub_seq
                print(
                    f"[rates] capture {(c - p[0]) / 3:.1f}  infer {(i - p[1]) / 3:.1f}  "
                    f"composite {(pub - p[2]) / 3:.1f} fps",
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
