"""Live-camera bring-up: run the configured detector and pose backends (ONNX/CPU) on a real
RTSP/RTMP feed.

This is the bring-up harness the base.RTSPSource stub always pointed at. It wires
vividly.sources.LiveRTSPSource into the same detector and pose factory the replay pipeline
uses, so a site's own camera runs the exact perception stack the appliance will. It writes
annotated frames only and records no continuous footage. The stream URL carries
credentials, so it is read from --url or the environment, never committed.

    REOLINK_RTSP_URL='rtsp://admin:PASS@172.16.25.10:554/h264Preview_01_sub' \
        uv run python scripts/live_camera.py --frames 20 --out outputs/live

Continuous mode: --frames 0 runs until interrupted. Pair it with --rolling so the
annotated output overwrites a single `latest.jpg` (disk stays bounded) and each processed
frame appends one line to `detections.log`, and with --stride N to process every Nth
source frame (throttle CPU: at 25 fps, --stride 5 is about 5 inferences per second). In
continuous mode the source is reopened with backoff if the stream drops, so it survives a
camera blip.

Models default to the smoke configuration's rtmlib zoo assets, which are evaluation-only: an
RTMDet-nano detector trained on COCO with Objects365, and an RTMPose-m body7 pose model. Point
--config at a configuration naming an exported lane graph, such as
configs/pipeline/replay_demo.yaml, to run the models of record instead.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vividly.models.factory import build_detector, build_pose  # noqa: E402
from vividly.sources import LiveRTSPSource  # noqa: E402

# COCO-17 skeleton edges.
SKELETON: tuple[tuple[int, int], ...] = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 12),
    (5, 11),
    (6, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
)


def annotate(img: np.ndarray, det, poses, kp_thr: float) -> np.ndarray:
    out = img.copy()
    for i in range(len(det)):
        x1, y1, x2, y2 = det.xyxy[i].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (120, 240, 120), 2)
        cv2.putText(
            out,
            f"person {det.scores[i]:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (120, 240, 120),
            1,
            cv2.LINE_AA,
        )
    if poses is not None:
        for i in range(len(poses)):
            kps, sc = poses.keypoints[i], poses.scores[i]
            for a, b in SKELETON:
                if sc[a] > kp_thr and sc[b] > kp_thr:
                    cv2.line(
                        out,
                        tuple(kps[a].astype(int)),
                        tuple(kps[b].astype(int)),
                        (235, 133, 140),
                        2,
                        cv2.LINE_AA,
                    )
            for j in range(len(kps)):
                if sc[j] > kp_thr:
                    cv2.circle(out, tuple(kps[j].astype(int)), 3, (180, 120, 255), -1, cv2.LINE_AA)
    return out


def _spec_label(spec: dict) -> str:
    """The backend key the factory will build, and the config's name for it when it has one."""
    name = spec.get("name")
    return f"{spec.get('backend')}" + (f" ({name})" if name else "")


def _write_atomic(path: Path, image: np.ndarray) -> None:
    """Overwrite `path` so a viewer never reads a half-written JPEG."""
    tmp = path.with_suffix(".tmp.jpg")
    cv2.imwrite(str(tmp), image)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--url",
        default=os.environ.get("REOLINK_RTSP_URL", ""),
        help="RTSP/RTMP URL (or set REOLINK_RTSP_URL)",
    )
    ap.add_argument("--config", default="configs/eval/smoke_mac.yaml")
    ap.add_argument(
        "--frames",
        type=int,
        default=10,
        help="number of frames to process; 0 = run continuously until interrupted",
    )
    ap.add_argument(
        "--stride", type=int, default=1, help="process every Nth source frame (throttle inference)"
    )
    ap.add_argument(
        "--rolling",
        action="store_true",
        help="overwrite one latest.jpg and append to detections.log (bounded disk)",
    )
    ap.add_argument("--out", default="outputs/live")
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--kp-thr", type=float, default=0.3)
    a = ap.parse_args()
    if not a.url:
        print("no stream URL: pass --url or set REOLINK_RTSP_URL", file=sys.stderr)
        return 2
    if a.stride < 1:
        print("--stride must be >= 1", file=sys.stderr)
        return 2

    cfg = yaml.safe_load((REPO / a.config).read_text())
    det_spec, pose_spec = dict(cfg["detector"]), dict(cfg["pose"])
    print(
        f"building detector {_spec_label(det_spec)} and pose {_spec_label(pose_spec)} "
        "on ONNX/CPU...",
        flush=True,
    )
    det = build_detector(det_spec)
    pose = build_pose(pose_spec)

    out_dir = REPO / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    unlimited = a.frames <= 0
    log_fh = (out_dir / "detections.log").open("a", encoding="utf-8") if a.rolling else None

    n = 0
    total_persons = 0
    backoff = 1.0
    try:
        while unlimited or n < a.frames:
            src = LiveRTSPSource(a.url)
            try:
                info = src.open()
                print(
                    f"stream open: {info.width}x{info.height} @ {info.fps:.1f} fps "
                    f"(fallback if 0){' [continuous]' if unlimited else ''}",
                    flush=True,
                )
                backoff = 1.0  # a clean open resets the reconnect backoff
                for frame in src.frames():
                    if frame.index % a.stride != 0:
                        continue
                    t0 = time.time()
                    d = det.detect_one(frame.image).filter(a.score)
                    p = pose.estimate(frame.image, d.xyxy) if len(d) else None
                    dt = (time.time() - t0) * 1000.0
                    annotated = annotate(frame.image, d, p, a.kp_thr)
                    if a.rolling:
                        _write_atomic(out_dir / "latest.jpg", annotated)
                        stamp = datetime.now(UTC).isoformat(timespec="seconds")
                        log_fh.write(
                            f"{stamp} src={frame.index} persons={len(d)} infer_ms={dt:.0f}\n"
                        )
                        log_fh.flush()
                    else:
                        _write_atomic(out_dir / f"frame_{n:04d}.jpg", annotated)
                    total_persons += len(d)
                    if not a.rolling or n % 25 == 0:
                        print(
                            f"  frame {n} (src idx {frame.index}): {len(d)} persons, {dt:.0f} ms",
                            flush=True,
                        )
                    n += 1
                    if not unlimited and n >= a.frames:
                        break
            except Exception as exc:
                if not unlimited:
                    raise
                print(
                    f"  stream error ({type(exc).__name__}: {exc}); reconnecting in {backoff:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                src.close()
            if not unlimited:
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)  # exponential backoff, capped
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    finally:
        if log_fh is not None:
            log_fh.close()
    print(f"done: {n} frames, {total_persons} person-detections total, out={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
