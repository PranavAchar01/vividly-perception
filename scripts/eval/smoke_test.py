#!/usr/bin/env python3
"""Smoke test: detector and pose on one image and one clip, processor only, with overlays.

    uv run python scripts/eval/smoke_test.py --config configs/eval/smoke_mac.yaml

Done when the image and the clip each report at least one detection. Exit 1 otherwise.
The two overlays are written beside them under the config's `out_dir`, and are not
themselves part of the pass condition.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs/eval/smoke_mac.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    import cv2

    from vividly.models.factory import build_detector, build_pose

    det = build_detector(cfg["detector"])
    pose = build_pose(cfg["pose"])
    print(f"det : {det.provenance.label()}")
    print(f"pose: {pose.provenance.label()}")

    out_dir = REPO_ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = True

    # -- image ---------------------------------------------------------------
    img = cv2.imread(str(REPO_ROOT / cfg["inputs"]["image"]))
    t0 = time.perf_counter()
    d = det.detect_one(img)
    p = pose.estimate(img, d.xyxy)
    ms = (time.perf_counter() - t0) * 1000
    vis = img.copy()
    for (x1, y1, x2, y2), kps in zip(d.xyxy, p.keypoints, strict=True):
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (80, 80, 240), 2)
        for x, y in kps:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 220, 0), -1)
    cv2.imwrite(str(out_dir / "smoke_image_overlay.jpg"), vis)
    print(f"image: {len(d)} people in {ms:.0f} ms -> smoke_image_overlay.jpg")
    ok &= len(d) > 0

    # -- clip ----------------------------------------------------------------
    cap = cv2.VideoCapture(str(REPO_ROOT / cfg["inputs"]["clip"]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    wr = cv2.VideoWriter(
        str(out_dir / "smoke_clip_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    n = people = 0
    t0 = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        d = det.detect_one(frame)
        p = pose.estimate(frame, d.xyxy)
        people += len(d)
        vis = frame.copy()
        for kps in p.keypoints:
            for x, y in kps:
                cv2.circle(vis, (int(x), int(y)), 2, (0, 220, 0), -1)
        wr.write(vis)
        n += 1
    cap.release()
    wr.release()
    wall = time.perf_counter() - t0
    print(
        f"clip : {n} frames, {people} person-detections, {n / wall:.1f} fps -> smoke_clip_overlay.mp4"
    )
    ok &= people > 0

    print("smoke", "pass" if ok else "fail")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
