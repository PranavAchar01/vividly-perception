"""AGPL quarantine. This is the only file in this repository that may import
ultralytics or boxmot.

Everything here is AGPL-3.0-licensed at runtime and produces internal reference
numbers only. Nothing in the vividly package and nothing in scripts/ may import this
module except through subprocess; enforced by .importlinter (agpl-quarantine,
runner-never-imports-agpl-ref) and scripts/guards/check_agpl_imports.py.

Run it, never import it. The reference configuration is a small YAML mapping written
for the run and kept outside the shipped config set, with the required key `weights`
and the optional keys `imgsz` (default 640), `conf` (default 0.05) and `device`
(default cpu):

    UV_PROJECT_ENVIRONMENT=.venv-agpl \\
      uv run --extra eval-agpl --no-default-groups \\
      python -m vividly.eval.agpl_ref --config outputs/eval/<run>/agpl_ref.yaml \\
      --images-json outputs/eval/<run>/images.json \\
      --out outputs/eval/<run>/preds_agpl.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ENV_FLAG = "VIVIDLY_ALLOW_AGPL"


def _require_agpl_env() -> None:
    """Refuse to run anywhere the AGPL wall might be broken."""
    if os.environ.get(_ENV_FLAG) != "1":
        raise RuntimeError(
            f"{_ENV_FLAG}=1 is required to run the AGPL reference harness. "
            "This code is AGPL-3.0 and must run only in .venv-agpl, as a subprocess, "
            "producing internal reference numbers (POLICY.md, rule 5, AGPL isolation)."
        )
    venv = Path(sys.prefix).name
    if venv != ".venv-agpl":
        raise RuntimeError(
            f"running in {sys.prefix!r}, expected .venv-agpl. Use: "
            "UV_PROJECT_ENVIRONMENT=.venv-agpl uv run --extra eval-agpl "
            "--no-default-groups python -m vividly.eval.agpl_ref ..."
        )


def run_yolo_pose(
    weights: str, image_paths: list[str], imgsz: int, conf: float, device: str
) -> dict:
    """YOLO11-pose reference numbers. Lazy import: the module import itself must not
    pull AGPL code into any process that merely reads this file's docstring."""
    _require_agpl_env()
    from ultralytics import YOLO  # AGPL-3.0, allowed only here

    model = YOLO(weights)
    dets, kpts, lat = [], [], []
    for idx, path in enumerate(image_paths):
        t0 = time.perf_counter()
        (r,) = model.predict(
            path, imgsz=imgsz, conf=conf, device=device, classes=[0], verbose=False
        )
        lat.append((time.perf_counter() - t0) * 1000.0)
        b = r.boxes.xyxy.cpu().numpy()
        s = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), sc in zip(b, s, strict=True):
            dets.append(
                {
                    "image_index": idx,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(sc),
                }
            )
        if r.keypoints is not None and len(r.keypoints) > 0:
            kk = r.keypoints.xy.cpu().numpy()
            vv = r.keypoints.conf.cpu().numpy()
            for k, v, sc in zip(kk, vv, s, strict=False):
                flat = [float(c) for xy, cf in zip(k, v, strict=True) for c in (xy[0], xy[1], cf)]
                kpts.append(
                    {"image_index": idx, "category_id": 1, "keypoints": flat, "score": float(sc)}
                )
    return {
        "backend": "ultralytics-yolo11-pose",
        "license": "AGPL-3.0",
        "shippable": "never",
        "weights": weights,
        "detections": dets,
        "keypoints": kpts,
        "latency_ms": lat,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--images-json", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    _require_agpl_env()
    import yaml

    cfg = yaml.safe_load(a.config.read_text())
    images = json.loads(a.images_json.read_text())["paths"]
    res = run_yolo_pose(
        cfg["weights"],
        images,
        int(cfg.get("imgsz", 640)),
        float(cfg.get("conf", 0.05)),
        cfg.get("device", "cpu"),
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res))
    print(f"[agpl-ref] wrote {len(res['detections'])} dets -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
