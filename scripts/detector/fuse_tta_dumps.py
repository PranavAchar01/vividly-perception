#!/usr/bin/env python
r"""I1 TTA/ensemble row: weighted-boxes-fuse two or more detection dumps into one.

    uv run python scripts/detector/fuse_tta_dumps.py \
        --gt  data/train_ok/physicalai_smartspaces/annotations/person_v2_devval.json \
        --det identity=outputs/eyes_b1/dumps/dets_b1_rtv4s_seed0_identity.json \
        --det hflip=outputs/eyes_b1/dumps/dets_b1_rtv4s_seed0_hflip.json \
        --out outputs/eyes_b1/dumps/dets_b1_rtv4s_seed0_tta_wbf.json

The paths under `data/` and `outputs/` are not in version control.

Pre-declared by research/eyes-instrument-protocol.md as the optional same-session
I1 row: hflip-only dumps for the pre-declared protocol row, at fuse.py defaults,
iou_thr 0.55, with no grid search, reported as an ensemble and TTA protocol row,
not a checkpoint and not promotable.

The fusion module is src/vividly/eval/wbf.py; this script is a thin driver over
`fuse_wbf` and uses `WBFParams()` exactly as declared, that is iou_thr 0.55,
skip_box_thr 0.05, conf_type 'avg', all weights 1.0, and `ScoreCal()` identity
for every input. There is no tuning knob here on purpose: exposing one would
invite the grid search the protocol forbids. The only flags are the file paths.

Two consequences of the reference WBF behaviour, recorded because they change how
the fused scores must be read:

- ensemble-boxes' `conf_type='avg'` rescales a cluster's score by
  (weight of the models that contributed) / (total weight). With two equally
  weighted views, a box seen by only one view keeps half its score. A fixed cut
  at 0.3/0.4/0.5 on fused scores is therefore not the same operating point as
  the same cut on a single view's scores.
- `skip_box_thr=0.05` drops every input box below 0.05 before clustering, so the
  fused list is much shorter than the 300-per-image dumps. That is the declared
  default, not a filter chosen here.

Boxes are pixel xywh on disk (pycocotools results-list format) and pixel xyxy
inside the fusion module; the ground-truth json supplies each image's width/height, which
fuse_wbf needs to normalize and to clip. Output is the same results-list format
plus a `.sidecar.json` recording every input's sha256 and checkpoint sha256, so
the fused file still carries its checkpoint hash downstream (POLICY.md, rule 10).

Fusion is single-class: the driver hands `fuse_wbf` boxes and scores only, so an input
dump's `category_id` never reaches it, and `fuse_wbf` labels every box 0 internally
(src/vividly/eval/wbf.py). Every fused row is therefore written with `category_id` 0,
which is the person id in `person_v2_devval.json`. A non-zero category in an input dump is
relabelled, not kept.

Exit codes: 0 ok, 3 config/prerequisite problem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vividly.eval.wbf import RawDetections, ScoreCal, WBFParams, fuse_wbf

EXIT_OK = 0
EXIT_CONFIG = 3


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_results(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a pycocotools results list")
    return raw


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gt", required=True, type=Path, help="COCO GT json (for image sizes)")
    ap.add_argument(
        "--det",
        required=True,
        action="append",
        metavar="NAME=PATH",
        help="a named detection dump; repeat once per view/model",
    )
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    if not args.gt.is_file():
        print(f"FATAL: no such file: {args.gt}", file=sys.stderr)
        return EXIT_CONFIG
    sources: dict[str, Path] = {}
    for spec in args.det:
        if "=" not in spec:
            print(f"FATAL: --det wants NAME=PATH, got {spec!r}", file=sys.stderr)
            return EXIT_CONFIG
        name, _, raw_path = spec.partition("=")
        path = Path(raw_path)
        if not path.is_file():
            print(f"FATAL: no such file: {path}", file=sys.stderr)
            return EXIT_CONFIG
        sources[name] = path
    if len(sources) < 2:
        print("FATAL: fusion needs at least two distinct --det sources", file=sys.stderr)
        return EXIT_CONFIG

    with args.gt.open() as fh:
        gt = json.load(fh)
    sizes = {img["id"]: (int(img["width"]), int(img["height"])) for img in gt["images"]}

    by_source: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for name, path in sources.items():
        per_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for rec in load_results(path):
            per_image[int(rec["image_id"])].append(rec)
        by_source[name] = per_image

    calibration = {name: ScoreCal() for name in sources}
    params = WBFParams()  # protocol: fuse.py defaults, no grid search

    fused_out: list[dict[str, Any]] = []
    for image_id in sorted(sizes):
        width, height = sizes[image_id]
        dets: list[RawDetections] = []
        for name in sources:
            recs = by_source[name].get(image_id, [])
            dets.append(
                RawDetections(
                    source=name,
                    boxes=[
                        [
                            float(r["bbox"][0]),
                            float(r["bbox"][1]),
                            float(r["bbox"][0]) + float(r["bbox"][2]),
                            float(r["bbox"][1]) + float(r["bbox"][3]),
                        ]
                        for r in recs
                    ],
                    scores=[float(r["score"]) for r in recs],
                )
            )
        boxes, scores = fuse_wbf(dets, width, height, calibration, params)
        for (x1, y1, x2, y2), score in zip(boxes, scores, strict=True):
            fused_out.append(
                {
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fused_out))

    inputs = []
    for name, path in sources.items():
        side = path.with_suffix(path.suffix + ".sidecar.json")
        ckpt_sha = None
        if side.is_file():
            with side.open() as fh:
                ckpt_sha = json.load(fh).get("checkpoint_sha256")
        inputs.append(
            {
                "source": name,
                "path": str(path),
                "sha256": sha256_file(path),
                "checkpoint_sha256": ckpt_sha,
            }
        )
    sidecar = {
        "written_by": "scripts/detector/fuse_tta_dumps.py",
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "gt": str(args.gt),
        "inputs": inputs,
        "checkpoint_sha256": (
            inputs[0]["checkpoint_sha256"]
            if len({i["checkpoint_sha256"] for i in inputs}) == 1
            else "multiple, see inputs[]"
        ),
        "wbf": {
            "iou_thr": params.iou_thr,
            "skip_box_thr": params.skip_box_thr,
            "conf_type": params.conf_type,
            "weights": "1.0 for every source (fuse.py default)",
            "calibration": "identity for every source",
        },
        "n_detections": len(fused_out),
        "n_images_with_detections": len({r["image_id"] for r in fused_out}),
        "detections_sha256": sha256_file(args.out),
        "note": "ensemble/TTA protocol, not a checkpoint, not promotable",
    }
    side_path = args.out.with_suffix(args.out.suffix + ".sidecar.json")
    side_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} ({len(fused_out)} detections)")
    print(f"wrote {side_path}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
