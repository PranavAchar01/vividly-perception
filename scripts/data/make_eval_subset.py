#!/usr/bin/env python3
r"""Freeze a deterministic, person-containing COCO val subset. Commit the output.

    uv run python scripts/data/make_eval_subset.py \
      --ann data/train_ok/coco/annotations/person_keypoints_val2017.json \
      --n 500 --seed 0 --out configs/eval/subset_coco_val_person500.json

Those are the flags that produced the committed subset
`configs/eval/subset_coco_val_person500.json`, fingerprint `2db41bbad264b7dc`: the frozen
pose exam surface read by the `coco500_*` evaluation configs under `configs/eval/`, and
quoted by that fingerprint in `final_models/joints/README.md` and `TRAINING.md`. The
annotation file it reads lives under `data/`, which is not in version control.

The fingerprint is the sha256 of the selected image ids, truncated to 16 hex characters,
so a re-run either reproduces it or announces that something upstream moved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ann", required=True, type=Path)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-keypoints", type=int, default=1)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)

    d = json.loads(a.ann.read_text())
    id2file = {im["id"]: im["file_name"] for im in d["images"]}
    ids = sorted(
        {
            an["image_id"]
            for an in d["annotations"]
            if an.get("num_keypoints", 0) >= a.min_keypoints and not an.get("iscrowd", 0)
        }
    )
    rng = random.Random(a.seed)
    picked = sorted(rng.sample(ids, min(a.n, len(ids))))
    payload = {
        "source_ann": str(a.ann),
        "seed": a.seed,
        "n_requested": a.n,
        "n_selected": len(picked),
        "min_keypoints": a.min_keypoints,
        "image_ids": picked,
        "file_names": [id2file[i] for i in picked],
    }
    payload["fingerprint"] = hashlib.sha256(",".join(map(str, picked)).encode()).hexdigest()[:16]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=1))
    print(f"{len(picked)} images, fingerprint {payload['fingerprint']} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
