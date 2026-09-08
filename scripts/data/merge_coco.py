#!/usr/bin/env python3
"""Build the v2 person training manifests from per-scene SmartSpaces conversions.

    uv run python scripts/data/merge_coco.py --config configs/convert/person_v2_manifests.yaml

Merging logic lives in `vividly.data.convert.coco_merge`; this file is config,
registry resolution, and exit codes (the convert_smartspaces.py pattern).

Train-side only, enforced twice: the registry resolves the annotations directory
inside the licence bucket, and this script refuses any input whose filename says
`_val_`, because the SmartSpaces val scenes are exam-side and never enter a training
or dev-val manifest (the registry exam rule; dev-val is Warehouse_005, a train scene
held out by scene in the config).

Exit codes: 0 ok, 2 usage, 3 config error, 4 counsel gate, 5 registry refusal,
7 merge error (missing scene JSON, mixed conventions, exam-side input).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXIT_OK = 0
EXIT_CONFIG = 3
EXIT_COUNSEL_GATE = 4
EXIT_REGISTRY_REFUSED = 5
EXIT_MERGE = 7


class ConfigError(ValueError):
    pass


def run(cfg: dict[str, Any]) -> int:
    from vividly.data import registry as R
    from vividly.data.convert.coco_merge import MergeError, MergeSource, merge_coco

    name = str(cfg.get("dataset") or "")
    if not name:
        raise ConfigError("config needs `dataset`")
    manifests = cfg.get("manifests")
    if not isinstance(manifests, dict) or not manifests:
        raise ConfigError("config needs a non-empty `manifests` mapping")

    reg = R.default_registry()
    ds = reg[name]
    bucket_root = reg.bucket_root(ds.bucket)
    dataset_root = reg.dataset_root(name)
    ann_dir = (dataset_root / str(cfg.get("annotations_dir", "annotations"))).resolve()
    # outside the licence bucket: the ValueError is caught in main() and reported as a
    # config error (exit 3)
    ann_dir.relative_to(bucket_root)

    import json

    for out_name, spec in manifests.items():
        scenes = spec.get("scenes")
        if not scenes:
            raise ConfigError(f"manifest {out_name!r} needs `scenes`")
        stride = spec.get("frame_stride")
        sources = []
        for scene in scenes:
            src = ann_dir / f"smartspaces_train_{scene}_coco.json"
            if "_val_" in src.name:
                print(f"merge error: {src.name} is exam-side (val split)", file=sys.stderr)
                return EXIT_MERGE
            if not src.is_file():
                print(
                    f"merge error: {src} missing - run convert_smartspaces.py first",
                    file=sys.stderr,
                )
                return EXIT_MERGE
            sources.append(MergeSource(path=src, prefix=scene, frame_stride=stride))
        try:
            merged, lines = merge_coco(sources)
        except MergeError as exc:
            print(f"merge error: {exc}", file=sys.stderr)
            return EXIT_MERGE
        out = ann_dir / f"{out_name}.json"
        out.write_text(json.dumps(merged), encoding="utf-8")
        for line in lines:
            print(f"  {line}")
        print(
            f"{out_name}: {len(merged['images'])} images / "
            f"{len(merged['annotations'])} boxes -> {out}"
        )
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args(argv)

    from vividly.data import registry as R

    try:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ConfigError(f"{args.config} did not parse to a mapping")
        return run(cfg)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except R.CounselGateOpen as exc:
        print(f"counsel gate: {exc}", file=sys.stderr)
        return EXIT_COUNSEL_GATE
    except R.RegistryError as exc:
        print(f"registry refused: {exc}", file=sys.stderr)
        return EXIT_REGISTRY_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
