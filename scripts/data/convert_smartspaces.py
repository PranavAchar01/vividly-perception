#!/usr/bin/env python3
"""PhysicalAI-SmartSpaces MTMC ground truth -> per-scene COCO person JSON + frames.

    uv run python scripts/data/convert_smartspaces.py \
        --config configs/convert/smartspaces_train_v1.yaml [--dry-run] [--only Warehouse_000]

The conversion logic (schema, sampling, visible-box emission, ffmpeg extraction and
its cross-checks) lives in `vividly.data.convert.mtmc_to_coco`; this file carries the
config, the registry resolution and the exit codes, which is the shape every converter
script in this directory follows.

Dataset resolution goes through the registry and nowhere else: `dataset:
physicalai_smartspaces` resolves the scene tree through `image_dir(name, split)` and
the outputs through the same containment check, so a symlink or `..` that leaves
data/train_ok/ raises PathEscapesBucket instead of being written. The val split is
convertible here, because val scenes are exam-side probe surfaces and their conversion
is an eval artefact, but the exam rule is unchanged: the scene lists in
`configs/convert/person_v2_manifests.yaml` name the training scenes, and
`scripts/data/merge_coco.py` refuses any per-scene JSON whose name carries `_val_`, so
val scenes stay out of every training manifest; converting bytes grants nothing.

Exit codes:
    0  converted, or a dry run completed
    2  usage error (argparse)
    3  config error
    4  CounselGateOpen: the dataset has unanswered counsel questions
    5  other registry refusal (legal hold, bucket, missing bytes, escaping path)
    7  conversion error: ground truth and videos disagree, or extraction mismatch
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
EXIT_CONVERT = 7


class ConfigError(ValueError):
    pass


def _require(cfg: dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise ConfigError(f"config needs `{key}`")
    return cfg[key]


def run(cfg: dict[str, Any], *, dry_run: bool, only: str | None) -> int:
    from vividly.data import registry as R
    from vividly.data.convert.mtmc_to_coco import (
        ExtractionMismatch,
        MTMCFormatError,
        convert_scene,
    )

    name = str(_require(cfg, "dataset"))
    split = str(_require(cfg, "split"))
    if split not in ("train", "val"):
        raise ConfigError(f"split must be train or val, got {split!r}")
    stride = int(cfg.get("stride", 30))
    object_types = tuple(cfg.get("object_types", ["Person"]))
    min_side_px = float(cfg.get("min_side_px", 2.0))
    keep_empty = bool(cfg.get("keep_empty", True))
    jpeg_q = int(cfg.get("jpeg_q", 2))

    reg = R.default_registry()
    ds = reg[name]
    split_dir = reg.image_dir(name, split)
    bucket_root = reg.bucket_root(ds.bucket)
    dataset_root = reg.dataset_root(name)

    def _confined_out(rel: str) -> Path:
        out = (dataset_root / rel).resolve()
        try:
            out.relative_to(bucket_root)
        except ValueError:
            raise ConfigError(f"output path {rel!r} escapes the licence bucket") from None
        return out

    frames_root = _confined_out(str(cfg.get("frames_dir", "frames")))
    ann_root = _confined_out(str(cfg.get("annotations_dir", "annotations")))

    scenes = sorted(p for p in split_dir.iterdir() if (p / "ground_truth.json").is_file())
    wanted = cfg.get("scenes")
    if wanted:
        missing = set(wanted) - {p.name for p in scenes}
        if missing:
            raise ConfigError(f"config names scenes not on disk: {sorted(missing)}")
        scenes = [p for p in scenes if p.name in set(wanted)]
    if only is not None:
        scenes = [p for p in scenes if p.name == only]
        if not scenes:
            raise ConfigError(f"--only {only!r}: no such scene in this config's selection")
    if not scenes:
        raise ConfigError(f"no scenes with ground_truth.json under {split_dir}")

    total_imgs = total_boxes = 0
    for scene_dir in scenes:
        try:
            report = convert_scene(
                scene_dir,
                frames_out=frames_root / split / scene_dir.name,
                ann_out=ann_root / f"smartspaces_{split}_{scene_dir.name}_coco.json",
                stride=stride,
                object_types=object_types,
                min_side_px=min_side_px,
                keep_empty=keep_empty,
                jpeg_q=jpeg_q,
                extract=not dry_run,
            )
        except (MTMCFormatError, ExtractionMismatch) as exc:
            print(f"conversion error: {exc}", file=sys.stderr)
            return EXIT_CONVERT
        mode = "planned" if dry_run else "wrote"
        print(
            f"{report.scene}: {mode} {report.images} images / {report.person_boxes} person "
            f"boxes ({report.cameras} cameras x {report.sampled_frames} sampled frames, "
            f"stride {stride}; dropped {report.dropped_boxes} sub-{min_side_px:g}px boxes; "
            f"types seen: { {k: v for k, v in sorted(report.object_types_seen.items())} })"
        )
        total_imgs += report.images
        total_boxes += report.person_boxes
    print(f"Total: {total_imgs} images, {total_boxes} person boxes across {len(scenes)} scenes")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true", help="plan and count; write nothing")
    ap.add_argument("--only", default=None, help="convert a single scene by name")
    args = ap.parse_args(argv)

    from vividly.data import registry as R

    try:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ConfigError(f"{args.config} did not parse to a mapping")
        return run(cfg, dry_run=args.dry_run, only=args.only)
    except ConfigError as exc:
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
