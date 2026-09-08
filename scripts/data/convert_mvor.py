#!/usr/bin/env python3
"""Build the MVOR local probe exam (v2 person COCO) from the released annotations.

    uv run python scripts/data/convert_mvor.py --config configs/convert/mvor_person_v2.yaml

Logic in vividly.data.convert.mvor_to_coco; this file is registry resolution and
exit codes (the convert_smartspaces.py pattern). MVOR is EVAL_ONLY and
clinical-adjacent: this output stays on this machine, is only ever read by the
local CPU eval of exported students, and nothing derived from it enters a
training manifest (the registry raises) or a pod bundle (the setup script
fetches only cloud-cleared sets).

Exit codes: 0 ok, 2 usage, 3 config error, 4 counsel gate, 5 registry refusal,
7 conversion error (schema mismatch, missing frames).
"""

from __future__ import annotations

import argparse
import json
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


def run(cfg: dict[str, Any]) -> int:
    from vividly.data import registry as R
    from vividly.data.convert.mvor_to_coco import MVORFormatError, convert_mvor

    name = str(cfg.get("dataset") or "")
    if name != "mvor":
        raise ConfigError(f"this converter is mvor-only, got dataset={name!r}")
    reg = R.default_registry()
    ds = reg[name]
    bucket_root = reg.bucket_root(ds.bucket)
    dataset_root = reg.dataset_root(name)
    source = reg.annotation_file(name, str(cfg.get("split", "test")))
    image_root = reg.image_dir(name, str(cfg.get("split", "test")))
    out = (dataset_root / str(cfg.get("out", "annotations/mvor_person_v2_coco.json"))).resolve()
    try:
        out.relative_to(bucket_root)
    except ValueError:
        raise ConfigError(f"output {out} escapes the licence bucket") from None

    try:
        coco, report = convert_mvor(
            source_json=source,
            image_root=image_root,
            min_side_px=float(cfg.get("min_side_px", 2.0)),
        )
    except MVORFormatError as exc:
        print(f"conversion error: {exc}", file=sys.stderr)
        return EXIT_CONVERT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(coco), encoding="utf-8")
    print(
        f"mvor: wrote {report.images} images / {report.boxes} person boxes "
        f"({report.negatives} negatives; dropped {report.dropped_boxes} sub-2px) -> {out}"
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
    except R.CounselGateOpen as exc:
        print(f"counsel gate: {exc}", file=sys.stderr)
        return EXIT_COUNSEL_GATE
    except R.RegistryError as exc:
        print(f"registry refused: {exc}", file=sys.stderr)
        return EXIT_REGISTRY_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
