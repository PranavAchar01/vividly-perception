#!/usr/bin/env python3
"""Detection and pose eval runner. Thin --config shell over vividly.eval.runner.

uv run python scripts/eval/eval_runner.py --config configs/eval/coco500_eyes_stockm.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    from vividly.eval.runner import run

    out_dir = run(cfg, REPO_ROOT)
    print(f"\nresults: {out_dir}/results.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
