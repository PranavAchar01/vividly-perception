"""Materialize the synthetic convulsion training set (the interim event lane admitted by
POLICY.md, rule 4; vividly.verifier.synth).

    uv run python scripts/events/make_synth_convulsion.py \
        --out outputs/events/synth_convulsion_v1 --seed 20260826 \
        --n-pos 400 --n-neg-per-kind 70

Output is the extractor layout (sequences/ + meta/ + extraction.json), so the trainer
consumes it exactly like a real extraction. Deterministic in --seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vividly.verifier.synth import write_synth_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-pos", type=int, default=400)
    ap.add_argument("--n-neg-per-kind", type=int, default=70)
    args = ap.parse_args(argv)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    summary = write_synth_dataset(
        out, n_positive=args.n_pos, n_negative_per_kind=args.n_neg_per_kind, seed=args.seed
    )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
