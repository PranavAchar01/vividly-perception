"""Per-seizure-type breakout of a WU-SAHZU convulsion-verifier eval report.

    uv run python scripts/events/wu_sahzu_breakout.py \
        --report outputs/verifier/convulsion_v2_real/eval_wusahzu_testsplit/report.json \
        --meta-dir outputs/events/wu_sahzu_tracks_v2/meta

The eval report scores every video identically; the honest presentation splits it:
convulsive PG events are the class the rhythmic convulsion verifier targets and give
the headline sensitivity, focal P events (motion often subtle or absent) are reported
separately and never folded in, and the non-seizure recordings give the false-alarm
side. Types come from the extraction meta: the split branches on the `has_seizure` and
`convulsive` keys the converter's index carries through.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_meta(meta_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(meta_dir.glob("*.json"))
    }


def split_rows(
    report: dict[str, Any], metas: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {"PG": [], "P": [], "negative": []}
    for source in report["results"]:
        for row in source["videos"]:
            meta = metas.get(row["key"])
            if meta is None:
                raise SystemExit(f"no extraction meta for scored video {row['key']!r}")
            if not meta.get("has_seizure"):
                groups["negative"].append(row)
            elif meta.get("convulsive"):
                groups["PG"].append(row)
            else:
                groups["P"].append(row)
    return groups


def describe(rows: list[dict[str, Any]], thresholds: tuple[float, ...]) -> dict[str, Any]:
    scores = sorted((r["key"], r["max_score"]) for r in rows)
    out: dict[str, Any] = {
        "n": len(rows),
        "videos": [{"key": k, "max_score": s} for k, s in scores],
    }
    for thr in thresholds:
        out[f"over_{thr:g}"] = sum(1 for _, s in scores if s >= thr)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--meta-dir", required=True, type=Path)
    ap.add_argument("--thresholds", default="0.5,0.8,0.9")
    ap.add_argument("--out", type=Path, default=None, help="write the breakout JSON here")
    args = ap.parse_args(argv)

    thresholds = tuple(float(t) for t in args.thresholds.split(","))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    groups = split_rows(report, load_meta(args.meta_dir))

    breakout = {
        "report": str(args.report),
        "model": report.get("model"),
        "disclosure": report.get("disclosure"),
        "convulsive_PG": describe(groups["PG"], thresholds),
        "focal_P": describe(groups["P"], thresholds),
        "non_seizure": describe(groups["negative"], thresholds),
    }
    text = json.dumps(breakout, indent=1, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)

    pg, p, neg = breakout["convulsive_PG"], breakout["focal_P"], breakout["non_seizure"]
    for thr in thresholds:
        k = f"over_{thr:g}"
        line = f"thr {thr:g}: PG {pg[k]}/{pg['n']} caught | focal P {p[k]}/{p['n']}"
        if neg["n"]:
            line += f" | non-seizure videos flagged {neg[k]}/{neg['n']}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
