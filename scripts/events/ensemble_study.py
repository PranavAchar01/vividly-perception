"""Two-model ensemble study: can final_v1 corroborate final_v2's fires?

The question posed on 2026-08-27 was whether combining the model with the highest
accuracy and the model with the lowest false-alarm rate helps. final_v1 (ONNX
`42dff31c`, stock eyes, conservative collapsed scores, 47 s median latency) and
final_v2 (ONNX `07c399be`, chained eyes, calibrated margins, 7 s) were evaluated on
the same held-out patients through different extractions, so their window streams
align only in time, never by track id. This instrument time-aligns the two streams
per video and measures the combination families:

  v2:<t>          final_v2 alone (baseline rows, recomputed per video)
  v1:<t>          final_v1 alone
  or:<a>|<b>      v2 >= a  or  v1 >= b        (the naive best of both)
  and:<p>&<c>     v2 >= p and v1 >= c within +/-TOL_S (second-opinion gate);
                  alert time = max(v2 fire, earliest corroborating v1 time),
                  that is, the alert waits for the corroboration, which is causal
                  latency
  tier:<hi>|<p>&<c>   v2 >= hi passes alone; below that, the and-gate applies

Catch, latency and false-alarm definitions mirror decision_layer_study.py, except
that episodes collapse per video (cross-model fires share no track), so numbers
here are only comparable within this table; the v2 baselines are re-derived under
the same definition for that reason.

    uv run python scripts/events/ensemble_study.py \
        --v2-dir outputs/verifier/convulsion_v2_real/eval_wusahzu_testsplit \
        --v1-dir outputs/verifier/convulsion_v1_real/eval_wusahzu_testsplit
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

TOL_S = 6.0  # two strides: how far apart the two models may fire and still agree
EPISODE_GAP_S = 30.0
CATCH_TAIL_S = 30.0

COMBOS: tuple[str, ...] = (
    "v2:0.5",
    "v2:0.8",
    "v2:0.9",
    "v1:0.5",
    "or:0.9|0.5",
    "or:0.8|0.5",
    "and:0.5&0.5",
    "and:0.65&0.5",
    "tier:0.8|0.5&0.5",
    "tier:0.9|0.5&0.5",
)


def load_stream(eval_dir: Path) -> dict[str, list[tuple[float, float]]]:
    """Per-video (t_end, score) points, all tracks merged, time-sorted."""
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for line in (eval_dir / "windows.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r["key"]].append((r["t_end"], r["score"]))
    for pts in out.values():
        pts.sort()
    return dict(out)


def solo_fires(pts: list[tuple[float, float]], thr: float) -> list[float]:
    return [t for t, s in pts if s >= thr]


def and_fires(
    v2_pts: list[tuple[float, float]], v1_pts: list[tuple[float, float]], p: float, c: float
) -> list[float]:
    v1_hot = solo_fires(v1_pts, c)
    fires = []
    for t in solo_fires(v2_pts, p):
        corroborating = [tv for tv in v1_hot if abs(tv - t) <= TOL_S]
        if corroborating:
            fires.append(max(t, min(corroborating)))
    return sorted(fires)


def combo_fires(
    combo: str, v2_pts: list[tuple[float, float]], v1_pts: list[tuple[float, float]]
) -> list[float]:
    kind, spec = combo.split(":")
    if kind == "v2":
        return solo_fires(v2_pts, float(spec))
    if kind == "v1":
        return solo_fires(v1_pts, float(spec))
    if kind == "or":
        a, b = spec.split("|")
        return sorted(set(solo_fires(v2_pts, float(a))) | set(solo_fires(v1_pts, float(b))))
    if kind == "and":
        p, c = spec.split("&")
        return and_fires(v2_pts, v1_pts, float(p), float(c))
    if kind == "tier":
        hi, rest = spec.split("|")
        p, c = rest.split("&")
        return sorted(
            set(solo_fires(v2_pts, float(hi))) | set(and_fires(v2_pts, v1_pts, float(p), float(c)))
        )
    raise ValueError(combo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", required=True, type=Path)
    ap.add_argument("--v1-dir", required=True, type=Path)
    ap.add_argument("--neg-keys", default="pat03__free,pat04__free,pat04__no-Sz2P")
    # Default derives from metrics_deep, not a constant: a hardcoded 2.8309 here
    # deflated every published FP-episodes/h by 3x until 2026-08-27 (the true
    # container duration of the three negatives is 0.944 h).
    ap.add_argument("--neg-hours", type=float, default=None)
    args = ap.parse_args()

    md = json.loads((args.v2_dir / "metrics_deep.json").read_text(encoding="utf-8"))
    if args.neg_hours is None:
        args.neg_hours = float(md["negatives"]["recorded_hours"])
        if not args.neg_hours:
            raise SystemExit("metrics_deep reports zero negative hours; pass --neg-hours")
    events = [
        {"key": r["key"], "onset": r["onset_s"], "end": r["end_s"], "cls": r["class"]}
        for r in md["events"]["rows"]
    ]
    neg_keys = set(args.neg_keys.split(","))
    v2 = load_stream(args.v2_dir)
    v1 = load_stream(args.v1_dir)
    keys = sorted(set(v2) | set(v1))
    if not neg_keys & set(keys):
        # Without this the false-alarm episode count is zero against a non-zero
        # denominator, so fp_episodes_per_h reads 0.00 as though it were measured.
        raise SystemExit(
            f"none of --neg-keys {sorted(neg_keys)} matches a scored video; scored keys are {keys}"
        )

    results = []
    for combo in COMBOS:
        fires = {k: combo_fires(combo, v2.get(k, []), v1.get(k, [])) for k in keys}

        caught, latencies, pg_caught, pg_total = 0, [], 0, 0
        for ev in events:
            if ev["cls"] == "convulsive_PG":
                pg_total += 1
            hit = next(
                (
                    t
                    for t in fires.get(ev["key"], [])
                    if ev["onset"] <= t <= ev["end"] + CATCH_TAIL_S
                ),
                None,
            )
            if hit is not None:
                caught += 1
                latencies.append(hit - ev["onset"])
                if ev["cls"] == "convulsive_PG":
                    pg_caught += 1

        episodes = 0
        for k in neg_keys:
            fs = fires.get(k, [])
            episodes += (1 if fs else 0) + sum(
                1 for a, b in itertools.pairwise(fs) if b - a > EPISODE_GAP_S
            )

        latencies.sort()
        med = latencies[len(latencies) // 2] if latencies else None
        results.append(
            {
                "combo": combo,
                "caught": caught,
                "of": len(events),
                "pg": f"{pg_caught}/{pg_total}",
                "median_latency_s": round(med, 2) if med is not None else None,
                "fp_episodes_per_h": round(episodes / args.neg_hours, 2),
            }
        )
        print(
            f"{combo:18s} caught {caught:2d}/{len(events)}  PG {pg_caught}/{pg_total}  "
            f"median lat {med if med is not None else '-':>6}  "
            f"FP eps/h {episodes / args.neg_hours:5.1f}"
        )

    out = args.v2_dir / "ensemble_study.json"
    out.write_text(json.dumps({"tol_s": TOL_S, "combos": results}, indent=1) + "\n", "utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
