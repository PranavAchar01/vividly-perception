"""Decision-layer study: what alerting rule on top of a verifier's window scores
best trades catches + latency against false-alarm episodes?

The question posed on 2026-08-26: whether final_v2's quietness at 0.9 can be had
together with its catches at 0.5. The model's scores are fixed; the free lever is
the decision rule applied to the window stream. This instrument replays an eval's
full-precision windows.jsonl through candidate rules and reports, for each: events
caught (per class), detection latency, and distinct false-alarm episodes per
negative-recording hour. Persistence was measured and rejected on final_v1 (its
collapsed scores flicker); final_v2's calibrated margins are a different regime, so
the family is re-measured rather than assumed dead.

Rules:
  single:<thr>            fire on any window >= thr
  persist:<k>@<thr>       fire when k consecutive windows of one track >= thr
                          (consecutive = same track, t_end gap <= stride + 0.5 s)
  hyst:<hi>|<k>@<lo>      fire immediately on >= hi, or on k consecutive >= lo

Catch = first fire of any track in [onset, event_end + 30 s]; latency = fire
t_end - onset. Episodes on negatives: fires of one track closer than 30 s apart
collapse into one episode (the engine's refire debounce scale).

    uv run python scripts/events/decision_layer_study.py \
        --eval-dir outputs/verifier/convulsion_v2_real/eval_wusahzu_testsplit

The eval directory is a run output under `outputs/` and is not in version control.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

# Both spans below are the seizure heads' window geometry (window 12.0 s, stride 3.0 s),
# fixed rather than read from the eval's report.json, so that every recorded run of this
# instrument reproduces. On a fall eval (window 8.0 s, stride 1.5 s) they are therefore
# not the head's own geometry: the slack spans a missing window, and each isolated fire is
# credited with 12 s of alarm rather than 8 s. research/operating-points.md records that
# as a standing caveat on the fall rows.
STRIDE_SLACK_S = 3.5  # windows 3 s apart are consecutive; anything larger is a break
EPISODE_GAP_S = 30.0
CATCH_TAIL_S = 30.0
# Span each fire's alarm covers for the time-in-alarm metric: the window that produced it.
# On the seizure heads single-threshold rules therefore reproduce the raw union-of-windows
# measurement exactly; persistence and hysteresis fires cover their k-th window.
ALARM_WINDOW_S = 12.0

RULES: tuple[str, ...] = (
    "single:0.5",
    "single:0.8",
    "single:0.9",
    "persist:2@0.5",
    "persist:3@0.5",
    "persist:2@0.65",
    "persist:2@0.8",
    "hyst:0.9|2@0.5",
    "hyst:0.9|2@0.65",
    "hyst:0.85|2@0.5",
)


def fires_for_track(ts: list[float], ss: list[float], rule: str) -> list[float]:
    """Fire times a rule produces on one track's chronological window stream."""
    if rule.startswith("single:"):
        thr = float(rule.split(":")[1])
        return [t for t, s in zip(ts, ss, strict=True) if s >= thr]
    if rule.startswith("persist:"):
        k, thr = rule.split(":")[1].split("@")
        return _persist(ts, ss, int(k), float(thr))
    if rule.startswith("hyst:"):
        hi, rest = rule.split(":")[1].split("|")
        k, lo = rest.split("@")
        merged = sorted(
            set(_persist(ts, ss, 1, float(hi))) | set(_persist(ts, ss, int(k), float(lo)))
        )
        return merged
    raise ValueError(rule)


def _persist(ts: list[float], ss: list[float], k: int, thr: float) -> list[float]:
    out, run = [], 0
    for i, (t, s) in enumerate(zip(ts, ss, strict=True)):
        if s >= thr and (run == 0 or t - ts[i - 1] <= STRIDE_SLACK_S):
            run += 1
        elif s >= thr:
            run = 1  # above threshold but a time gap broke the streak
        else:
            run = 0
        if run >= k:
            out.append(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, type=Path)
    # Default: the WU-SAHZU testsplit's three negative recordings. Any other eval must
    # pass its own; the run refuses to continue when none of them is a scored recording,
    # because an empty negative set silently reports 0.00 episodes per hour as if measured.
    ap.add_argument("--neg-keys", default="pat03__free,pat04__free,pat04__no-Sz2P")
    # No hardcoded default: a 2.8309 constant here (origin unreconstructable, 3x the
    # true container duration) silently deflated every published FP-episodes/h by 3x
    # until 2026-08-27. The denominator now comes from metrics_deep's own
    # negatives.recorded_hours (sum of the negative videos' container durations).
    ap.add_argument(
        "--neg-hours",
        type=float,
        default=None,
        help="override footage hours of the neg keys (default: metrics_deep recorded_hours)",
    )
    args = ap.parse_args()

    md = json.loads((args.eval_dir / "metrics_deep.json").read_text(encoding="utf-8"))
    if args.neg_hours is None:
        args.neg_hours = float(md["negatives"]["recorded_hours"])
        if not args.neg_hours:
            raise SystemExit("metrics_deep reports zero negative hours; pass --neg-hours")
    events = [
        {"key": r["key"], "onset": r["onset_s"], "end": r["end_s"], "cls": r["class"]}
        for r in md["events"]["rows"]
    ]
    neg_keys = set(args.neg_keys.split(","))

    tracks: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for line in (args.eval_dir / "windows.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        tracks[(r["key"], r["track_id"])].append((r["t_end"], r["score"]))
    for tw in tracks.values():
        tw.sort()

    if not neg_keys & {key for key, _tid in tracks}:
        raise SystemExit(
            f"none of --neg-keys {sorted(neg_keys)} names a scored recording in "
            f"{args.eval_dir}; pass this eval's own negative keys"
        )

    results = []
    for rule in RULES:
        fires: dict[str, list[float]] = defaultdict(list)
        for (key, _tid), tw in tracks.items():
            ts = [t for t, _ in tw]
            ss = [s for _, s in tw]
            fires[key].extend(fires_for_track(ts, ss, rule))
        for key in fires:
            fires[key].sort()

        caught, latencies, pg_caught, pg_total = 0, [], 0, 0
        for ev in events:
            if ev["cls"] == "convulsive_PG":
                pg_total += 1
            hit = next(
                (t for t in fires[ev["key"]] if ev["onset"] <= t <= ev["end"] + CATCH_TAIL_S),
                None,
            )
            if hit is not None:
                caught += 1
                latencies.append(hit - ev["onset"])
                if ev["cls"] == "convulsive_PG":
                    pg_caught += 1

        episodes = 0
        for (key, _tid), tw in tracks.items():
            if key not in neg_keys:
                continue
            fs = fires_for_track([t for t, _ in tw], [s for _, s in tw], rule)
            episodes += (1 if fs else 0) + sum(
                1 for a, b in itertools.pairwise(fs) if b - a > EPISODE_GAP_S
            )

        # Same collapse per camera: simultaneous fires from every track on one
        # video merge before the 30 s gap rule, giving the operator's alarm count
        # (the ensemble-study side finding, measured uniformly here since 2026-08-28).
        episodes_cam = 0
        for key in neg_keys:
            fs = fires.get(key, [])
            episodes_cam += (1 if fs else 0) + sum(
                1 for a, b in itertools.pairwise(fs) if b - a > EPISODE_GAP_S
            )

        # Fraction of the negative footage with the alarm lit: each fire covers the
        # window that produced it ([t - ALARM_WINDOW_S, t]), unioned across every
        # track on the camera. Episode counts alone invert meaning at high fire
        # rates (an alarm that never turns off is "one episode"); this is the
        # honest quiet metric next to them (the 2026-08-27 v4 lesson).
        alarm_s = 0.0
        for key in neg_keys:
            ivs = sorted((t - ALARM_WINDOW_S, t) for t in fires.get(key, []))
            end = None
            for s, e in ivs:
                if end is None or s > end:
                    alarm_s += e - s
                    end = e
                elif e > end:
                    alarm_s += e - end
                    end = e
        alarm_frac = alarm_s / (args.neg_hours * 3600.0)

        latencies.sort()
        med = latencies[len(latencies) // 2] if latencies else None
        results.append(
            {
                "rule": rule,
                "caught": caught,
                "of": len(events),
                "pg": f"{pg_caught}/{pg_total}",
                "median_latency_s": round(med, 2) if med is not None else None,
                "fp_episodes_per_h": round(episodes / args.neg_hours, 2),
                "fp_episodes_per_h_camera": round(episodes_cam / args.neg_hours, 2),
                "neg_time_in_alarm": round(alarm_frac, 4),
            }
        )
        print(
            f"{rule:18s} caught {caught:2d}/{len(events)}  PG {pg_caught}/{pg_total}  "
            f"median lat {med if med is not None else '-':>6}  "
            f"FP eps/h {episodes / args.neg_hours:5.1f}  "
            f"cam/h {episodes_cam / args.neg_hours:5.1f}  in-alarm {100 * alarm_frac:4.1f}%"
        )

    out = args.eval_dir / "decision_layer_study.json"
    out.write_text(json.dumps({"rules": results}, indent=1) + "\n", encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
