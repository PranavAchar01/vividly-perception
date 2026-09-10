"""Run the substance analyzer over a video file and report what it decided, when.

    uv run python scripts/substance_check.py --in clip.mp4 [--render out.mp4] [--json out.json]

For every confirmed detection it prints the moment of confirmation, the delay from the pool's
appearance when a substance_synth sidecar sits beside the clip, the peak area fraction and how
long the detection lasted, plus the time spent in each state and the per-update cost at p50
and p95. That last pair is the number the spec's 3 ms budget is judged on, so it is measured
around update() alone and never includes decode or draw.

Run it on an unaltered take as well as a spilled one: the room's red step ladder and red shelf
handles are the false positives that matter, and the only proof they are handled is a clean
take that reports no detections at all.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

STATE_BGR = {
    "idle": (170, 170, 170),
    "candidate": (0, 190, 255),
    "confirmed": (60, 60, 255),
    "priming": (200, 140, 90),
}


def _load_analyzer(spec: str):
    """Instantiate the analyzer named by 'module' or 'module:Class'.

    Indirection exists so this script can be exercised against a stand-in while
    vividly.analysis.substance is still being written; the default is the real thing.
    """
    mod_name, _, cls_name = spec.partition(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name or "SubstanceAnalyzer")()


def _load_overlay():
    """vividly.hud.substance.draw_substance(img, state, t_now), drawn in place.

    Imported here rather than at module scope so this script still runs, on a plain contour,
    against a checkout where the HUD module is not present yet.
    """
    try:
        from vividly.hud.substance import draw_substance
    except ImportError:
        return None
    return draw_substance


def _plain_overlay(frame: np.ndarray, st, t: float) -> None:
    bucket = "priming" if getattr(st, "priming", False) else st.state
    colour = STATE_BGR.get(bucket, (255, 255, 255))
    for poly in st.polygons:
        cv2.polylines(frame, [np.asarray(poly, np.int32).reshape(-1, 1, 2)], True, colour, 3)
    label = (
        f"{t:6.2f}s  {bucket:9s}  area {st.area_frac:.5f}  pools {st.n_pools}  "
        f"held {st.held_s:.1f}s  growth {st.growth_frac_per_s:+.5f}/s"
    )
    cv2.putText(frame, label, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
    cv2.putText(frame, label, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--render", type=Path, help="write an annotated copy here")
    ap.add_argument("--json", dest="json_out", type=Path, help="write the report here")
    ap.add_argument(
        "--analyzer",
        default="vividly.analysis.substance:SubstanceAnalyzer",
        help="module[:Class] to instantiate",
    )
    args = ap.parse_args(argv)

    src = args.src if args.src.is_absolute() else REPO_ROOT / args.src
    analyzer = _load_analyzer(args.analyzer)

    sidecar_path = src.with_suffix(".substance.json")
    truth = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else None

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    dt = 1.0 / fps

    writer = None
    overlay = _load_overlay() if args.render else None
    overlay_name = "vividly.hud.substance.draw_substance" if overlay else "plain contour"
    if args.render:
        dst = args.render if args.render.is_absolute() else REPO_ROOT / args.render
        dst.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    costs_ms: list[float] = []
    totals: dict[str, float] = {}
    detections: list[dict] = []
    open_det: dict | None = None
    peak = {"area_frac": 0.0, "t": 0.0}
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = n / fps
        t0 = time.perf_counter()
        st = analyzer.update(frame, t)
        costs_ms.append((time.perf_counter() - t0) * 1e3)

        bucket = "priming" if getattr(st, "priming", False) else st.state
        totals[bucket] = totals.get(bucket, 0.0) + dt
        if st.area_frac > peak["area_frac"]:
            peak = {"area_frac": float(st.area_frac), "t": t}

        # A detection is one unbroken run of "confirmed": the analyzer's own release_s already
        # bridges the dropouts, so a second run here means the event genuinely cleared.
        if st.state == "confirmed":
            if open_det is None:
                open_det = {"confirmed_at_s": t, "peak_area_frac": 0.0, "peak_n_pools": 0}
            open_det["peak_area_frac"] = max(open_det["peak_area_frac"], float(st.area_frac))
            open_det["peak_n_pools"] = max(open_det["peak_n_pools"], int(st.n_pools))
            open_det["last_s"] = t
        elif open_det is not None:
            open_det["duration_s"] = open_det["last_s"] - open_det["confirmed_at_s"] + dt
            detections.append(open_det)
            open_det = None

        if writer is not None:
            if overlay is not None:
                overlay(frame, st, t)
            else:
                _plain_overlay(frame, st, t)
            writer.write(frame)
        n += 1
    cap.release()
    if open_det is not None:
        open_det["duration_s"] = open_det["last_s"] - open_det["confirmed_at_s"] + dt
        detections.append(open_det)
    if writer is not None:
        writer.release()

    if truth is not None:
        for d in detections:
            d["delay_s"] = d["confirmed_at_s"] - truth["appearance_s"]

    c = np.asarray(costs_ms) if costs_ms else np.zeros(1)
    timing = {
        "n_updates": len(costs_ms),
        "p50_ms": float(np.percentile(c, 50)),
        "p95_ms": float(np.percentile(c, 95)),
        "max_ms": float(c.max()),
    }
    report = {
        "video": {"path": str(src), "width": w, "height": h, "fps": fps, "n_frames": n},
        "analyzer": args.analyzer,
        "ground_truth": (
            None
            if truth is None
            else {
                "sidecar": str(sidecar_path),
                "appearance_s": truth["appearance_s"],
                "stable_s": truth["stable_s"],
                "area_frac": truth["pool"]["area_frac"],
            }
        ),
        "detections": detections,
        "state_totals_s": totals,
        "peak": peak,
        "timing_ms": timing,
    }

    print(f"video      {src}")
    print(f"           {n} frames  {w}x{h}  {fps:.2f} fps  {n / fps:.2f} s")
    print(f"analyzer   {args.analyzer}")
    if args.render:
        print(f"overlay    {overlay_name}")
    if truth is not None:
        print(
            f"truth      pool appears {truth['appearance_s']:.2f}s, stable "
            f"{truth['stable_s']:.2f}s, area_frac {truth['pool']['area_frac']:.5f}"
        )
    else:
        print("truth      no sidecar beside this clip, delays not scored")
    if detections:
        for i, d in enumerate(detections, 1):
            delay = f"delay {d['delay_s']:+6.2f} s   " if "delay_s" in d else ""
            print(
                f"detect #{i}  confirmed {d['confirmed_at_s']:6.2f} s   {delay}"
                f"peak area_frac {d['peak_area_frac']:.5f}   pools {d['peak_n_pools']}   "
                f"held {d['duration_s']:.2f} s"
            )
    else:
        print("detect     none confirmed")
    print("states     " + "  ".join(f"{k} {v:.2f}s" for k, v in sorted(totals.items())))
    print(f"peak       area_frac {peak['area_frac']:.5f} at {peak['t']:.2f} s")
    print(
        f"per-update p50 {timing['p50_ms']:.2f} ms  p95 {timing['p95_ms']:.2f} ms  "
        f"max {timing['max_ms']:.2f} ms  over {timing['n_updates']} updates"
    )

    if args.json_out:
        jd = args.json_out if args.json_out.is_absolute() else REPO_ROOT / args.json_out
        jd.parent.mkdir(parents=True, exist_ok=True)
        jd.write_text(json.dumps(report, indent=1))
        print(f"wrote      {jd}")
    if args.render:
        print(f"wrote      {args.render}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
