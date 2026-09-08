#!/usr/bin/env python3
"""Assemble the AJILE12 negative diet into verifier track sequences.

Reads the harvested sessions (scripts/data/harvest_ajile12.py: per-session pose npz
+ epochs.json) and emits an extraction-shaped directory (sequences/ + meta/) that
scripts/events/train_event_verifier.py can consume as a `force_label: 0` source.

Diet rule (configs/datasets/ajile12.yaml, non-negotiable): windows come only from
spans positively labelled as normal activity; a span qualifies if and only if every
comma-part of its label is in the allowed set; any "Blocklist" label and all
unlabelled time are excluded. Patients' seizure spans never become positives, and
nothing here is ever a positive: the emitted metas carry no event keys and every
consuming training configuration forces label 0.

Row synthesis (fixed, documented; the per-frame contract mirrors the row schema of
scripts/events/extract_track_sequences.py, its frozen 9 fields plus the `kp` block that
`emit_keypoints: true` adds):
  kp      COCO-17: [nose, ears, shoulders, elbows, wrists] mapped from the 9 AJILE
          joints; eyes/hips/knees/ankles absent -> conf 0. Present joint conf = 1.0
          (the deposit has no per-joint confidence; non-NaN == tracked).
  kpc     mean confidence over all 17 entries (same reduction as extraction).
  angle   posture_angle_deg needs hips -> always null here. Recorded so the gap is
          visible: WU-SAHZU bed footage under blankets also loses hips, so
          angle_valid=0 is a real state, not an assembly artefact.
  box     from the per-frame extent of present joints: h = 3.0 x vertical extent
          (upper body ~ 1/3 of a person), w = 1.4 x horizontal extent, floors 40 px,
          top padded 0.15 x extent above the highest joint. A proxy, not a detector
          box; whether the diet transfers is decided on patient-held-out val AUC.
  gaps    frames with <3 present joints emit no row (tracker gaps are data).
One segment = one sequence file with a single track (id 0), t re-zeroed.

Volume is capped (per-session and total seconds, subject round-robin, seeded) so
the diet stays a diet. Usage:
    uv run python scripts/data/assemble_ajile12_diet.py --config configs/events/ajile12_diet_v1.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

# AJILE joint name -> COCO-17 index
COCO_IDX = {
    "Nose": 0,
    "L_Ear": 3,
    "R_Ear": 4,
    "L_Shoulder": 5,
    "R_Shoulder": 6,
    "L_Elbow": 7,
    "R_Elbow": 8,
    "L_Wrist": 9,
    "R_Wrist": 10,
}
ALLOWED = {"Sleep/rest", "Inactive", "Eat", "Talk", "TV", "Computer/phone", "Other activity"}


def normal_spans(ep: dict, min_s: float) -> list[tuple[float, float]]:
    spans = []
    for lab, t0, t1 in zip(ep["labels"], ep["start_time_s"], ep["stop_time_s"], strict=True):
        parts = [p.strip() for p in str(lab).split(",")]
        if all(p in ALLOWED for p in parts) and (t1 - t0) >= min_s:
            spans.append((float(t0), float(t1)))
    return spans


def rows_for_segment(pose: dict[str, np.ndarray], hz: float, t0: float, t1: float) -> list[dict]:
    i0, i1 = int(t0 * hz), int(t1 * hz)
    rows = []
    for i in range(i0, i1):
        kp = [[0.0, 0.0, 0.0]] * 17
        xs, ys, present = [], [], 0
        for name, j in COCO_IDX.items():
            x, y = pose[name][i]
            if np.isnan(x) or np.isnan(y):
                continue
            kp[j] = [round(float(x), 2), round(float(y), 2), 1.0]
            xs.append(float(x))
            ys.append(float(y))
            present += 1
        if present < 3:
            continue
        ext_x, ext_y = max(xs) - min(xs), max(ys) - min(ys)
        w = max(1.4 * ext_x, 40.0)
        h = max(3.0 * ext_y, 40.0)
        top = min(ys) - 0.15 * ext_y
        rows.append(
            {
                "f": i - i0 + 1,
                "t": round((i - i0 + 1) / hz, 4),
                "track": 0,
                "cx": round((min(xs) + max(xs)) / 2.0, 2),
                "cy": round(top + h / 2.0, 2),
                "w": round(w, 2),
                "h": round(h, 2),
                "angle": None,  # torso angle needs hips; AJILE12 is upper-body only
                "kpc": round(present / 17.0, 4),
                "kp": kp,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    sessions_dir = Path(cfg["sessions_dir"])
    out_dir = Path(cfg["out_dir"])
    seg_s = float(cfg["segment_s"])
    min_span_s = float(cfg["min_span_s"])
    per_session_cap = float(cfg["per_session_cap_s"])
    total_cap = float(cfg["total_cap_s"])
    rng = np.random.default_rng(int(cfg["seed"]))

    (out_dir / "sequences").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)

    stems = sorted(p.name[: -len(".pose.npz")] for p in sessions_dir.glob("*.pose.npz"))
    # subject round-robin so the cap spreads across all 12 patients
    by_sub: dict[str, list[str]] = {}
    for s in stems:
        by_sub.setdefault(s.split("_")[0], []).append(s)
    order = []
    while any(by_sub.values()):
        for sub in sorted(by_sub):
            if by_sub[sub]:
                order.append(by_sub[sub].pop(0))

    landed_s, n_seq = 0.0, 0
    for stem in order:
        if landed_s >= total_cap:
            break
        ep = json.loads((sessions_dir / f"{stem}.epochs.json").read_text())
        npz = np.load(sessions_dir / f"{stem}.pose.npz", allow_pickle=False)
        hz = float(np.median(npz["rate_hz"]))
        pose = {n: npz[f"pose__{n}"] for n in COCO_IDX}
        spans = normal_spans(ep, min_span_s)
        if not spans:
            continue
        rng.shuffle(spans)
        sess_s = 0.0
        for t0, t1 in spans:
            if sess_s >= per_session_cap or landed_s >= total_cap:
                break
            # deterministic segment start inside the span
            max_start = (t1 - t0) - seg_s
            s0 = t0 + (rng.uniform(0.0, max_start) if max_start > 0 else 0.0)
            s1 = min(s0 + seg_s, t1)
            rows = rows_for_segment(pose, hz, s0, s1)
            if len(rows) < 0.5 * seg_s * hz:  # too gappy to be a useful negative
                continue
            key = f"{stem}_t{int(s0)}"
            with (out_dir / "sequences" / f"{key}.jsonl").open("w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            (out_dir / "meta" / f"{key}.json").write_text(
                json.dumps(
                    {
                        "key": key,
                        "location": stem.split("_")[0],
                        "fps": hz,
                        "has_seizure": False,
                        "has_fall": False,
                        "events_t": [],
                        "source_span_s": [round(s0, 1), round(s1, 1)],
                        "n_rows": len(rows),
                        "n_frames_processed": int((s1 - s0) * hz),
                        "n_tracks": 1,
                        "provenance": {
                            "dataset": "ajile12 (DANDI:000055, CC-BY-4.0, negative diet)",
                            "assembler": "scripts/data/assemble_ajile12_diet.py",
                            "diet_rule": (
                                "labelled-normal spans only; blocklist and unlabelled excluded"
                            ),
                        },
                    },
                    indent=1,
                )
            )
            sess_s += s1 - s0
            landed_s += s1 - s0
            n_seq += 1
    print(
        json.dumps(
            {"sequences": n_seq, "minutes": round(landed_s / 60.0, 1), "out_dir": str(out_dir)}
        )
    )


if __name__ == "__main__":
    main()
