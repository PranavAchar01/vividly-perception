#!/usr/bin/env python3
"""Pose v2 3-seed curves from the committed per-epoch CSVs.

Reads the three committed per-epoch CSVs under final_models/joints/curves/:
pose_rtmpose_m_p0_seed0.csv, pose_rtmpose_m_p1_seed1.csv and
pose_rtmpose_m_p2_seed2.csv. Writes
  pose_v2_ap_curves.png    validation AP by epoch, 3 seeds and gate
  pose_v2_loss_curves.png  epoch-average train loss by epoch

Neither figure is committed. The destination defaults to outputs/pod_backup, the
location research/pose-seed-floor-and-champion.md records for them; it is not
tracked, so it is absent in a fresh clone and the script creates it before writing.

Usage: uv run python scripts/analysis/plot_pose_curves.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
CURVES = REPO / "final_models" / "joints" / "curves"
OUT = REPO / "outputs" / "pod_backup"

# Raw body7 init on the identical gt-bbox protocol, log outputs/pod_backup/
# body7_baseline_eval.log; the value is carried with its hash e48f03d0 on
# final_models/joints/README.md.
BASELINE = 0.764499

RUNS = [  # (csv stem, label, color)
    (
        "pose_rtmpose_m_p0_seed0",
        "seed 0, model of record (best 0.7741 @ e38, a0a035fe)",
        "#18398a",
    ),
    ("pose_rtmpose_m_p1_seed1", "seed 1 (best 0.7740 @ e30, 8d990435)", "#6c8fce"),
    ("pose_rtmpose_m_p2_seed2", "seed 2 (best 0.7741 @ e32, 5bcca5bd)", "#c2410c"),
]


def load(curves: Path, stem: str):
    eps, losses, vep, vap = [], [], [], []
    with (curves / f"{stem}.csv").open() as f:
        for r in csv.DictReader(f):
            ep = int(r["epoch"])
            if r["train_loss"]:
                eps.append(ep)
                losses.append(float(r["train_loss"]))
            if r["val_ap"]:
                vep.append(ep)
                vap.append(float(r["val_ap"]))
    return eps, losses, vep, vap


def render(curves: Path, out: Path) -> int:
    fig1, ax1 = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig2, ax2 = plt.subplots(figsize=(9, 5.5), dpi=150)

    for stem, label, color in RUNS:
        eps, losses, vep, vap = load(curves, stem)
        lw = 2.0 if "model of record" in label else 1.4
        ax1.plot(vep, vap, marker="o", ms=3, lw=lw, color=color, label=label)
        ax2.plot(eps, losses, lw=lw, color=color, label=label)

    ax1.axhline(BASELINE, color="#b91c1c", lw=1.4, ls="--")
    ax1.text(
        1,
        BASELINE + 0.0004,
        "gate: raw body7 init = 0.7645 (identical eval)",
        fontsize=8,
        color="#b91c1c",
    )

    for ax in (ax1, ax2):
        ax.set_xlabel("epoch (60 per run; ~7 min each on the RTX 5090)")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(loc="lower right" if ax is ax1 else "upper right", fontsize=8)
    ax1.set_ylabel("COCO val2017 AP, gt-bbox protocol (5,000 imgs)")
    ax1.set_title("v2 pose: RTMPose-m person fine-tune, 3 seeds, accuracy by epoch")
    ax2.set_ylabel("training loss (epoch average of logged iterations)")
    ax2.set_title("v2 pose: training loss by epoch")

    fig1.tight_layout()
    fig2.tight_layout()
    out.mkdir(parents=True, exist_ok=True)  # not tracked, so absent in a fresh clone
    fig1.savefig(out / "pose_v2_ap_curves.png")
    fig2.savefig(out / "pose_v2_loss_curves.png")
    plt.close(fig1)
    plt.close(fig2)
    print("wrote", out / "pose_v2_ap_curves.png")
    print("wrote", out / "pose_v2_loss_curves.png")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--curves",
        type=Path,
        default=CURVES,
        help=f"directory holding the per-epoch CSVs (default {CURVES.relative_to(REPO)})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help=f"directory the two figures are written to (default {OUT.relative_to(REPO)})",
    )
    args = parser.parse_args(argv)
    return render(args.curves, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
