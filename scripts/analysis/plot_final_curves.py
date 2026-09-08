#!/usr/bin/env python3
"""Render the committed validation figures for every model lane.

Reads final_models/<lane>/curves/*.csv and writes two PNG files into
final_models/<lane>/: validation_curves.png, which puts every run of the lane on one
axis with the run of record emphasised and labelled with its hash, and
champion_vs_baseline.png, which carries only the model of record and the baseline or
bar holder it is compared with. A lane with no CSV file is skipped, which is not an
error: a lane whose first run has not landed yet has no curve.

Usage:
  uv run python scripts/analysis/plot_final_curves.py --lanes eyes joints fall seizure reid
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend("Agg")

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "final_models"
DPI, FIGSIZE, DASH = 120, (9.0, 5.4), (0, (6, 3))
GREY, CHAMPION, GATE = "#b7bec8", "#12355b", "#7a6ea8"
BASELINE = ("#b8471f", "#3d7a52")
REST = ("#8fa3b8", "#a89179", "#7f9c8c", "#a08ba5")


class Lane(NamedTuple):
    title: str
    ylabel: str
    note: str
    claim: str
    champion: str | None = None
    champion_label: str = ""
    champion_value: float | None = None
    baselines: tuple[tuple[str, str], ...] = ()
    gate: tuple[float, str] | None = None


# The one table of lane facts; everything outside it is layout. The third CSV column is
# named val_ap in every one of the files, but for the fall and seizure verifier runs it
# carries a validation AUC rather than an average precision, so the y axis label is
# taken per lane from this table and never from the column name. A gate value is drawn
# as a horizontal line only where it sits on the same axis as the curves; the fall and
# seizure exam bars are quoted in the footnote, being measured on held-out video.
LANES: dict[str, Lane] = {
    "eyes": Lane(
        title="Eyes lane, person detector",
        ylabel="Validation AP at IoU 0.50 to 0.95",
        note="Exam: person_v2 dev-val, 1800 images and 33755 boxes, Warehouse_005 held out by scene.",
        claim="The run of record reaches 0.8356 on the person_v2 dev-val exam, above the pre-registered bar of 0.8353 and above the retained fallback checkpoint e98de401 at 0.8341.",
        champion="detector_b1_rtv4s_seed0",
        champion_label="RT-DETRv4-S seed 0, checkpoint 345c30a5",
        baselines=(("detector_deim_seed1", "DEIM-D-FINE-S seed 1, checkpoint e98de401"),),
        gate=(0.8353, "Pre-registered bar"),
    ),
    "joints": Lane(
        title="Joints lane, pose",
        ylabel="Validation keypoint AP, COCO val2017",
        note="Exam: COCO val2017 under the ground-truth box protocol. Validation ran every second epoch, so each curve carries 30 points across 60 epochs. The body7 initialisation is a fixed weight scored once under the identical protocol, so it has no per-epoch curve and is drawn as a gate line.",
        claim="The run of record reaches keypoint AP 0.774115 at epoch 38 against the body7 initialisation e48f03d0 at 0.764499 on the identical protocol, a gain of 0.009616.",
        champion="pose_rtmpose_m_p0_seed0",
        champion_label="RTMPose-m seed 0, checkpoint a0a035fe",
        gate=(0.764499, "Body7 initialisation e48f03d0 at"),
    ),
    "fall": Lane(
        title="Fall lane, event verifier",
        ylabel="Validation AUC, train-time holdout",
        note="The model of record has no per-epoch record in this repository, so its best validation AUC of 0.99589 is drawn as a single reference value. The fall_push F0, F1, F2 and F3 arms validate on the push v3 pinned holdout of 18 positive and 151 negative windows, a different surface from the train-time holdout of fall_v3 and fall_v2pose. The three fall_v2c_F0 curves are the low band: they validate on the v2c-tree holdout of 169 positive and 1,832 negative windows, a third surface again, and their per-seed values and hashes are recorded in final_models/fall/README.md.",
        claim="On the multicam all-cameras exam of 192 videos the model of record scores window AUC 0.79276 against 0.78474 for fall_v2pose. Those exam values are measured on held-out video and are not on this axis.",
        champion_label="fall_v3, checkpoint 3d2d17ee, ONNX 67954c04",
        champion_value=0.99589,
        baselines=(("fall_verifier_v2pose", "fall_v2pose, ONNX b66c3a0a"),),
    ),
    "seizure": Lane(
        title="Seizure lane, event verifier",
        ylabel="Validation AUC, train-time holdout",
        note="The model of record has no per-epoch record in this repository, so its best validation AUC of 0.77152 is drawn as a single reference value. The push 2 and push 3 arms are cross-validation folds that hold out different patients, so curves from different folds measure different validation sets. One curve, conv_v2c_finalB_s20260826, is the v2c refit and validates on the v2c clinical tree holdout of 154 positive and 5,120 negative windows, a different surface again, so its best does not stand against the curves beside it; that best and the hashes behind it are recorded in final_models/seizure/README.md.",
        claim="On the wusahzu testsplit exam of 16 videos the model of record scores window AUC 0.69841 against 0.53728 for convulsion_v2pose. The standing bars are E1 0.87422 held by final B and E2 0.72916 held by 07c399be, both measured on held-out video and not on this axis.",
        champion_label="convulsion_v2_real, checkpoint 715d172c, ONNX 07c399be",
        champion_value=0.77152,
        baselines=(
            ("seizure_verifier_v2pose", "convulsion_v2pose, ONNX 614b92fa"),
            ("seizure_push_finalB", "final B, checkpoint 535141c1, ONNX b9e04d03"),
        ),
    ),
    "reid": Lane(
        title="Re-identification lane, appearance embedder",
        ylabel="MEVID test mean average precision, general protocol",
        note="The trainer evaluates every ten epochs. The run of record carries three points over 30 epochs, 0.137423, 0.156771 and 0.204782, the last being the epoch the model of record is taken from. Rerun A2, the same recipe with the learning-rate step moved from epoch 20 to 40 over 60 epochs, reproduces the first two points exactly and ends at 0.188552 at epoch 60, below the run of record, so it was not promoted. The published same-clothes floor of 0.39 is an external value with no checkpoint hash and is drawn as a reference line, not as a measured comparator.",
        claim="The model of record reaches general-protocol mAP 0.204782 with rank-1 0.490506 on the MEVID test split at epoch 30. The external same-clothes floor of 0.39 is not reached under the general protocol; the in-house same-clothes approximation scores 0.293091 and the camera-debiased protocol 0.348949, neither of which is comparable to the published figure.",
        champion="reid_mevid_a1",
        champion_label="OSNet-x1_0 MEVID Phase A, checkpoint 66623239, ONNX 60164a99",
        baselines=(
            ("reid_mevid_a2_long", "rerun A2, 60 epochs, checkpoint 78ba6b55, ONNX 151eda30"),
        ),
        gate=(0.39, "External same-clothes floor, no hash"),
    ),
}


def load_curves(root: Path, lane: str) -> dict[str, pd.DataFrame]:
    """Return every curve of a lane keyed by file stem, empty val_ap cells dropped."""
    out: dict[str, pd.DataFrame] = {}
    folder = root / lane / "curves"
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("*.csv")):
        # The pose CSVs have an empty val_ap on odd epochs because validation ran every
        # second epoch, so those cells are dropped rather than read as zero.
        frame = pd.read_csv(path)
        if "val_ap" not in frame.columns and "test_map" in frame.columns:
            # The re-identification trainer writes its own seven columns; the evaluated
            # mean average precision is the one plotted, under the shared column name.
            frame = frame.rename(columns={"test_map": "val_ap"})
        frame = frame.dropna(subset=["val_ap"])
        if not frame.empty:
            out[path.stem] = frame
    return out


def _draw(ax, frame, colour: str, width: float, label: str | None, z: int) -> None:
    ax.plot(frame["epoch"], frame["val_ap"], color=colour, linewidth=width, label=label, zorder=z)


def _champion(ax, cfg: Lane, curves: dict) -> None:
    """Emphasise the run of record, or its recorded best value when it has no curve."""
    if cfg.champion in curves:
        frame = curves[cfg.champion]
        _draw(ax, frame, CHAMPION, 2.4, cfg.champion_label, 6)
        best = frame.loc[frame["val_ap"].idxmax()]
        style = {"textcoords": "offset points", "fontsize": 8, "color": CHAMPION, "ha": "right"}
        ax.annotate(f"best {best['val_ap']:g}", (best["epoch"], best["val_ap"]), (-6, 8), **style)
    elif cfg.champion_value is not None:
        label = f"{cfg.champion_label}, best {cfg.champion_value:g}"
        ax.axhline(cfg.champion_value, color=CHAMPION, linestyle=DASH, linewidth=2.0, label=label)


def _finish(ax, cfg: Lane, note: str, out: Path) -> int:
    fig = ax.figure
    ax.set_xlabel("Epoch")
    ax.set_ylabel(cfg.ylabel)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.margins(y=0.09)  # headroom, so a reference line or annotation clears the frame
    ax.legend(loc="best", fontsize=8, framealpha=0.92)
    lines = textwrap.wrap(note, 128)
    fig.tight_layout(rect=(0.0, 0.035 + 0.030 * len(lines), 1.0, 1.0))
    fig.text(0.010, 0.012, "\n".join(lines), fontsize=7.2, va="bottom", ha="left", color="#3d4450")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out.stat().st_size


def render(lane: str, cfg: Lane, curves: dict, out: Path, gate: float | None, every: bool) -> int:
    """Draw one figure. With every, all runs of the lane; otherwise the compared pair."""
    _, ax = plt.subplots(figsize=FIGSIZE)
    head = "validation metric per epoch" if every else "model of record against the baseline"
    ax.set_title(f"{cfg.title}: {head}", fontsize=12.5, loc="left")
    if every:
        named = {cfg.champion} | {stem for stem, _ in cfg.baselines}
        rest = [stem for stem in curves if stem not in named]
        itemised = len(curves) <= 6
        for index, stem in enumerate(rest):
            colour = REST[index % len(REST)] if itemised else GREY
            _draw(ax, curves[stem], colour, 1.1, stem if itemised else None, 1)
        if rest and not itemised:
            ax.plot([], [], color=GREY, linewidth=1.1, label=f"other runs ({len(rest)})")
        note = f"Source: final_models/{lane}/curves, {len(curves)} files. {cfg.note}"
    else:
        stems = [s for s in (cfg.champion, *(b[0] for b in cfg.baselines)) if s in curves]
        files = ", ".join(f"final_models/{lane}/curves/{stem}.csv" for stem in stems)
        note = f"Source: {files}. {cfg.claim}"
    for index, (stem, label) in enumerate(cfg.baselines):
        if stem in curves:
            _draw(ax, curves[stem], BASELINE[index % len(BASELINE)], 1.9, label, 4)
    if cfg.gate is not None:
        value = cfg.gate[0] if gate is None else gate
        ax.axhline(value, color=GATE, linestyle="--", label=f"{cfg.gate[1]} {value:g}")
    _champion(ax, cfg, curves)
    return _finish(ax, cfg, note, out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lanes", nargs="+", default=list(LANES), choices=list(LANES))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    # Only a lane whose Lane.gate is set draws a gate line; the fall and seizure lanes draw a
    # recorded reference value instead, so an override aimed at one of them has nothing to move.
    gated = [name for name, cfg in LANES.items() if cfg.gate is not None]
    gate_help = (
        "override one lane gate line, so the drawn value is data and not a source constant. "
        f"Only {', '.join(gated)} draw one"
    )
    parser.add_argument("--gate", action="append", default=[], metavar="LANE=VALUE", help=gate_help)
    args = parser.parse_args()
    overrides = {pair.split("=", 1)[0]: float(pair.split("=", 1)[1]) for pair in args.gate}
    for name in overrides:
        if name not in gated:
            raise SystemExit(
                f"--gate {name}=... would be discarded: {name} draws no gate line. "
                f"The lanes that draw one are {', '.join(gated)}."
            )
    for lane in args.lanes:
        cfg, curves = LANES[lane], load_curves(args.root, lane)
        if not curves:
            print(f"{lane}: no curve file, skipped")
            continue
        gate, folder = overrides.get(lane), args.root / lane
        one = render(lane, cfg, curves, folder / "validation_curves.png", gate, True)
        two = render(lane, cfg, curves, folder / "champion_vs_baseline.png", gate, False)
        print(f"{lane}: {len(curves)} curves, validation_curves.png {one} B, pair {two} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
