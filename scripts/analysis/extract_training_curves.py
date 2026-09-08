#!/usr/bin/env python3
"""Extract per-epoch accuracy and loss curves from backed-up training logs to CSV.

The practice recorded in research/pose-seed-floor-and-champion.md, section "Curves
stored (2026-09-04)": accuracy and loss by epoch are stored for every model trained
in this repository. Raw logs live in outputs/pod_backup/ (gitignored,
large); this script distills each into a small committed CSV under
final_models/<lane>/curves/ with columns: epoch, train_loss, val_ap. An empty
val_ap means no validation ran that epoch (pose validates every 2 epochs). The
reid lane is the exception: its trainer writes its own seven-column curve, so
final_models/reid/curves/reid_mevid_a1.csv does not come from here.

Log dialects handled:
  deim    - DEIM detector: one "Averaged stats: ... loss: x (avg)" per epoch,
            one COCO AP block per epoch.
  b1      - RT-DETRv4 no-teacher: per-iter "Epoch: [N] ... loss: x (running)",
            the last line of each epoch carries the epoch average; one COCO AP
            block per epoch.
  mmpose  - mmengine: "Epoch(train) [N][i/total] ... loss: x" every 50 iters
            (averaged per epoch here), "Epoch(val) [N][last] ... coco/AP: x".

Usage:  uv run python scripts/analysis/extract_training_curves.py
Re-run any time; output is deterministic from the logs.
"""

import csv
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "outputs" / "pod_backup"
FINAL_MODELS = REPO / "final_models"

# Lane of record for a run, read from its curve name: detector runs belong to the eyes
# lane, pose runs to joints, fall runs to fall, and seizure or convulsion runs to seizure.
LANE_PREFIXES = (
    ("detector", "eyes"),
    ("pose", "joints"),
    ("fall", "fall"),
    ("seizure", "seizure"),
    ("conv", "seizure"),
)

COCO_AP_RE = re.compile(
    r"Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=\s+all \| maxDets=100 \] = ([0-9.]+)"
)
DEIM_LOSS_RE = re.compile(r"Averaged stats:.*?loss: [0-9.]+ \(([0-9.]+)\)")
B1_ITER_RE = re.compile(r"Epoch: \[(\d+)\].*?loss: [0-9.]+ \(([0-9.]+)\)")
MM_TRAIN_RE = re.compile(r"Epoch\(train\)\s*\[\s*(\d+)\].*?loss: ([0-9.]+)")
MM_VAL_RE = re.compile(r"Epoch\(val\)\s*\[\s*(\d+)\].*?coco/AP: ([0-9.]+)")

RUNS = [  # (csv name, log file, dialect). Paths are relative to outputs/pod_backup
    # unless the dialect is "verifier", whose path is relative to the repo root
    # (the trainer's own report.json carries per-epoch history).
    ("detector_deim_seed0", "train_seed0.log", "deim"),
    ("detector_deim_seed1", "train_seed1.log", "deim"),
    ("detector_deim_seed2", "train_seed2.log", "deim"),
    ("detector_b1_rtv4s_seed0", "train_rtv4_s0_final.log", "b1"),
    ("pose_rtmpose_m_p0_seed0", "train_pose_p0_final.log", "mmpose"),
    ("pose_rtmpose_m_p1_seed1", "train_pose_p1_final.log", "mmpose"),
    ("pose_rtmpose_m_p2_seed2", "train_pose_p2_final.log", "mmpose"),
    ("fall_verifier_v2pose", "outputs/verifier/fall_v2pose/report.json", "verifier"),
    ("seizure_verifier_v2pose", "outputs/verifier/convulsion_v2pose/report.json", "verifier"),
    ("seizure_push_finalA", "outputs/verifier/conv_finalA/report.json", "verifier"),
    ("seizure_push_finalB", "outputs/verifier/conv_finalB/report.json", "verifier"),
    ("seizure_push_finalC", "outputs/verifier/conv_finalC/report.json", "verifier"),
    # Fall push v3 (research/fall-push-v3-protocol.md): F0 control floor and
    # F1 label-geometry, 3 seeds each on the pinned holdout.
    ("fall_push_F0_s20260826", "outputs/verifier/fall_F0_s20260826/report.json", "verifier"),
    ("fall_push_F0_s20260901", "outputs/verifier/fall_F0_s20260901/report.json", "verifier"),
    ("fall_push_F0_s20260905", "outputs/verifier/fall_F0_s20260905/report.json", "verifier"),
    ("fall_push_F1_s20260826", "outputs/verifier/fall_F1_s20260826/report.json", "verifier"),
    ("fall_push_F1_s20260901", "outputs/verifier/fall_F1_s20260901/report.json", "verifier"),
    ("fall_push_F1_s20260905", "outputs/verifier/fall_F1_s20260905/report.json", "verifier"),
    # Arm F2 (fall_kpgeo_v1): keypoint channels on `contain` labelling (F1 failed the
    # promotion gate), 5 seeds, plus the two control-extension F0 seeds that complete
    # the 5-seed control arm (the first 3 control seeds are the F0 runs above).
    ("fall_push_F0_s20260910", "outputs/verifier/fall_F0_s20260910/report.json", "verifier"),
    ("fall_push_F0_s20260915", "outputs/verifier/fall_F0_s20260915/report.json", "verifier"),
    ("fall_push_F2kp_s20260826", "outputs/verifier/fall_F2kp_s20260826/report.json", "verifier"),
    ("fall_push_F2kp_s20260901", "outputs/verifier/fall_F2kp_s20260901/report.json", "verifier"),
    ("fall_push_F2kp_s20260905", "outputs/verifier/fall_F2kp_s20260905/report.json", "verifier"),
    ("fall_push_F2kp_s20260910", "outputs/verifier/fall_F2kp_s20260910/report.json", "verifier"),
    ("fall_push_F2kp_s20260915", "outputs/verifier/fall_F2kp_s20260915/report.json", "verifier"),
    # Seizure push 2 (research/seizure-push-2-arms.md). Arm HN is a per-window loss
    # weight on the out-of-fold diagnosis's class (a) (pat04 seizure-free windows),
    # ablated on fold F2 only at alpha 2 and 4; both failed the F2 gate, so neither has
    # a full grid. Arm AUG is track-space augmentation, F2 gate passed, full 4x3 grid.
    ("seizure_p2_hn2_f2_s20260826", "outputs/verifier/conv_p2hn2/f2_s1/report.json", "verifier"),
    ("seizure_p2_hn2_f2_s20260901", "outputs/verifier/conv_p2hn2/f2_s2/report.json", "verifier"),
    ("seizure_p2_hn2_f2_s20260905", "outputs/verifier/conv_p2hn2/f2_s3/report.json", "verifier"),
    ("seizure_p2_hn4_f2_s20260826", "outputs/verifier/conv_p2hn4/f2_s1/report.json", "verifier"),
    ("seizure_p2_hn4_f2_s20260901", "outputs/verifier/conv_p2hn4/f2_s2/report.json", "verifier"),
    ("seizure_p2_hn4_f2_s20260905", "outputs/verifier/conv_p2hn4/f2_s3/report.json", "verifier"),
    ("seizure_p2_aug_f1_s20260826", "outputs/verifier/conv_p2aug/f1_s1/report.json", "verifier"),
    ("seizure_p2_aug_f1_s20260901", "outputs/verifier/conv_p2aug/f1_s2/report.json", "verifier"),
    ("seizure_p2_aug_f1_s20260905", "outputs/verifier/conv_p2aug/f1_s3/report.json", "verifier"),
    ("seizure_p2_aug_f2_s20260826", "outputs/verifier/conv_p2aug/f2_s1/report.json", "verifier"),
    ("seizure_p2_aug_f2_s20260901", "outputs/verifier/conv_p2aug/f2_s2/report.json", "verifier"),
    ("seizure_p2_aug_f2_s20260905", "outputs/verifier/conv_p2aug/f2_s3/report.json", "verifier"),
    ("seizure_p2_aug_f3_s20260826", "outputs/verifier/conv_p2aug/f3_s1/report.json", "verifier"),
    ("seizure_p2_aug_f3_s20260901", "outputs/verifier/conv_p2aug/f3_s2/report.json", "verifier"),
    ("seizure_p2_aug_f3_s20260905", "outputs/verifier/conv_p2aug/f3_s3/report.json", "verifier"),
    ("seizure_p2_aug_f4_s20260826", "outputs/verifier/conv_p2aug/f4_s1/report.json", "verifier"),
    ("seizure_p2_aug_f4_s20260901", "outputs/verifier/conv_p2aug/f4_s2/report.json", "verifier"),
    ("seizure_p2_aug_f4_s20260905", "outputs/verifier/conv_p2aug/f4_s3/report.json", "verifier"),
]
# Fall push v3 arm F3 (research/fall-geometric-augmentation-arm.md): camera-geometry
# augmentation, 3 pre-declared strengths x K in {4, 8} augmented draws per train positive
# x 3 seeds.
RUNS += [
    (
        f"fall_push_F3geo_{label}_k{k}_s{seed}",
        f"outputs/verifier/fall_F3geo_{label}_k{k}_s{seed}/report.json",
        "verifier",
    )
    for label in ("weak", "med", "strong")
    for k in (4, 8)
    for seed in (20260826, 20260901, 20260905)
]
# Plus the post-hoc identity-K duplication control (8 byte-identical copies per train
# positive, no warp) that separates oversampling from camera geometry. Recorded because
# every trained model stores its curve, not because it selected anything.
RUNS += [
    (
        f"fall_push_F3geo_dupctrl_k8_s{seed}",
        f"outputs/verifier/fall_F3geo_dupctrl_k8_s{seed}/report.json",
        "verifier",
    )
    for seed in (20260826, 20260901, 20260905)
]
# Seizure push 3 (research/seizure-push-3-protocol.md): the harmonized AJILE12 negative
# diet as a training arm, 5 seeds per arm on fold F2. Recorded because every trained
# model stores its curve; the arm was not promoted.
RUNS += [
    (
        f"seizure_p3_{arm}_f2_s{seed}",
        f"outputs/verifier/conv_p3{arm}/f2_s{i}/report.json",
        "verifier",
    )
    for arm in ("base", "harm")
    for i, seed in enumerate((20260826, 20260901, 20260905, 20260910, 20260915), 1)
]
# The 2026-09-07 refits on the v2c track trees: the fall F0 control at three seeds
# (final_models/fall/README.md) and the seizure final B recipe at seed 20260826
# (final_models/seizure/README.md). Neither refit was promoted; the curves are stored
# because every trained model stores its curve.
RUNS += [
    (
        f"fall_v2c_F0_s{seed}",
        f"outputs/verifier/fall_F0_v2c_s{seed}/report.json",
        "verifier",
    )
    for seed in (20260826, 20260901, 20260905)
]
RUNS += [
    ("conv_v2c_finalB_s20260826", "outputs/verifier/conv_finalB_v2c/report.json", "verifier"),
]


def extract(txt: str, dialect: str):
    """Return {epoch: {"train_loss": float|None, "val_ap": float|None}} (epochs 0- or 1-based as logged)."""
    rows: dict[int, dict] = {}

    def row(ep: int):
        return rows.setdefault(ep, {"train_loss": None, "val_ap": None})

    if dialect == "deim":
        for i, loss in enumerate(DEIM_LOSS_RE.findall(txt)):
            row(i)["train_loss"] = float(loss)
        for i, ap in enumerate(COCO_AP_RE.findall(txt)):
            row(i)["val_ap"] = float(ap)
    elif dialect == "b1":
        for ep, avg in B1_ITER_RE.findall(txt):  # last hit per epoch wins = epoch average
            row(int(ep))["train_loss"] = float(avg)
        for i, ap in enumerate(COCO_AP_RE.findall(txt)):
            row(i)["val_ap"] = float(ap)
    elif dialect == "mmpose":
        sums: dict[int, list] = {}
        for ep, loss in MM_TRAIN_RE.findall(txt):
            sums.setdefault(int(ep), []).append(float(loss))
        for ep, vals in sums.items():
            row(ep)["train_loss"] = round(sum(vals) / len(vals), 6)
        for ep, ap in MM_VAL_RE.findall(txt):  # last val line per epoch wins (final summary)
            row(int(ep))["val_ap"] = float(ap)
    elif dialect == "verifier":
        import json

        for h in json.loads(txt)["history"]:  # epoch/train_loss/val_auc per row
            r = row(int(h["epoch"]))
            r["train_loss"] = h.get("train_loss")
            r["val_ap"] = h.get("val_auc")  # verifier metric is AUC; column name kept
    else:
        raise ValueError(dialect)
    return rows


def lane_for(name: str) -> str:
    """Return the lane directory under final_models/ that owns this run's curve."""
    for prefix, lane in LANE_PREFIXES:
        if name.startswith(prefix):
            return lane
    raise ValueError(f"no lane is defined for run name {name!r}")


def main():
    for name, logname, dialect in RUNS:
        log = (REPO / logname) if dialect == "verifier" else (LOGS / logname)
        if not log.exists():
            print(f"skipped {name}: {log} not present")
            continue
        rows = extract(log.read_text(errors="replace"), dialect)
        out_dir = FINAL_MODELS / lane_for(name) / "curves"
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"{name}.csv"
        with dest.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_ap"])
            for ep in sorted(rows):
                r = rows[ep]
                w.writerow([ep, r["train_loss"], r["val_ap"]])
        n_ap = sum(1 for r in rows.values() if r["val_ap"] is not None)
        best = max((r["val_ap"] for r in rows.values() if r["val_ap"] is not None), default=None)
        print(f"wrote {dest.relative_to(REPO)}: {len(rows)} epochs, {n_ap} vals, best AP {best}")


if __name__ == "__main__":
    main()
