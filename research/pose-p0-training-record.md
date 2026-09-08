# Pose P0: RTMPose-m person_v2 fine-tune, seed 0, gate passed

> **P1 (seed 1) completed 2026-09-04 10:49 UTC.** See the addendum at the
> bottom. Seeds 0/1 landed 0.0001 apart (0.7741 / 0.7740). P2 (seed 2) has since
> completed at AP 0.774057, and seed 0 was selected as the model of record; the
> three-seed verdict is research/pose-seed-floor-and-champion.md.

Dated 2026-09-04. Run 1 of the 3-seed pose protocol (P0/P1/P2). **Best
COCO-val AP 0.7741** (epoch 38) against the recipe gate, which is body7's own
score under the identical eval, measured this run at **0.7645**, a **+0.0096**
gain over the init. The fine-tune scores above its starting weights.

## The run

| item | value |
|---|---|
| model | RTMPose-m (SimCC head), top-down 256x192 |
| config | `third_party/configs/rtmpose_m_person_v2_ft.py` (60 epochs, stage-2 last 5, lr 1e-4, warmup 500, batch 256; provenance card in-file) |
| init | body7 `rtmpose-m_simcc-body7…e48f03d0…pth` (hash-verified; top-level `model.init_cfg`, load confirmed in-log: `load model from` + per-module `PretrainedInit` stamps, no missing/unexpected keys) |
| data | COCO 2017 person keypoints (train 118,287 imgs / 262,465 person anns; val 5,000 / 11,004; counts re-verified at launch) |
| steering eval | COCO val2017, **gt-bbox** protocol, every 2 epochs (30 vals) |
| best checkpoint | `best_coco_AP_epoch_38.pth` sha256 `a0a035fe65909980447560a70ae2f8d13400c1d12490a929e7cf3c5a7b26f102` (backed up: `outputs/pod_backup/p0_best_e38_a0a035fe.pth`) |
| runtime / cost | 20:30 to 03:36 UTC, approximately 7.1 h on the RTX 5090 (about $4.90); dataloader-bound (data_time approximately 0.4/0.7 s at 16 workers; P1/P2 run 32 workers via `--cfg-options`, recipe-neutral) |
| log | `outputs/pod_backup/train_pose_p0_final.log` |

Every `outputs/` path on this page is a pod backup or a run artefact under a gitignored
directory, so none of them is in version control and a clone does not contain them.

## Scores

- **Baseline (gate):** raw body7 on the identical protocol = **AP 0.7645**
  (AP50 0.927, AP75 0.831, AR 0.791; `outputs/pod_backup/body7_baseline_eval.log`).
  Measured post-run with `tools/test.py` (test loader needed explicit
  `--cfg-options` data_root overrides, because the override file only re-pointed
  train/val loaders).
- **P0 trajectory:** e2 0.7638 (approximately init level, as expected), then a plateau near
  0.767 to 0.772 through the peak-lr phase, then a slow climb to
  **0.7741 @ e38**, then a flat top 0.7734 to 0.7741 through e60 (final val
  0.7739). Stage-2 (no-aug, e55+) added
  nothing here, unlike the detector lane.
- Env fixes required at launch (recorded for reproducibility): the pose venv
  needed `numpy<2` (xtcocotools wheel is numpy-1 ABI) and `albumentations<2`
  (mmpose 1.3.2 uses the 1.x module layout), installed in a single `uv pip`
  command (separate installs let uv re-upgrade numpy).

## Standing caveats

1. COCO-val is the **steering** surface only. No frozen pose exam on hospital-style
   footage exists for the v2 lane (the exam once designed on MEVA footage was never
   built), and no shipping claim rides on COCO-val.
2. Selection of the model of record waits for P1/P2 (seeds 1 to 2, same recipe), under the
   same 3-seed noise-floor discipline as the detector. P1 launched 03:40 UTC 2026-09-04.
3. Provenance (POLICY.md, rule 8): body7 init carries the `pt-body7` backbone
   lineage recorded in its provenance card; shippability review is counsel's.

## Addendum: P1 (seed 1), completed 2026-09-04

Identical recipe, `randomness.seed=1`, `train_dataloader.num_workers=32`
(recipe-neutral throughput fix; data_time confirmed the bottleneck in P0).
Init re-verified at launch: 191 `PretrainedInit` stamps from body7 e48f03d0,
no missing/unexpected keys.

| item | value |
|---|---|
| best | **AP 0.7740 @ epoch 30**, `best_coco_AP_epoch_30.pth` sha256 `8d99043597076a672e7f4bc1ce886111d06a62de287d4801e489080561fa40ae` (backed up: `outputs/pod_backup/p1_best_e30_8d990435.pth`) |
| final (e60) | AP 0.7731, AP50 0.9366, AP75 0.8394, AR 0.7971 |
| runtime / cost | 03:40 to 10:49 UTC, approximately 7.15 h (about $4.93); 32 workers cut iter time from 0.70 to 0.56 s but val overhead kept wall-clock at P0's level |
| log | `outputs/pod_backup/train_pose_p1_final.log` |

Trajectory mirrored P0: climb to a ~e30 peak, then a flat 0.773x plateau
through e60; stage-2 (no-aug, e55+) again added nothing. **Seed spread so far
is 0.0001** (P0 0.7741 / P1 0.7740), so the recipe is seed-stable on this
eval. P2 (seed 2) launched 10:54 UTC the same day and completed at AP 0.774057
(epoch 32, checkpoint `5bcca5bd`); selection of the model of record and the
noise-floor statement are recorded in research/pose-seed-floor-and-champion.md.
