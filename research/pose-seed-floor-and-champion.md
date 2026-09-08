# Pose v2 three-seed verdict, model of record selected, recipe maximally seed-stable

Dated 2026-09-04. All three seeds of the RTMPose-m person_v2 fine-tune are
complete. The model of record is seed 0, `best_coco_AP_epoch_38.pth`, AP 0.774115
(sha256 `a0a035fe65909980447560a70ae2f8d13400c1d12490a929e7cf3c5a7b26f102`).

## The floor

| run | seed | best AP (COCO val2017, gt-bbox) | epoch | checkpoint sha256 (short) | final e60 | runtime |
|---|---|---|---|---|---|---|
| P0 | 0 | **0.774115** | 38 | `a0a035fe` | 0.773865 | 7.1 h / $4.90 |
| P1 | 1 | 0.774000 | 30 | `8d990435` | 0.773100 | 7.15 h / $4.93 |
| P2 | 2 | 0.774057 | 32 | `5bcca5bd` | 0.773165 | 7.3 h / $5.04 |

- The noise floor is 0.77406 plus or minus 0.00006 (spread 0.000115; the recipe is
  essentially deterministic on this eval; compare the detector's 0.0012 spread).
- The gate is the measured body7-init baseline 0.764499 on the identical
  protocol, and every seed clears it by +0.0095 to +0.0096.
- The bar for any future pose rival recipe is the model of record plus two sigma,
  approximately 0.7742 (same pre-registration rule as the detector bake-off).
- All three runs: identical config (`rtmpose_m_person_v2_ft.py`, 60 epochs,
  stage-2 no-aug at e55), differing only in `randomness.seed` and (P1/P2)
  `train_dataloader.num_workers=32` (throughput-only). Loss curves for the
  three seeds overlap within line width; see
  `outputs/pod_backup/pose_v2_loss_curves.png`.

## The export the event lane consumes

- `pose_v2_champion_p0e38.onnx` sha256
  `6ba9628506a8899d3a3e940dcc8af5d30342921f09d2605afa505b57453d3527`
  (54.4 MB, opset 17, outputs `simcc_x` (1,17,384) / `simcc_y` (1,17,512),
  decode outside the graph, which is the rtmlib contract).
- Exported by `scripts/pose/export_rtmpose_onnx.py` from the checkpoint of
  record; the maximum absolute difference between the ONNX and torch outputs is
  2.876e-06 and 5.230e-06.
- Backed up: `outputs/pod_backup/pose_v2_champion_p0e38_6ba96285.onnx`
  (hash-verified after transfer), alongside all three best checkpoints and
  full training logs.

Every `outputs/pod_backup/` artefact named on this page, the loss-curve figure included, is a
pod backup under a gitignored directory and is not in version control. The committed record of
the curves is the CSVs under `final_models/joints/curves/`, listed below.

## Curves stored (2026-09-04)

Per-epoch accuracy and loss for every run lives in committed CSVs:
`final_models/joints/curves/pose_rtmpose_m_p0_seed0.csv`, `pose_rtmpose_m_p1_seed1.csv`
and `pose_rtmpose_m_p2_seed2.csv` (the four detector runs have their own curves under
`final_models/eyes/curves/`), extracted by
`scripts/analysis/extract_training_curves.py`; plots regenerate via
`scripts/analysis/plot_pose_curves.py`.

## Standing caveats (unchanged from the P0 record)

1. COCO-val is the steering surface; no frozen hospital-style pose exam exists in
   this repository (the exam once designed on MEVA footage was never built).
   No shipping claim rides on these numbers.
2. Provenance (POLICY.md, rule 8, checkpoint provenance is recorded, never
   assumed): the model of record inherits body7's `pt-body7` backbone lineage, and
   shippability is a question for counsel.
3. Next consumer: Mac-local keypoint re-extraction over the event datasets
   (fall, then seizure verifiers) via `RtmlibPose` with this ONNX. An
   acceptance parity test on a real image precedes any extraction.
