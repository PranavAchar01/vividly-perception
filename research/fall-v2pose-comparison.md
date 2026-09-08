# Fall verifier, v2 lane: verdict

Dated 2026-09-04. The model is fall_v2pose in the v2 top-view lane.

The v2-lane fall verifier, retrained on the v2 pose model of record, is a statistical wash with the frozen model of record `fall_v3`: it wins one of six exam metrics and is marginally behind on the other five. It does not beat `fall_v3` on the headline all-cameras exam. `fall_v3` remains the fall model of record.

Numbers below carry the checkpoint sha256, as POLICY.md, rule 10 requires. No number here is clinical detection performance; every exam set is public, licence-reviewed acted-fall footage measured as a motion signature under the acted-event carve-out of POLICY.md, rule 4, and the prototype disclosure is carried word for word in each report.json.

## What was trained

Same recipe and same three public training datasets as the model of record, with one deliberate change: the pose front-end that produces the track sequences the verifier reads.

| | Model of record `fall_v3` | New `fall_v2pose` |
|---|---|---|
| verifier onnx sha256 | `67954c04eeef301ff2ab3a1238a79dba2bfa397b3eba5a93a1d3102d7e00010e` | `b66c3a0abc1ce214d93757a7b5b1269ea5027e3ae48f3cecbd33f7ef515e63e3` |
| pose front-end (extraction) | v1 pose | v2 model of record `pose_v2_champion_p0e38` (onnx `6ba9628506a8899d3a3e940dcc8af5d30342921f09d2605afa505b57453d3527`, AP 0.774115) |
| train sources | le2i + gmdcsa24 + caucafall (v1 tracks) | le2i + gmdcsa24 + caucafall (v2pose tracks) |
| best val AUC (train-time holdout) | 0.99589 | 0.99558 |
| architecture | small temporal convolutional network, 60 epochs (identical) | small temporal convolutional network, 60 epochs (identical) |
| window | 8.0 s / 1.5 s stride / 64 steps / contain | identical |

The train-time holdout column is a seed-drawn grouped_split; research/fall-push-v3-protocol.md later recorded that draw as confounded, because the seed-20260826 draw put 126 of 190 positives in validation, and replaced it with a pinned holdout, on which the measured F0 seed floor is 0.00184 (research/fall-noise-floor.md). The exam rows below are unaffected, and no exam-side seed spread has been measured.

Per-epoch accuracy and loss are stored at `final_models/fall/curves/fall_verifier_v2pose.csv` (60 epochs).

## Held-out exams (pre-declared protocol: one look each, no variant selection on test)

Three untouched fall exams, run once after training. `video AUC` is the per-video decision metric the model-of-record reports carry; `window AUC` is the per-window metric.

| Exam | n videos | Metric | Model of record `fall_v3` | New `fall_v2pose` | difference, new minus record |
|---|---|---|---|---|---|
| ur_fall | 52 | video AUC | 0.96711 | 0.94925 | -0.01786 |
| ur_fall | 52 | window AUC | 0.98413 | 0.97698 | -0.00715 |
| multicam (subset) | 24 | video AUC | 0.82609 | **0.86957** | **+0.04348** |
| multicam (subset) | 24 | window AUC | 0.81595 | 0.80814 | -0.00781 |
| multicam **allcams** (headline) | 192 | video AUC | 0.75815 | 0.73234 | -0.02581 |
| multicam **allcams** (headline) | 192 | window AUC | 0.79276 | 0.78474 | -0.00802 |

## Reading

- A tie, not a win. Five of six metrics move by less than an estimated seed-to-seed spread of about plus or minus 0.02 at this dataset size, which is a judgement rather than a spread measured on this page; the one clear move is multicam video-AUC (+0.043 for v2pose). On the headline all-cameras exam `fall_v3` is ahead on both metrics by a small margin.
- A better pose front-end did not buy better falls. The v2 pose model of record is a genuine joint-accuracy upgrade (AP 0.764 to 0.774), yet it did not translate into fall-detection gains. Falls are dominated by gross body motion and trajectory collapse, which the v1 pose front-end already captured; fine joint precision is not the bottleneck for this event. This is a finding, not a null result: it places the fall lane's headroom in the verifier and the data, not the pose front-end.
- The model of record is unchanged. `fall_v3` (sha `67954c04…`) stays the fall model of record. `fall_v2pose` (sha `b66c3a0a…`) is a validated, reproducible alternative on the new v2 pose front-end, kept as the recorded alternative for the v2 lane.

## Provenance and reproduction

- Extraction configs: the superseded `extract_{le2i,gmdcsa24,caucafall,ur_fall,multicam,multicam_allcams}_v2pose` set (v1-stock configs, the pose graph of record and `emit_keypoints: true`; run on the migration pod's RTX 5090, `device: cuda`), which is not carried in this repository. Its detector hash, pose hash, tracker and declared deltas are recorded in `final_models/fall/README.md`, section "Track trees"; the surviving configurations are `configs/events/extract_*_v2c.yaml`.
- Train: `uv run python scripts/events/train_event_verifier.py --config configs/events/train_fall_verifier_v2pose.yaml`
- Exams, one invocation per exam because `--config` takes a single path: `for c in configs/events/eval_fall_{urfall,multicam,multicam_allcams}_v2pose.yaml; do uv run python scripts/events/eval_event_verifier.py --config "$c"; done`
- Reports: `outputs/verifier/fall_v2pose/{report.json, eval_ur_fall, eval_multicam, eval_multicam_allcams}/report.json`, build outputs of the recorded run under the gitignored `outputs/` tree
