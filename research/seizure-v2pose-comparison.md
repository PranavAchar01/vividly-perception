# Convulsion (seizure) verifier, v2 lane: outcome

Dated 2026-09-04. The model is convulsion_v2pose in the v2 top-view lane.

The v2-lane convulsion verifier, retrained on the v2 pose model of record, does not beat the frozen model of record `convulsion_v2_real` on the primary exam (testsplit window AUC 0.537 against 0.698) though it edges the secondary exam (unallocated 0.699 against 0.643). The result is dominated by the lane's known data starvation (four training patients) and is high-variance. `convulsion_v2_real` remains the seizure model of record.

This is not a seizure detector and no number here is clinical detection performance. The verifier measures a convulsive-motion signature; WU-SAHZU is public (ungated HuggingFace, Apache-2.0, faces masked, patient consent plus hospital ethics approval, counsel-cleared 2026-08-26), used as an interim prototype train and test surface under the carve-out in POLICY.md, rule 4. Nothing here ships without counsel (CQ-23). Numbers carry the checkpoint sha256, as POLICY.md, rule 10 requires.

## What was trained

Same recipe and same real clinical training patients as the model of record, with one deliberate change: the pose front-end.

| | Model of record `convulsion_v2_real` | New `convulsion_v2pose` |
|---|---|---|
| verifier onnx sha256 | `07c399be24f89a44670262ea75879cc67f60f46c5cd52e5d2cc4218e914b7e22` | `614b92fa729b670738e1521b2c3a7f04046d32444f16a08fdf7e58d7740eb0d4` |
| detector (extraction) | chained prototype detector `55c2c242` | chained prototype detector `55c2c242` (unchanged) |
| pose front-end (extraction) | v1 pose | v2 model of record `pose_v2_champion_p0e38` (onnx `6ba9628506a8899d3a3e940dcc8af5d30342921f09d2605afa505b57453d3527`, keypoint AP 0.774115) |
| train patients | WU-SAHZU pat01/02/05/13 (real clinical PG tracks) | same |
| best val AUC (tiny holdout) | 0.77152 | 0.74420 |
| architecture | small temporal convolutional network (hidden 64, 3 blocks, dropout 0.15), 60 epochs | identical |
| window | 12.0 s / 3.0 s stride / 64 steps / overlap | identical |

Extraction ran on the migration pod's RTX 5090 (public data pulled direct from HuggingFace, raw footage deleted from the pod after the de-identified track sequences were retrieved). Per-epoch accuracy and loss stored at `final_models/seizure/curves/seizure_verifier_v2pose.csv` (60 epochs).

## Held-out exams (pre-declared protocol: one look each, no variant selection on test)

Two untouched exams over patient sets disjoint from training. **`window AUC` is the metric for this lane**, because per-video AUC is meaningless here (tiny n, almost all videos are positives: 13/16 and 12/12), which is why the bar set by the model of record is itself a window AUC.

| Exam | n videos | Model of record `convulsion_v2_real` (window AUC) | New `convulsion_v2pose` (window AUC) | Delta |
|---|---|---|---|---|
| wusahzu **testsplit** (primary): pat03/07/14/04/09/11 | 16 | 0.69841 | 0.53728 | **-0.161** |
| wusahzu **unallocated**: pat06/08/10/12 | 12 | 0.64270 | 0.69921 | +0.056 |

For context, the historical real-data variants whose runs are recorded in `research/stats-compendium.md` scored testsplit window AUC 0.71543 (`convulsion_v1_real`, onnx `42dff31c`), 0.69841 (`convulsion_v2_real`, onnx `07c399be`), 0.48426 (`convulsion_v3_real`, onnx `d85f1a20`), 0.66889 (`convulsion_v4_real`, onnx `6d205fee`) and 0.67447 (`convulsion_v6_real`, onnx `10cdbe78`); `convulsion_v5_real` (onnx `c8bc5c70`) carries no recorded testsplit run. On the unallocated exam the recorded values are 0.64270 (`07c399be`), 0.72618 (`6d205fee`) and 0.53100 (`10cdbe78`). The spread across attempts is larger than the gap measured here, which is the whole point below.

## Reading

- This is a loss on the primary exam, and the lane is too data-poor to trust a single run. Testsplit fell to 0.537 (barely above chance, 0.5); unallocated rose to 0.699. With only four training patients, the tiny validation holdout swung between 0.42810 and 0.56196 across the last ten epochs, so the checkpoint selection is itself noisy, and the five other real variants that carry a recorded testsplit look span a wider band than this result moves. This is a genuinely underdetermined lane.
- A better pose front-end did not help seizure, which is the same finding as the fall lane. The pose AP upgrade (0.764 to 0.774) did not translate into convulsion-signature gains; the binding constraint here is the quantity of data (four patients), not the front-end; `TRAINING.md`, section "Fall and seizure: event verifiers", records the same constraint.
- The model of record is unchanged. `convulsion_v2_real` (sha `07c399be...`) stays the seizure model of record. `convulsion_v2pose` (sha `614b92fa...`) is recorded as the v2-lane run of record; it is not promoted.
- The real unlock is more clinical data, not another architecture or pose front-end swap. Until that exists this lane stays an interim internal prototype under the carve-outs in POLICY.md, rule 4, never quoted as detection performance.

## Provenance and reproduction

- Extraction config: the superseded `extract_wu_sahzu_v2pose` configuration, which is not carried in this repository (chained prototype detector `55c2c242` with the pose graph of record; run on the pod GPU with `device: cuda`). Its detector hash, pose hash, tracker and declared deltas are recorded in `final_models/seizure/README.md`, section "Track trees"; the surviving extraction configuration is `configs/events/extract_wu_sahzu_v2c.yaml`.
- Train: `uv run python scripts/events/train_event_verifier.py --config configs/events/train_convulsion_v2pose.yaml`
- Exams, one invocation per exam because `--config` takes a single path: `for c in configs/events/eval_wusahzu_{testsplit,unallocated}_v2pose.yaml; do uv run python scripts/events/eval_event_verifier.py --config "$c"; done`
- Reports: `outputs/verifier/convulsion_v2pose/{report.json, eval_wusahzu_testsplit, eval_wusahzu_unallocated}/report.json`, build outputs of the recorded run under the gitignored `outputs/` tree
