# Fall lane seed floor v1 (arm F0 of the fall push v3 protocol)

Dated 2026-09-05. The protocol is research/fall-push-v3-protocol.md, pre-declared. The
arm is F0 control, the unchanged fall_v2pose recipe
(configs/events/train_fall_verifier_v2pose.yaml) with one change: the seed-drawn
grouped_split is replaced by the protocol's pinned holdout, `holdout_locations: [Home_02, S4_ADL,
S4_Fall, S9, S10]` (le2i Home_02, gmdcsa24 S4_ADL and S4_Fall, caucafall S9 and S10).
All five names were verified present in the v2pose track trees, with no substitutions.
Configs: configs/events/train_fall_F0_s{20260826,20260901,20260905}.yaml. Trained on Mac
CPU, 60 epochs each; selection is the best-epoch validation AUC on the pinned holdout
only.

Dataset, identical across all seeds: 881 windows, 190 positive and 691 negative, in 22
groups; the pinned holdout is 169 windows, 18 positive and 151 negative, which matches
the protocol's pre-declared count exactly; the train side is 712 windows in 17 groups.

## The floor

| Seed | Holdout-val AUC | checkpoint sha256 | onnx sha256 | calib T |
|---|---|---|---|---|
| 20260826 | 0.98675 | 4dffa823 | bb819c30 | 1.3796 |
| 20260901 | 0.98823 | 6aac471a | a1c5b7eb | 1.6818 |
| 20260905 | 0.98639 | a0c28672 | e4bfd7e7 | 1.0508 |

**Seed floor, maximum minus minimum: 0.00184** on the pinned holdout. The median-value
seed is **20260826** (0.98675, onnx bb819c30), the F0 seed that took the arm's one exam
look; the ledger rows are in the protocol document. Under the protocol's bar, no arm's
result counts unless its delta exceeds this spread, and the two-part false-positive gate
governs regardless of AUC.

## Window-count verification (protocol step 2, measured on the real trees)

Same geometry (8.0 s window, 1.5 s stride, min_present 0.35), with the labelling policy
as the only variable. `contain` uses contain_margin_s 0.5; `overlap` uses
overlap_min_s 4.0. Negatives are labelling-policy-independent, 691 either way, as
expected.

| Source | contain pos/neg | overlap(4.0s) pos/neg | holdout contain | holdout overlap |
|---|---|---|---|---|
| le2i_tracks_v2pose | 163 / 378 | 174 / 378 | 10 / 65 | 11 / 65 |
| gmdcsa24_tracks_v2pose | 26 / 183 | 87 / 183 | 7 / 51 | 16 / 51 |
| caucafall_tracks_v2pose | 1 / 130 | 95 / 130 | 1 / 35 | 31 / 35 |
| **Total** | **190 / 691** | **356 / 691** | **18 / 151** | **58 / 151** |

The protocol's premise is confirmed with real numbers: caucafall contributes 1
positive window under `contain` against 95 under `overlap`, and the totals match the
protocol's audit (le2i 174, gmdcsa24 87, caucafall 95, total 356).

## F1 label-geometry holdout results, recorded

Same three seeds, window.policy `overlap` (overlap_min_s 4.0), everything else
identical (configs/events/train_fall_F1_s*.yaml). The holdout is 209 windows,
58 positive and 151 negative, which is not AUC-comparable to the F0 column above
because the positive set differs; selection was within-arm only.

| Seed | Holdout-val AUC | checkpoint sha256 | onnx sha256 |
|---|---|---|---|
| 20260826 | 0.97008 | b2e0e28b | afff9dab |
| 20260901 | 0.98322 | bb4fcd8a | 864e04a6 |
| 20260905 | 0.97671, the median, which took the F1 exam look | 2ccc0c15 | 916332b2 |

F1 spread: 0.01314. The exam-look numbers and the two-part false-positive gate verdict,
under which F1 fails the gate and is dropped despite its all-cameras AUC gain, live in
the protocol document's append-only ledger.

Disclosure: these are prototype-class interim event verifiers under the recorded carve-out
in POLICY.md, rule 4, and nothing here is clinical detection performance. Every number
above carries its checkpoint and onnx sha256, as POLICY.md, rule 10 requires; per-epoch
curves for all six runs are in final_models/fall/curves/ (fall_push_F{0,1}_s*.csv).
