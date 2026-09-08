# Fall push v3, pre-declared protocol

Dated 2026-09-05. Recorded before any run, with phase A cleared to proceed under the
augmentation carve-out of POLICY.md, rule 4. Lineage: everything here trains and selects
on the v2pose track trees; the like-for-like baseline is fall_v2pose `b66c3a0a`
(allcams window 0.78474 / video 0.73234, multicam window 0.80814, UR window 0.97698).
Context on the model of record: fall_v3 `67954c04` allcams window 0.79276 on the v1 trees.

## The finding this push attacks (measured 2026-09-05, read-only audit)

The trainer's `contain` window-labelling policy requires the whole fall span inside the
window. CAUCAFall's fall spans run to each clip's final frame (median clip_end -
fall_end = 0.00 s), so CAUCAFall contributes **1 positive window from 50 fall videos**;
GMDCSA-24 contributes 26 (58% of its spans end inside the 0.5 s trailing margin). Net:
163 of 190 training positives come from one eye-level room (le2i), while the model of record's
allcams misses are all high-angle cameras (cam4/6/7). `overlap` labelling (min 4.0 s)
yields le2i 174 / gmdcsa24 87 / caucafall 95 = **356 positives**, including 41
CAUCAFall high-angle videos. The policy is labelling only (featurize_window never reads
it): the ONNX contract and the exams are untouched.

## Fixed choices (before any run)

- **Pinned holdout** (checkpoint selection and all train-side comparisons):
  `holdout_locations`: le2i Home_02, gmdcsa24 S4_ADL and S4_Fall, caucafall S9 and S10
  (18 pos / 151 neg windows under `contain`). Pinned because the seed-20260826
  grouped_split draw put 126 of 190 positives in validation, so the fall_v2pose
  no-effect reading is confounded and is not treated as evidence against the v2 pose.
- **Seeds:** 20260826, 20260901, 20260905; every arm runs all three.
- **Exam budget:** one look per promoted final (at most one final per arm) on each of
  ur_fall / multicam / multicam_allcams. Selection never touches the exams.
- **False-alarm gate (must pass or the candidate is dropped regardless of AUC):** UR
  lie-transition false positives @0.8 not above the control arm's, and chute24 in-alarm
  fraction @0.8 not above the control's (metrics_deep on the same look).

## Arms, in run order

1. **F0 floor (control):** the unchanged fall_v2pose recipe on the pinned holdout,
   3 seeds, giving the lane's first seed floor (research/fall-noise-floor.md); one
   exam look per seed is not taken, because only the median-validation seed takes the
   baseline look. Also: metrics_deep for the existing b66c3a0a look, and one look of
   the model of record `67954c04` on the v2pose trees so both lineages share one negative pool.
2. **F1 label-geometry:** window.policy `overlap` (overlap_min_s 4.0), everything else
   identical to F0. 3 seeds; select on the pinned holdout; the median seed takes one look
   per exam, plus the false-alarm gate.
3. **F2 kp-geometry channels (fall_kpgeo_v1):** `kp_channels` with per-channel validity
   flags (torso cos/sin, spine compression, head vertical velocity in torso lengths,
   hip-centred wrist/ankle), on whichever labelling F1's result keeps. 5 seeds by
   (control plus kp). Same selection and gate. Deployment note recorded: a winning kp head
   needs a keypoint ring in TrackState and an event_verifier row rebuild before it runs
   live.
4. **F3 camera-geometry augmentation (the augmentation carve-out):** vertical
   foreshortening s_y 0.55 to 1.0, perspective trapezoid, roll plus or minus 12 degrees, mirror; angle
   recomputed from transformed keypoints; v2pose lineage only. 3 pre-declared strengths by
   K in {4,8}, selected on the pinned holdout over 3 seeds; one look for the single chosen
   setting.

## Bars and honesty

Bars to beat: allcams window 0.78474 (`b66c3a0a`); a result counts only if the delta
exceeds the F0-measured seed spread. Expected outcome 0.80 to 0.82 allcams, with roughly a
40% chance that every arm lands inside the floor. Not reachable and not promised:
allcams at or above 0.85 (no overhead positive source exists, because the overhead sets are
exam-only), any ward false-alarm rate, recall above 87.5% on allcams (25/200 events
produce no window, which is a tracker lever and out of scope here).

## Exam looks ledger (append-only)

This ledger records every look taken on ur_fall, multicam and multicam_allcams,
whichever protocol declared it, not only looks declared by push v3. Rows 6 to 8 were
declared under the v2c coupling protocol on `final_models/fall/README.md`, section
"Fall F0 refit on the v2c track trees, measured 2026-09-07", and are charged to that
budget, so push v3's own spend remains 12 looks.

| # | Arm/candidate | ur_fall | multicam | allcams | False-alarm gate | Date |
|---|---|---|---|---|---|---|
| 1 | F0 median-validation seed 20260826 (ckpt 4dffa823 / onnx bb819c30), the control look | window 0.97063 / video 0.95207 | window 0.80198 / video 0.86957 | window 0.78165 / video 0.71332 | control values: UR lie-transition @0.8 = 15/46 false-positive windows (9/16 videos); chute24 in-alarm @0.8 = 0/74 (reference camera), 11/654 = 0.0168 (allcams) | 2026-09-05 |
| 2 | F1 median-validation seed 20260905 (ckpt 2ccc0c15 / onnx 916332b2; overlap 4.0 s labelling, exam surface unchanged) | window 0.93889 / video 0.89474 | window 0.81608 / video 0.91304 | window 0.80012 / video 0.72045 | **Rejected.** UR lie-transition @0.8 = 16/46 false-positive windows (11/16 videos) > control's 15 (9); chute24 @0.8 = 0/74 reference (tie), 13/654 = 0.0199 allcams > control's 0.0168. The gate governs regardless of the +0.0185 allcams window delta (which did exceed the 0.00184 F0 floor). The UR window -0.032 against control corroborates: overlap labelling trades UR lie-transition specificity for angle recall | 2026-09-05 |
| 3 | Model of record fall_v3 `67954c04` (ckpt 3d2d17ee), absolute-onnx reslice onto the v2pose trees, giving one shared negative pool for both lineages | window 0.98095 / video 0.96147 | window 0.82238 / video 0.82609 | window 0.79283 / video 0.74253 | context: UR lie-transition @0.8 = 18/46 false-positive windows (13/16 videos); chute24 @0.8 = 0/74 reference, 10/654 = 0.0153 allcams. (v1-tree allcams was 0.79276, and the reslice moves it by +0.00007) | 2026-09-05 |
| 4 | `b66c3a0a` (fall_v2pose), no new look; metrics_deep completed on the existing eval_multicam outputs (eval_ur_fall and allcams already had it) | (existing look: 0.97698) | (existing look: 0.80814) | (existing look: 0.78474 / 0.73234) | context: UR lie-transition @0.8 = 5/46 false-positive windows (5/16 videos); chute24 @0.8 = 0/74 reference, 6/654 = 0.0092 allcams | 2026-09-05 |
| 5 | F2 (fall_kpgeo_v1) median-validation seed 20260826 (ckpt 0d6f4eb3 / onnx e7cb3777; `kp_channels` on the unchanged `contain` labelling, exam surface unchanged) | window 0.97381 / video 0.95865 | window 0.82319 / video 0.82609 | window 0.81143 / video 0.80842 | **Rejected.** UR lie-transition @0.8 = 22/46 false-positive windows (12/16 videos) > control's 15 (9); chute24 @0.8 = 0/74 reference (tie), 9/654 = 0.0138 allcams at or below control's 0.0168 (that half passes). The gate governs regardless of the +0.0298 allcams window delta against the control look, the largest in this push, and of the promotion margin (+0.01148 holdout, above the 0.00442 control spread). Same trade as F1 from the feature side: kp geometry buys angle recall and pays in lie-transition specificity, because the kp channels make a downward transition more convincing, not less | 2026-09-05 |
| none | F3 (camera-geometry augmentation), chosen setting strong / K = 8. **No look taken, so no ledger row is spent.** The candidate that would have taken it is the median-validation seed 20260901 (ckpt 2f97dc17 / onnx 119c6f84) | not scored | not scored | not scored | **Not promoted.** Holdout mean 0.99068 against the F0 control mean 0.98639 = **+0.00429**, which does not exceed the pre-declared promotion margin 0.00442 (the F0 control's 5-seed spread, its two extra seeds trained alongside arm F2), missing it by 0.00013. The false-alarm gate was never reached, because the promotion condition gates the look. Exam budget unchanged at 12 looks. Full record: research/fall-geometric-augmentation-arm.md | 2026-09-05 |
| 6 | F0 refit on the v2c track trees, median-validation seed 20260901 (ckpt a5da25da / onnx baa93141; validation 0.83552, seed spread 0.04334 across 0.85832, 0.83552 and 0.81498), the only look of the three seeds; exam surface is the v2c exam trees, not the v2pose trees of rows 1 to 5 | window 0.81818 / video 0.69123 | window 0.64961 / video 0.52174 | window 0.55615 / video 0.23600 | **Not promoted.** Bar 0.78474 missed by 0.22859, larger than the seed spread. UR lie-transition @0.8 = 13/84 false-positive windows, not above the control's 15 by count (the v2c UR tree carries 84 lie-transition windows against 46); chute24 in-alarm @0.8 = 0/205 reference camera, 143/1644 = 0.0870 all cameras, above the control's 0.0168, so the second part of the gate fails. Rows 7 and 8 show the surface moved under every head; record in final_models/fall/README.md | 2026-09-07 |
| 7 | Model of record fall_v3 `67954c04` (ckpt 3d2d17ee), re-sliced onto the v2c exam trees with its weights untouched | window 0.76578 / video 0.64035 | window 0.40267 / video 0.13043 | window 0.51600 / video 0.23463 | context: UR lie-transition @0.8 = 15/84 false-positive windows; chute24 @0.8 = 57/205 reference camera, 319/1644 = 0.1940 all cameras. The same head scored 0.79283 all cameras on the v2pose trees (row 3) | 2026-09-07 |
| 8 | `b66c3a0a` (fall_v2pose), re-sliced onto the v2c exam trees with its weights untouched | window 0.77326 / video 0.58421 | window 0.43223 / video 0.19565 | window 0.49847 / video 0.24863 | context: UR lie-transition @0.8 = 9/84 false-positive windows; chute24 @0.8 = 44/205 reference camera, 258/1644 = 0.1569 all cameras. The same head holds the bar at 0.78474 on the v2pose trees (row 4). Attribution of the whole movement to the v2c track trees, with the detector probe behind it, is in research/eyes-coupling-probe.md | 2026-09-07 |

Ledger notes (2026-09-05, arms F0 and F1 executed): pinned-holdout floor recorded in
research/fall-noise-floor.md, with F0 seed AUCs 0.98675 / 0.98823 / 0.98639 and
spread 0.00184. Window-count premise verified on the real trees: contain = le2i 163 /
gmdcsa24 26 / caucafall 1 (= 190 pos), overlap(4.0) = 174 / 87 / 95 (= 356 pos),
negatives 691 under both; pinned holdout 18/151 contain, 58/151 overlap. All five
pre-declared holdout group names existed exactly as declared, with no substitutions.
**F1 outcome: the false-alarm gate failed, so the label-geometry candidate is dropped and
`contain` labelling stands for F2.** Exam budget spent by arms F0 and F1: 9 looks (3 per
candidate over F0-median, F1-median and the reslice of the model of record), exactly the pre-declared
set; selection touched only the pinned holdout.

Ledger notes (2026-09-05, arm F2 fall_kpgeo_v1 executed): F1's gate failure pinned the
labelling, so F2 ran on `contain` (windows 881 = 190 pos / 691 neg, identical to F0;
pinned holdout groups unchanged). `kp_channels` (15 per-step channels: torso cos/sin,
head-hip and shoulder-ankle spine extents, head vertical velocity in torso-lengths/s,
hip-centred wrist/ankle offsets, each with a validity flag; per-joint gate
kp_conf_min 0.3, which is the posture-angle gate) added behind `window.kp_channels: true`,
default off and byte-identical when off (golden-hash tested,
tests/test_verifier_kp_channels.py). Five seeds per arm on the pinned holdout:

- control (F0 recipe): 20260826 = 0.98675 (ckpt 4dffa823/onnx bb819c30),
  20260901 = 0.98823 (6aac471a/a1c5b7eb), 20260905 = 0.98639 (a0c28672/e4bfd7e7),
  20260910 = 0.98381 (dd52a7b4/61141c60), 20260915 = 0.98675 (23119e5a/2a96a475),
  giving mean 0.98639 and spread (max-min) 0.00442. The first three are the existing F0
  runs (identical config); only the two new seeds were trained.
- kp arm (F2kp): 20260826 = 0.99779 (0d6f4eb3/e7cb3777),
  20260901 = 0.99926 (423ab6b2/57c7e0f8), 20260905 = 0.99669 (f3bc0354/eaa19172),
  20260910 = 0.99706 (244910c4/97635a3c), 20260915 = 0.99853 (666b9eda/8fc3adcc),
  giving mean 0.99787 and spread 0.00257.

Promotion: the +0.01148 kp minus control holdout mean delta exceeds the 0.00442 control
spread, so the median-validation kp seed (20260826) took its one pre-declared look per
exam (row 5). **F2 outcome: the false-alarm gate failed on the UR lie-transition side, so
the kp-geometry candidate is dropped and `contain` labelling with the 8-channel geometry
stands for F3.** Exam budget spent by this push so far: 12 looks (the 9 above plus 3 for
F2-median). Deployment note, recorded as pre-declared even though the candidate dropped:
any winning kp head needs a keypoint ring in TrackState and an evidence-row rebuild in
vividly.models.event_verifier (its `_features` reconstructs the frozen 9-field
rows from the position and box rings, and no `kp` field exists in the rings today)
before it could ever run live.

Ledger notes (2026-09-05, arm F3 camera-geometry augmentation executed): the gate
failures in F1 and F2 pinned both levers, so F3 ran on `contain` labelling with the
original 8 channels, leaving the augmentation as the only variable (windows 881 = 190 pos
/ 691 neg, pinned holdout 169 = 18/151, identical to F0). The transforms (s_y in [0.55,
1.0], perspective trapezoid, roll plus or minus 12 degrees, mirror) are applied to the raw keypoint and box
rows and the window is re-featurized, with the posture angle recomputed from the warped
keypoints (a test pins that recompute equal to `vividly.tracks.posture_angle_deg`);
boxes travel through the same map. Default off is the same bytes at training scale:
re-running `train_fall_F0_s20260826.yaml` under the modified trainer reproduces
val_auc_best 0.98675 and onnx `bb819c30` exactly. A pre-epoch guard re-featurizes every
source window and requires byte-equality with the recorded row (0 mismatches in all 21
runs). Pre-declared ablation, 3 strengths by K in {4, 8} by 3 seeds = 18 runs, selection on
the pinned holdout only:

- weak/k4 0.98835, weak/k8 0.98810, med/k4 0.98872, med/k8 0.98994, strong/k4 0.98982,
  **strong/k8 0.99068** (seed spread 0.00405; per-seed 0.98859 `87fec2bf` / 0.99080
  `119c6f84` / 0.99264 `14b1dc50`). Monotone in strength at K = 8.

Promotion: +0.00429 over the 0.98639 control mean, against the pre-declared 0.00442
margin (the F0 control's 5-seed spread, its two extra seeds trained alongside arm F2), so the arm **misses by 0.00013,
is not promoted, and no look is taken.** The narrower 3-seed F0 floor (0.00184) would have
cleared it; the wider floor was fixed in advance and governs, and switching rulers after
seeing the number is not available. **F3 outcome: not promoted; exam budget still 12
looks.** Negative-side read on the pinned holdout (the axis F1 and F2 failed on), chosen
setting against the 3 matched control seeds: negatives at or above 0.5 rise from 6.00 to 10.67 of
151 (+78%) and their mean score from 0.0741 to 0.1191, while negatives at or above 0.8 stay flat
(4.33 to 5.00) and positives at or above 0.8 rise from 14.67 to 17.00 of 18. That is the same trade
F1 and F2 were dropped for, louder everywhere with the positives loudest, which is
corroborating evidence that a look would most likely have been lost on the UR
lie-transition half of the gate. A post-hoc identity-K control (8 byte-identical copies
per positive, no warp) scores 0.98688 = -0.00025 against control, so the movement is the
camera geometry and not positive oversampling.
Full record: research/fall-geometric-augmentation-arm.md.
