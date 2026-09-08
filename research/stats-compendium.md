# Vividly model statistics compendium

This is the numeric record of the event-verifier and detector evaluation runs that were
present when it was generated on 2026-08-28, produced by `scripts/events/stats_compendium.py`
from run artefacts under `outputs/`, which is not in version control. Regenerating it needs
those run directories, so a clone of this repository cannot reproduce it; the numbers below
are the frozen state at that date. It therefore carries no rows for the eyes, joints and reid
models of record, and none for the verifier arms measured after 2026-08-28. The lane pages
under `final_models/` are the record for the models of record, and any number quoted
elsewhere should be taken from them.

All event-verifier weights are interim prototypes under the carve-out in
POLICY.md, rule 4: internal demonstration only, never quoted as clinical
detection performance. Numbers travel with checkpoint hashes
(POLICY.md, rule 10). Narrative, verdicts, and
caveats live in `research/event-verifier-v1-record.md`.

The detector rows at the end are the exception, and they are not the eyes lane of record:
they are person AP on a 500-image COCO val subset, a different exam from the person_v2
dev-val split the eyes lane's 0.8356 is measured on. One of those runs has a surviving
configuration, `coco500_eyes_stockm`, which is the stock upstream rtmdet-m person graph
`4f4d7e07` behind body7 pose `5c0a4bf6` (`configs/eval/coco500_eyes_stockm.yaml`); the
weights behind the other rows belong to retired prototype detector generations whose
configurations and registry entries were removed under POLICY.md, rule 1, so those rows
carry a run name and no hash. `[clean]` is the generator's condition tag for undegraded
imagery, and the degraded conditions it once contrasted with were removed under the same
rule.

# Event verifier models

## convulsion_v1, task `convulsion`, onnx `9ed32801`

- weight class **prototype**, git `c6f1100`, checkpoint `77847a1c`
- training data: 12013 windows (3447 pos / 8566 neg) from 944 videos
- best in-training val AUC: **0.99681** (same-domain holdout split, not a held-out-test number)

### eval re-run: `eval_wu_sahzu`
- source `outputs/events/wu_sahzu_tracks_v1`: **window AUC 0.38922** / video AUC n/a (1874 windows, 564 pos; 33 videos, 33 with events)
  - video ops @0.5: TP 4 FN 29 FP 0 TN 0
  - video ops @0.8: TP 4 FN 29 FP 0 TN 0
  - video ops @0.9: TP 4 FN 29 FP 0 TN 0
  - WU-SAHZU breakout: see `outputs/verifier/convulsion_v1/eval_wu_sahzu/breakout.json`

### eval re-run: `eval_wusahzu_testsplit`
- source `outputs/events/wu_sahzu_tracks_v1`: **window AUC 0.40761** / video AUC 0.20513 (2163 windows, 216 pos; 16 videos, 13 with events)
  - video ops @0.5: TP 1 FN 12 FP 2 TN 1
  - video ops @0.8: TP 1 FN 12 FP 2 TN 1
  - video ops @0.9: TP 1 FN 12 FP 1 TN 2
- deep: **ROC 0.40761 / PR 0.09866** (216+/1947- windows); pos score median 0.000 vs neg 0.000
  - sweep: @0.3: sens 0.005 spec 0.993 prec 0.071 F1 0.009; @0.5: sens 0.000 spec 0.994 prec 0.000 F1 n/a; @0.8: sens 0.000 spec 0.994 prec 0.000 F1 n/a; @0.9: sens 0.000 spec 0.996 prec 0.000 F1 n/a
  - best F1 0.009 at thr 0.1 (sens 0.005, spec 0.993)
  - events @0.5: 0/13 caught
  - events @0.8: 0/13 caught
  - negatives: 3 videos / 0.944 h; FP windows/hr @0.5 10.6, @0.8 10.6, @0.9 7.4
  - per-class videos: convulsive_PG: 0/6 over 0.5; focal_P: 1/7 over 0.5; non_seizure: 2/3 over 0.5
  - coverage: 16/16 videos produced windows; footage covered 0.885 of 1.569 h; 0 events with no overlapping window
  - alarm experience (negatives): @0.5: eps/h 4.2, cam/h 3.2, in-alarm 1.9%; @0.8: eps/h 4.2, cam/h 3.2, in-alarm 1.9%; @0.9: eps/h 3.2, cam/h 2.1, in-alarm 1.6%
  - WU-SAHZU breakout: see `outputs/verifier/convulsion_v1/eval_wusahzu_testsplit/breakout.json`


## convulsion_v1_real, task `convulsion`, onnx `42dff31c`

- weight class **prototype**, git `de31e26`, checkpoint `aeb813ea`
- training data: 518 windows (171 pos / 347 neg) from 8 videos
- best in-training val AUC: **0.65589** (same-domain holdout split, not a held-out-test number)

### eval re-run: `eval_wusahzu_testsplit`
- source `outputs/events/wu_sahzu_tracks_v1`: **window AUC 0.71543** / video AUC 0.41026 (2163 windows, 216 pos; 16 videos, 13 with events)
  - video ops @0.5: TP 11 FN 2 FP 3 TN 0
  - video ops @0.8: TP 0 FN 13 FP 0 TN 3
  - video ops @0.9: TP 0 FN 13 FP 0 TN 3
- deep: **ROC 0.71543 / PR 0.19346** (216+/1947- windows); pos score median 0.494 vs neg 0.492
  - sweep: @0.3: sens 1.000 spec 0.000 prec 0.100 F1 0.182; @0.5: sens 0.185 spec 0.913 prec 0.191 F1 0.188; @0.8: sens 0.000 spec 1.000 prec n/a F1 n/a; @0.9: sens 0.000 spec 1.000 prec n/a F1 n/a
  - best F1 0.188 at thr 0.5 (sens 0.185, spec 0.913)
  - events @0.5: 10/13 caught, latency median 47.0 s (mean 46.7, max 79.3)
  - events @0.8: 0/13 caught
  - negatives: 3 videos / 0.944 h; FP windows/hr @0.5 144.0, @0.8 0.0, @0.9 0.0
  - per-class videos: convulsive_PG: 5/6 over 0.5; focal_P: 6/7 over 0.5; non_seizure: 3/3 over 0.5
  - coverage: 16/16 videos produced windows; footage covered 0.885 of 1.569 h; 0 events with no overlapping window
  - alarm experience (negatives): @0.5: eps/h 28.6, cam/h 10.6, in-alarm 16.5%; @0.8: eps/h 0.0, cam/h 0.0, in-alarm 0.0%; @0.9: eps/h 0.0, cam/h 0.0, in-alarm 0.0%
  - WU-SAHZU breakout: see `outputs/verifier/convulsion_v1_real/eval_wusahzu_testsplit/breakout.json`


## convulsion_v1_synthreal, task `convulsion`, onnx `a95616f2`

- weight class **prototype**, git `de31e26`, checkpoint `094ddddc`
- training data: 12531 windows (3618 pos / 8913 neg) from 952 videos
- best in-training val AUC: **0.98629** (same-domain holdout split, not a held-out-test number)

### eval re-run: `eval_wusahzu_testsplit`
- source `outputs/events/wu_sahzu_tracks_v1`: **window AUC 0.55937** / video AUC 0.23077 (2163 windows, 216 pos; 16 videos, 13 with events)
  - video ops @0.5: TP 12 FN 1 FP 3 TN 0
  - video ops @0.8: TP 9 FN 4 FP 3 TN 0
  - video ops @0.9: TP 7 FN 6 FP 3 TN 0
- deep: **ROC 0.55937 / PR 0.13201** (216+/1947- windows); pos score median 0.054 vs neg 0.037
  - sweep: @0.3: sens 0.241 spec 0.850 prec 0.151 F1 0.186; @0.5: sens 0.194 spec 0.888 prec 0.162 F1 0.177; @0.8: sens 0.102 spec 0.926 prec 0.133 F1 0.115; @0.9: sens 0.079 spec 0.947 prec 0.141 F1 0.101
  - best F1 0.215 at thr 0.1 (sens 0.407, spec 0.736)
  - events @0.5: 10/13 caught, latency median 24.9 s (mean 35.3, max 76.3)
  - events @0.8: 7/13 caught, latency median 53.0 s (mean 41.4, max 74.0)
  - negatives: 3 videos / 0.944 h; FP windows/hr @0.5 209.7, @0.8 139.8, @0.9 100.6
  - per-class videos: convulsive_PG: 5/6 over 0.5; focal_P: 7/7 over 0.5; non_seizure: 3/3 over 0.5
  - coverage: 16/16 videos produced windows; footage covered 0.885 of 1.569 h; 0 events with no overlapping window
  - alarm experience (negatives): @0.5: eps/h 41.3, cam/h 25.4, in-alarm 30.8%; @0.8: eps/h 27.5, cam/h 19.1, in-alarm 22.5%; @0.9: eps/h 26.5, cam/h 20.1, in-alarm 16.2%
  - WU-SAHZU breakout: see `outputs/verifier/convulsion_v1_synthreal/eval_wusahzu_testsplit/breakout.json`


## convulsion_v2_real, task `convulsion`, onnx `07c399be`

- weight class **prototype**, git `8fbebe2`, checkpoint `715d172c`
- training data: 1106 windows (164 pos / 942 neg) from 8 videos
- best in-training val AUC: **0.77152** (same-domain holdout split, not a held-out-test number)

### eval re-run: `eval_wusahzu_testsplit`
- source `outputs/events/wu_sahzu_tracks_v2`: **window AUC 0.69841** / video AUC 0.33333 (4880 windows, 285 pos; 16 videos, 13 with events)
  - video ops @0.5: TP 13 FN 0 FP 3 TN 0
  - video ops @0.8: TP 10 FN 3 FP 3 TN 0
  - video ops @0.9: TP 8 FN 5 FP 2 TN 1
- deep: **ROC 0.69841 / PR 0.11484** (285+/4595- windows); pos score median 0.545 vs neg 0.256
  - sweep: @0.3: sens 0.747 spec 0.545 prec 0.092 F1 0.165; @0.5: sens 0.547 spec 0.738 prec 0.115 F1 0.190; @0.8: sens 0.175 spec 0.911 prec 0.109 F1 0.135; @0.9: sens 0.070 spec 0.979 prec 0.170 F1 0.099
  - best F1 0.190 at thr 0.5 (sens 0.547, spec 0.738)
  - events @0.5: 12/13 caught, latency median 7.0 s (mean 17.3, max 53.2)
  - events @0.8: 8/13 caught, latency median 19.1 s (mean 35.6, max 100.9)
  - negatives: 3 videos / 0.944 h; FP windows/hr @0.5 962.6, @0.8 272.1, @0.9 57.2
  - per-class videos: convulsive_PG: 6/6 over 0.5; focal_P: 7/7 over 0.5; non_seizure: 3/3 over 0.5
  - coverage: 16/16 videos produced windows; footage covered 0.949 of 1.569 h; 0 events with no overlapping window
  - alarm experience (negatives): @0.5: eps/h 103.8, cam/h 21.2, in-alarm 72.4%; @0.8: eps/h 39.2, cam/h 23.3, in-alarm 33.2%; @0.9: eps/h 13.8, cam/h 9.5, in-alarm 8.1%
  - WU-SAHZU breakout: see `outputs/verifier/convulsion_v2_real/eval_wusahzu_testsplit/breakout.json`

### eval re-run: `eval_wusahzu_unallocated`
- source `outputs/events/wu_sahzu_tracks_v2`: **window AUC 0.64270** / video AUC n/a (1386 windows, 178 pos; 12 videos, 12 with events)
  - video ops @0.5: TP 12 FN 0 FP 0 TN 0
  - video ops @0.8: TP 10 FN 2 FP 0 TN 0
  - video ops @0.9: TP 6 FN 6 FP 0 TN 0
- deep: **ROC 0.64270 / PR 0.17017** (178+/1208- windows); pos score median 0.518 vs neg 0.310
  - sweep: @0.3: sens 0.708 spec 0.490 prec 0.170 F1 0.274; @0.5: sens 0.517 spec 0.653 prec 0.180 F1 0.267; @0.8: sens 0.219 spec 0.858 prec 0.185 F1 0.201; @0.9: sens 0.101 spec 0.920 prec 0.158 F1 0.123
  - best F1 0.287 at thr 0.4 (sens 0.629, spec 0.595)
  - events @0.5: 11/12 caught, latency median 38.8 s (mean 42.4, max 109.0)
  - events @0.8: 6/12 caught, latency median 40.9 s (mean 40.9, max 80.7)
  - per-class videos: focal_P: 12/12 over 0.5
  - coverage: 12/12 videos produced windows; footage covered 0.790 of 0.668 h; 0 events with no overlapping window


## convulsion_v3_real, task `convulsion`, onnx `d85f1a20`

- weight class **prototype**, git `b1dfe79`, checkpoint `a0804c51`
- training data: 935 windows (142 pos / 793 neg) from 8 videos
- best in-training val AUC: **0.93659** (same-domain holdout split, not a held-out-test number)

### eval re-run: `eval_wusahzu_testsplit`
- source `outputs/events/wu_sahzu_tracks_v3`: **window AUC 0.48426** / video AUC 0.20513 (3869 windows, 272 pos; 16 videos, 13 with events)
  - video ops @0.5: TP 13 FN 0 FP 3 TN 0
  - video ops @0.8: TP 0 FN 13 FP 0 TN 3
  - video ops @0.9: TP 0 FN 13 FP 0 TN 3
- deep: **ROC 0.48426 / PR 0.12262** (272+/3597- windows); pos score median 0.497 vs neg 0.498
  - sweep: @0.3: sens 1.000 spec 0.000 prec 0.070 F1 0.131; @0.5: sens 0.279 spec 0.796 prec 0.094 F1 0.141; @0.8: sens 0.000 spec 1.000 prec n/a F1 n/a; @0.9: sens 0.000 spec 1.000 prec n/a F1 n/a
  - best F1 0.141 at thr 0.5 (sens 0.279, spec 0.796)
  - events @0.5: 10/13 caught, latency median 32.4 s (mean 36.4, max 75.1)
  - events @0.8: 0/13 caught
  - negatives: 3 videos / 0.944 h; FP windows/hr @0.5 707.4, @0.8 0.0, @0.9 0.0
  - per-class videos: convulsive_PG: 6/6 over 0.5; focal_P: 7/7 over 0.5; non_seizure: 3/3 over 0.5
  - coverage: 16/16 videos produced windows; footage covered 0.934 of 1.569 h; 0 events with no overlapping window
  - alarm experience (negatives): @0.5: eps/h 95.3, cam/h 20.1, in-alarm 62.2%; @0.8: eps/h 0.0, cam/h 0.0, in-alarm 0.0%; @0.9: eps/h 0.0, cam/h 0.0, in-alarm 0.0%


## convulsion_v4_real, task `convulsion`, onnx `6d205fee`

- weight class **prototype**, git `b1dfe79`, checkpoint `727fc726`
- training data: 4852 windows (164 pos / 4688 neg) from 14 videos
- best in-training val AUC: **0.98139** (same-domain holdout split, not a held-out-test number)

### eval re-run: `eval_wusahzu_testsplit`
- source `outputs/events/wu_sahzu_tracks_v2`: **window AUC 0.66889** / video AUC 0.12821 (4880 windows, 285 pos; 16 videos, 13 with events)
  - video ops @0.5: TP 13 FN 0 FP 3 TN 0
  - video ops @0.8: TP 13 FN 0 FP 3 TN 0
  - video ops @0.9: TP 10 FN 3 FP 3 TN 0
- deep: **ROC 0.66889 / PR 0.12322** (285+/4595- windows); pos score median 0.681 vs neg 0.442
  - sweep: @0.3: sens 0.839 spec 0.386 prec 0.078 F1 0.143; @0.5: sens 0.681 spec 0.536 prec 0.084 F1 0.149; @0.8: sens 0.383 spec 0.808 prec 0.110 F1 0.171; @0.9: sens 0.267 spec 0.914 prec 0.161 F1 0.201
  - best F1 0.202 at thr 0.95 (sens 0.193, spec 0.955)
  - events @0.5: 13/13 caught, latency median 7.2 s (mean 16.4, max 53.2)
  - events @0.8: 13/13 caught, latency median 24.0 s (mean 28.7, max 84.2)
  - negatives: 3 videos / 0.944 h; FP windows/hr @0.5 1902.9, @0.8 800.6, @0.9 355.8
  - per-class videos: convulsive_PG: 6/6 over 0.5; focal_P: 7/7 over 0.5; non_seizure: 3/3 over 0.5
  - coverage: 16/16 videos produced windows; footage covered 0.949 of 1.569 h; 0 events with no overlapping window
  - alarm experience (negatives): @0.5: eps/h 109.1, cam/h 6.4, in-alarm 95.7%; @0.8: eps/h 104.9, cam/h 18.0, in-alarm 73.7%; @0.9: eps/h 71.0, cam/h 22.2, in-alarm 46.1%

### eval re-run: `eval_wusahzu_unallocated`
- source `outputs/events/wu_sahzu_tracks_v2`: **window AUC 0.72618** / video AUC n/a (1386 windows, 178 pos; 12 videos, 12 with events)
  - video ops @0.5: TP 12 FN 0 FP 0 TN 0
  - video ops @0.8: TP 11 FN 1 FP 0 TN 0
  - video ops @0.9: TP 10 FN 2 FP 0 TN 0
- deep: **ROC 0.72618 / PR 0.26187** (178+/1208- windows); pos score median 0.667 vs neg 0.330
  - sweep: @0.3: sens 0.848 spec 0.473 prec 0.192 F1 0.313; @0.5: sens 0.685 spec 0.652 prec 0.225 F1 0.339; @0.8: sens 0.354 spec 0.825 prec 0.229 F1 0.278; @0.9: sens 0.270 spec 0.901 prec 0.286 F1 0.278
  - best F1 0.339 at thr 0.5 (sens 0.685, spec 0.652)
  - events @0.5: 10/12 caught, latency median 30.6 s (mean 35.5, max 83.1)
  - events @0.8: 9/12 caught, latency median 29.8 s (mean 47.6, max 122.7)
  - per-class videos: focal_P: 12/12 over 0.5
  - coverage: 12/12 videos produced windows; footage covered 0.790 of 0.668 h; 0 events with no overlapping window


## convulsion_v5_real, task `convulsion`, onnx `c8bc5c70`

- weight class **prototype**, git `b1dfe79`, checkpoint `b2c49bee`
- training data: 4852 windows (164 pos / 4688 neg) from 14 videos
- best in-training val AUC: **0.97211** (same-domain holdout split, not a held-out-test number)


## convulsion_v6_real, task `convulsion`, onnx `10cdbe78`

- weight class **prototype**, git `b1dfe79`, checkpoint `9ec2bf47`
- training data: 1485 windows (164 pos / 1321 neg) from 134 videos
- best in-training val AUC: **0.98338** (same-domain holdout split, not a held-out-test number)

### eval re-run: `eval_wusahzu_testsplit`
- source `outputs/events/wu_sahzu_tracks_v2`: **window AUC 0.67447** / video AUC 0.43590 (4880 windows, 285 pos; 16 videos, 13 with events)
  - video ops @0.5: TP 12 FN 1 FP 3 TN 0
  - video ops @0.8: TP 12 FN 1 FP 3 TN 0
  - video ops @0.9: TP 12 FN 1 FP 3 TN 0
- deep: **ROC 0.67447 / PR 0.18399** (285+/4595- windows); pos score median 0.910 vs neg 0.284
  - sweep: @0.3: sens 0.723 spec 0.508 prec 0.084 F1 0.150; @0.5: sens 0.681 spec 0.583 prec 0.092 F1 0.162; @0.8: sens 0.568 spec 0.739 prec 0.119 F1 0.197; @0.9: sens 0.512 spec 0.796 prec 0.135 F1 0.213
  - best F1 0.241 at thr 0.95 (sens 0.463, spec 0.852)
  - events @0.5: 11/13 caught, latency median 6.7 s (mean 15.8, max 53.2)
  - events @0.8: 11/13 caught, latency median 10.2 s (mean 17.7, max 53.2)
  - negatives: 3 videos / 0.944 h; FP windows/hr @0.5 1628.7, @0.8 903.3, @0.9 649.1
  - per-class videos: convulsive_PG: 5/6 over 0.5; focal_P: 7/7 over 0.5; non_seizure: 3/3 over 0.5
  - coverage: 16/16 videos produced windows; footage covered 0.949 of 1.569 h; 0 events with no overlapping window
  - alarm experience (negatives): @0.5: eps/h 110.2, cam/h 9.5, in-alarm 90.1%; @0.8: eps/h 120.8, cam/h 20.1, in-alarm 68.8%; @0.9: eps/h 100.6, cam/h 27.5, in-alarm 57.3%

### eval re-run: `eval_wusahzu_unallocated`
- source `outputs/events/wu_sahzu_tracks_v2`: **window AUC 0.53100** / video AUC n/a (1386 windows, 178 pos; 12 videos, 12 with events)
  - video ops @0.5: TP 12 FN 0 FP 0 TN 0
  - video ops @0.8: TP 11 FN 1 FP 0 TN 0
  - video ops @0.9: TP 11 FN 1 FP 0 TN 0
- deep: **ROC 0.53100 / PR 0.13032** (178+/1208- windows); pos score median 0.702 vs neg 0.533
  - sweep: @0.3: sens 0.657 spec 0.413 prec 0.142 F1 0.233; @0.5: sens 0.590 spec 0.490 prec 0.146 F1 0.234; @0.8: sens 0.444 spec 0.588 prec 0.137 F1 0.209; @0.9: sens 0.354 spec 0.631 prec 0.124 F1 0.183
  - best F1 0.239 at thr 0.4 (sens 0.629, spec 0.464)
  - events @0.5: 12/12 caught, latency median 27.2 s (mean 39.2, max 103.0)
  - events @0.8: 10/12 caught, latency median 27.1 s (mean 38.4, max 106.0)
  - per-class videos: focal_P: 12/12 over 0.5
  - coverage: 12/12 videos produced windows; footage covered 0.790 of 0.668 h; 0 events with no overlapping window


## fall_v1, task `fall`, onnx `080b5d86`

- weight class **prototype**, git `c6f1100`, checkpoint `7d21e906`
- training data: 541 windows (163 pos / 378 neg) from 122 videos
- best in-training val AUC: **0.99924** (same-domain holdout split, not a held-out-test number)

### eval re-run: `eval_multicam`
- source `outputs/events/multicam_tracks_v1`: **window AUC 0.82092** / video AUC 0.82609 (533 windows, 80 pos; 24 videos, 23 with events)
  - video ops @0.5: TP 21 FN 2 FP 1 TN 0
  - video ops @0.8: TP 20 FN 3 FP 1 TN 0
  - video ops @0.9: TP 20 FN 3 FP 1 TN 0
- deep: **ROC 0.82092 / PR 0.48680** (80+/453- windows); pos score median 0.947 vs neg 0.103
  - sweep: @0.3: sens 0.875 spec 0.631 prec 0.295 F1 0.442; @0.5: sens 0.825 spec 0.669 prec 0.306 F1 0.446; @0.8: sens 0.675 spec 0.766 prec 0.338 F1 0.450; @0.9: sens 0.600 spec 0.826 prec 0.378 F1 0.464
  - best F1 0.464 at thr 0.9 (sens 0.600, spec 0.826)
  - events @0.5: 19/25 caught, latency median 2.6 s (mean 3.0, max 7.3)
  - events @0.8: 17/25 caught, latency median 2.7 s (mean 3.0, max 5.2)
  - negatives: 1 videos / 0.034 h; FP windows/hr @0.5 234.5, @0.8 117.3, @0.9 88.0
  - per-class videos: adl_other: 1/1 over 0.5; fall: 21/23 over 0.5
  - coverage: 24/24 videos produced windows; footage covered 0.886 of 0.3 h; 4 events with no overlapping window
  - alarm experience (negatives): @0.5: eps/h 117.7, cam/h 58.8, in-alarm 46.6%; @0.8: eps/h 58.8, cam/h 58.8, in-alarm 22.1%; @0.9: eps/h 29.4, cam/h 29.4, in-alarm 12.2%

### eval re-run: `eval_multicam_allcams`
- source `outputs/events/multicam_allcams_tracks_v1`: **window AUC 0.77752** / video AUC 0.70007 (4588 windows, 609 pos; 192 videos, 184 with events)
  - video ops @0.5: TP 174 FN 10 FP 8 TN 0
  - video ops @0.8: TP 167 FN 17 FP 7 TN 1
  - video ops @0.9: TP 155 FN 29 FP 7 TN 1
- deep: **ROC 0.77752 / PR 0.34521** (609+/3979- windows); pos score median 0.931 vs neg 0.216
  - sweep: @0.3: sens 0.837 spec 0.542 prec 0.219 F1 0.347; @0.5: sens 0.791 spec 0.617 prec 0.240 F1 0.368; @0.8: sens 0.660 spec 0.747 prec 0.285 F1 0.398; @0.9: sens 0.543 spec 0.824 prec 0.321 F1 0.404
  - best F1 0.404 at thr 0.9 (sens 0.543, spec 0.824)
  - events @0.5: 151/200 caught, latency median 2.7 s (mean 3.1, max 7.3)
  - events @0.8: 135/200 caught, latency median 2.8 s (mean 3.3, max 7.2)
  - negatives: 8 videos / 0.273 h; FP windows/hr @0.5 428.6, @0.8 212.4, @0.9 106.2
  - per-class videos: adl_other: 8/8 over 0.5; fall: 174/184 over 0.5
  - coverage: 192/192 videos produced windows; footage covered 0.898 of 2.418 h; 25 events with no overlapping window
  - alarm experience (negatives): @0.5: eps/h 87.9, cam/h 51.3, in-alarm 51.1%; @0.8: eps/h 62.3, cam/h 36.6, in-alarm 31.1%; @0.9: eps/h 44.0, cam/h 33.0, in-alarm 20.1%

### eval re-run: `eval_ur_fall`
- source `outputs/events/ur_fall_tracks_v1`: **window AUC 0.95714** / video AUC 0.93139 (104 windows, 14 pos; 52 videos, 14 with events)
  - video ops @0.5: TP 14 FN 0 FP 21 TN 17 (lie-transition FP 16/16)
  - video ops @0.8: TP 14 FN 0 FP 19 TN 19 (lie-transition FP 16/16)
  - video ops @0.9: TP 14 FN 0 FP 17 TN 21 (lie-transition FP 15/16)
- deep: **ROC 0.95714 / PR 0.72040** (14+/90- windows); pos score median 0.999 vs neg 0.198
  - sweep: @0.3: sens 1.000 spec 0.522 prec 0.246 F1 0.394; @0.5: sens 1.000 spec 0.578 prec 0.269 F1 0.424; @0.8: sens 1.000 spec 0.633 prec 0.298 F1 0.459; @0.9: sens 1.000 spec 0.711 prec 0.350 F1 0.518
  - best F1 0.609 at thr 0.95 (sens 1.000, spec 0.800)
  - events @0.5: 14/30 caught, latency median 2.5 s (mean 2.5, max 3.1)
  - events @0.8: 14/30 caught, latency median 2.5 s (mean 2.5, max 3.1)
  - negatives: 38 videos / 0.081 h; FP windows/hr @0.5 467.2, @0.8 405.7, @0.9 319.7
  - per-class videos: adl_lie_transition: 16/16 over 0.5; adl_other: 5/22 over 0.5; fall: 14/14 over 0.5
  - coverage: 52/70 videos produced windows; footage covered 0.851 of 0.111 h; 16 events with no overlapping window; zero-window videos: 18 (0.013 h)
  - alarm experience (negatives): @0.5: eps/h 333.3, cam/h 259.3, in-alarm 92.0%; @0.8: eps/h 271.6, cam/h 234.6, in-alarm 83.5%; @0.9: eps/h 222.2, cam/h 209.9, in-alarm 74.0%


## fall_v2, task `fall`, onnx `e9cebe4d`

- weight class **prototype**, git `b1dfe79`, checkpoint `d99c4fb7`
- training data: 5894 windows (163 pos / 5731 neg) from 128 videos
- best in-training val AUC: **0.99779** (same-domain holdout split, not a held-out-test number)

### eval re-run: `eval_multicam`
- source `outputs/events/multicam_tracks_v1`: **window AUC 0.84603** / video AUC 0.82609 (533 windows, 80 pos; 24 videos, 23 with events)
  - video ops @0.5: TP 15 FN 8 FP 0 TN 1
  - video ops @0.8: TP 15 FN 8 FP 0 TN 1
  - video ops @0.9: TP 11 FN 12 FP 0 TN 1
- deep: **ROC 0.84603 / PR 0.63376** (80+/453- windows); pos score median 0.645 vs neg 0.009
  - sweep: @0.3: sens 0.637 spec 0.945 prec 0.671 F1 0.654; @0.5: sens 0.562 spec 0.956 prec 0.692 F1 0.621; @0.8: sens 0.463 spec 0.965 prec 0.698 F1 0.556; @0.9: sens 0.388 spec 0.971 prec 0.705 F1 0.500
  - best F1 0.671 at thr 0.4 (sens 0.637, spec 0.954)
  - events @0.5: 14/25 caught, latency median 3.0 s (mean 3.2, max 6.4)
  - events @0.8: 12/25 caught, latency median 3.4 s (mean 3.6, max 5.9)
  - negatives: 1 videos / 0.034 h; FP windows/hr @0.5 0.0, @0.8 0.0, @0.9 0.0
  - per-class videos: adl_other: 0/1 over 0.5; fall: 15/23 over 0.5
  - coverage: 24/24 videos produced windows; footage covered 0.886 of 0.3 h; 4 events with no overlapping window
  - alarm experience (negatives): @0.5: eps/h 0.0, cam/h 0.0, in-alarm 0.0%; @0.8: eps/h 0.0, cam/h 0.0, in-alarm 0.0%; @0.9: eps/h 0.0, cam/h 0.0, in-alarm 0.0%

### eval re-run: `eval_multicam_allcams`
- source `outputs/events/multicam_allcams_tracks_v1`: **window AUC 0.77043** / video AUC 0.72826 (4588 windows, 609 pos; 192 videos, 184 with events)
  - video ops @0.5: TP 125 FN 59 FP 3 TN 5
  - video ops @0.8: TP 105 FN 79 FP 0 TN 8
  - video ops @0.9: TP 90 FN 94 FP 0 TN 8
- deep: **ROC 0.77043 / PR 0.46688** (609+/3979- windows); pos score median 0.315 vs neg 0.010
  - sweep: @0.3: sens 0.507 spec 0.928 prec 0.519 F1 0.513; @0.5: sens 0.432 spec 0.944 prec 0.542 F1 0.481; @0.8: sens 0.373 spec 0.964 prec 0.614 F1 0.464; @0.9: sens 0.317 spec 0.970 prec 0.621 F1 0.420
  - best F1 0.515 at thr 0.2 (sens 0.547, spec 0.912)
  - events @0.5: 86/200 caught, latency median 2.8 s (mean 3.3, max 7.2)
  - events @0.8: 75/200 caught, latency median 2.9 s (mean 3.5, max 7.2)
  - negatives: 8 videos / 0.273 h; FP windows/hr @0.5 18.3, @0.8 0.0, @0.9 0.0
  - per-class videos: adl_other: 3/8 over 0.5; fall: 125/184 over 0.5
  - coverage: 192/192 videos produced windows; footage covered 0.898 of 2.418 h; 25 events with no overlapping window
  - alarm experience (negatives): @0.5: eps/h 11.0, cam/h 11.0, in-alarm 4.6%; @0.8: eps/h 0.0, cam/h 0.0, in-alarm 0.0%; @0.9: eps/h 0.0, cam/h 0.0, in-alarm 0.0%

### eval re-run: `eval_ur_fall`
- source `outputs/events/ur_fall_tracks_v1`: **window AUC 1.00000** / video AUC 1.00000 (104 windows, 14 pos; 52 videos, 14 with events)
  - video ops @0.5: TP 14 FN 0 FP 11 TN 27 (lie-transition FP 10/16)
  - video ops @0.8: TP 14 FN 0 FP 4 TN 34 (lie-transition FP 4/16)
  - video ops @0.9: TP 14 FN 0 FP 1 TN 37 (lie-transition FP 1/16)
- deep: **ROC 1.00000 / PR 1.00000** (14+/90- windows); pos score median 0.999 vs neg 0.089
  - sweep: @0.3: sens 1.000 spec 0.733 prec 0.368 F1 0.538; @0.5: sens 1.000 spec 0.833 prec 0.483 F1 0.651; @0.8: sens 1.000 spec 0.956 prec 0.778 F1 0.875; @0.9: sens 1.000 spec 0.989 prec 0.933 F1 0.966
  - best F1 0.966 at thr 0.9 (sens 1.000, spec 0.989)
  - events @0.5: 14/30 caught, latency median 2.5 s (mean 2.5, max 3.1)
  - events @0.8: 14/30 caught, latency median 2.5 s (mean 2.5, max 3.1)
  - negatives: 38 videos / 0.081 h; FP windows/hr @0.5 184.4, @0.8 49.2, @0.9 12.3
  - per-class videos: adl_lie_transition: 10/16 over 0.5; adl_other: 1/22 over 0.5; fall: 14/14 over 0.5
  - coverage: 52/70 videos produced windows; footage covered 0.851 of 0.111 h; 16 events with no overlapping window; zero-window videos: 18 (0.013 h)
  - alarm experience (negatives): @0.5: eps/h 148.2, cam/h 135.8, in-alarm 46.9%; @0.8: eps/h 49.4, cam/h 49.4, in-alarm 16.5%; @0.9: eps/h 12.3, cam/h 12.3, in-alarm 4.1%


## fall_v3, task `fall`, onnx `67954c04`

- weight class **prototype**, git `b1dfe79`, checkpoint `3d2d17ee`
- training data: 881 windows (190 pos / 691 neg) from 282 videos
- best in-training val AUC: **0.99589** (same-domain holdout split, not a held-out-test number)

### eval re-run: `eval_multicam`
- source `outputs/events/multicam_tracks_v1`: **window AUC 0.81595** / video AUC 0.82609 (533 windows, 80 pos; 24 videos, 23 with events)
  - video ops @0.5: TP 20 FN 3 FP 1 TN 0
  - video ops @0.8: TP 17 FN 6 FP 0 TN 1
  - video ops @0.9: TP 14 FN 9 FP 0 TN 1
- deep: **ROC 0.81595 / PR 0.46137** (80+/453- windows); pos score median 0.784 vs neg 0.036
  - sweep: @0.3: sens 0.625 spec 0.795 prec 0.350 F1 0.448; @0.5: sens 0.600 spec 0.832 prec 0.387 F1 0.471; @0.8: sens 0.475 spec 0.909 prec 0.481 F1 0.478; @0.9: sens 0.388 spec 0.940 prec 0.534 F1 0.449
  - best F1 0.497 at thr 0.7 (sens 0.550, spec 0.883)
  - events @0.5: 14/25 caught, latency median 2.6 s (mean 2.9, max 4.4)
  - events @0.8: 12/25 caught, latency median 3.4 s (mean 3.5, max 5.6)
  - negatives: 1 videos / 0.034 h; FP windows/hr @0.5 58.6, @0.8 0.0, @0.9 0.0
  - per-class videos: adl_other: 1/1 over 0.5; fall: 20/23 over 0.5
  - coverage: 24/24 videos produced windows; footage covered 0.886 of 0.3 h; 4 events with no overlapping window
  - alarm experience (negatives): @0.5: eps/h 29.4, cam/h 29.4, in-alarm 14.7%; @0.8: eps/h 0.0, cam/h 0.0, in-alarm 0.0%; @0.9: eps/h 0.0, cam/h 0.0, in-alarm 0.0%

### eval re-run: `eval_multicam_allcams`
- source `outputs/events/multicam_allcams_tracks_v1`: **window AUC 0.79276** / video AUC 0.75815 (4588 windows, 609 pos; 192 videos, 184 with events)
  - video ops @0.5: TP 160 FN 24 FP 8 TN 0
  - video ops @0.8: TP 143 FN 41 FP 5 TN 3
  - video ops @0.9: TP 133 FN 51 FP 3 TN 5
- deep: **ROC 0.79276 / PR 0.36323** (609+/3979- windows); pos score median 0.769 vs neg 0.052
  - sweep: @0.3: sens 0.672 spec 0.770 prec 0.309 F1 0.424; @0.5: sens 0.598 spec 0.829 prec 0.349 F1 0.441; @0.8: sens 0.479 spec 0.888 prec 0.396 F1 0.434; @0.9: sens 0.414 spec 0.912 prec 0.418 F1 0.416
  - best F1 0.441 at thr 0.6 (sens 0.565, spec 0.847)
  - events @0.5: 118/200 caught, latency median 2.8 s (mean 3.3, max 7.2)
  - events @0.8: 103/200 caught, latency median 2.9 s (mean 3.5, max 7.5)
  - negatives: 8 videos / 0.273 h; FP windows/hr @0.5 120.9, @0.8 40.3, @0.9 25.6
  - per-class videos: adl_other: 8/8 over 0.5; fall: 160/184 over 0.5
  - coverage: 192/192 videos produced windows; footage covered 0.898 of 2.418 h; 25 events with no overlapping window
  - alarm experience (negatives): @0.5: eps/h 62.3, cam/h 47.6, in-alarm 25.5%; @0.8: eps/h 29.3, cam/h 22.0, in-alarm 9.1%; @0.9: eps/h 14.7, cam/h 14.7, in-alarm 5.3%

### eval re-run: `eval_ur_fall`
- source `outputs/events/ur_fall_tracks_v1`: **window AUC 0.98413** / video AUC 0.96711 (104 windows, 14 pos; 52 videos, 14 with events)
  - video ops @0.5: TP 14 FN 0 FP 17 TN 21 (lie-transition FP 15/16)
  - video ops @0.8: TP 14 FN 0 FP 13 TN 25 (lie-transition FP 12/16)
  - video ops @0.9: TP 14 FN 0 FP 6 TN 32 (lie-transition FP 6/16)
- deep: **ROC 0.98413 / PR 0.81536** (14+/90- windows); pos score median 0.998 vs neg 0.161
  - sweep: @0.3: sens 1.000 spec 0.611 prec 0.286 F1 0.444; @0.5: sens 1.000 spec 0.678 prec 0.326 F1 0.491; @0.8: sens 1.000 spec 0.789 prec 0.424 F1 0.596; @0.9: sens 1.000 spec 0.911 prec 0.636 F1 0.778
  - best F1 0.839 at thr 0.95 (sens 0.929, spec 0.956)
  - events @0.5: 14/30 caught, latency median 2.5 s (mean 2.5, max 3.1)
  - events @0.8: 14/30 caught, latency median 2.5 s (mean 2.5, max 3.1)
  - negatives: 38 videos / 0.081 h; FP windows/hr @0.5 356.6, @0.8 233.6, @0.9 98.4
  - per-class videos: adl_lie_transition: 15/16 over 0.5; adl_other: 2/22 over 0.5; fall: 14/14 over 0.5
  - coverage: 52/70 videos produced windows; footage covered 0.851 of 0.111 h; 16 events with no overlapping window; zero-window videos: 18 (0.013 h)
  - alarm experience (negatives): @0.5: eps/h 259.3, cam/h 209.9, in-alarm 74.3%; @0.8: eps/h 172.8, cam/h 160.5, in-alarm 56.5%; @0.9: eps/h 74.1, cam/h 74.1, in-alarm 26.2%

### eval re-run: `study_multicam_allcams_overlap`
- source `outputs/events/multicam_allcams_tracks_v1`: **window AUC 0.79003** / video AUC 0.76766 (4691 windows, 712 pos; 192 videos, 184 with events)
  - video ops @0.5: TP 162 FN 22 FP 8 TN 0
  - video ops @0.8: TP 144 FN 40 FP 5 TN 3
  - video ops @0.9: TP 134 FN 50 FP 3 TN 5
- deep: **ROC 0.79003 / PR 0.39057** (712+/3979- windows); pos score median 0.736 vs neg 0.052
  - sweep: @0.3: sens 0.673 spec 0.770 prec 0.344 F1 0.455; @0.5: sens 0.593 spec 0.829 prec 0.383 F1 0.466; @0.8: sens 0.465 spec 0.888 prec 0.426 F1 0.445; @0.9: sens 0.403 spec 0.912 prec 0.450 F1 0.425
  - best F1 0.466 at thr 0.5 (sens 0.593, spec 0.829)
  - events @0.5: 124/200 caught, latency median 2.4 s (mean 3.1, max 7.9)
  - events @0.8: 105/200 caught, latency median 2.6 s (mean 3.2, max 7.5)
  - negatives: 8 videos / 0.273 h; FP windows/hr @0.5 120.9, @0.8 40.3, @0.9 25.6
  - per-class videos: adl_other: 8/8 over 0.5; fall: 162/184 over 0.5
  - coverage: 192/192 videos produced windows; footage covered 0.900 of 2.418 h; 19 events with no overlapping window
  - alarm experience (negatives): @0.5: eps/h 62.3, cam/h 47.6, in-alarm 25.5%; @0.8: eps/h 29.3, cam/h 22.0, in-alarm 9.1%; @0.9: eps/h 14.7, cam/h 14.7, in-alarm 5.3%

### eval re-run: `study_ur_fall_overlap`
- source `outputs/events/ur_fall_tracks_v1`: **window AUC 0.98497** / video AUC 0.96827 (107 windows, 17 pos; 55 videos, 17 with events)
  - video ops @0.5: TP 17 FN 0 FP 17 TN 21 (lie-transition FP 15/16)
  - video ops @0.8: TP 17 FN 0 FP 13 TN 25 (lie-transition FP 12/16)
  - video ops @0.9: TP 17 FN 0 FP 6 TN 32 (lie-transition FP 6/16)
- deep: **ROC 0.98497 / PR 0.83934** (17+/90- windows); pos score median 0.998 vs neg 0.161
  - sweep: @0.3: sens 1.000 spec 0.611 prec 0.327 F1 0.493; @0.5: sens 1.000 spec 0.678 prec 0.370 F1 0.540; @0.8: sens 1.000 spec 0.789 prec 0.472 F1 0.641; @0.9: sens 1.000 spec 0.911 prec 0.680 F1 0.809
  - best F1 0.865 at thr 0.95 (sens 0.941, spec 0.956)
  - events @0.5: 17/30 caught, latency median 2.4 s (mean 2.3, max 3.1)
  - events @0.8: 17/30 caught, latency median 2.4 s (mean 2.3, max 3.1)
  - negatives: 38 videos / 0.081 h; FP windows/hr @0.5 356.6, @0.8 233.6, @0.9 98.4
  - per-class videos: adl_lie_transition: 15/16 over 0.5; adl_other: 2/22 over 0.5; fall: 17/17 over 0.5
  - coverage: 55/70 videos produced windows; footage covered 0.879 of 0.111 h; 13 events with no overlapping window; zero-window videos: 15 (0.009 h)


## Detector (eyes): COCO person-500 perspective controls

- `coco500_eyes_960_20260826_133034_2db41bba` [clean]: det AP 0.362, kpt AP 0.626
- `coco500_eyes_chained_20260826_111643_2db41bba` [clean]: det AP 0.365, kpt AP 0.637
- `coco500_eyes_lr25_20260826_211623_2db41bba` [clean]: det AP 0.438, kpt AP 0.677
- `coco500_eyes_stockm_20260826_111338_2db41bba` [clean]: det AP 0.682, kpt AP 0.729
- `cpu_proof_coco500_20260807_074619_2db41bba` [clean]: det AP 0.481, kpt AP 0.695
