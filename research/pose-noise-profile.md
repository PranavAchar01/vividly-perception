# Pose noise profile: the v2pose track trees, joint by joint

Dated 2026-09-05. Scope: report only. Instrument: `scripts/analysis/pose_noise_profile.py`. Every number below is a count over rows that were already on disk; no model was loaded, no window was scored, nothing was trained. **Pose weights: `6ba96285`** (`rtmpose-m-person-v2`, the P0-e38 model of record of research/pose-seed-floor-and-champion.md); all 7 trees were extracted with that one graph, and the per-tree table records how each tree attests it.

Gate used throughout: **conf at or above 0.3**, which is `features.KP_CONF_MIN`, the `tracks.kp_conf_min` of 0.3 that the extraction configurations carry (`configs/events/extract_*_v2c.yaml`; the superseded v2pose configurations, not carried in this repository, used the same value), and the gate `posture_angle_deg` applies. Jitter is the per-frame displacement of a joint between rows of one track whose frame indices differ by exactly 1, with both endpoints above the gate, divided by the mean of the two box heights (box-heights/frame; the per-second column makes 15/20/25/30 fps trees comparable). Definitions in full sit in the instrument's docstring.

## 1. Pooled over every v2pose tree

1,512,537 rows, 4,208 tracks and 711 recordings across 7 trees.

| joint | n obs | conf p10 | conf median | conf p90 | at or above gate | jitter median (bh/frame) | jitter p90 | jitter median (bh/s) | n pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nose | 1,512,537 | 0.194 | **0.546** | 0.831 | 74.4% | 0.009 | 0.041 | 0.266 | 1,068,956 |
| left_eye | 1,512,537 | 0.198 | **0.548** | 0.845 | 74.6% | 0.009 | 0.041 | 0.256 | 1,070,462 |
| right_eye | 1,512,537 | 0.195 | **0.571** | 0.854 | 75.5% | 0.009 | 0.040 | 0.252 | 1,086,015 |
| left_ear | 1,512,537 | 0.168 | **0.555** | 0.845 | 69.6% | 0.007 | 0.037 | 0.212 | 1,003,493 |
| right_ear | 1,512,537 | 0.170 | **0.549** | 0.844 | 72.3% | 0.007 | 0.038 | 0.213 | 1,043,495 |
| left_shoulder | 1,512,537 | 0.149 | **0.545** | 0.780 | 67.8% | 0.007 | 0.034 | 0.209 | 978,547 |
| right_shoulder | 1,512,537 | 0.150 | **0.532** | 0.775 | 68.6% | 0.007 | 0.033 | 0.209 | 988,545 |
| left_elbow | 1,512,537 | 0.128 | **0.440** | 0.757 | 63.1% | 0.008 | 0.038 | 0.235 | 906,625 |
| right_elbow | 1,512,537 | 0.127 | **0.428** | 0.762 | 63.0% | 0.009 | 0.041 | 0.263 | 902,764 |
| left_wrist | 1,512,537 | 0.134 | **0.376** | 0.747 | 59.4% | 0.008 | 0.045 | 0.241 | 841,803 |
| right_wrist | 1,512,537 | 0.133 | **0.371** | 0.743 | 59.0% | 0.009 | 0.047 | 0.272 | 834,633 |
| left_hip | 1,512,537 | 0.117 | **0.415** | 0.667 | 60.0% | 0.009 | 0.034 | 0.272 | 865,132 |
| right_hip | 1,512,537 | 0.121 | **0.411** | 0.666 | 60.6% | 0.009 | 0.034 | 0.276 | 871,569 |
| left_knee | 1,512,537 | 0.114 | **0.345** | 0.731 | 55.1% | 0.009 | 0.041 | 0.260 | 786,972 |
| right_knee | 1,512,537 | 0.114 | **0.374** | 0.727 | 58.7% | 0.009 | 0.040 | 0.252 | 836,403 |
| left_ankle | 1,512,537 | 0.167 | **0.357** | 0.756 | 60.2% | 0.008 | 0.047 | 0.234 | 855,192 |
| right_ankle | 1,512,537 | 0.163 | **0.389** | 0.762 | 63.2% | 0.007 | 0.044 | 0.220 | 902,644 |

By group:

| group | conf p10 | conf median | conf p90 | at or above gate | jitter median (bh/frame) | jitter p90 |
|---|---:|---:|---:|---:|---:|---:|
| face (nose/eyes/ears) | 0.182 | **0.553** | 0.844 | 73.3% | 0.008 | 0.040 |
| shoulders | 0.150 | **0.538** | 0.778 | 68.2% | 0.007 | 0.033 |
| elbows | 0.128 | **0.433** | 0.760 | 63.1% | 0.008 | 0.040 |
| wrists | 0.133 | **0.373** | 0.745 | 59.2% | 0.009 | 0.046 |
| hips | 0.119 | **0.413** | 0.666 | 60.3% | 0.009 | 0.034 |
| knees | 0.114 | **0.360** | 0.729 | 56.9% | 0.009 | 0.040 |
| ankles | 0.165 | **0.371** | 0.759 | 61.7% | 0.008 | 0.045 |

## 2. Per tree, joint groups

Confidence median (top line of each cell) and share at or above gate (bottom).

| tree | rows | tracks | recs | face (nose/eyes/ears) | shoulders | elbows | wrists | hips | knees | ankles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `caucafall_tracks_v2pose` | 20,770 | 142 | 100 | 0.686<br>97.1% | 0.690<br>96.2% | 0.678<br>92.9% | 0.676<br>89.0% | 0.644<br>95.7% | 0.722<br>93.6% | 0.719<br>91.9% |
| `gmdcsa24_tracks_v2pose` | 34,388 | 206 | 159 | 0.832<br>97.8% | 0.776<br>98.8% | 0.726<br>97.9% | 0.672<br>95.2% | 0.682<br>98.6% | 0.743<br>97.8% | 0.739<br>96.1% |
| `le2i_tracks_v2pose` | 45,370 | 227 | 130 | 0.842<br>92.0% | 0.784<br>92.3% | 0.759<br>91.6% | 0.725<br>90.0% | 0.676<br>91.9% | 0.747<br>89.8% | 0.765<br>87.7% |
| `multicam_allcams_tracks_v2pose` | 365,587 | 1,953 | 192 | 0.633<br>82.2% | 0.644<br>78.8% | 0.542<br>72.1% | 0.433<br>67.4% | 0.562<br>73.2% | 0.577<br>71.5% | 0.596<br>74.5% |
| `multicam_tracks_v2pose` | 45,162 | 301 | 24 | 0.645<br>85.1% | 0.651<br>84.2% | 0.554<br>76.0% | 0.438<br>70.3% | 0.573<br>78.5% | 0.599<br>75.4% | 0.626<br>80.7% |
| `ur_fall_tracks_v2pose` | 19,457 | 222 | 70 | 0.752<br>87.0% | 0.694<br>86.5% | 0.646<br>84.0% | 0.576<br>79.0% | 0.582<br>83.1% | 0.613<br>81.2% | 0.595<br>85.1% |
| `wu_sahzu_tracks_v2pose` | 981,803 | 1,157 | 36 | 0.457<br>66.9% | 0.413<br>60.3% | 0.339<br>55.5% | 0.312<br>51.9% | 0.305<br>50.6% | 0.277<br>46.4% | 0.310<br>52.6% |

Jitter median, box-heights per frame (gate-passing adjacent pairs only):

| tree | median fps | face (nose/eyes/ears) | shoulders | elbows | wrists | hips | knees | ankles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `caucafall_tracks_v2pose` | 20.00 | 0.017 | 0.013 | 0.013 | 0.014 | 0.010 | 0.009 | 0.008 |
| `gmdcsa24_tracks_v2pose` | 29.82 | 0.009 | 0.008 | 0.010 | 0.011 | 0.007 | 0.008 | 0.008 |
| `le2i_tracks_v2pose` | 25.00 | 0.009 | 0.007 | 0.008 | 0.009 | 0.007 | 0.007 | 0.006 |
| `multicam_allcams_tracks_v2pose` | 30.00 | 0.017 | 0.014 | 0.016 | 0.019 | 0.014 | 0.014 | 0.013 |
| `multicam_tracks_v2pose` | 30.00 | 0.017 | 0.014 | 0.016 | 0.019 | 0.014 | 0.014 | 0.013 |
| `ur_fall_tracks_v2pose` | 30.00 | 0.011 | 0.009 | 0.010 | 0.011 | 0.009 | 0.009 | 0.010 |
| `wu_sahzu_tracks_v2pose` | 29.97 | 0.005 | 0.004 | 0.005 | 0.005 | 0.007 | 0.006 | 0.005 |

## 3. Scalar channels the verifiers actually read

`kpc` is the row's mean confidence over all 17 entries, which is feature channel `kpc`. `angle_valid` is the share of rows whose `angle` field is non-null, the row-level population behind feature channel `angle_valid` (and, because `angle_norm` is that same field over 90 with nulls filled to 0, behind `angle_norm` too).

| tree | rows | kpc p10 | kpc median | kpc p90 | kpc at or above gate | angle_valid | pose weights (attested by) |
|---|---:|---:|---:|---:|---:|---:|---|
| `caucafall_tracks_v2pose` | 20,770 | 0.491 | **0.669** | 0.784 | 96.4% | 94.9% | `6ba96285` (from the superseded `extract_caucafall_v2pose` configuration, not carried in this repository; no extraction.json on disk) |
| `gmdcsa24_tracks_v2pose` | 34,388 | 0.547 | **0.741** | 0.845 | 99.0% | 97.7% | `6ba96285` (from the superseded `extract_gmdcsa24_v2pose` configuration, not carried in this repository; no extraction.json on disk) |
| `le2i_tracks_v2pose` | 45,370 | 0.486 | **0.757** | 0.858 | 92.2% | 91.4% | `6ba96285` (extraction.json) |
| `multicam_allcams_tracks_v2pose` | 365,587 | 0.177 | **0.602** | 0.730 | 77.2% | 71.2% | `6ba96285` (extraction.json) |
| `multicam_tracks_v2pose` | 45,162 | 0.194 | **0.609** | 0.726 | 81.3% | 76.8% | `6ba96285` (from the superseded `extract_multicam_v2pose` configuration, not carried in this repository; no extraction.json on disk) |
| `ur_fall_tracks_v2pose` | 19,457 | 0.251 | **0.652** | 0.818 | 86.4% | 80.7% | `6ba96285` (extraction.json) |
| `wu_sahzu_tracks_v2pose` | 981,803 | 0.162 | **0.378** | 0.667 | 58.5% | 47.8% | `6ba96285` (extraction.json) |
| **pooled** | 1,512,537 | 0.167 | **0.493** | 0.719 | 66.5% | 57.8% | not applicable |

## 4. Contrast with the one recorded body7 figure

`research/seizure-limb-channel-gate.md` measured **wrist+ankle confidence median 0.305, 25.1% > 0.5** over all 976,659 rows of `wu_sahzu_tracks_v2kp`, and that single pooled number is the measured cause it gives for the limb channels' 8/8-negative result. `v2kp` and `v2pose` are the same 36 recordings through the same detector (`55c2c242`, sha `6a6e4d90`) and the same tracker; the only difference between them is the pose graph, stock body7 `5c0a4bf6` against the v2 model of record `6ba96285`. So the two computed rows below isolate the pose upgrade and nothing else.

| population (wrist + ankle, every track) | n obs | conf p10 | conf median | conf p90 | at or above 0.5 | at or above 0.8 | at or above gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `wu_sahzu_tracks_v2kp`, as recorded in research/seizure-limb-channel-gate.md | 976,659 rows x 4 | not recorded | 0.305 | not recorded | 25.1% | 1.8% | not recorded |
| `wu_sahzu_tracks_v2kp` (recomputed here) | 3,932,416 | 0.121 | **0.303** | 0.675 | 24.8% | 1.8% | 50.6% |
| `wu_sahzu_tracks_v2pose` (v2 pose model of record) | 3,927,212 | 0.137 | **0.311** | 0.693 | 26.1% | 2.1% | 52.3% |

`n obs` counts joint observations (rows x 4 joints), which is why the recorded row is given as `rows x 4`. The `v2kp` tree on disk today holds 983,104 rows against the 976,659 research/seizure-limb-channel-gate.md counted, a 0.7% difference whose most likely cause is that the earlier count dropped the `h <= 0` rows `tracks_from_rows` discards. It is far too small to move a median, and the recomputed 0.303 against the recorded 0.305 is the check that both instruments are measuring the same population.

The columns marked at or above are inclusive where research/seizure-limb-channel-gate.md wrote `>`; at 0.001 bin width the two differ by less than the last digit shown.

**Exam-side rows are included, as statistics only.** The WU tree's 14 patients include the 6 allocated exam-side (pat03, pat06, pat07, pat08, pat10, pat12). This instrument runs no model and produces no score, so their rows are data profiling, not an exam look; no number on this page is or implies a held-out metric.

## 5. Reading

**Nothing here is a per-joint accuracy.** RTMPose emits a confidence for all 17 joints on
every observation, so `n obs` is the row count, not a detection count, and a 0.1 entry is
the estimator guessing rather than a joint that went missing. What the table ranks is how
often each joint is *worth reading*.

1. **The trustworthy end is the head and the shoulder girdle; the untrustworthy end is
   every distal joint.** Pooled medians fall monotonically outward from the torso: face
   0.553, shoulders 0.538, elbows 0.433, hips 0.413, wrists 0.373, ankles 0.371, knees
   0.360. The share clearing the 0.3 gate falls with them, 73.3% at the face to 56.9%
   at the knees. Roughly two in five wrist, knee and ankle observations in this corpus are
   below the gate that `posture_angle_deg` and `KP_CHANNELS` already apply, so any feature
   built on a distal joint is running on about half the frames it appears to have.
2. **Which tree the footage comes from matters more than which joint.** `wu_sahzu`, the
   seizure lane's own EMU footage and the only clinical set here, sits at face 0.457 /
   wrists 0.312 / knees 0.277, while every acted-fall tree sits between 0.55 and 0.85 on
   the same joints. gmdcsa24 and le2i clear the gate on 88-99% of observations at every
   joint; `wu_sahzu` clears it on 46-67%. Bedding, blankets, low light and a wall-mounted
   oblique view are the difference. A gate tuned on the fall trees is a different gate on
   the seizure tree.
3. **Jitter is small and, at the median, nearly joint-independent.** Conditioned on both
   endpoints clearing the gate, the median frame-to-frame displacement is 0.007-0.009
   box-heights everywhere, about 0.2-0.3 box-heights per second, well under the motion any
   event rule looks for. The joints separate only in the tail: p90 is 0.033-0.034 for
   shoulders and hips against 0.045-0.047 for wrists and ankles, i.e. the distal joints are
   not noisier most of the time, they are noisier *sometimes*, and that tail is where a
   spectral limb feature would live. Read those numbers as conditional: a joint that clears
   the gate 57% of the time contributes jitter only from its better half.
4. **Per tree the jitter ordering is a property of the footage, not of the estimator.**
   `wu_sahzu` is the quietest tree at every joint (0.004-0.007 bh/frame) because its
   subjects are in bed; the multicam trees are the loudest (0.013-0.019) because their
   subjects walk across the room. This is the same fact the seizure lane already pays for
   elsewhere, that a patient convulsing in bed barely moves their box, arriving from the
   opposite direction.
5. **The v2 pose model of record did not fix the limb-confidence problem.** On identical footage
   the wrist+ankle median moves from 0.303 (stock body7, `v2kp`) to 0.311 (v2 model of record,
   `v2pose`), and the share above 0.5 moves from 24.8% to 26.1%. The limb-channel gate
   finding therefore survives its own pose upgrade intact: sub-gate limb confidence on
   this footage is a property of the recordings, not of the weights, and re-extracting will
   not buy a limb feature the confidence it needs. The two independently recomputed rows
   also reproduce that document's headline (0.305 recorded, 0.303 recomputed here over a row
   population that differs by 0.7%), which is the cheapest available check that both
   instruments are measuring the same thing.
6. **What this does not say.** No number here is evidence for or against any model, gate
   value or feature; it is the distribution those decisions are taken against, which had
   never been written down per joint. Confidence is not calibrated, and 0.6 from RTMPose is
   not a 60% probability of correct placement, so these are comparisons within one
   estimator's own scale, never absolute accuracy. And jitter here is displacement, not
   error: a fast-moving joint and a badly-estimated one look the same to it.
