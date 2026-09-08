# Negative-side bookkeeping for the v2-lane event verifiers (push-2 item 4)

Dated 2026-09-05. Scope: report only (research/seizure-push-2-protocol.md item 4).
No new inference, no training and no exam look: every number below is computed from the
windows.jsonl and report.json each eval run left on disk. **Every number in this document
is a count on recorded windows, not a ward false-alarm rate.** All weights are interim
prototypes under the carve-outs in POLICY.md, rule 4, and nothing here is clinical detection
performance. No selection decision is made or implied here.

Instruments (all over recorded artefacts only):
`scripts/events/metrics_deep.py`, `scripts/events/decision_layer_study.py`,
`scripts/events/wu_sahzu_breakout.py`; the numeric record is written into
`research/stats-compendium.md` by `scripts/events/stats_compendium.py`, which was not
re-run for this pass (see the closing note).

Definitions (fixed by the instruments, not tuned here):
- **in-alarm** = fraction of the negative recordings' footage covered by the union of
  [fire minus 12 s, fire] spans for fires at or above the threshold. The instrument
  hard-codes the 12 s span, so the fall rows, whose windows are 8.0 s, are overstated by up
  to 4 s per fire (decision_layer_study `neg_time_in_alarm`; single:thr rules reproduce the
  raw union of windows).
- **FP win/h** = windows at or above the threshold on the negative recordings divided by their recorded hours
  (metrics_deep `fp_windows_per_hour`).
- **eps/h** = distinct false-alarm episodes per negative hour, fires < 30 s apart merged;
  quoted as track-level / camera-level (camera = all tracks on one recording merged first).
- **persist:2@0.5** = fire only on 2 consecutive same-track windows at or above 0.5.
- **catches** = metrics_deep event rows (a window at or above the threshold overlapping the labelled event);
  where a second figure is given "(dls N/M)" it is decision_layer_study's catch (any-track
  fire in [onset, end + 30 s]); the definitions differ and both are recorded.

Denominators, stated once and repeated per table:
- **pat03__free = 662.3 s (0.184 h)**, the only negative recording in E2.
- **E1 has no negative videos** (pat06/08/10/12 are all seizure recordings): every
  per-hour column is n/a and the eval report's `video_fp 0` is vacuous, not quiet.
- **WU-SAHZU testsplit negatives = 3 recordings, 0.944 h** (pat03__free 662.3 s +
  pat04__free 1781.6 s + pat04__no-Sz2P 955.7 s).
- **chute24 = 0.273 h** (8 cameras of the one no-fall chute).
- **UR ADL = 0.081 h** (38 of the 40 ADL recordings; adl-24 and adl-28 produced zero
  windows and sit outside every denominator; see the coverage note).

---

## Lane 1: convulsive exam E2 (pat03 and pat07, locked)

Negative denominator per row: **pat03__free, 662.3 s (0.184 h)**, one recording, one
camera. All values are counts on recorded windows, not a ward false-alarm rate.
On-video non-event windows (pre-onset and post-ictal, including staff activity, on the four
PG seizure videos) are tabulated separately below and are not in the per-hour columns.

| model (onnx sha8) | in-alarm @0.5/0.8/0.9 | FP win/h @0.5/0.8/0.9 | eps/h track and cam @0.5 | persist:2@0.5 catch, eps/h, in-alarm | catches @0.5 (event) |
|---|---|---|---|---|---|
| model of record `07c399be` | 75.2% / 39.3% / 9.1% | 891.4 / 331.6 / 65.2 | 81.5, 27.2 | 4/4 PG, 70.7, 68.2% | 4/4, median 6.0 s |
| finalA `a373127b` | 55.5% / 11.0% / 7.1% | 679.5 / 141.3 / 54.4 | 97.8, 27.2 | 3/4 PG, 70.7, 45.3% | 3/4, median 22.8 s |
| finalB `b9e04d03` | 68.7% / 17.2% / 9.8% | 891.4 / 206.6 / 97.8 | 114.1, 27.2 | 3/4 PG, 76.1, 54.0% | 4/4, median 30.9 s |
| finalC `d3e6d8c6` | 87.5% / 81.8% / 80.0% | 1212.1 / 962.1 / 875.1 | 92.4, 10.9 | 2/4 PG, 76.1, 79.8% | 3/4, median 9.0 s |

- finalC `d3e6d8c6` is the episode-count inversion the 2026-08-27 record warned about:
  its camera-episodes/h is the lowest of the four (10.9) because on pat03__free its alarm
  is lit 80 to 88% of the time and essentially never turns off. in-alarm is the honest quiet
  metric next to the episode counts; finalC is the loudest model on this recording at
  every threshold.
- Raising the threshold to 0.9 buys quiet only by dropping PG catches (model of record 4/4 to 2/4,
  finalA 3/4 to 1/4, finalB 4/4 to 1/4 across single:0.5 to single:0.9); finalC keeps 2/4 at
  0.9 while staying 80% in alarm.
- On-video non-event windows (130 per model, the same recorded windows for all four):
  at or above 0.5/0.8/0.9, model of record 97/25/10, finalA 35/16/5, finalB 36/20/12, finalC 36/17/11.
  These are pre-onset and post-ictal windows on seizure recordings (staff response
  included) and are kept separate from the seizure-free recording above.
- AJILE12 fingerprint caveat for arm B, `b9e04d03`: arm B's training diet includes the
  2 h AJILE12 proxy-negative dose, and every proxy row carries the fingerprint
  `angle_valid 0` plus constant `kpc 0.5294`. A net can learn to separate the proxy source
  rather than the behaviour, so arm B's quietness on its own training diet certifies nothing.
  The rows above are real WU-SAHZU windows (no proxy rows in any eval), so they are the
  honest measurement; what the AJILE12 dose actually contributed is not verifiable from
  recorded windows. The authorised, roughly 20-minute channel-masking control (H2h) is a separate
  report-only run outside this step, which carries no new inference; that control was run and
  is recorded in `research/seizure-ajile12-negative-control.md`.
- The model-of-record row is the E2 reslice of the `convulsion_v2_real` model through
  the v2pose tracks (`outputs/verifier/conv_champion_reslice/eval_E2`, a build output that is
  not in version control). That run directory has no training report in this repository, so
  `research/stats-compendium.md` does not list it; this table and its eval-directory artefacts
  are its record.

## Lane 2: focal exam E1 (pat06/08/10/12)

**E1 has no negative videos.** There is no negative-recording denominator, no in-alarm
fraction, no FP-windows/h and no episodes/h, and the eval reports' `video_fp 0` is vacuous,
so it must never be quoted as quietness. What the recorded windows do carry is the on-video
non-event population: 1253 label-0 windows on the 12 focal seizure recordings
(pre-onset, post-ictal and non-owner tracks). Counts on recorded windows, not a ward
false-alarm rate.

| model (onnx sha8) | non-event windows at or above 0.5/0.8/0.9 (of 1253) | catches @0.5 (event) | catches @0.8 | video-level at or above 0.5/0.8/0.9 (of 12) |
|---|---|---|---|---|
| finalA `a373127b` | 109 / 15 / 3 | 10/12, median 15.6 s | 5/12 | 12 / 10 / 4 |
| finalB `b9e04d03` | 103 / 18 / 6 | 10/12, median 12.6 s | 7/12 | 12 / 10 / 4 |
| finalC `d3e6d8c6` | 219 / 119 / 47 | 9/12, median 41.8 s | 6/12 | 11 / 10 / 10 |

- decision_layer_study is not runnable here from recorded data: it needs negative
  recording hours, and metrics_deep correctly reports 0.0 for E1. The value is reported as
  measured, not improvised.
- The AJILE12 fingerprint caveat above applies to finalB's row identically.
- finalC's E1 look exists on disk (`conv_finalC/eval_E1`, window AUC 0.58613) and is
  included so the recorded exam-look ledger and this table agree.

## Lane 3: v2pose seizure verifier on the WU-SAHZU test split

Model: `convulsion_v2pose` onnx `614b92fa`. Negative denominator per row: **3 seizure-free
recordings, 0.944 h** (pat03__free 662.3 s, pat04__free 1781.6 s, pat04__no-Sz2P 955.7 s).
Counts on recorded windows, not a ward false-alarm rate.

| model (onnx sha8) | in-alarm @0.5/0.8/0.9 | FP win/h @0.5/0.8/0.9 | eps/h track and cam @0.5 | persist:2@0.5 catch, eps/h, in-alarm | catches @0.5 (event) |
|---|---|---|---|---|---|
| v2pose `614b92fa` | 13.5% / 0.0% / 0.0% | 82.6 / 0.0 / 0.0 | 27.5, 16.9 | 3/13 (0/6 PG), 20.1, 8.8% | 5/13, median 14.3 s (dls 5/13, PG 1/6) |

- Per-recording FP windows at or above 0.5: pat03__free 13, pat04__free 57, pat04__no-Sz2P 8;
  nothing on any negative recording ever reaches 0.8. The quiet above 0.8 is bought with
  0/13 catches at 0.8, so the score range is compressed, not selective.
- On-video non-event windows (657 on the 13 seizure recordings): 13 at or above 0.5 and 0 at or above 0.8, which is
  reported separately from the seizure-free recordings per the breakout discipline.
- WU-SAHZU breakout (video max-score): PG 2/6, focal 5/7, non-seizure flagged 3/3 @0.5
  (`outputs/verifier/convulsion_v2pose/eval_wusahzu_testsplit/breakout.json`).

## Lane 4: fall verifier `fall_v2pose` (onnx `b66c3a0a`)

`fall_v2pose` is a recorded comparison head, not the fall model of record: the fall lane of
record is `fall_v3`, onnx `67954c04` (`final_models/fall/README.md`), so the counts below are
not the shipped model's.

Counts on recorded windows, not a ward false-alarm rate. Two exam surfaces, two
denominators; same model both rows.

| exam (denominator) | in-alarm @0.5/0.8/0.9 | FP win/h @0.5/0.8/0.9 | eps/h track and cam @0.5 | persist:2@0.5 catch, eps/h, in-alarm | catches @0.5 (event) |
|---|---|---|---|---|---|
| multicam chute24 (**0.273 h**, 8 cams) | 9.4% / 5.2% / 5.0% | 51.3 / 22.0 / 18.3 | 25.6, 14.7 | 114/200, 7.3, 4.1% | 107/200, median 2.9 s (dls 141/200) |
| UR ADL (**0.081 h**, 38 recordings) | 60.1% / 20.6% / 4.1% | 258.2 / 61.5 / 12.3 | 197.5, 172.8 | 0/30, 37.0, 13.3% | 14/30, median 2.55 s (dls 14/30) |

- UR persist:2 removes every catch (0/30): only 14 of the 30 UR fall events produced any
  positive window (16 fall videos are zero-window under the recorded contain-policy
  extraction), and the surviving events yield too few consecutive windows for a 2-streak.
  14/30 is the recorded ceiling at any threshold; the misses are extraction coverage, not
  scoring.
- Coverage holes carried into every denominator above: for UR, 18 zero-window videos
  (16 fall plus adl-24 and adl-28), footage covered 0.851 of 0.111 h; for multicam, 25 events
  with no overlapping window, footage covered 0.897. Zero-window ADL footage is outside the
  0.081 h denominator (that is why it is 0.081 h and not the 0.083 h container total).
- On-video non-event windows on fall recordings (outside the labelled fall interval):
  multicam 551/327/239 of 3338 at or above 0.5/0.8/0.9 (post-fall lying-still and recovery motion
  dominate); UR fall recordings contribute 0 label-0 windows under the recorded contain
  extraction.
- chute24's per-camera spread @0.5: cam1 8, cam8 5, cam5 1 FP windows; cams 2/3/4/6/7
  have zero windows at or above 0.5.

---

Artefacts written by this pass (all derived from recorded windows): per-eval
`metrics_deep.json` (10 dirs), `decision_layer_study.json` (7 dirs: 4 for E2, testsplit,
2 for fall) and `breakout.json` (8 conv evals), all under `outputs/`, which is not in version
control. `research/stats-compendium.md` was not regenerated for this pass: it carries the
numeric record as it stood when it was generated on 2026-08-28, so the runs above are not in
it and this document is their record.
