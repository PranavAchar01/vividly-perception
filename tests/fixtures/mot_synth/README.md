# mot_synth: synthetic MOTChallenge fixture

Two tiny sequences of MOTChallenge-format ground truth plus three tracker outputs, used
by `tests/test_mot.py`. About 7 KB in total.

Nothing here is derived from any dataset. Every box is an arithmetic function of the
frame index, there is no RNG anywhere, and there are no images: TrackEval never opens
`imDir`, so `seqinfo.ini` describes a frame size that no file has. That means the whole
tracking-metrics path is testable today, with no download, no counsel gate, and no bytes
from any EVAL_ONLY set.

Guardrails, stated explicitly because a fixture is still data: no real people, no
recorded footage, and no simulated or acted medical event of any kind. These are
rectangles moving in straight lines.

## Layout

    seqs/<SEQ>/gt/gt.txt        9 columns: frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,visibility
    seqs/<SEQ>/seqinfo.ini      [Sequence] block; TrackEval reads seqLength from it
    trackers/perfect/<SEQ>.txt  10 columns: ...,conf,-1,-1,-1
    trackers/corrupted/<SEQ>.txt
    trackers/empty/<SEQ>.txt    zero bytes

Frames and track ids are 1-based in every file. GT `conf` is the consider flag (1), not a
detector score; `class` is 1 (pedestrian) and `visibility` is 1.

## Ground truth

Frame size 1280x720, frame rate 15, in both sequences. Track `t` occupies
`left = x0 + dx * (frame - 1)`, `top = y`, for the whole sequence. Tracks are separated in
`y`, so the IoU between two different tracks is exactly 0 and matching is never ambiguous.

| seq | frames | track | x0 | dx | y | w | h |
|---|---|---|---|---|---|---|---|
| SEQ01 | 12 | 1 | 100 | +20 | 100 | 40 | 90 |
| SEQ01 | 12 | 2 | 600 | -20 | 300 | 44 | 96 |
| SEQ01 | 12 | 3 | 300 | +10 | 500 | 50 | 100 |
| SEQ02 | 10 | 1 | 80 | +15 | 150 | 42 | 94 |
| SEQ02 | 10 | 2 | 500 | +12 | 420 | 46 | 98 |

## Trackers

- **perfect** is a box-for-box copy of the ground truth with `conf = 1.0`. It scores
  `HOTA = MOTA = IDF1 = 1.0` on both sequences. Anything else means the interchange
  format, the directory tree, or `DO_PREPROC` is wrong, which is exactly what the fixture
  is for.
- **corrupted** carries two named defects, one per sequence, so each shows up in a
  different metric:
  - SEQ01: the ids of tracks 1 and 2 are exchanged from frame 7 onward. Detection is
    untouched (`DetA` stays 1.0); association pays for it (`IDSW = 2`, `IDF1 < 1`).
  - SEQ02: frames 4, 5 and 6 are dropped entirely. Association is untouched
    (`IDSW = 0`); detection pays for it (`DetA < 1`).
  The tests assert the ordering (corrupted below perfect, `IDF1 < 1`), never a magic
  constant, so a trackeval version bump shows up as a real signal rather than a diff.
- **empty** is two zero-byte files. It must score 0.0 and must not raise. That branch is
  why `trackeval` is pinned to the maintained PyPI fork rather than the upstream git URL,
  whose `np.float` on the empty-tracker path was removed in NumPy 1.24; a run that takes
  the normal path never reaches it.

## Regenerating

The files are committed, and the tests read the committed files, never a generator. To
change them, edit the table above, rewrite the text with
`vividly.eval.mot.write_gt_file` / `write_tracker_file` / `write_seqinfo`, and re-run
`uv run --no-sync pytest tests/test_mot.py -q`.
