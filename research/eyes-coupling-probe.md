# The eyes model on side-view event footage: a detector probe

Measured 2026-09-07 on training-side footage only. No exam set was touched and nothing was
trained. This document is the evidence behind the attribution on `final_models/fall/README.md`
that the v2c fall refit moved because of its track trees rather than its weights, and behind
the same movement recorded on `final_models/seizure/README.md` for the clinical tree.

## The question

The v2c track-tree generation replaces the event lanes' earlier detector with the eyes lane
model of record, checkpoint `345c30a5`, ONNX
`4026204967aa01a7680f7ce0c6169463d654a519f8093383a10537321f56fd49`, at the detector score
threshold 0.3 that every earlier generation used. That detector was trained on warehouse
footage from above and at a distance. The fall refit on the v2c trees fell to a validation
AUC of 0.83552 at the median seed against 0.98639 to 0.98823 for the same recipe on the
v2pose trees, and the two frozen heads that hold the fall bar lost about 0.28 of window AUC
when re-sliced onto the v2c exam trees. The question was whether the detector's behaviour on
ordinary side-view footage explains that.

## Track-tree inflation

Counts read from the sequence files of each tree generation. The earlier generation's detector
is not one graph across these rows: the five fall rows ran the upstream rtmdet-m person
release, ONNX `4f4d7e07350b1753299111d1ae500fd64447a5b0e38e4bacbefab6573c742d30`
(`final_models/fall/README.md`, section Track trees), and the WU-SAHZU pat03 row belongs to the
seizure lane, whose earlier generations ran the chained prototype detector `55c2c242`, ONNX
`6a6e4d90a3774522d7e1127122a5a00be41a947e93c8940a3fc6a0c09f97c3ce`
(`final_models/seizure/README.md`, section Track trees).

| track tree | boxes per frame, the earlier generation's detector | boxes per frame, eyes model at 0.3 | tracks, the earlier generation's detector | tracks, eyes model at 0.3 |
|---|---|---|---|---|
| LE2I, training side, one person per video | 1.02 | 1.78 | 227 | 1,283 |
| GMDCSA-24, training side, one person per video | 1.04 | 4.86 | 206 | 2,203 |
| CAUCAFall, training side | not measured | not measured | 142 | 635 |
| UR Fall, exam | 1.63 | 3.31 | 222 | 773 |
| multicam, all cameras, exam | 1.40 | 3.02 | 1,953 | 10,735 |
| WU-SAHZU pat03 free-running video, seizure | not measured | not measured | 54 | 331 |

On GMDCSA-24 the extra boxes are mostly not duplicates of the main box: 16 percent overlap it
at intersection over union 0.5 or more, 44 percent sit elsewhere in the frame at below 0.1,
and 25 percent are small fragments. On LE2I the same three figures are 20, 53 and 41 percent.

## Boxes per frame at candidate thresholds

Eight videos per set and sixteen frames per video, spread evenly through each video, 128
frames per set, the same frames for both detectors. The comparison detector is the stock COCO
RT-DETRv4-S checkpoint, sha256
`238a3f6537bf3b75b55e73f91f9d4cec8d21259b4908b3f21896f3e038b5a3ee`, exported to ONNX
`b4c3ff04a89b4938bfa6fe39ee458ea00fa9b52e407954fa4b21af5d65e53390` from the pinned RT-DETRv4
source with the patches this repository carries; a general-purpose person detector that has
never seen the warehouse footage. Both graphs share the interface, 300 queries, no
non-maximum suppression, and only the person class is counted.

The box counts here are not the box counts in the table above and are not expected to agree.
That table counts rows in the extracted sequence files of each tree generation, over every
frame of every video in the set; this one counts raw detector output on the 128-frame sample,
with nothing downstream of the detector applied. They are different populations of the same
name. The lane pages quote the sequence-file pair, 1.78 boxes per frame on LE2I and 4.86 on
GMDCSA-24 (`final_models/fall/README.md`).

| set | detector | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.80 | frames with no box at 0.5 | median top score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LE2I | eyes model `345c30a5` | 3.023 | 2.117 | 1.656 | 1.305 | 1.164 | 0.984 | 0.859 | 0.719 | 0.664 | 0.531 | 14 of 128 | 0.816 |
| LE2I | stock COCO RT-DETRv4-S `238a3f65` | 0.898 | 0.898 | 0.852 | 0.820 | 0.797 | 0.797 | 0.797 | 0.781 | 0.781 | 0.750 | 26 of 128 | 0.894 |
| GMDCSA-24 | eyes model `345c30a5` | 6.109 | 4.406 | 3.195 | 2.492 | 2.023 | 1.680 | 1.367 | 1.164 | 0.977 | 0.422 | 8 of 128 | 0.776 |
| GMDCSA-24 | stock COCO RT-DETRv4-S `238a3f65` | 0.953 | 0.945 | 0.914 | 0.906 | 0.875 | 0.859 | 0.836 | 0.828 | 0.812 | 0.695 | 17 of 128 | 0.864 |
| CAUCAFall | eyes model `345c30a5` | 1.891 | 1.570 | 1.391 | 1.164 | 1.008 | 0.906 | 0.812 | 0.727 | 0.578 | 0.367 | 32 of 128 | 0.731 |
| CAUCAFall | stock COCO RT-DETRv4-S `238a3f65` | 1.008 | 0.922 | 0.898 | 0.859 | 0.805 | 0.766 | 0.727 | 0.711 | 0.609 | 0.516 | 26 of 128 | 0.814 |

The earlier generation's detector produced 1.02 to 1.04 boxes per frame in the sequence files
of these sets. No single threshold brings the eyes model back to that without losing people:
at 0.5 GMDCSA-24 still carries two boxes per frame while CAUCAFall has no box in a quarter of
its frames.

## Frame-by-frame agreement

The same 128 frames per set. A person counts as found at score 0.5 or above; where both
detectors are confident, their top boxes are compared by intersection over union.

| set | frames | both confident | same person, IoU at least 0.5 | boxes at 0.3 when both confident, eyes model | boxes at 0.3 when both confident, stock | stock low, eyes model confident | eyes model low, stock confident | both low |
|---|---|---|---|---|---|---|---|---|
| LE2I | 128 | 97 | 94 | 3.14 | 1.06 | 17 | 5 | 9 |
| GMDCSA-24 | 128 | 104 | 53 | 6.63 | 1.05 | 16 | 7 | 1 |
| CAUCAFall | 128 | 81 | 68 | 1.95 | 1.07 | 15 | 21 | 11 |

## What the disagreement frames show

Every disagreement frame was drawn with both detectors' boxes and read by eye. The contact
sheets are held with the run outputs outside this repository, because they reproduce frames
of the licensed datasets.

- LE2I, frames where the eyes model is confident and the stock detector is not: the room is
  empty. The eyes model puts boxes at 0.50 to 0.74 on a sofa, a floor lamp and an armchair;
  the stock detector reports nobody. Those boxes become persistent tracks, which is where the
  five to ten times inflation comes from.
- GMDCSA-24, both confident but on different boxes: the person is lying on the bed or the
  floor. The stock detector boxes the whole lying body at 0.54 to 0.79; the eyes model's top
  box is a fragment of the body or a jacket hanging on the wall at 0.5 to 0.6.
- CAUCAFall, frames where the stock detector is confident and the eyes model is not: the
  person is lying on the floor after a fall. The stock detector scores 0.56 to 0.77 on the
  full body; the eyes model scores 0.30 to 0.46 and also boxes a chair. The lying state is
  exactly what the fall verifier needs.
- Where the stock detector is weaker: prone press-up and plank poses in GMDCSA-24, scored
  0.06 to 0.20 while the eyes model is also wrong there, on a jacket; and crouched or fallen
  people in CAUCAFall's night infrared clips, scored 0.10 to 0.49 where the eyes model is
  right at 0.67 to 0.81.

## Reading

The eyes model is out of domain on ordinary side-view footage: the boxes it keeps are often
furniture and the boxes it drops are often lying people, and both effects reach the verifiers
through every window they train and test on. Raising the threshold alone is not a fix. The
practical option is a further declared extraction generation built on the stock COCO
RT-DETRv4-S at a threshold of 0.4, which keeps the 0.43 to 0.49 cases; the durable option is
a re-fit of the eyes model that includes side-view, lying and infrared footage. Either is an
owner decision, and nothing has been re-extracted.

## How it was measured

Three probe scripts, `probe_detector_thr.py`, `probe_pair.py` and `probe_pair_sheets.py`, are
held with the run outputs and are not carried in this repository. The first two build each
detector through the repository's `rtdetr_onnx` backend on the CPU with the score threshold set
to zero and read only the scores and boxes; the third draws the disagreement frames. Frames are
keyed by repository-relative path, because the video basenames repeat across subject folders in
all three sets.
