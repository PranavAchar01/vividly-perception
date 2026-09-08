# The model record

This folder is the single record of which model holds each lane, what it was measured on,
and which bytes produced the number.

Three statements bind every page in this folder.

1. Every weight listed here is the prototype weight class. It is internal demonstration
   material only, it is not a medical device, and no number on any of these pages is
   clinical detection performance.
2. Every number carries the hash that produced it. A number without its hash is not
   quotable, because one value can identify two different checkpoints.
3. A model is compared only on an exam it never trained on, variants are selected on the
   train-side validation split, each version gets one pre-declared look at the test set,
   and an exam is never edited to flatter a model.

## What is in this folder

```
final_models/
  README.md                this index and the rules
  eyes/     README.md  model.onnx  model.onnx.sha256  model.json  curves/
            validation_curves.png  champion_vs_baseline.png
  joints/   README.md  model.onnx  model.onnx.sha256  model.json  curves/
            validation_curves.png  champion_vs_baseline.png
  fall/     README.md  model.onnx  model.onnx.sha256  model.json  report.json  curves/
            report_v2c_F0_s20260826.json  report_v2c_F0_s20260901.json  report_v2c_F0_s20260905.json
            validation_curves.png  champion_vs_baseline.png
  seizure/  README.md  model.onnx  model.onnx.sha256  model.json  report.json  curves/
            report_v2c_finalB.json  validation_curves.png  champion_vs_baseline.png
  reid/     README.md  model.onnx  model.onnx.sha256  model.json  eval_summary.json  curves/
            validation_curves.png  champion_vs_baseline.png
```

Each lane directory holds the weight of record as `model.onnx`, its digest as
`model.onnx.sha256` in `sha256sum -c` form, and a `model.json` sidecar that carries the
lane, the kind, the loader, the display name, both hashes, the input size where one
applies, the weight class, the training data and the backbone pretrain lineage. The
sidecar's `input_size` is written as width then height, the same order every pipeline
configuration uses. Its `backend` field names the loader: `rtdetr_onnx` and `rtmlib` are
detector and pose keys in `src/vividly/models/factory.py`, `onnx` is the embedder key
there, and `event_verifier` means the head is built from the lane directory by
`vividly.models.event_verifier` through a pipeline config's `verify.heads.<rule_id>.run_dir`
rather than through the factory. The demo model picker reads the detector and pose
sidecars; the two event heads attach to every demo run from their lane directories, and the
re-identification lane is reached through the pipeline configuration, not the picker. The
two event lanes also keep the `report.json` their trainer wrote, which carries the window
geometry, the calibration temperature and the graph digest the loader checks; the fall lane
keeps the three `report_v2c_F0_*.json` files and the seizure lane `report_v2c_finalB.json`
from the refits its page records and did not promote. The re-identification lane keeps
`eval_summary.json`, its trainer's measurement record, in place of a `report.json`. The two
figures in every lane are drawn from `curves/` by `scripts/analysis/plot_final_curves.py`.

The `.onnx` files are stored with Git LFS. After cloning, run `git lfs pull`; if a weight
looks like a short text file rather than a binary, that pull has not happened. Verify a
weight in place with:

    cd final_models/<lane> && shasum -a 256 -c model.onnx.sha256

## The index

| Lane | Model of record | Checkpoint sha8 | ONNX sha8 | Headline metric | Exam | Value | Compared against | Page |
|---|---|---|---|---|---|---|---|---|
| eyes | RT-DETRv4-S person v2, seed 0 | `345c30a5` | `40262049` | AP 0.50:0.95 | person_v2 dev-val | 0.8356 | DEIM `e98de401` at 0.8341; pre-registered bar 0.8353 | [eyes/README.md](eyes/README.md) |
| joints | RTMPose-m person v2, seed 0 epoch 38 | `a0a035fe` | `6ba96285` | keypoint AP, ground-truth boxes | COCO val2017 | 0.774115 | body7 init `e48f03d0` at 0.764499 | [joints/README.md](joints/README.md) |
| fall | fall_v3 | `3d2d17ee` | `67954c04` | window AUC | multicam, all cameras, 192 videos | 0.79276 | fall_v2pose `b66c3a0a` at 0.78474 | [fall/README.md](fall/README.md) |
| seizure | convulsion_v2_real | `715d172c` | `07c399be` | window AUC | wusahzu testsplit, 16 videos | 0.69841 | convulsion_v2pose `614b92fa` at 0.53728 | [seizure/README.md](seizure/README.md) |
| reid | OSNet-x1_0 MEVID Phase A, epoch 30 | `66623239` | `60164a99` | mAP, general protocol | MEVID test split, 316 queries against 1,438 gallery tracklets | 0.204782 | external same-clothes floor 0.39, not reached; published general-protocol figure 0.271, not reached | [reid/README.md](reid/README.md) |

The table shows the first eight characters of each digest. The full sha256 of every weight
is in that lane's `model.onnx.sha256` and `model.json`, and the full checkpoint sha256 is
on the lane page. Every eight-character short form used anywhere in this folder resolves to
a full sha256 on one of the five lane pages; where a digest is on record only as eight
characters, the lane page says so and says why. The re-identification row is the lane's
first measured embedder for this domain and sits below both external figures it is
compared against; its page says so, records the two collapsed runs that preceded it, and
reports the camera-debiased and in-house same-clothes protocols beside the general one
without treating either as comparable to the published values. It is compared against an
external published value, which carries no hash, because no earlier in-house embedder
exists for this domain. Its rank-1 of 0.490506 is on the lane page beside the mAP. A rerun
of the same recipe on a 60-epoch schedule, declared before it ran, scored 0.188552 and was
not promoted (`research/reid-rerun-a2-protocol.md`).

## How to read a number

The two event lanes were measured with their own pinned detectors, not with the eyes model
of record. The seizure exam ran on eyes `55c2c242` and the fall exams on the stock
person detector `235e8209`; both digests are recorded in full on the seizure lane page, and
the stock detector is recorded in full on the fall and joints pages as well. Both swaps to
the newer eyes were measured and both were rejected. The seizure swap is arm v3 on the
seizure lane page, at testsplit window AUC 0.48426 against the model of record's 0.69841.
The fall swap is the first-lever probe of 2026-08-27 recorded in
`research/event-verifier-v1-record.md`: the fall v1 verifier `080b5d86`, re-tested on
re-extractions in which only the detector changed, fell from window AUC 0.821 to 0.675 on
multicam and from 0.957 to 0.822 on UR Fall, and it is the only fall-side detector swap on
record, which is why the fall arm table on that lane page holds none. Each event lane
therefore stays pinned to the detector it was trained and tested
against. The demo may drive every lane off the eyes model of record visually, but the two
event AUC values above are claims only under their pinned detectors. Repointing a lane at
a different detector without re-running that lane's exam is the mistake this paragraph
exists to prevent. The direct test of that repointing was run on 2026-09-07: both
verifiers were refitted on track trees extracted with the eyes and joints models of
record, and neither refit was promoted. The fall refit, ONNX `baa93141`, scored
all-cameras window AUC 0.55615 against the bar of 0.78474, and the seizure refit, ONNX
`978a1c31`, missed its focal bar outright
while catching none of E2's four convulsive events at the declared threshold, where the
model of record re-sliced onto the same trees catches three and final B four; both are
recorded in full on their lane pages, with the frozen heads beside them.

The `val_auc_best` field inside the two `report.json` files, 0.99589 for fall and 0.77152
for seizure, is the train-side validation number, not the exam number. Quote the exam
values from the index instead.

Detector accuracy and pose accuracy are not the same measurement even within one lane. The
joints value above is a keypoint AP measured with ground-truth boxes, which isolates the
pose model; the deployed path, where the pose model consumes detector boxes, is a different
and lower number, and the lane page records both with the protocol beside each.

Latency is measured on the host that produced it and is quoted with that host, the
execution provider and the input size. No figure is quoted for hardware the stack has not
been measured on.

## The rules that bind every page

- A model is compared only on an exam it never trained on. A dataset that trains loses its
  test role permanently.
- Variants are selected on the train-side validation split. Selection never touches an exam
  set.
- Each version gets one pre-declared look at the test set. A look is spent whether or not
  the result is favourable, and an unspent look does not carry over.
- An exam is never edited to flatter a model. When a number reads badly, the model is the
  thing that is bad.
- Every accuracy number is quoted with the checkpoint hash that produced it.
- Every checkpoint carries a provenance record giving its sha256, its training data and its
  backbone pretrain data. The provenance section of each lane page is that record.
- Every re-identification number is provisional and is quoted with the embedder sha256
  beside it.
- The standing rules that govern what may be trained on, what may be measured, and what may
  be claimed are in `POLICY.md`. The reproduction commands for each lane are in
  `TRAINING.md`, the dataset registry summary is in `DATASETS.md`, and the full run by run
  evidence, winners and losers alike, is in `research/`, beginning with
  `research/stats-compendium.md`.
