# Eyes lane: person detector

The model of record for the eyes lane, the measurements that earned it the slot, the arms
that lost, and the operating points the rest of the stack is configured against. Every
number on this page carries the hash of the checkpoint or graph that produced it, because
one accuracy value can identify two different checkpoints.

## Identity and provenance

| field | value |
|---|---|
| Model of record | RT-DETRv4-S, no-teacher recipe, seed 0, best checkpoint at epoch 47 |
| Checkpoint sha256 | `345c30a539c2b6bb665302825f418322284b98f6cbcf55d58d84ffeb75b16e7e` |
| ONNX sha256 | `4026204967aa01a7680f7ce0c6169463d654a519f8093383a10537321f56fd49` |
| Shipped as | `final_models/eyes/model.onnx`, digest in `final_models/eyes/model.onnx.sha256` |
| Parameters | 10.2M |
| Graph | ONNX opset 17, two inputs (`images` (N,3,640,640) float32, `orig_target_sizes` (N,2) int64), three outputs (`labels`, `boxes` in original pixels, `scores`), 300 queries, postprocessor inside the graph |
| Backend | `src/vividly/models/rtdetr_backend.py`, factory key `rtdetr_onnx`, input 640x640, plain resize with aspect ratio not preserved, scale to 0 to 1, no mean and standard deviation normalisation, one image per session call |
| Weight class | prototype |
| Shippable | counsel |

Training data. The person_v2 corpus derived from the NVIDIA PhysicalAI-SmartSpaces
MTMC_Tracking_2025 slice (registry entry `physicalai_smartspaces`, bucket TRAIN_OK,
CC BY 4.0): train 55,200 images and 828,230 boxes; dev-val 1,800 images and 33,755 boxes,
with Warehouse_005 held out by scene so no scene appears on both sides. One category,
`id: 0`, person, zero `iscrowd`, 30 cameras by 60 frames on the dev-val side. The dev-val
annotation file has sha256
`bc9bedadbcf33cc7d06e5171cd08f091d6a8f1f974ba2e3e1083a34195dd89dc`.

Recipe. 48 epochs, batch 32, automatic mixed precision, seed 0, about 13.0 hours on an
RTX 5090 pod. The augmentation policy engages at epoch 4, ramps down at 28 and turns off
at 40; the best epoch is 47 in every seed of both recipes. The distillation teacher block
is absent from the configuration rather than disabled, so no teacher weights are ever
downloaded and no additional terms are ever accepted.

Backbone pretrain data and initialisation. The initialisation weight is `rtv4_s_coco.pth`,
sha256 `238a3f6537bf3b75b55e73f91f9d4cec8d21259b4908b3f21896f3e038b5a3ee`, read from the
checkpoint manifest of the pod backup the training run was staged from.
Its lineage is ImageNet through the PPHGNetV2_B0_stage1 backbone, then COCO train2017
through the RT-DETRv4 recipe, with no Objects365 stage: the shipped training log covers
118,272 images per epoch, which is COCO train2017 with the final partial batch dropped
(118,287 images, 3,696 full batches of 32) and nowhere near the scale of an Objects365
stage. The stage-1 best checkpoint of this run is recorded as `365b6925`, eight hex
characters only. At initialisation the only unmatched keys were the class heads
(`denoising_class_embed`, `enc_score_head`, `dec_score_head.0-2`), which is the expected
shape of a COCO to person-only transfer; everything else loaded from the initialisation
weight.

Shippability. RT-DETRv4 initialises from COCO, and the released RT-DETRv4-S weights this
run initialises from were themselves trained with a DINOv3 teacher, which is recorded as
an open counsel item in `third_party/PINS.yaml`. Whether that inheritance matters is a
question for counsel, not one this record resolves.

Provenance record. The checkpoint registry `configs/checkpoints/registry.yaml` records the
five baseline checkpoints downloaded on 2026-08-07 and carries no entry for `345c30a5`, so
this section is that checkpoint's provenance record: it carries the checkpoint sha256, the
ONNX sha256, the training corpus and its split, the initialisation weight and its lineage,
and the counsel position, which is what the checkpoint provenance rule in `POLICY.md`
requires of every checkpoint.

## Before and after

| stage | checkpoint | dev-val AP 0.50:0.95 |
|---|---|---|
| COCO base, zero shot, no fine-tune | `a089c9eb` | 0.584 |
| DEIM-D-FINE-S fine-tune, three-seed mean | `5185d500`, `e98de401`, `32a783a3` | 0.8334, spread 0.0012 |
| Model of record, RT-DETRv4-S seed 0 | `345c30a5` | 0.8356 |

The zero-shot row is the untouched COCO base evaluated on the same split, the same
transforms and the same evaluator, with the 80-class heads intact, class remapping off and
the exponential moving average off. It reads AP small 0.003 and AR small 0.035: the base
model is effectively blind to the small and distant people this domain is made of. The DEIM
fine-tune's gain over it is 0.249 AP, roughly 200 times the seed spread; the model of
record's gain over the same row is 0.2516.

## Arms measured

Every row is one look at the dev-val selection split. No exam was consulted by anything in
this table.

Two pre-registered bars govern this lane and they are different quantities. The promotion
bar is 0.8353: the retained fallback's best seed `e98de401` at 0.8341 plus the 0.0012
noise floor, which is the full spread of the three DEIM seeds, 0.8329 to 0.8341. That is
the bar the model of record had to clear to take the slot, and it cleared it by 0.0003 at
0.8356. It is the value the index table in `final_models/README.md` carries for this lane,
and the value the dashed reference line marks in both figures listed below. The
displacement bar is 0.8368: the model of record's own 0.8356 plus max(0.0012, 0.00075),
where 0.00075 is the two-seed spread of the promoted recipe. That is the bar a later arm
has to clear to take the slot away, and it is the bar the weight soup and the teacher arm
were measured against.

| arm | checkpoint | dev-val AP | outcome |
|---|---|---|---|
| B1 RT-DETRv4-S seed 0 | `345c30a5` | 0.8356 | promoted, model of record |
| B1 RT-DETRv4-S seed 1 | `17362dca` | 0.8348 | recorded, seed replication |
| B2 RT-DETRv4-S with a frozen DINOv2 ViT-B/14 teacher | `9ecb4627` | 0.8345 | not promoted, bar 0.8368 |
| DEIM-D-FINE-S seed 1 | `e98de401` | 0.8341 | retained as the fallback |
| DEIM-D-FINE-S seed 2 | `32a783a3` | 0.8333 | recorded |
| DEIM-D-FINE-S seed 0 | `5185d500` | 0.8329 | recorded |
| Uniform-2 weight soup over B1 seeds 0 and 1 | `0d80cb4c` | 0.830 | not promoted, bar 0.8368 |
| COCO base, zero shot | `a089c9eb` | 0.584 | baseline, not a candidate |

Full digests where they exist: B1 seed 0
`345c30a539c2b6bb665302825f418322284b98f6cbcf55d58d84ffeb75b16e7e`; DEIM seed 1
`e98de401670d34d61d03100974eb72449607b319611ff454e8a865cd07b81147`; the B2 teacher arm
`9ecb46272c838cf9032d53db73e491dc82de7a1d9c9a6e098b517d106b5382a3`; the soup
`0d80cb4c860bd737ee098291cce5bfdcbf53fe9ccac82c47443d59cfd169cf43`; the COCO base, which
is the released `deim_dfine_s_coco.pth`,
`a089c9eb798f8a0322e05f757fe8e8188eee34aa578788a35ec253bcec8a6b38`, read from the
checkpoint manifest of the pod backup. B1 seed 1, DEIM seed 0 and DEIM seed 2 carry eight
hex characters only.

### Why each rejected arm was rejected

**Uniform-2 weight soup, `0d80cb4c`, 0.830.** The pre-declared bar was the best single seed
plus the larger of the measured noise floor and the observed seed spread: 0.8356 plus
max(0.0012, 0.00075), that is 0.8368. The soup scored about 0.007 below the bar and below
both of its own ingredients, and it was worse in every size bucket than seed 1 alone (small
0.118 against 0.124, medium 0.714 against 0.721, large 0.928 against 0.930). Weight
averaging pays when the ingredients sit in one loss basin; these two seeds are full
retrainings from a generic COCO initialisation whose data order and augmentation draws
differ from the first step, so they land in nearby but different basins and the midpoint
sits slightly off both. Greedy souping was not run and would change nothing: greedy starts
from the best single ingredient and keeps only additions that help, and the only available
addition measured minus 0.005.

**Frozen DINOv2 ViT-B/14 teacher, `9ecb4627`, 0.8345.** The gated distillation arm ran to
completion over 16.8 hours against the same 0.8368 bar and missed it. Teacher repository
pin `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, teacher weights sha256
`0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`, load-clean assertion
passed, peak video memory 23,758 MiB against a 30,500 MiB gate. Three independent full
trainings, two plain and one with a foundation teacher, land within 0.0011 of one another
and the teacher run sits below both plain seeds. The conclusion of record is that DINOv2-B
distillation adds nothing on this training set, that 0.8356 is the data-era ceiling for
this corpus, and that the next real lever for this lane is data, not architecture or
teachers.

**Horizontal-flip test-time augmentation with weighted box fusion.** A pre-declared
protocol row, not a checkpoint and not promotable. Run at the declared defaults
(`iou_thr` 0.55, `skip_box_thr` 0.05, `conf_type` average, all weights 1.0, identity score
calibration per view) it cost 0.0423 AP for `345c30a5` (0.7933 against 0.8356) and 0.0427
for `e98de401` (0.7914 against 0.8341). The single flipped view on its own is marginally
better than identity (plus 0.0005 and plus 0.0001); it is the fusion that destroys the
number, because average confidence rescales each cluster by the contributing weight over
the total weight, so a box seen by only one of two equally weighted views keeps half its
score, and average precision is a score-ranking metric. The same rescaling means a fixed
score cut on fused scores is not the same operating point as that cut on a single view.
The row is recorded and closed.

**B1 seed 2.** Never run. The arm was closed on 2026-09-06; there is no checkpoint and no
hash.

## Operating points

Instrument: `scripts/eval/detector_precision_at_thr.py`, intersection over union 0.5, no
maximum-detections cap, no area ranges, camera depth 2, ground truth as hashed above. All
rows are the identity view on the dev-val split, 1,800 images. The demo platform's default
score cut is 0.4, which is the row to read for what a viewer sees; the replay pipeline and
the track-tree extraction configurations run at 0.3.

| checkpoint | score at least | true positives | false positives | false negatives | precision | recall | F1 | false positives per image |
|---|---|---|---|---|---|---|---|---|
| `345c30a5` | 0.3 | 31951 | 4718 | 1804 | 0.8713 | 0.9466 | 0.9074 | 2.621 |
| `345c30a5` | 0.4 | 31413 | 2302 | 2342 | 0.9317 | 0.9306 | 0.9312 | 1.279 |
| `345c30a5` | 0.5 | 30704 | 1059 | 3051 | 0.9667 | 0.9096 | 0.9373 | 0.588 |
| `e98de401` | 0.3 | 31960 | 4709 | 1795 | 0.8716 | 0.9468 | 0.9076 | 2.616 |
| `e98de401` | 0.4 | 31429 | 2311 | 2326 | 0.9315 | 0.9311 | 0.9313 | 1.284 |
| `e98de401` | 0.5 | 30709 | 1041 | 3046 | 0.9672 | 0.9098 | 0.9376 | 0.578 |

At the 0.4 point, 93.2 percent of drawn boxes are real, 93.1 percent of people are drawn,
and an average frame carries 1.28 boxes that correspond to nothing. The two detectors are
indistinguishable at this operating point, which is the same conclusion the average
precision comparison reached by a margin of 0.0003. Moving the cut to 0.3 trades precision
for recall, 0.8713 against 0.9466, and more than doubles the false positives per image, to
2.621.

The per-camera spread is what average precision hides. At 0.4 for `345c30a5`, precision
runs from 0.8906 on Camera_0022 (3.717 false positives per image over 1,976 ground-truth
boxes) to 0.9873 on Camera_0001 (0.100 false positives per image over 467 boxes). The worst
three cameras by false positives per image are 0022 at 3.717, 0004 at 3.100 and 0026 at
3.000; the worst three by recall are 0004 at 0.8584, 0003 at 0.8623 and 0010 at 0.8681.
Every one of them is a crowded view. The overall figure of 1.28 false positives per image
covers a 55-fold spread across cameras, 0.067 on Camera_0007 to 3.717 on Camera_0022.

Full dev-val detection quality for `345c30a5`: AP 0.50:0.95 0.8356 (exact
0.8355529763877473), AP50 0.958, AP75 0.883, AP small 0.128, AP medium 0.722, AP large
0.931, AR at 100 detections 0.860 with AR50 0.977 and AR75 0.907.

Export fidelity. The exported ONNX scores 0.8355776016 on the full dev-val split against
the trainer's 0.8355529764, a difference of 0.0000246, which is the accumulated 32-bit
floating point drift over the split. Per image, every ONNX box has a twin in the
checkpoint's detection dump at intersection over union at least 0.9964 and the median box
disagreement is 0.006 pixels. The detection dump itself reproduces the trainer to every
printed digit, a difference of exactly 0.0.

## Latency

Apple M5 Pro, arm64, macOS 26.6.2, Python 3.11.15, numpy 2.4.6, onnxruntime 1.28.0,
CPU execution provider, default thread count, batch 1, input 640x640. The CoreML provider
crashes on this graph and is never configured.

| measurement | mean | p50 | p95 |
|---|---|---|---|
| `session.run` in isolation, warm input tensor | 55.04 ms | 54.77 ms | 56.72 ms |
| `session.run` with a decode interleaved between calls | 63.5 ms | 64.8 ms | 70.0 ms |
| end to end per frame, including decode and resize | 76.3 ms | | |

The end-to-end figure is the one to quote for throughput. It is 167.5 seconds over a
2,196-frame pass on the same host. The isolation figure excludes preprocessing and reuses
one already-warm input tensor. There is no measured figure for any embedded accelerator,
so none is quoted.

## Real-footage transfer

One authorised internal-only look at real clinical-adjacent footage exists and is recorded
in `research/eyes-instrument-grading.md`. It is marked internal only in that document, its
numbers are not restated here, and the corpus it uses trains nothing and gates nothing. The
reading of record is that a detector trained entirely on synthetic warehouse renders
transfers poorly to a real operating room, that recall holds up better than precision, and
that the number is the field any future real-footage claim must be measured against.

## Figures

- `final_models/eyes/validation_curves.png`: validation average precision per epoch, every
  run of this lane on one axis, the model of record emphasised. Source CSVs:
  `final_models/eyes/curves/detector_b1_rtv4s_seed0.csv`,
  `detector_deim_seed0.csv`, `detector_deim_seed1.csv`, `detector_deim_seed2.csv`.
  The `val_ap` column of these four files is an average precision.
- `final_models/eyes/champion_vs_baseline.png`: the model of record against the retained
  fallback and the promotion bar. The dashed line legended as the pre-registered bar at
  0.8353 in both figures is that promotion bar, not the 0.8368 displacement bar.

No per-epoch CSV exists for B1 seed 1 `17362dca`, for the weight soup `0d80cb4c` or for the
teacher arm `9ecb4627`, so those three rows of the arms table have no plottable per-epoch
backing. They are single end-of-run values.

## Provenance gaps

Recorded so they can be closed rather than rediscovered.

- The stage-1 best checkpoint `365b6925`, B1 seed 1 `17362dca`, DEIM seed 0 `5185d500` and
  DEIM seed 2 `32a783a3` are recorded with eight hex characters only. No run report, export
  sidecar or checkpoint manifest available in this repository carries the full digest of
  any of the four, so each should be captured from the pod at the next opportunity.
- The dev-val frame trees on the development machine and on the training pod are different
  JPEG encodes of the same split. Any pixel-level comparison against a pod-produced
  detection dump must use pod bytes; the full-split parity result above was produced that
  way.

## Detail

- Training record and the arm comparison: `research/eyes-rtdetrv4-training-record.md`.
- Three-seed noise floor and the zero-shot baseline: `research/eyes-noise-floor.md`.
- Seed replication, the weight soup and the teacher arm verdict:
  `research/eyes-seed-spread-and-soup.md` and `research/eyes-dinov2-teacher-arm.md`.
- Precision instrument, export parity and latency: `research/eyes-instrument-grading.md`,
  with the pre-declared instrument protocol in `research/eyes-instrument-protocol.md`.
- Repository pins, licences and the counsel items behind the initialisation weight:
  `third_party/PINS.yaml`.
- Measurement rules that govern every number above: `POLICY.md`.
