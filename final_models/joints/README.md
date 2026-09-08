# Joints lane: pose estimator

The model of record for the joints lane, the measurements that earned it the slot, the arms
that lost, and the two different protocols under which this lane is measured. Every number
on this page carries the hash of the checkpoint or graph that produced it.

## Identity and provenance

| field | value |
|---|---|
| Model of record | RTMPose-m person_v2 fine-tune, seed 0, best checkpoint at epoch 38 |
| Checkpoint sha256 | `a0a035fe65909980447560a70ae2f8d13400c1d12490a929e7cf3c5a7b26f102` |
| Checkpoint size | 112,155,925 bytes |
| ONNX sha256 | `6ba9628506a8899d3a3e940dcc8af5d30342921f09d2605afa505b57453d3527` |
| ONNX size | 54,369,587 bytes, 54.4 MB |
| Shipped as | `final_models/joints/model.onnx`, digest in `final_models/joints/model.onnx.sha256` |
| Architecture | RTMPose-m, SimCC head with `simcc_split_ratio` 2.0, top-down 256x192, COCO-17 keypoints |
| Graph | opset 17, input `input` (N,3,256,192) normalised crop batch with a dynamic batch axis, outputs `simcc_x` (batch,17,384) and `simcc_y` (batch,17,512) |
| Decode | outside the graph: argmax over the SimCC bins at split ratio 2.0, then the affine back-projection. This is the rtmlib contract, which is why the pose backend drives the file unmodified |
| Backend | `src/vividly/models/rtmlib_backend.py`, factory key `rtmlib`, input 256x192 |
| Export parity | worst absolute difference between ONNX and torch 2.876e-06 on `simcc_x` and 5.230e-06 on `simcc_y`, tolerance 1e-4, reported by the exporter at export time |
| Weight class | prototype |
| Shippable | counsel |

Training data. COCO 2017 person keypoints only (registry entry `coco`, bucket TRAIN_OK):
118,287 train images and 262,465 person annotations, counts re-verified at launch. No
rendered imagery, no prototype overhead data and no event dataset enter this fine-tune.

Recipe. `third_party/configs/rtmpose_m_person_v2_ft.py`: 60 epochs, learning rate 1e-4,
warmup 500, batch 256, stage 2 with augmentation off from epoch 55. The recipe sets no
randomness key; the seed is supplied at launch through `--cfg-options`, and this run was
`randomness.seed` 0. The full configuration text is embedded in the checkpoint's own
metadata. The shipped state dictionary is the exponential-moving-average weight set, which
is what the best-epoch selection ran on and what the ONNX export used; the raw online
weights are kept beside it.

Initialisation and backbone pretrain data. The initialisation weight is
`rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth`, sha256
`e48f03d0cfe1285ee8b6d3457ac3ce33a4594a92b080053ab1ec4a7e300975f2`. Load confirmed in the
training log: 191 pretrained-initialisation stamps, no missing keys and no unexpected keys.
The backbone pretrain data is therefore `pt-body7`, inherited through that initialisation:
body7 is COCO plus AI Challenger plus CrowdPose plus MPII plus sub-JHMDB plus Halpe plus
PoseTrack18. MPII is free to researchers from educational or research institutes for
non-commercial purposes; CrowdPose is CC BY 4.0 for academic and non-commercial research
purposes; AI Challenger and PoseTrack carry their own custom terms. Fine-tuning a
commercial product from body7 inherits that mixture. The non-commercial wording is flagged
for counsel here, not absorbed.

Shippability. No published RTMPose checkpoint is free of the AI Challenger corpus: all of
them carry an AI Challenger and COCO backbone lineage, and this checkpoint's lineage is
`pt-body7`, which is further from AI Challenger free than the alternatives, not closer.
Fine-tuning on COCO-only data does not launder the initialisation. The clean-room fallback
is priced in the pose stack research: an ImageNet-1k-only backbone initialisation
(`cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth`) followed by COCO and own top-view data,
at some accuracy cost. Shippability is a question for counsel.

Naming. The generation is not christened. Naming a generation is an owner decision and has
not been made, so every artefact is named after its recipe.

Note on the filename token. The `a0a035fe` in this checkpoint's filename is the leading
eight hex characters of its sha256. That is not true of upstream RTMPose artefacts: the
`e48f03d0` token is a digest prefix of the release `.pth`, while the cached ONNX of the
same model hashes to
`5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c`. Only a full sha256
verifies identity.

## Before and after

Trainer-side COCO val2017, ground-truth box protocol, mmpose evaluation loop with flip test
on, torch weights on GPU. Identical protocol on both rows.

| stage | checkpoint | keypoint AP | delta |
|---|---|---|---|
| body7 initialisation | `e48f03d0` | 0.764499 | baseline |
| Model of record, seed 0 epoch 38 | `a0a035fe` | 0.774115 | plus 0.009616 |

Seed floor, three full trainings of the identical configuration differing only in the seed
and in a throughput-only worker count:

| run | seed | best keypoint AP | best epoch | checkpoint | final epoch 60 |
|---|---|---|---|---|---|
| P0 | 0 | 0.774115 | 38 | `a0a035fe` | 0.773865 |
| P1 | 1 | 0.774000 | 30 | `8d990435` | 0.773100 |
| P2 | 2 | 0.774057 | 32 | `5bcca5bd` | 0.773165 |

Noise floor 0.77406 with a standard deviation of 0.00006 and a spread of 0.000115. The
recipe is essentially deterministic on this evaluation; the detector lane's spread on its
own selection split is 0.0012 by comparison. Every seed clears the body7 gate by 0.0095 to
0.0096. The bar for any future rival recipe on this lane is the model of record plus two
standard deviations, approximately 0.7742.

## The deployed path is a different measurement

The artefact the event lanes consume is the ONNX, driven through the pose backend on
onnxruntime CPU, with boxes from a detector and no flip test-time augmentation. That is not
the protocol the 0.774115 was measured under, and the two must never be subtracted from
each other. The comparisons this table supports are the ones inside it: same subset, same
detector, same decode, one variable at a time.

Frozen subset `configs/eval/subset_coco_val_person500.json`, fingerprint `2db41bbad264b7dc`,
500 COCO val2017 images. Boxes for every row come from the stock rtmdet-m person release,
token `235e8209`, at 640 with a score threshold of 0.05, cached ONNX sha256
`4f4d7e07350b1753299111d1ae500fd64447a5b0e38e4bacbefab6573c742d30`. 15,571 detections in
every row, so the detector is a control and not a variable; detection AP is 0.6823 in every
row to four decimal places. Intervals are 1000-resample per-image bootstraps at seed 0.

| row | pose ONNX | colour fed to pose | flip | keypoint AP | 95 percent interval | AP50 | AR |
|---|---|---|---|---|---|---|---|
| 1, body7 control | `5c0a4bf6` | BGR, the deployed path | off | 0.7285 | 0.7038 to 0.7532 | 0.8940 | 0.7961 |
| 2, model of record | `6ba96285` | BGR, the deployed path | off | 0.7196 | 0.6939 to 0.7440 | 0.8846 | 0.7866 |
| 3, model of record with flip | `6ba96285` | BGR, the deployed path | on | 0.7262 | 0.7015 to 0.7508 | 0.8916 | 0.7918 |
| diagnostic 1, body7 | `5c0a4bf6` | RGB | off | 0.7393 | 0.7139 to 0.7636 | | 0.8046 |
| diagnostic 2, model of record | `6ba96285` | RGB | off | 0.7473 | 0.7242 to 0.7715 | | 0.8107 |

Paired deltas on the same resample draws, so the shared-image and shared-detector
correlation cancels rather than inflating the interval:

| delta | point | 95 percent interval | probability the delta is positive |
|---|---|---|---|
| row 1 to row 2, body7 to the model of record, both without flip | minus 0.0089 | minus 0.0148 to minus 0.0037 | 0.001 |
| row 2 to row 3, flip test-time augmentation | plus 0.0066 | plus 0.0030 to plus 0.0109 | 1.000 |
| model of record, BGR to RGB | plus 0.0277 | plus 0.0215 to plus 0.0342 | 1.000 |
| body7, BGR to RGB | plus 0.0108 | plus 0.0062 to plus 0.0154 | 1.000 |
| RGB without flip, body7 to the model of record | plus 0.0081 | plus 0.0023 to plus 0.0137 | 0.995 |

Row 1 is the gate and it passed: it reproduces a leg with a recorded answer of detection AP
0.682 and keypoint AP 0.729, measuring 0.6823 and 0.7285, so rows 2 and 3 are
interpretable. The gate is declared in the configuration and enforced by the runner.

Three things follow, and they must be stated together.

**0.7196 against 0.774115 is not a regression.** It is the cost of detector boxes at a
score threshold of 0.05 plus ONNX and pose-backend preprocessing, and the body7 row pays a
similar toll (0.7285 here against 0.764499 trainer-side). Neither number is a claim about
the other.

**The inversion on the deployed path was a preprocessing defect, not a property of the
model.** The pose backend never converts colour: it affine-crops the OpenCV image, which is
in blue-green-red order, normalises with the ImageNet statistics and feeds that to the
graph. The training preprocessor converts blue-green-red to red-green-blue, and neither
this graph nor the upstream software-development-kit graph carries the swap, because the
export wraps the feature extractor and head, which sit after the preprocessor. Feeding the
correct colour order moves the model of record to 0.7473 and restores the trainer-side
ordering, the model of record beating body7 by 0.0081. The defect costs this model 2.6
times what it costs body7, which is what a harder fine-tune on one colour convention
predicts.

**The colour-order correction reaches the live path only through a full track-tree
re-extraction, never as a backend hot fix.** Every event verifier in this repository was
trained on features extracted in the current colour order, so changing the backend
underneath a trained verifier changes its input distribution without retraining it. The
correction therefore travels as a declared delta of a track-tree generation, alongside the
detector swap, and the verifiers are refitted on the new trees. The live and demo pipeline
configurations are fenced against setting the colour-order key directly by
`tests/test_pipeline_color_order.py`.

## Arms measured and rejected

| arm | artefact | outcome |
|---|---|---|
| Seed replication, three seeds | `a0a035fe`, `8d990435`, `5bcca5bd` | floor measured, seed 0 promoted |
| Uniform three-seed weight soup | sha256 `46e6a8e8` | unscored and parked; it is not the model of record and has no accuracy number |
| Flip test-time augmentation on the deployed path | `6ba96285` with flip | measured, not adopted |

The weight soup is parked rather than rejected: scoring it needs a training-side evaluation
environment that the harness environment does not satisfy. Its bar, if it is ever scored,
is 0.7742.

Flip test-time augmentation buys 0.0066 keypoint AP on the deployed path and costs 1.63
times the pose stage, 20.21 to 32.86 milliseconds per person crop, because only the network
forward is duplicated while the affine crop, the normalisation and the SimCC decode happen
once. The colour-order correction buys 0.0277 for roughly one colour conversion per image.
A colour-corrected run without flip beats a flipped run in the current colour order by
0.0211 while being cheaper. Any decision about shipping flip test-time augmentation is
taken after the colour order is corrected, not before.

## Operating points

This lane has no score threshold of its own: the pose stage runs on whatever boxes the
detector hands it and emits a keypoint and a confidence per joint. The points that matter
downstream are set elsewhere and are recorded here so the chain is legible.

- Boxes for the exam rows: stock rtmdet-m person `235e8209` at 640, score threshold 0.05.
- Boxes on the live and extraction path: the eyes lane model of record at score threshold
  0.3.
- Keypoint confidence gate used by the track layer and by the posture angle:
  `kp_conf_min` 0.3.
- Colour order on the deployed path: blue-green-red, since no pipeline configuration sets
  the colour-order key; the newest track-tree generation extracts in red-green-blue.

Latency, Apple M5 Pro, macOS 26.6.2, onnxruntime 1.28.0, CPU execution provider, default
thread count, batch 1 per person crop. The pose stage is billed per image as well as per
crop; this subset averages 31.1 person crops per image at a score threshold of 0.05.

| mode | p50 per image | p95 per image | per crop |
|---|---|---|---|
| no flip | 344.8 ms | 1964.0 ms | 20.21 ms |
| flip test-time augmentation | 488.9 ms | 3432.4 ms | 32.86 ms |

These were measured on a shared machine: the byte-identical detector stage spread from 166
to 213 milliseconds p50 across the three rows, which is the honest error bar on every cell.
There is no measured figure for any embedded accelerator, so none is quoted.

## Figures

- `final_models/joints/validation_curves.png`: validation keypoint average precision per
  epoch, all three seeds on one axis, the model of record emphasised, with the body7 gate
  line at 0.764499. Source CSVs: `final_models/joints/curves/pose_rtmpose_m_p0_seed0.csv`,
  `pose_rtmpose_m_p1_seed1.csv`, `pose_rtmpose_m_p2_seed2.csv`. The `val_ap` column of
  these three files is an average precision, and it is empty on odd epochs because
  validation ran every second epoch.
- `final_models/joints/champion_vs_baseline.png`: the model of record against the body7
  initialisation on the trainer-side ground-truth-box protocol only. The axis carries the
  seed 0 validation keypoint AP curve with the body7 gate line at 0.764499, which is the one
  protocol on which those two artefacts are directly comparable. Source CSV:
  `final_models/joints/curves/pose_rtmpose_m_p0_seed0.csv`. The deployed-path rows of the
  table above (body7 control 0.7285, model of record 0.7196, model of record with flip
  0.7262) are a different measurement and are deliberately absent from this figure.

## Full digests

Every short form used in the tables above resolves here, so no number on this page is
quoted against a hash that exists only as a prefix.

| artefact | sha256 |
|---|---|
| Model of record, seed 0 epoch 38 | `a0a035fe65909980447560a70ae2f8d13400c1d12490a929e7cf3c5a7b26f102` |
| Model of record, exported graph | `6ba9628506a8899d3a3e940dcc8af5d30342921f09d2605afa505b57453d3527` |
| Seed 1, best epoch 30 | `8d99043597076a672e7f4bc1ce886111d06a62de287d4801e489080561fa40ae` |
| Seed 2, best epoch 32 | `5bcca5bd3a33c18fbba70db89a6d9d8f9c2703679b3c820dc267090546cf5f07` |
| Parked three-seed weight soup | `46e6a8e87ea0f8e0b6a5ff370ba747f29cc669f40a7f645b4d05c43d16d85966` |
| body7 initialisation checkpoint | `e48f03d0cfe1285ee8b6d3457ac3ce33a4594a92b080053ab1ec4a7e300975f2` |
| body7 cached graph, the exam control | `5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c` |
| Stock rtmdet-m person release checkpoint, token `235e8209` | `235e820939cb2ff33c505441e71f7e9532958c281636a963c6829d100867aed9` |
| Stock rtmdet-m person cached graph, the box source for every exam row | `4f4d7e07350b1753299111d1ae500fd64447a5b0e38e4bacbefab6573c742d30` |

The seed 1 and seed 2 digests were read from the pod backup checkpoints and confirmed
independently against the ingredient list recorded in the weight soup sidecar, which also
carries the soup's own output digest. The body7 baseline of 0.764499 is quoted in the
research set without a hash beside it; it resolves to the initialisation checkpoint digest
in the table above, and that resolution is recorded here so the number is quotable.

## Detail

- Seed floor and the selection of the model of record:
  `research/pose-seed-floor-and-champion.md`.
- The seed 0 training record: `research/pose-p0-training-record.md`.
- Weight soup: `research/pose-weight-soup.md`.
- Deployed-path ladder, the colour-order finding and the latency table:
  `research/pose-deployed-path-cpu-proof.md`.
- Keypoint confidence behaviour across track trees: `research/pose-noise-profile.md`.
- Backbone licence position and the clean-room fallback: `research/pose-stack.md`.
- Measurement rules that govern every number above: `POLICY.md`.
