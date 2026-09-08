# Re-identification lane: session-scoped appearance embedder

The model of record is the Phase A run of 2026-09-07: OSNet-x1_0 trained on MEVID from an
ImageNet-only initialisation, checkpoint sha256 beginning `66623239`, exported to
`model.onnx` with sha256 beginning `60164a99`; both digests are given in full below. Every
measured number on this page comes from the run's own artefacts, `eval_summary.json` and
`curves/reid_mevid_a1.csv` beside this page, and is quoted with the embedder hash. The
supporting figures are drawn from named sources instead: the corpus totals from the registry
entry `configs/datasets/mevid.yaml`, the split counts from what the loader reported for this
run, the initialisation digests and byte counts from the export sidecar that
`scripts/reid/export_osnet_onnx.py` wrote beside that graph, the trained checkpoint's byte
count from the checkpoint on the training host, and the two collapsed runs from that host's
logs. It is the model of record because it is the only measured embedder for this domain,
not because it clears the external floor: it does not, and the result section says so. The
weights are prototype class and the numbers are internal measurements, never clinical
performance.

## Scope, which applies to everything below

Re-identification here is a session-scoped appearance embedding and nothing more. It
computes a descriptor from person crops, proposes a leave-and-return match to a person, and
never binds one itself: a human confirms every proposed match on the dashboard. There is no
pose, gait, body-shape or face signal in any part of it, and no face detection, embedding,
matching or face-localisation branch exists anywhere in the repository. Every
re-identification number is provisional and is quoted with the embedder sha256 beside it;
a number without its hash is not quotable.

## Identity and provenance

| field | value |
|---|---|
| Model of record | OSNet-x1_0 MEVID Phase A, run `reid_mevid_a1`, the epoch-30 weights of a 30-epoch run trained 2026-09-07 from 16:57Z to 17:15Z on one RTX 5090 |
| Architecture | OSNet-x1_0, 512-dimensional embedding head, classifier dropped on export, 2.2M parameters and 0.98 GFLOPs at 256x128 |
| Initialisation checkpoint | `osnet_x1_0_imagenet.pth`, sha256 `fe2d63f9157c28a4a8d8ca29bec12d5b2988ac0346d712025789ea9174968e79`, 10,910,553 bytes |
| Initialisation ONNX | sha256 `73e76e2d0003f6b31eb1d47ecdbe991faf1327704599fc70c449079e3c80e6cf`, 8,729,450 bytes |
| Trained checkpoint sha256 | `66623239b93fc7a807b986b07fd2fcb2047ccf1e77d1988d96ba2575b8390bb7`, `model-best.pth.tar`, 36,233,078 bytes, epoch 30, best test mAP of the run |
| Trained ONNX sha256 | `60164a9945445efbc5ea3d2abafb3fe0d78f1bcd2b4616ffd47634d87ef808a7`, `model.onnx` beside this page, 8,729,490 bytes, sidecar `model.onnx.sha256` |
| Trained export | opset 17, input `images` at 1x3x256x128 float32 with batch as the only free axis, output `embeddings` at batch by 512. Parity gate: torch against onnxruntime cosine similarity 1.000000 on random input, threshold above 0.999, passed on the pod at export and again on a development Mac against `model.onnx` after the copy, with a maximum absolute difference of 2.1e-6. Exported 2026-09-07 17:21Z from the checkpoint above by the trainer's own export function; the in-process export at the end of training had failed on a missing package, recorded in the result section |
| Library | torchreid, MIT, pin `f8cd150fdf77e8d9e1ed143b7f308c2c609ded50`, recorded in `third_party/PINS.yaml` |
| Initialisation source | `https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY`, which is the entry for `osnet_x1_0` in the library's own `pretrained_urls` table, read from the installed canonical distribution |
| Initialisation fetched | 2026-08-08, deliberately, from the canonical distribution; sha256 recorded on the same run |
| Initialisation export | opset 13, input `images` at 1x3x256x128 float32 with batch as the only free axis, output `embeddings` at batch by 512, written 2026-08-09 by `scripts/reid/export_osnet_onnx.py` with a sidecar carrying both digests |
| Weight class | prototype |
| Shippable | counsel |

Training data. MEVID, registry entry `configs/datasets/mevid.yaml`, bucket
TRAIN_PROTOTYPE, status available, the three archive digests recorded in the entry and in
`DATASETS.md`. Corpus totals: 158 identities, 598 outfits, 8,092 tracklets, 33 camera
views, 17 locations. Train split 104 identities, 485 outfits, 6,338 tracklets. Test split
54 identities, 113 outfits, 1,754 tracklets. The loader reported the splits it built for
the run: 104 identities in 6,338 training tracklets over 22 cameras; 52 query identities
in 316 tracklets over 9 cameras and 54 gallery identities in 1,438 tracklets over 11. The
test split was held out entirely and touched only by the three evaluation protocols. The
actor check-in photo package is barred by name because it contains faces, is absent from
the entry's download list and is never fetched.

Backbone pretrain data. ImageNet-1k classification pretrain. This checkpoint is the
pretrain: it is the `_imagenet` variant, not a Market-1501, MSMT17 or Duke-family
re-identification fine-tune. That variant was chosen deliberately. Every torchreid
re-identification fine-tune of this architecture was trained on corpora that sit behind
their own licence questions, so an ImageNet pretrain is the weakest-claim starting point in
the menu: worse absolute numbers, cleanest available provenance. The refusal of the banned
weight sources is enforced at load time and at export time by
`scripts/guards/check_reid_weight_sources.py`, and the refusal message names
`docs/LICENCE_ALLOWLIST.md`.

Shippability. Two questions are open and neither is one this record resolves. First, the
initialisation: the original access terms of the ImageNet corpus carry non-commercial
research clauses, which is the standing question for any commercial weight whose lineage
passes through it. Second, the training corpus: the underlying MEVA video is CC BY 4.0, and
the MEVID annotation and chip layer, which is the part that actually trains an embedder,
ships with no licence statement at all, verified 2026-09-07. Silence is not a grant, so the
conservative reading governs, commercial use is recorded as conflicting rather than
permitted, and every weight this data touches stays in the prototype weight class. Counsel
items in play: CQ-11 on appearance and biometric statutes, CQ-20 on freedom to operate
including the overhead re-identification patent and the waiting-room patient-identification
cluster, and CQ-27 on the publisher position attached to the synthetic corpus used
elsewhere in this repository. Shippability is a question for counsel.

## What was run

The recipe below is the one the run of record used, read back from the `recipe` block of
`eval_summary.json` beside this page. The two collapsed runs that preceded it under the
triplet engine are recorded in the result section.

Trainer `scripts/reid/train_reid_mevid_a1.py`, configuration
`configs/reid/train_mevid_a1.yaml`. Declared settings: `weight_class` prototype, which the
trainer refuses to start without, so a checkpoint can never end up unlabelled; seed
20260907; 30 epochs; `eval_freq` 10, so evaluation runs at epochs 10, 20 and 30 plus the
unconditional final pass; sequence length 8 frames per tracklet with `sample_method`
`random` for training and `evenly` for evaluation, because one sampler otherwise reaches
both and makes every evaluation a fresh random draw; batch of 32 tracklets with
`num_instances` 4 under `RandomIdentitySampler`, eight identities per step; the first five
epochs train only the classifier with the ImageNet trunk frozen in eval mode
(`fixbase_epoch` 5, `open_layers` classifier, torchreid's two-stepped transfer learning);
crops at height 256 and width 128; transforms `random_flip` and `random_erase` only;
optimiser `amsgrad` at learning rate 0.0015 and weight decay 0.0005; `single_step` schedule
with `stepsize` 20 and `gamma` 0.1; loss `softmax`, cross-entropy over the training
identities with label smoothing on and no metric term, which is the library's own OSNet
recipe; tracklet features pooled by mean over the sequence; distance metric euclidean. The
triplet keys (`weight_t`, `weight_x`, `margin`) remain in the configuration and are read
only under `loss: triplet`, the engine the two collapsed runs recorded below used.

Curve columns, seven of them: `epoch`, `loss_t`, `loss_x`, `acc`, `lr`, `test_map`,
`test_rank1`. `loss_t` is written only by the triplet engine and stays empty under the
softmax recipe. The last two are populated only on evaluation epochs, so the accuracy curve
carries three points over 30 epochs while the training columns are per epoch.

Three evaluations are reported side by side over one feature extraction, so the differences
between them are attributable to the protocol rather than to a re-run.

1. **General.** The library's standard protocol, euclidean distance on raw features. This
   is the number that is comparable to published results. A self-check compares it against
   the library's own evaluation on the same tensors.
2. **Camera debias.** The per-camera mean embedding, taken over the pooled query and
   gallery test features grouped by camera, subtracted from every feature, then
   renormalised, then cosine distance. Reported next to a plain normalised cosine baseline
   so the debias effect is separated from the metric change.
3. **Same-clothes approximation.** An in-house definition, not the published same-clothes
   protocol, and labelled as in-house everywhere it is printed. A positive is the same
   person identifier and the same outfit identifier; a gallery tracklet of the same person
   in a different outfit is removed from the ranking as neutral rather than scored as a
   negative, because counting it as a negative would reward an embedder for telling two
   outfits of one person apart, which is the opposite of what a session-scoped
   re-identification is asked to do. Same person on the same camera stays neutral, as in
   the standard protocol.

Export. ONNX opset 17, input `images` of shape (1,3,256,128) with a dynamic batch axis,
output `embeddings`, shipped beside this page as `model.onnx` with its `model.onnx.sha256`
sidecar. The trainer refuses to declare success unless torch and onnxruntime agree to a
cosine similarity above 0.999 on random input. That is the parity gate the exported graph
must clear before any number measured with the torch weights may be attributed to the graph.

## The bar, and what it is not

The bar is external and carries no hash, because no checkpoint hash exists or can exist for
a published external number. It is labelled external wherever it appears.

| value | what it is |
|---|---|
| 39.0 mAP | the published same-clothes floor for this corpus, on a test set of 54 identities and 113 outfits with clothes-change distractors. External, no hash. This is the floor to beat |
| 27.1 mAP | the published general-protocol figure for the same corpus. External, no hash. This is the wrong floor to use for a session-scoped, unchanged-clothing case |
| rank-1 0.70 to 0.85, mAP 0.55 to 0.70, rank-5 recall 0.90 to 0.96 | a stated expectation for Phase A. A projection, not a measurement, and never presented as one |

Two earlier internal re-identification baselines exist in the repository's history and are
void: both were measured on a superseded imaging domain, and the acceptance criteria that
bound on them are void with them. Neither is a comparator for this lane and neither is carried
here.

## The corpus hashes are recorded

The registry entry for the training corpus carries the three archive sha256 values,
recorded on first download on 2026-09-07, and its status is `available`; `DATASETS.md`
carries the same three digests. That precondition for quoting a number from this corpus is
met, and every number below is quoted with the embedder sha256 beginning `60164a99`, which
is the graph beside this page, exported from the checkpoint beginning `66623239`.

### Result: re-identification Phase A on MEVID, measured 2026-09-07

Status: measured. The bar is not met.

Bar: the published same-clothes floor of 39.0 mAP on the 54-identity, 113-outfit test set
with clothes-change distractors. It is an external published value and carries no holder
hash, because none exists; it is labelled external wherever it is quoted. The general
protocol figure of 27.1 mAP is not the floor for this case. Separately, the exported graph
must clear the parity gate: torch against onnxruntime cosine similarity above 0.999 on
random input.
Protocol as declared: the corpus test split, held out entirely, evaluated at epochs 10, 20
and 30 under the three protocols above over one feature extraction; seed 20260907; one
training run of record, and the general-protocol number is the one compared against the
external floor. The evaluation at each epoch uses the fixed span-covering frame draw, so
the three points are comparable to each other.

Evaluation epochs, embedder `60164a99`, from `curves/reid_mevid_a1.csv`. The cross-entropy
and accuracy columns are training-set values averaged over the epoch; the two test columns
are the general protocol.

| epoch | cross-entropy | train accuracy | learning rate | test mAP | test rank-1 |
|---|---|---|---|---|---|
| 10 | 2.4224 | 55.72 | 0.0015 | 0.137423 | 0.389241 |
| 20 | 1.9821 | 68.39 | 0.0015 | 0.156771 | 0.439873 |
| 30 | 1.4373 | 85.45 | 0.00015 | 0.204782 | 0.490506 |

The `loss_t` column of the curve file is empty for every epoch, as the softmax engine
writes no triplet term.

Protocol tables at epoch 30, one feature extraction, embedder `60164a99`, from
`eval_summary.json`. Percentages, as the trainer prints them.

| protocol | mAP | rank-1 | rank-5 | rank-10 | queries scored |
|---|---|---|---|---|---|
| General: euclidean distance on raw features, the library's standard protocol | 20.48 | 49.05 | 61.71 | 70.89 | 316 |
| Normalised cosine, the comparison point for the camera debias | 23.83 | 49.05 | 64.24 | 69.62 | 316 |
| Camera debias: per-camera mean subtracted, renormalised, cosine | 34.89 | 52.85 | 73.73 | 76.58 | 316 |
| Same-clothes approximation, in-house definition, not the published protocol | 29.31 | 51.74 | 65.62 | 75.35 | 288 |

Self-check on the general protocol: the library's own evaluation of the same tensors gives
mAP 20.48 and rank-1 49.05, a difference of 3.1e-8 from the in-house computation. The
camera debias saw 11 cameras across the pooled query and gallery features; camera 301
holds a single tracklet, so its mean is the feature itself and the debias zeroes it. The
same-clothes approximation scores 288 of the 316 queries, the rest having no same-outfit
gallery match once different-outfit tracklets of the same person are removed as neutral.

Against the bar. General-protocol mAP 20.48 is 18.5 points below the external same-clothes
floor of 39.0 and 6.6 points below the published general-protocol figure of 27.1. The
camera-debiased 34.89 and the in-house same-clothes approximation of 29.31 are reported for
what they are and are not comparable to either published figure, because their definitions
differ from the published protocols. The Phase A expectation stated above, rank-1 0.70 to
0.85 and mAP 0.55 to 0.70, was a projection and was not met. What the run does establish:
the recipe trains without collapse from an ImageNet-only initialisation on this corpus, the
embedding is a working one with rank-1 at 49 percent over a gallery of 1,438 tracklets, and
the camera debias is worth 14 points of mAP over raw features, which is the largest single
effect on this page and a property of the test cameras rather than of the weights.

Parity gate: passed, cosine similarity 1.000000 on the pod and on a development Mac.

The runs behind this result. Three runs were launched on 2026-09-07, all from the same
initialisation, seed and data.

1. 16:32Z, the library's triplet engine at 16 tracklets per batch with no classifier-only
   phase. Collapsed by epoch 3 to a constant feature: triplet loss at its margin of 0.3,
   cross-entropy at ln(104), accuracy at chance, and an epoch-10 evaluation of mAP 5.6
   with rank-1, rank-5, rank-10 and rank-20 all at 6.3. Stopped at epoch 11.
2. 16:48Z, the triplet engine with five classifier-only epochs and 32 tracklets per batch.
   The frozen phase held, with accuracy at 18 percent on ImageNet features; once the trunk
   opened the collapse resumed, the triplet loss falling from 5.76 towards the margin over
   epochs 6 to 9 while the cross-entropy rose from 4.23 to 4.57 and accuracy fell from
   12.5 to 7.9 percent. Stopped at epoch 8.
3. 16:57Z, the softmax recipe above. Cross-entropy 4.59 to 3.55 over the five
   classifier-only epochs, 3.58 to 1.98 over epochs 6 to 20 with the trunk open, and 1.82
   to 1.44 over epochs 21 to 30 after the learning-rate step, ending at 85.4 percent
   training accuracy. This is the run of record.

The mechanism of the two collapses is the one the configuration file records: with hard
mining on unnormalised features and clothes-change positives that sit far from their
anchor, the triplet term is satisfied cheapest by shrinking every feature to one point, and
the cross-entropy term did not hold it. Both collapsed runs are kept on the training host
with their logs; no number from either is quoted anywhere except as the collapse evidence
above.

The export. The in-process export at the end of the run raised
`torch.onnx.OnnxExporterError` because the training environment lacked the `onnx` package:
the locked environment sync removes packages installed by hand, and the environment gate
checked for onnxruntime but not for onnx. The trainer had already written the evaluation
summary, by design, so the measurement survived. The graph was exported six minutes later
from `model-best.pth.tar`, the same epoch-30 weights, through the trainer's own export
function and its parity gate, and the summary's `onnx.exported_from` block records the
checkpoint digest, the time and the reason. TRAINING.md now lists `onnx` among the packages
to install on the training host.

Preconditions, both satisfied: the three corpus archive sha256 values are recorded with the
registry status `available`, and the parity gate passed.

### Rerun A2 on a 60-epoch schedule, measured 2026-09-07: not promoted

Status: measured. The internal promotion bar is not met; the model of record stands.

Declared before it ran in `research/reid-rerun-a2-protocol.md`: a clone of the recipe above
with exactly two deltas, 60 epochs and the learning-rate step at epoch 40, everything else
identical, one run, the final-epoch weights as the candidate, promotion only if the
general-protocol mAP exceeds 0.204782 and the export clears the parity gate.

Result. Final-epoch general-protocol mAP 0.188552, rank-1 0.458861, rank-5 0.658228,
rank-10 0.705696 over the same 316 queries; parity gate passed at cosine similarity
1.000000. Checkpoint sha256 `78ba6b55f2884ed04855e7ccda389fa82069787088c88c3478b7b37ccfa64145` (epoch 60, 36,233,078 bytes), exported graph sha256
`151eda30c44fe61bf3c0abf2f25c0feb99c2b10270629595593eacfad7a0e5d8` (8,729,490 bytes, opset 17, 512-dimensional). The graph is not shipped beside
this page; `model.onnx` remains embedder `60164a99`. The curve is
`curves/reid_mevid_a2_long.csv`, and the six evaluation points are 0.137423, 0.156771,
0.130376, 0.130791, 0.183268 and 0.188552 at epochs 10 to 60. The first two reproduce the
run of record to six decimals, the two runs being identical until their schedules diverge
after epoch 20. Under the other protocols at epoch 60: normalised cosine 22.23, camera
debias 34.25 with rank-1 54.11, in-house same-clothes approximation 27.24, each below the
run of record except the camera-debiased rank-1.

Reading. Holding the high learning rate for 40 epochs instead of 20 cost test mAP while
training accuracy rose, 91.1 percent at the end against 85.4 for the run of record, and the
later step did not recover the difference: the extra high-rate epochs fit the 104 training
identities more tightly and generalised worse. A longer schedule is not this recipe's lever.
The training host was rebuilt from nothing for this run, so the rerun also re-verified the
three corpus archives against the registry digests and the ImageNet initialisation
`fe2d63f9157c28a4a8d8ca29bec12d5b2988ac0346d712025789ea9174968e79` against the run of record's summary.

## Figures

- `final_models/reid/validation_curves.png`: the general-protocol test mAP at every
  evaluated epoch for the run of record (three points over 30 epochs) and for rerun A2
  (six points over 60 epochs). The external same-clothes floor of 0.39 is drawn as a
  dashed reference line and labelled external; it is not a measured comparator.
- `final_models/reid/champion_vs_baseline.png`: the run of record against rerun A2, the
  one in-house comparator this lane has, with the same reference line and the claim under
  the axis.

Both are rendered by `scripts/analysis/plot_final_curves.py --lanes reid` from
`curves/reid_mevid_a1.csv` and `curves/reid_mevid_a2_long.csv`.

## Detail

- Run artefacts beside this page: `eval_summary.json` (identity, recipe, the three
  protocols, the export record) and `curves/reid_mevid_a1.csv` (the seven columns, one row
  per epoch). The trained checkpoint itself is not in the repository; its digest is above.
  The `caveat` string in the summary was normalised on 2026-09-07 to the wording the
  current trainer emits; every metric value, digest and timestamp in the file is as the
  run wrote it.
- The rerun of this recipe on a longer schedule, declared before it ran and measured
  above, with its bars, looks, promotion rule and outcome: `research/reid-rerun-a2-protocol.md`.
- Corpus survey, the licence position on every candidate corpus, the choice of floor and
  the freedom-to-operate exposure: `research/reid-topview-corpus.md`.
- Licence review of the re-identification data landscape: `research/reid-data-licences.md`,
  with the per-family reports beside it under `research/reid-licence-*.md`.
- Why the synthetic mesh corpus was not used for training:
  `research/reid-smartspaces-mesh-audit.md`.
- Weight source refusals: `docs/LICENCE_ALLOWLIST.md` and
  `scripts/guards/check_reid_weight_sources.py`.
- Corpus entry, licence text and the recorded archive digests: `configs/datasets/mevid.yaml`,
  summarised in `DATASETS.md`.
- Measurement rules and the re-identification scope rule: `POLICY.md`.
